"""浮盈回吐摸底 + 退出规则单笔口径重放(2026-09-28 立项)。

计划文档: `task/kalman/plan_profit_protection.md`(§1 现象量化 / §2 归因 / §4 候选规则对照)。

**口径(必须遵守, 否则结论无效)**:
1. 只用**收盘价**决策 —— 本系统收盘算信号、次日开盘成交, 没有任何盘中动作路径;
2. 新规则的成交价 = **触发次日开盘价**(与 `fill_policy` = NextOpen 一致);
3. 新规则**只允许让出场早于原出场**(不模拟资金再投/替代效应) → 结果是**单笔口径**,
   不能直接当作组合收益; 组合裁决见计划文档 §6 门槛;
4. 峰值与决策均从**成交次日起**(成交当日 on_bar 已执行过, 引擎在当天看不到持仓) ——
   校准证明该口径与实际出场 bar 一致率 >= 90%。

**自检(不过则拒绝出结论, 参照 `sweep_params.py` 的先例)**:
- 校准: 用现行 `trailing_stop_pct`(默认 0.04) 重放, 与实际出场 bar 完全一致率需 >= 90%;
- 一致性: 校准行的 Δ金额需落在 |Δ| <= 1% × 基线净收益(现行规则已在生产, 重放应为近零差)。

用法:
    python study_giveback.py --report                    # §1 现象量化(口径对比 + 分解)
    python study_giveback.py --calibrate                 # 校准自检(退出码非 0 = 口径失效)
    python study_giveback.py --sweep                     # §4 候选规则单笔对照表
    python study_giveback.py --report --sweep --out /tmp/gb.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Optional

import numpy as np
import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRADES = os.path.join(TASK_DIR, "portfolio_trades.csv")
DEFAULT_CATALOG = os.path.join(TASK_DIR, ".catalog")

# 触发函数签名: (peak_profit, cur_profit, entry_price, peak_price) -> 触发价 or None
Rule = Callable[[float, float, float, float], Optional[float]]


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def load_context(trades_path: str, catalog: str, verbose: bool = True) -> pd.DataFrame:
    """把每笔已完成交易与日线拼接: 记录入场/出场 bar 下标、收盘/开盘数组。"""
    trades = pd.read_csv(trades_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    bars_cache: dict = {}
    rows = []
    skipped = 0
    for r in trades.itertuples():
        sym = str(r.symbol).strip().zfill(6)
        if sym not in bars_cache:
            p = os.path.join(catalog, sym, "data.parquet")
            bars_cache[sym] = pd.read_parquet(p) if os.path.exists(p) else None
        bars = bars_cache[sym]
        if bars is None:
            skipped += 1
            continue
        idx = bars.index
        i0 = idx.get_indexer([pd.Timestamp(r.entry_time.tz_localize(None).normalize())])[0]
        i1 = idx.get_indexer([pd.Timestamp(r.exit_time.tz_localize(None).normalize())])[0]
        if i0 < 0 or i1 < 0 or i1 <= i0:
            skipped += 1
            continue
        close = bars["close"].values
        entry = float(r.entry_price)
        # 口径定义(三个, 报告里必须写明用的是哪个):
        #  - peak_all: 收盘价全窗口 i0..i1(含入场 bar 与出场 bar 的收盘) —— 上界口径
        #  - peak_dec: **可决策口径** i0+1..i1-1 —— 引擎能看到持仓、且能在该收盘做决策的区间
        #              (成交当日 on_bar 已跑过看不到持仓; 出场 bar 的收盘发生在卖出成交之后)
        #    报告与结论一律用 peak_dec; peak_all 仅作敏感性对照。
        dec_lo, dec_hi = i0 + 1, i1
        peak_all = (close[i0:i1 + 1].max() / entry - 1.0) * 100.0
        if dec_hi > dec_lo:
            peak_dec = (close[dec_lo:dec_hi].max() / entry - 1.0) * 100.0
        else:
            peak_dec = np.nan
        rows.append(
            dict(
                symbol=sym, pool=r.pool, year=int(r.exit_time.year), i0=int(i0), i1=int(i1),
                entry=entry, exit=float(r.exit_price), qty=float(r.quantity or 0.0),
                ret=float(r.return_pct), net=float(r.net_pnl), pnl=float(r.pnl),
                mfe=float(r.mfe), mae=float(r.mae), dur=int(r.duration_bars),
                close=close, open=bars["open"].values,
                peak_all=peak_all, peak_dec=peak_dec,
            )
        )
    df = pd.DataFrame(rows)
    df["gap"] = df["peak_dec"] - df["ret"]
    if verbose:
        print(f"载入 {len(df)} 笔(跳过 {skipped}) | {trades_path}")
    return df


# ----------------------------------------------------------------------
# 重放
# ----------------------------------------------------------------------
def replay(df: pd.DataFrame, rule: Rule, name: str = "") -> dict:
    """逐笔重放一条退出规则。返回 {触发数, Δ金额, 明细 DataFrame}。"""
    hits = []
    for r in df.itertuples():
        close, open_, entry = r.close, r.open, r.entry
        peak = entry
        new_px = None
        for i in range(r.i0 + 1, r.i1):            # 决策: 成交次日起, 严格早于原出场 bar
            peak = max(peak, close[i])
            lvl = rule(peak / entry - 1.0, close[i] / entry - 1.0, entry, peak)
            if lvl is not None and close[i] <= lvl:
                new_px = open_[i + 1]              # 成交: 次日开盘
                break
        if new_px is None:
            continue
        hits.append(dict(symbol=r.symbol, pool=r.pool, year=r.year, entry=entry,
                         act_ret=r.ret, new_ret=(new_px / entry - 1.0) * 100.0,
                         peak_profit=(peak / entry - 1.0) * 100.0,
                         act_exit=r.exit, new_exit=new_px, qty=r.qty,
                         d_price=new_px - r.exit, d_yuan=(new_px - r.exit) * r.qty))
    det = pd.DataFrame(hits)
    if det.empty:
        return dict(name=name, n=0, d_yuan=0.0, better=0, worse=0, detail=det)
    return dict(name=name, n=len(det), d_yuan=float(det.d_yuan.sum()),
                better=int((det.d_price > 0).sum()), worse=int((det.d_price <= 0).sum()),
                detail=det)


# ----------------------------------------------------------------------
# 规则族(每条都是"收紧"或"移动保护", 只可能让出场更早)
# ----------------------------------------------------------------------
def rule_trail(base: float) -> Rule:
    """现行: 峰值回撤 base 卖出(峰值初始 = 入场价)。"""
    return lambda pp, ret, ep, peak: peak * (1.0 - base)


def rule_breakeven(arm: float, buf: float = 0.002) -> Rule:
    """保本止损: 峰值浮盈 >= arm 后, 跌破 成本×(1+buf) 离场。"""
    return lambda pp, ret, ep, peak: ep * (1.0 + buf) if pp >= arm else None


def rule_scaled_trail(base: float, floor: float, g: float) -> Rule:
    """随浮盈收紧的移动止盈: 回撤容忍 = min(base, max(floor, g×峰值浮盈))。"""
    return lambda pp, ret, ep, peak: peak * (1.0 - min(base, max(floor, g * pp)))


def rule_hard_stop(x: float) -> Rule:
    """固定硬止损(替代现行 -4%): 跌破 成本×(1-x) 离场。"""
    return lambda pp, ret, ep, peak: ep * (1.0 - x)


def rule_struct(base: float, floor_stop: float) -> Rule:
    """结构止损: 取 min(峰值回撤线, 固定 floor_stop) —— 未走出浮盈的单子更早离场。"""
    return lambda pp, ret, ep, peak: min(peak * (1.0 - base), ep * (1.0 - floor_stop))


CANDIDATES: list[tuple[str, Rule]] = [
    ("现状 trail 4%", rule_trail(0.04)),
    ("保本 arm=1% +0.2%", rule_breakeven(0.01)),
    ("保本 arm=2% +0.2%", rule_breakeven(0.02)),
    ("保本 arm=3% +0.2%", rule_breakeven(0.03)),
    ("收紧 trail min(4%,max(2.5%,0.25×峰值))", rule_scaled_trail(0.04, 0.025, 0.25)),
    ("收紧 trail min(4%,max(2%,0.5×峰值))", rule_scaled_trail(0.04, 0.02, 0.5)),
    ("结构止损 min(峰值-4%, 入场-2%)", rule_struct(0.04, 0.02)),
    ("硬止损 -2%", rule_hard_stop(0.02)),
    ("硬止损 -3%", rule_hard_stop(0.03)),
]


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def report_observation(df: pd.DataFrame) -> None:
    """§1: 现象量化 —— 三个口径并列, 结论一律用"可决策口径 peak_dec"。"""
    n = len(df)
    losing = df[df.net < 0]
    valid = df[df.peak_dec.notna()]
    print("\n" + "=" * 78)
    print("§1 现象量化(基线生产交易; 结论用「可决策口径」= 引擎能看到持仓并做决策的收盘区间)")
    print("=" * 78)
    net_sum, gross_sum = df.net.sum(), df.pnl.sum()
    win_rate = (df.net > 0).mean() * 100
    mfe_hit = (df.mfe > 0.5).mean() * 100
    mfe_lose = ((df.mfe > 0.5) & (df.net < 0)).mean() * 100
    print(f"  样本: {n} 笔(其中 {n - len(valid)} 笔无可决策 bar, 已从浮盈统计剔除) "
          f"| 净收益合计 ¥{net_sum:,.0f}(毛 ¥{gross_sum:,.0f}) | 胜率 {win_rate:.1f}%")
    print(f"  ① 极值口径(日内 high, akquant mfe): 曾浮盈>=0.5% {mfe_hit:.1f}% "
          f"| 曾浮盈且净亏 {mfe_lose:.1f}%  ← 含不可执行盘中价, 仅对照")
    print(f"  ② 收盘全窗口口径(含入场/出场 bar 收盘): 曾浮盈>=0.5% {(df.peak_all > 0.5).mean() * 100:.1f}% "
          f"| 曾浮盈且净亏 {((df.peak_all > 0.5) & (df.net < 0)).mean() * 100:.1f}%  ← 上界口径")
    print(f"  ③ **可决策口径**(本次结论口径): 曾浮盈>=0.5% {(valid.peak_dec > 0.5).mean() * 100:.1f}% "
          f"| 曾浮盈且净亏 {((valid.peak_dec > 0.5) & (valid.net < 0)).mean() * 100:.1f}%")
    print()
    for t in (0.5, 1, 2, 3, 5):
        sub = valid[(valid.peak_dec > t) & (valid.net < 0)]
        print(f"    浮盈曾>={t:<4}%: {(valid.peak_dec > t).mean() * 100:5.1f}%"
              f" | 其中净亏 {len(sub):4d} 笔({len(sub) / n * 100:4.1f}%)"
              f" 回吐合计 {sub.gap.sum():7.0f}pp 中位 {sub.gap.median():4.2f}pp")
    never = valid[valid.peak_dec <= 0.5]
    nv = never[never.net < 0]
    nv_share = nv.net.sum() / losing.net.sum() * 100 if len(losing) else 0.0
    print(f"\n  从未浮盈(可决策口径峰值<=+0.5%): {len(never)} 笔({len(never) / n * 100:.1f}%), "
          f"其中亏损 {len(nv)} 笔, 净亏 ¥{nv.net.sum():,.0f} "
          f"= 总净亏 {nv_share:.1f}%(亏损单中位 {nv.ret.median():.2f}%)")
    big = valid[(valid.peak_dec > 4) & (valid.net < 0)]
    print(f"  浮盈>4% 后亏损(真回吐): {len(big)} 笔({len(big) / n * 100:.1f}%), "
          f"净亏 ¥{big.net.sum():,.0f} = 总净亏的 {big.net.sum() / losing.net.sum() * 100:.1f}%")
    print("\n  按池(曾浮盈且净亏占比): " + " | ".join(
        f"{p}: {((g.peak_dec > 0.5) & (g.net < 0)).mean() * 100:.1f}%(n={len(g)})"
        for p, g in valid.groupby("pool")))


def calibrate(df: pd.DataFrame, trail: float, tol: float = 0.01) -> bool:
    """校准自检: 现行规则重放必须与实际出场 bar 高度一致, 且 Δ金额近似为零。"""
    print("\n" + "=" * 78)
    print(f"校准自检(trailing_stop_pct={trail})")
    print("=" * 78)
    exact = total = 0
    for r in df.itertuples():
        close, open_, entry = r.close, r.open, r.entry
        peak = entry
        pred = None
        for i in range(r.i0 + 1, r.i1):
            peak = max(peak, close[i])
            if close[i] <= peak * (1.0 - trail):
                pred = i + 1
                break
        if pred is None:
            continue
        total += 1
        if pred == r.i1:
            exact += 1
    rate = exact / max(total, 1) * 100.0
    res = replay(df, rule_trail(trail), "calibrate")
    net = float(df.net.sum())
    ratio = abs(res["d_yuan"]) / max(abs(net), 1.0) * 100.0
    ok = rate >= 90.0 and ratio <= tol * 100.0
    print(f"  规则触发 {total}/{len(df)} 笔 | 与实际出场 bar 完全一致 {exact} = {rate:.1f}% (门槛 >=90%)")
    print(f"  重放 Δ金额 ¥{res['d_yuan']:,.0f} = 基线净收益的 {ratio:.2f}% (门槛 <=1%)")
    print(f"  → {'✅ 口径通过, 可用于候选筛选' if ok else '❌ 口径失效, 结论不可用'}")
    return ok


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    """§4: 候选规则单笔对照表。"""
    print("\n" + "=" * 78)
    print("§4 候选退出规则(单笔口径; 正值 = 比现状卖得更好, 但未计资金再投)")
    print("=" * 78)
    print(f"  {'规则':<40} {'触发':>5} {'优':>5} {'劣':>5} {'Δ¥合计':>12} {'股票池':>11} {'ETF池':>10}")
    rows = []
    for name, rule in CANDIDATES:
        res = replay(df, rule, name)
        det = res["detail"]
        st = det[det.pool == "stock"].d_yuan.sum() if res["n"] else 0.0
        et = det[det.pool == "etf"].d_yuan.sum() if res["n"] else 0.0
        print(f"  {name:<40} {res['n']:>5} {res['better']:>5} {res['worse']:>5} "
              f"{res['d_yuan']:>12,.0f} {st:>11,.0f} {et:>10,.0f}")
        rows.append(dict(rule=name, n=res["n"], better=res["better"], worse=res["worse"],
                         d_yuan=res["d_yuan"], d_stock=st, d_etf=et))
    print("\n  读法: 现行 trail 4% 行是**校准行**(规则已在生产, Δ 应近似 0); "
          "其余行的 Δ 为负 = 收紧退出是负期望。")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="浮盈回吐摸底 + 退出规则单笔口径重放")
    ap.add_argument("--trades", default=DEFAULT_TRADES)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--trail", type=float, default=0.04, help="现行峰值回撤阈值(校准用)")
    ap.add_argument("--report", action="store_true", help="§1 现象量化")
    ap.add_argument("--calibrate", action="store_true", help="校准自检")
    ap.add_argument("--sweep", action="store_true", help="§4 候选规则对照")
    ap.add_argument("--out", default=None, help="把 §4 对照表写入 CSV")
    args = ap.parse_args()
    if not (args.report or args.calibrate or args.sweep):
        args.report = args.calibrate = args.sweep = True

    df = load_context(args.trades, args.catalog)
    if df.empty:
        print("❌ 无可用交易", file=sys.stderr)
        return 1

    failed = False
    if args.calibrate or args.sweep:
        if not calibrate(df, args.trail):
            failed = True
    if args.report:
        report_observation(df)
    table = None
    if args.sweep:
        if failed:
            print("\n❌ 校准未通过 → **拒绝输出候选对照结论**(先修口径)", file=sys.stderr)
            return 2
        table = sweep(df)
    if args.out is not None:
        if table is None:
            table = sweep(df)
        table.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n对照表已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
