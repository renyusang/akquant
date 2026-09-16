"""名额争抢优先级摸底(2026-09-17)。

研究问题: 多只标的同日出现买入信号、但持仓名额有限时, 应优先买
"上升趋势"还是"偏离大"的标的?

本脚本产出**描述性统计 + 争抢频率 + 争抢日配对比较**(决策性结论由
组合 A/B 给出, 见 kalman.md)。相对 task/select/study_volume.py 骨架
修正了 5 处口径缺陷(不修则测的不是被测策略):

  1. 入场/出场时点: 骨架用"信号日收盘入场 / 次日收盘出场"(不对称且
     非回测口径) → 改为两端对称的**次根 bar 开盘价** o[i+1], 与
     backtest._run_pool 的 fill_policy(open/next_event/bar_offset=1) 一致
  2. 止损被静默禁用: 骨架 set_position(has_pos, 0.0) 传入场价 0 →
     signal_engine 的 5% 止损门控在 entry_price > 0 上, 永不触发
     (实测 22.5% 的真实交易是止损出场) → 传信号日收盘价(与回测
     strategy.py:419 `_entry_prices[symbol] = close_price` 同口径)
  3. MFI 静默失效: 骨架 update() 未传 volume → _refresh_mfi 恒 None →
     过滤器不生效 → 传 v[i]
  4. 整手可行性未建模: 生产预算 down=¥18,000 / up=¥57,000, 科创板
     200 股整手, 高价股买不起(2026-09-16 的 688234 即此类) → 记录
     feasible 并分列报告
  5. 统计窗口: 骨架 main() 用全池共同窗口(2023-08 起, 由最晚上市的
     301511 决定) → 改为**每只标的自身全历史**, 冷启动给 burn-in 敏感性

统计口径: 交易按日/按标的强聚类且分布重尾 → 组间比较用月度 block
bootstrap 置信区间与按日中对的 t 检验(不用裸 ttest_ind); 同时报告
中位数/胜率; 收益率**扣除往返费**(按实际仓位金额精算佣金/印花/过户)。

用法:
  python study_priority.py                 # 默认股票池(stocks.yaml)
  python study_priority.py --pool etf      # ETF 池
  python study_priority.py --burn-in 150   # 冷启动敏感性
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

CATALOG_DIR = os.path.join(TASK_DIR, ".catalog")
OUT_CSV = os.path.join(TASK_DIR, "priority_signals.csv")

# 手续费(与 backtest._run_pool / portfolio.calc_fee 同口径)
COMMISSION_RATE = 0.0003      # 佣金万3(双边, 最低 5 元)
STAMP_TAX_RATE = 0.001        # 印花税千1(仅卖出)
TRANSFER_FEE_RATE = 0.00001   # 过户费万0.1(双边)
MIN_COMMISSION = 5.0


def _fee(amount: float, side: str) -> float:
    """单边手续费。"""
    fee = max(MIN_COMMISSION, amount * COMMISSION_RATE)
    fee += amount * TRANSFER_FEE_RATE
    if side == "sell":
        fee += amount * STAMP_TAX_RATE
    return fee


def load_engine_params(config_path: str, pool: str) -> dict:
    """从 stocks.yaml 读取策略参数 + 池参数(与实盘/回测同源)。"""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg.get("strategy", {}))
    pool_cfg = cfg.get(pool, {})
    params["initial_cash"] = float(pool_cfg.get("initial_cash", 300000))
    params["single_position_pct"] = float(
        pool_cfg.get("single_position_pct", 0.20))
    params["max_positions"] = int(pool_cfg.get("max_positions", 5))
    return params


def _symbols_from_config(config_path: str, pool: str) -> list:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    wl = cfg.get("watchlist", {})
    key = "stocks" if pool == "stock" else "etfs"
    items = wl.get("stocks", []) + wl.get("etfs", []) if isinstance(wl, list) else wl.get(key, [])
    return list(dict.fromkeys(str(s["symbol"]).zfill(6) for s in items))


def _atr(h, l, c, period=14):
    """ATR(period), 用于波动控制(NATR = ATR/close)。"""
    n = len(c)
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])
    out = np.full(n, np.nan)
    for i in range(period, n):
        out[i] = np.nanmean(tr[i - period + 1: i + 1])
    return out


def scan_symbol(args):
    """扫描单只标的的完整买卖配对, 返回交易记录列表。

    返回字段: symbol, signal_i, signal_date, entry_date, exit_date,
    trend, target_pct, dev, natr, entry_price, exit_price, hold_bars,
    round_trip(扣费后), gross_return, feasible, lot, position_value
    """
    symbol, engine_params, burn_in = args
    path = os.path.join(CATALOG_DIR, symbol, "data.parquet")
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    if len(df) < 100:
        return []
    from signal_engine import SignalEngine

    # .catalog 的 parquet 以 DatetimeIndex 为索引(非 date 列) —— 两种都兼容
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    else:
        dates = pd.to_datetime(df.index).strftime("%Y-%m-%d").to_numpy()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    natr = _atr(h, l, c, 14) / np.where(c > 0, c, np.nan)

    lot = 200 if symbol.startswith("688") else 100
    cash = float(engine_params.get("initial_cash", 300000))
    max_pct = float(engine_params.get("single_position_pct", 0.20))
    eng_kwargs = {k: val for k, val in engine_params.items()
                  if k not in ("initial_cash", "single_position_pct",
                               "max_positions", "lot_based_position")}

    engine = SignalEngine(**eng_kwargs)
    has_pos = False
    ref_price = 0.0        # 引擎侧参考价(信号日收盘, 同 strategy.py:419)
    entry_i = -1
    entry_target = 0.95    # 入场信号的目标仓位(卖出分支的 target 恒 0)
    entry_reason = ""
    rows = []
    for i in range(n):
        ma20 = float(c[max(0, i - 19): i + 1].mean())
        ma20p = float(c[max(0, i - 20): i].mean())
        engine.set_position(has_pos, ref_price)
        r = engine.update(c[i], ma20, ma20p, h[i], l[i], v[i])

        if r["signal"] == "buy" and not has_pos and i >= burn_in and i + 1 < n:
            has_pos = True
            ref_price = c[i]
            entry_i = i
            # 入场信号的目标仓位(0.95 up / 0.30 down)必须在入场 bar 捕获:
            # 卖出分支返回的 target_pct 恒为 0(2026-09-17 修复)
            entry_target = float(r.get("target_pct", 0.95))
            entry_reason = str(r.get("reason", ""))
            # ⚠️ 偏离度/NATR 必须在入场 bar 捕获: 若留到卖出分支再算, 会误用
            # 卖出 bar 的卡尔曼价(2026-09-17 修复 —— 曾使 dev 中位数错到 0.2%)
            _kp = float(r.get("kalman_price") or 0.0)
            entry_dev = (c[i] / _kp - 1.0) if _kp > 0 else np.nan
            entry_natr = (float(natr[i]) if np.isfinite(natr[i]) else np.nan)
            entry_trend = str(r.get("trend", ""))
        elif r["signal"] == "sell" and has_pos:
            exit_i = i + 1
            if exit_i >= n:
                has_pos = False
                continue
            target_pct = entry_target
            capped = round(target_pct * max_pct, 4)
            buy_qty = int(cash * capped / c[entry_i] / lot) * lot
            entry_price = float(o[entry_i + 1])
            exit_price = float(o[exit_i])
            pos_value = buy_qty * entry_price
            if buy_qty > 0:
                fee = (_fee(pos_value, "buy")
                       + _fee(buy_qty * exit_price, "sell"))
                net = (buy_qty * (exit_price - entry_price) - fee) / pos_value
            else:
                fee = 0.0
                net = exit_price / entry_price - 1.0   # 不可行样本不计费
            rows.append({
                "symbol": symbol,
                "signal_date": dates[entry_i],
                "entry_date": dates[entry_i + 1],
                "exit_date": dates[exit_i],
                "trend": entry_trend,
                "target_pct": target_pct,
                "dev": entry_dev,
                "natr": entry_natr,
                "gross_return": exit_price / entry_price - 1.0,
                "round_trip": net,
                "fee": fee,
                "position_value": pos_value,
                "lot": lot,
                "feasible": buy_qty > 0,
                "hold_bars": i - entry_i,
                "entry_reason": entry_reason,
                "exit_reason": str(r.get("reason", "")),
            })
            has_pos = False
            ref_price = 0.0
    return rows


def _worker(args):
    try:
        return scan_symbol(args)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 统计工具(聚类稳健)
# ---------------------------------------------------------------------------
def block_bootstrap_ci(values, dates, n_boot=3000, seed=7):
    """按月分块的 bootstrap 均值置信区间(交易按日/月强聚类)。"""
    if len(values) == 0:
        return (np.nan, np.nan)
    d = pd.to_datetime(pd.Series(dates))
    blocks = d.dt.to_period("M").astype(str).to_numpy()
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        means.append(np.mean(values[idx]))
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def day_paired_test(dates_a, ret_a, dates_b, ret_b, n_boot=3000, seed=11):
    """同日配对差的 bootstrap 检验(按日聚类稳健; 环境无 scipy)。

    两规则收益按日聚合 → 逐日差 → 对"日"重采样。返回
    (平均差, CI下界, CI上界, 配对日数); 差 > 0 表示规则 A 更优。
    """
    da = pd.DataFrame({"date": dates_a, "ret": ret_a}).groupby("date")["ret"].mean()
    db = pd.DataFrame({"date": dates_b, "ret": ret_b}).groupby("date")["ret"].mean()
    joined = pd.concat([da.rename("a"), db.rename("b")], axis=1).dropna()
    if len(joined) < 5:
        return np.nan, np.nan, np.nan, len(joined)
    diff = (joined["a"] - joined["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(diff.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(joined))


def _fmt_pct(x):
    return "  n/a" if x is None or not np.isfinite(x) else f"{x*100:+.2f}%"


def report_group_table(df, title, key_col, bins, labels):
    print(f"\n=== {title} ===")
    print(f"{'分组':<18} {'笔数':>5} {'均值':>8} {'中位':>8} {'胜率':>6} "
          f"{'bootstrap95%CI':>20} {'偏离区间':>14}")
    binned = pd.cut(df[key_col], bins=bins, labels=labels, include_lowest=True)
    for lab in labels:
        sub = df[binned == lab]
        if len(sub) == 0:
            continue
        vals = sub["round_trip"].to_numpy()
        lo, hi = block_bootstrap_ci(vals, sub["signal_date"].to_numpy())
        dev = sub["dev"].dropna()
        dev_rng = (f"[{dev.min()*100:.1f}%,{dev.max()*100:.1f}%]"
                   if len(dev) else "n/a")
        print(f"{lab:<18} {len(sub):>5} {_fmt_pct(vals.mean()):>8} "
              f"{_fmt_pct(np.median(vals)):>8} {(vals>0).mean()*100:>5.1f}% "
              f"[{lo*100:+.2f}%,{hi*100:+.2f}%]".rjust(20) + f" {dev_rng:>14}")


def main():
    ap = argparse.ArgumentParser(description="名额争抢优先级摸底")
    ap.add_argument("--config", default=os.path.join(TASK_DIR, "stocks.yaml"))
    ap.add_argument("--pool", default="stock", choices=["stock", "etf"])
    ap.add_argument("--burn-in", type=int, default=50,
                    help="冷启动 burn-in 根数(默认 50)")
    ap.add_argument("--symbols", default="",
                    help="逗号分隔的标的(调试用, 默认全池)")
    args = ap.parse_args()

    params = load_engine_params(args.config, args.pool)
    symbols = ([s.zfill(6) for s in args.symbols.split(",")]
               if args.symbols else _symbols_from_config(args.config, args.pool))
    print(f"扫描 {len(symbols)} 只({args.pool} 池), burn-in={args.burn_in}, "
          f"预算 up={params['initial_cash']*0.95*params['single_position_pct']:,.0f} "
          f"down={params['initial_cash']*0.30*params['single_position_pct']:,.0f}")

    tasks = [(s, params, args.burn_in) for s in symbols]
    rows = []
    with mp.Pool(max(1, min(8, os.cpu_count() or 4))) as pool:
        for part in pool.imap_unordered(_worker, tasks, chunksize=4):
            rows.extend(part)
    df = pd.DataFrame(rows)
    if df.empty:
        print("无交易记录")
        return
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[保存] {OUT_CSV}: {len(df)} 笔配对交易")

    # ---- 0. 总览 ----
    rt = df["round_trip"]
    print(f"\n全部配对: {len(df)} 笔 | 均值 {_fmt_pct(rt.mean())} | "
          f"中位 {_fmt_pct(np.median(rt))} | 胜率 {(rt>0).mean()*100:.1f}% | "
          f"可行性 {df['feasible'].mean()*100:.1f}% | 平均持仓 {df['hold_bars'].mean():.1f} 根")

    # ---- 1. trend 分组 ----
    print(f"\n=== 按 trend 分组 ===")
    print(f"{'趋势':<18} {'笔数':>5} {'均值':>8} {'中位':>8} {'胜率':>6} "
          f"{'bootstrap95%CI':>20} {'偏离区间':>14}")
    for tr, sub in df.groupby("trend"):
        vals = sub["round_trip"].to_numpy()
        lo, hi = block_bootstrap_ci(vals, sub["signal_date"].to_numpy())
        dev = sub["dev"].dropna()
        print(f"{tr:<18} {len(sub):>5} {_fmt_pct(vals.mean()):>8} "
              f"{_fmt_pct(np.median(vals)):>8} {(vals>0).mean()*100:>5.1f}% "
              f"[{lo*100:+.2f}%,{hi*100:+.2f}%]".rjust(20)
              + f" [{dev.min()*100:.1f}%,{dev.max()*100:.1f}%]".rjust(15))
    d_pt, lo_pt, hi_pt, n_pt = day_paired_test(
        df.loc[df["trend"] == "up", "signal_date"], df.loc[df["trend"] == "up", "round_trip"],
        df.loc[df["trend"] == "down", "signal_date"], df.loc[df["trend"] == "down", "round_trip"])
    print(f"  up vs down 按日配对: 平均差 {_fmt_pct(d_pt)} "
          f"95%CI[{_fmt_pct(lo_pt)},{_fmt_pct(hi_pt)}] (配对日 {n_pt})")

    # ---- 2. 偏离分位 ----
    report_group_table(df, "按偏离度分位", "dev",
                       [-1, 0.03, 0.04, 0.05, 1], ["Q1(<3%)", "Q2(3-4%)", "Q3(4-5%)", "Q4(>5%)"])

    # ---- 3. 2×2 核心表(含样本量与共同支撑) ----
    print("\n=== 2×2: trend × 偏离(描述性) ===")
    hi = df["dev"] >= 0.04
    for tr in ("up", "down"):
        for lvl, mask in (("低(<4%)", ~hi), ("高(≥4%)", hi)):
            sub = df[(df["trend"] == tr) & mask]
            if len(sub) == 0:
                continue
            vals = sub["round_trip"].to_numpy()
            print(f"  {tr:<4} × 偏离{lvl:<8} n={len(sub):>4} "
                  f"均值 {_fmt_pct(vals.mean())} 中位 {_fmt_pct(np.median(vals))} "
                  f"胜率 {(vals>0).mean()*100:.0f}% "
                  f"名义期望 {_fmt_pct(vals.mean()*sub['target_pct'].mean()*params['single_position_pct'])}")
    common = df[(df["dev"] >= 0.02) & (df["dev"] <= 0.06)]
    print(f"  共同支撑(dev∈[2%,6%], n={len(common)}): ", end="")
    for tr, sub in common.groupby("trend"):
        print(f"{tr} {_fmt_pct(sub['round_trip'].mean())}(n={len(sub)}) ", end="")
    print()

    # ---- 4. 争抢频率 + 争抢日配对 ----
    print("\n=== 名额争抢(按信号日聚合, max_positions=%d) ===" % params["max_positions"])
    max_pos = int(params["max_positions"])
    cand = df[df["feasible"]].copy()
    cand = cand.sort_values("signal_date")
    dates = sorted(set(cand["signal_date"]) | set(cand["exit_date"]))
    occupied = 0
    contended_days, picked_a, picked_b = 0, [], []
    n_cand_days = 0
    by_date = {d: g for d, g in cand.groupby("signal_date")}
    exits_by_date = {d: g for d, g in cand.groupby("exit_date")}
    for d in dates:
        occupied -= len(exits_by_date.get(d, []))
        occupied = max(0, occupied)
        g = by_date.get(d)
        if g is None or len(g) == 0:
            continue
        n_cand_days += 1
        free = max(0, max_pos - occupied)
        k = min(free, len(g))
        occupied += k
        if len(g) > free:
            contended_days += 1
            order_a = g.sort_values("dev", ascending=False)
            order_b = g.assign(_u=(g["trend"] == "up").astype(int)).sort_values(
                ["_u", "dev"], ascending=[False, False])
            for _, row in order_a.head(k).iterrows():
                picked_a.append(row)
            for _, row in order_b.head(k).iterrows():
                picked_b.append(row)
    print(f"  有候选日 {n_cand_days} 天, 其中争抢日(候选>空名额) {contended_days} 天 "
          f"({contended_days/max(1,n_cand_days)*100:.1f}%)")
    if picked_a:
        a = pd.DataFrame(picked_a)
        b = pd.DataFrame(picked_b)
        same = len(set(zip(a["signal_date"], a["symbol"]))
                   & set(zip(b["signal_date"], b["symbol"])))
        print(f"  争抢日两规则选中的 (日期,标的) 重合率: "
              f"{same/max(1,len(a))*100:.1f}%")
        d_ab, lo_ab, hi_ab, n_ab = day_paired_test(
            a["signal_date"], a["round_trip"], b["signal_date"], b["round_trip"])
        print(f"  规则A(偏离最大) {_fmt_pct(a['round_trip'].mean())}(n={len(a)}) vs "
              f"规则B(up优先) {_fmt_pct(b['round_trip'].mean())}(n={len(b)})")
        print(f"  规则A vs 规则B 按日配对: 平均差 {_fmt_pct(d_ab)} "
              f"95%CI[{_fmt_pct(lo_ab)},{_fmt_pct(hi_ab)}] (配对日 {n_ab})")


if __name__ == "__main__":
    main()
