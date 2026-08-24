"""科学选股筛选脚本: 双窗口样本外验证。

对候选池每只股票跑卡尔曼策略双窗口回测:
  - 筛选期(in):  默认 2020-01-01 ~ 2024-12-31
  - 验证期(out): 默认 2025-01-01 ~ 2026-12-31
筛选"筛选期盈利且验证期也盈利"的股票(样本外稳健, 防过拟合),
替代人工拍脑袋选股。策略参数取自 stocks.yaml(与实盘一致)。

用法:
  python screen_pool.py --symbols-file candidates.csv        # 妙想选股器导出
  python screen_pool.py --symbols 002594,600900,603259       # 命令行指定
  python screen_pool.py --symbols-file candidates.csv \
      --in-start 20200101 --in-end 20241231 --out-start 20250101 --out-end 20261231

输出:
  screen_pool_result.csv  每只股票双窗口指标 + in_watchlist 标记
  推荐列表打印到终端(双窗口盈利 + 交易频率适中 + 回撤可控)
"""

import argparse
import contextlib
import io
import logging
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalman"))

logging.disable(logging.CRITICAL)

KALMAN_DIR = os.path.join(os.path.dirname(__file__), "..", "kalman")
CONFIG_FILE = os.path.join(KALMAN_DIR, "stocks.yaml")
CATALOG_DIR = os.path.join(KALMAN_DIR, ".catalog")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "screen_pool_result.csv")

IN_WINDOW = ("20200101", "20241231")
OUT_WINDOW = ("20250101", "20261231")

# 推荐过滤阈值(依据四象限研究实证: 低频交易股全池大幅恶化 -32.5pp)
MIN_IN_NTRADES = 30    # 筛选期至少 30 笔(交易太少=样本不足/信号不适配)
MAX_IN_NTRADES = 300   # 筛选期最多 300 笔(交易过频=磨损)
MIN_OUT_NTRADES = 15   # 验证期至少交易过
MAX_OUT_MDD = 40.0     # 验证期最大回撤上限(%)


# =============================================================================
# 候选池解析
# =============================================================================
def is_etf(symbol: str) -> bool:
    """ETF 判断: 51/15/58/56 开头(与实盘一致)。"""
    return str(symbol).zfill(6).startswith(("51", "15", "58", "56"))


def parse_candidates(symbols_file: str | None,
                     symbols: str | None) -> list[dict]:
    """解析候选池: CSV(code,name 列) 或命令行逗号分隔, 合并去重。"""
    cands: dict[str, dict] = {}
    if symbols_file and os.path.exists(symbols_file):
        df = pd.read_csv(symbols_file, dtype={"code": str})
        for _, r in df.iterrows():
            code = str(r["code"]).strip().zfill(6)
            if code and code != "nan":
                cands[code] = {
                    "symbol": code,
                    "name": str(r.get("name", code)).strip(),
                    "asset_type": "etf" if is_etf(code) else "stock",
                }
    if symbols:
        for s in str(symbols).split(","):
            code = s.strip().zfill(6)
            if code and code not in cands:
                cands[code] = {
                    "symbol": code,
                    "name": code,
                    "asset_type": "etf" if is_etf(code) else "stock",
                }
    return list(cands.values())


def load_watchlist_symbols(config_path: str = CONFIG_FILE) -> set:
    """读 stocks.yaml 当前 watchlist 的 symbol 集合(用于对比标注)。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    syms = set()
    wl = cfg.get("watchlist", {})
    for item in (wl.get("stocks", []) or []):
        syms.add(str(item["symbol"]).zfill(6))
    for item in (wl.get("etfs", []) or []):
        syms.add(str(item["symbol"]).zfill(6))
    return syms


def parse_risk_flags(risk_file: str) -> dict:
    """解析风险标记 CSV → {code: {"severity": str, "note": str}}。

    CSV 列: code, severity(red/yellow/light), note
    - red:   红线(财务造假/重大行政处罚/立案/退市风险) → 前置剔除
    - yellow: 黄线(会计差错/近期监管函等) → 推荐时标注 ⚠️ 供人工判断
    - light:  轻线(历史问询/个人违规) → 忽略
    分级由人工/AI 基于妙想公司事件查询结果判定, 脚本只做合并过滤。
    """
    flags: dict[str, dict] = {}
    if not risk_file or not os.path.exists(risk_file):
        return flags
    df = pd.read_csv(risk_file, dtype={"code": str})
    for _, r in df.iterrows():
        code = str(r["code"]).strip().zfill(6)
        sev = str(r.get("severity", "")).strip().lower()
        if code and sev in ("red", "yellow", "light"):
            flags[code] = {
                "severity": sev,
                "note": str(r.get("note", "")).strip(),
            }
    return flags


def load_strategy_params(config_path: str = CONFIG_FILE) -> dict:
    """从 stocks.yaml 读策略参数(与实盘一致)并映射为 KalmanStrategy 参数。

    单标的回测口径与 validate_adx 一致: single_position_pct=1.0(满仓评估),
    max_positions=5(组合上限, 单标的回测不触发)。
    """
    import yaml
    from backtest import _map_strategy_params
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = _map_strategy_params(dict(cfg.get("strategy", {})))
    params.setdefault("single_position_pct", 1.0)
    params.setdefault("max_positions", 5)
    params.setdefault("initial_cash", 100000)
    return params


# =============================================================================
# 双窗口回测
# =============================================================================
def _backtest_one(args: tuple) -> dict:
    """对单只股票跑一个窗口的回测, 返回指标 dict(进程 worker)。"""
    symbol, name, asset_type, params, start, end, window = args
    from backtest import run_kalman_backtest

    path = os.path.join(CATALOG_DIR, symbol, "data.parquet")
    if not os.path.exists(path):
        return {"symbol": symbol, "name": name, "ok": False, "window": window,
                "error": "无数据"}
    try:
        df = pd.read_parquet(path).reset_index()
        df["symbol"] = symbol
        if len(df) < 100:
            return {"symbol": symbol, "name": name, "ok": False,
                    "window": window, "error": "数据不足"}
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_kalman_backtest(df, symbol=symbol,
                                    strategy_params=dict(params),
                                    start_time=start, end_time=end,
                                    show_progress=False)
        m = r.metrics
        t = r.trades_df
        winrate = (float((t["return_pct"] > 0).mean() * 100)) if len(t) else 0.0
        return {
            "symbol": symbol, "name": name, "window": window,
            "ok": True,
            "ret": float(m.total_return_pct),
            "sharpe": float(m.sharpe_ratio),
            "mdd": float(m.max_drawdown_pct),
            "ntrades": int(len(t)),
            "winrate": float(winrate),
        }
    except Exception as e:
        return {"symbol": symbol, "name": name, "ok": False, "window": window,
                "error": str(e)}


def run_screening(candidates: list[dict], params: dict,
                  in_window=IN_WINDOW, out_window=OUT_WINDOW,
                  n_procs: int | None = None) -> pd.DataFrame:
    """并行跑双窗口回测, 返回长表(每标的×每窗口一行, 含 window 列)。"""
    tasks = []
    for c in candidates:
        tasks.append((c["symbol"], c["name"], c["asset_type"],
                      params, in_window[0], in_window[1], "in"))
        tasks.append((c["symbol"], c["name"], c["asset_type"],
                      params, out_window[0], out_window[1], "out"))

    rows = []
    pool_n = n_procs or max(1, min(8, os.cpu_count() or 4))
    with mp.Pool(pool_n) as pool:
        for i, res in enumerate(pool.imap_unordered(_backtest_one, tasks,
                                                    chunksize=3)):
            rows.append(res)
            if (i + 1) % 40 == 0:
                print(f"  进度 {i + 1}/{len(tasks)}", flush=True)
    return pd.DataFrame(rows)


def pivot_results(long_df: pd.DataFrame, in_window=IN_WINDOW,
                  out_window=OUT_WINDOW) -> pd.DataFrame:
    """长表 → 宽表: 每只股票一行, in_/out_ 指标列。"""
    if long_df.empty:
        return pd.DataFrame()
    out = long_df.pivot(index=["symbol", "name"], columns="window",
                        values=["ret", "sharpe", "mdd", "ntrades",
                                "winrate"])
    cols = []
    for metric in ["ret", "sharpe", "mdd", "ntrades", "winrate"]:
        for w in ["in", "out"]:
            cols.append(f"{w}_{metric}")
    flat = out.copy()
    flat.columns = cols
    return flat.reset_index()


def rank_and_recommend(df: pd.DataFrame,
                       watchlist: set | None = None) -> pd.DataFrame:
    """过滤+排名: 双窗口盈利, 交易频率适中, 回撤可控; 标注 in_watchlist。"""
    if df.empty:
        return df
    d = df.copy()
    d["in_watchlist"] = d["symbol"].isin(watchlist or set())
    ok = (
        (d["in_ret"] > 0) & (d["out_ret"] > 0)
        & (d["in_ntrades"] >= MIN_IN_NTRADES)
        & (d["in_ntrades"] <= MAX_IN_NTRADES)
        & (d["out_ntrades"] >= MIN_OUT_NTRADES)
        & (d["out_mdd"] < MAX_OUT_MDD)
    )
    d["recommended"] = ok
    d = d.sort_values(["recommended", "out_ret"],
                      ascending=[False, False])
    return d


# =============================================================================
# 输出
# =============================================================================
def print_summary(d: pd.DataFrame, watchlist: set) -> None:
    """打印推荐列表与 watchlist 对比。"""
    print("\n" + "=" * 78)
    print("  推荐列表 (双窗口盈利 + 交易适中 + 回撤可控, 按验证期收益排序)")
    print("=" * 78)
    rec = d[d["recommended"]]
    if not len(rec):
        print("  无推荐(候选池中无双窗口盈利标的)")
    else:
        print(f"{'代码':<8}{'名称':<10}{'筛选期收益':<10}{'验证期收益':<10}"
              f"{'验证期夏普':<10}{'验证期回撤':<10}{'现列表':<6}风险")
        for _, r in rec.iterrows():
            tag = "✅" if r["in_watchlist"] else "🆕"
            risk = ""
            note = str(r.get("risk_note", "") or "")
            if note:
                risk = f"⚠️{note[:14]}"
            print(f"{r['symbol']:<8}{str(r['name'])[:10]:<10}"
                  f"{r['in_ret']:>+7.1f}%  {r['out_ret']:>+7.1f}%  "
                  f"{r['out_sharpe']:>7.2f}   {r['out_mdd']:>6.1f}%   "
                  f"{tag:<6}{risk}")

    print("\n" + "=" * 78)
    print("  当前 watchlist 覆盖情况")
    print("=" * 78)
    wl = d[d["in_watchlist"]]
    passed = wl[wl["recommended"]]
    failed = wl[~wl["recommended"]]
    print(f"  现有列表 {len(wl)} 只: 通过推荐 {len(passed)} 只, "
          f"未通过 {len(failed)} 只")
    if len(failed):
        print("  未通过:")
        for _, r in failed.iterrows():
            reason = []
            if r["in_ret"] <= 0:
                reason.append(f"筛选期{r['in_ret']:+.1f}%")
            if r["out_ret"] <= 0:
                reason.append(f"验证期{r['out_ret']:+.1f}%")
            if not (MIN_IN_NTRADES <= r["in_ntrades"] <= MAX_IN_NTRADES):
                reason.append(f"交易{r['in_ntrades']}笔")
            if r["out_mdd"] >= MAX_OUT_MDD:
                reason.append(f"回撤{r['out_mdd']:.0f}%")
            print(f"    {r['symbol']} {r['name']:<8} "
                  f"{'/'.join(reason) if reason else '其他'}")
    new_rec = rec[~rec["in_watchlist"]]
    if len(new_rec):
        print(f"\n  🆕 新候选入选 {len(new_rec)} 只(建议加入 watchlist):")
        for _, r in new_rec.iterrows():
            print(f"    {r['symbol']} {r['name']}  "
                  f"筛选期{r['in_ret']:+.1f}% → 验证期{r['out_ret']:+.1f}%")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="科学选股: 双窗口样本外验证筛选策略适配股票")
    parser.add_argument("--symbols-file", default=None,
                        help="候选池 CSV(code,name 列, 妙想选股器导出)")
    parser.add_argument("--symbols", default=None,
                        help="逗号分隔的代码列表(调试用)")
    parser.add_argument("--config", default=CONFIG_FILE,
                        help="策略参数来源(默认 stocks.yaml)")
    parser.add_argument("--in-start", default=IN_WINDOW[0])
    parser.add_argument("--in-end", default=IN_WINDOW[1])
    parser.add_argument("--out-start", default=OUT_WINDOW[0])
    parser.add_argument("--out-end", default=OUT_WINDOW[1])
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help="结果 CSV 路径")
    parser.add_argument("--procs", type=int, default=None,
                        help="并行进程数(默认 min(8, cpu))")
    parser.add_argument("--risk-file", default=None,
                        help="风险标记 CSV(code,severity,note): "
                             "red=红线前置剔除, yellow=黄线推荐标注, "
                             "light=忽略")
    args = parser.parse_args()

    candidates = parse_candidates(args.symbols_file, args.symbols)
    if not candidates:
        print("错误: 候选池为空, 请用 --symbols-file 或 --symbols")
        sys.exit(1)
    print(f"候选池: {len(candidates)} 只 "
          f"({sum(1 for c in candidates if c['asset_type']=='stock')} 股 + "
          f"{sum(1 for c in candidates if c['asset_type']=='etf')} ETF)")

    # 红线前置过滤(2026-08-14): 财务造假/重大行政处罚/立案/退市风险,
    # 在数据下载与回测之前剔除, 避免浪费算力且防止污染推荐
    risk = parse_risk_flags(args.risk_file)
    red = [c for c in candidates
           if risk.get(c["symbol"], {}).get("severity") == "red"]
    for c in red:
        note = risk[c["symbol"]]["note"]
        print(f"  ⛔ 前置剔除(红线) {c['symbol']} {c['name']}: {note}")
        c["skip"] = True
    if red:
        print(f"  红线剔除 {len(red)} 只, 剩余 {len(candidates) - len(red)} 只")

    # 数据准备: 缺失的并行下载到 .catalog(2026-08-24 由串行改 8 路线程池;
    # 下载无副作用各写各文件可并行, 失败标的标记 skip 与串行版一致)
    print("数据准备(缺失自动下载到 .catalog, 8 路并行)...")
    from concurrent.futures import ThreadPoolExecutor
    from data_utils import load_cached_data
    missing = [c for c in candidates
               if not os.path.exists(os.path.join(CATALOG_DIR, c["symbol"],
                                                  "data.parquet"))]

    def _prepare(c: dict):
        try:
            load_cached_data(c["symbol"], args.in_start, args.out_end,
                             asset_type=c["asset_type"])
        except Exception as e:
            c["skip"] = True
            return f"  [跳过] {c['symbol']} 数据下载失败: {e}"
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for msg in ex.map(_prepare, missing):
            if msg:
                print(msg)
    if missing:
        print(f"  新下载 {len(missing)} 只")

    valid = [c for c in candidates if not c.get("skip")]
    if not valid:
        print("错误: 无有效候选(数据全部缺失)")
        sys.exit(1)

    params = load_strategy_params(args.config)
    print(f"回测窗口: 筛选期 {args.in_start}-{args.in_end}, "
          f"验证期 {args.out_start}-{args.out_end}")
    print("并行回测...")
    long_df = run_screening(
        valid, params,
        in_window=(args.in_start, args.in_end),
        out_window=(args.out_start, args.out_end),
        n_procs=args.procs)

    wide = pivot_results(long_df)
    if wide.empty:
        print("错误: 无回测结果")
        sys.exit(1)
    watchlist = load_watchlist_symbols(args.config)
    ranked = rank_and_recommend(wide, watchlist)
    # 风险标记列: pre_filtered(红线剔除原因, 已剔除的不会出现在结果,
    # 保留列供审计) / risk_note(黄线, 推荐时标注)
    ranked["pre_filtered"] = ranked["symbol"].map(
        lambda s: risk.get(s, {}).get("note", "")
        if risk.get(s, {}).get("severity") == "red" else "")
    ranked["risk_note"] = ranked["symbol"].map(
        lambda s: risk.get(s, {}).get("note", "")
        if risk.get(s, {}).get("severity") == "yellow" else "")
    ranked.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {args.output} ({len(ranked)} 只)")

    print_summary(ranked, watchlist)


if __name__ == "__main__":
    main()
