"""HS300 全池 ADX 阈值稳健性验证。

对 stocks_hs300.yaml 全部 300 只股票分别回测 3 种配置:
  基线(无ADX) / ADX≥20 / ADX≥25
对比收益/夏普/回撤/交易数/胜率,评估 ADX 门控在全池的稳健性。

用法:
  python validate_adx.py                     # 全池 3 配置并行回测
  python validate_adx.py --config my.yaml    # 自定义池
  python validate_adx.py --symbols 002594,600519  # 指定标的(调试)

输出:
  adx_validation.csv  每只股票的 3 配置明细
  汇总统计打印到终端
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
CATALOG_DIR = os.path.join(KALMAN_DIR, ".catalog")

BASE_CFG = dict(
    initial_cash=100000,
    trend_filter_enabled=True,
    entry_threshold=0.02,
    exit_threshold=0.005,
    stop_loss_pct=0.05,
    trend_bear_position_pct=0.3,
    downtrend_entry_threshold=0.03,
    single_position_pct=1.0,
    max_positions=5,
)

CONFIGS = {
    "baseline": dict(BASE_CFG),
    "rc2": dict(BASE_CFG, trend_recover_confirm_bars=2),
    "rc3": dict(BASE_CFG, trend_recover_confirm_bars=3),
    "rc3+atr": dict(BASE_CFG, trend_recover_confirm_bars=3,
                    atr_adaptive_exit_enabled=True, exit_atr_factor=1.0),
}


def _backtest_one(symbol: str, cfg: dict) -> dict:
    """对单只股票跑一次回测,返回指标 dict。失败返回 None 标记。"""
    from backtest import run_kalman_backtest

    path = os.path.join(CATALOG_DIR, symbol, "data.parquet")
    try:
        df = pd.read_parquet(path).reset_index()
    except Exception:
        return {"symbol": symbol, "ok": False, "error": "no-data"}
    df["symbol"] = symbol
    if len(df) < 100:
        return {"symbol": symbol, "ok": False, "error": "too-short"}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_kalman_backtest(
                df, symbol=symbol, strategy_params=dict(cfg), show_progress=False
            )
        m = r.metrics
        t = r.trades_df
        nt = len(t) if t is not None else 0
        win = float((t["return_pct"] > 0).mean() * 100) if nt else 0.0
        return {
            "symbol": symbol,
            "ok": True,
            "ret": float(m.total_return_pct),
            "sharpe": float(m.sharpe_ratio),
            "mdd": float(m.max_drawdown_pct),
            "ntrades": nt,
            "winrate": win,
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)[:80]}


def _worker(args):
    """multiprocessing worker: (symbol, cfg_name) → 结果 dict。"""
    symbol, cfg_name = args
    res = _backtest_one(symbol, CONFIGS[cfg_name])
    res["cfg"] = cfg_name
    return res


def run(symbols: list) -> pd.DataFrame:
    """并行跑全池 3 配置,返回透视表 DataFrame。"""
    # 去重: stocks_hs300.yaml 存在重复 symbol
    symbols = list(dict.fromkeys(symbols))
    tasks = [(s, cfg) for s in symbols for cfg in CONFIGS]
    rows = []
    with mp.Pool(max(1, min(8, os.cpu_count() or 4))) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=3)):
            rows.append(res)
            if (i + 1) % 90 == 0:
                print(f"  进度 {i + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)

    # 透视: 每只股票一行, 3 配置列(先按组合去重防重复)
    ok = df[df["ok"]].drop_duplicates(subset=["symbol", "cfg"])
    piv = ok.pivot(index="symbol", columns="cfg")
    out = pd.DataFrame(index=piv.index)
    for metric, col in [
        ("ret", "ret"),
        ("sharpe", "sharpe"),
        ("mdd", "mdd"),
        ("ntrades", "ntrades"),
        ("winrate", "winrate"),
    ]:
        for cfg in CONFIGS:
            out[f"{metric}_{cfg}"] = piv[col][cfg]
    failed = df[~df["ok"]]
    if len(failed):
        print(f"[警告] {len(failed)} 次回测失败:")
        for _, f in failed.head(10).iterrows():
            print(f"  {f['symbol']} {f['cfg']}: {f.get('error', '?')}")
    return out.sort_index()


def summarize(out: pd.DataFrame) -> None:
    """打印汇总统计。"""
    n = len(out)
    print(f"\n{'=' * 70}")
    print(f"HS300 全池 ADX 阈值稳健性验证: {n} 只股票, 2020-01 ~ 2026-07")
    print(f"{'=' * 70}")

    cfgs = list(CONFIGS)

    print(f"\n--- 收益分布(%) ---")
    hdr = f"{'':>9} {'均值':>8} {'中位数':>8} {'P25':>8} {'P75':>8} {'盈利占比':>8}"
    print(hdr)
    for cfg in cfgs:
        r = out[f"ret_{cfg}"]
        print(
            f"{cfg:>9} {r.mean():8.2f} {r.median():8.2f} {r.quantile(.25):8.2f} "
            f"{r.quantile(.75):8.2f} {(r > 0).mean() * 100:8.1f}%"
        )

    print(f"\n--- 夏普分布 ---")
    for cfg in cfgs:
        s = out[f"sharpe_{cfg}"]
        print(
            f"{cfg:>9} 均值 {s.mean():6.3f}  中位数 {s.median():6.3f}  "
            f"正夏普占比 {(s > 0).mean() * 100:5.1f}%"
        )

    print(f"\n--- 配对对比(每只股票 vs 基线) ---")
    for cfg in cfgs[1:]:
        better = (out[f"ret_{cfg}"] > out["ret_baseline"]).sum()
        worse = (out[f"ret_{cfg}"] < out["ret_baseline"]).sum()
        same = n - better - worse
        med_gain = (out[f"ret_{cfg}"] - out["ret_baseline"]).median()
        mean_gain = (out[f"ret_{cfg}"] - out["ret_baseline"]).mean()
        print(
            f"  {cfg}: 优于基线 {better} 只 / 劣于 {worse} 只 / 持平 {same} 只 | "
            f"收益差 均值 {mean_gain:+.2f}pp / 中位数 {med_gain:+.2f}pp"
        )

    print(f"\n--- 回撤对比(中位数, %) ---")
    for cfg in cfgs:
        print(f"  {cfg}: {out[f'mdd_{cfg}'].median():6.2f}%")

    print(f"\n--- 交易次数(中位数) ---")
    for cfg in cfgs:
        print(f"  {cfg}: {out[f'ntrades_{cfg}'].median():5.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="stocks_hs300.yaml")
    parser.add_argument("--symbols", default=None, help="逗号分隔标的(调试用)")
    parser.add_argument("--output", default="adx_validation.csv")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        import yaml

        with open(os.path.join(KALMAN_DIR, args.config)) as f:
            cfg = yaml.safe_load(f)
        symbols = [s["symbol"] for s in cfg.get("watchlist", {}).get("stocks", [])]

    print(f"标的数: {len(symbols)}, 配置: {list(CONFIGS)}, 回测总数: {len(symbols) * 3}")
    out = run(symbols)
    out.to_csv(os.path.join(os.path.dirname(__file__), args.output),
               encoding="utf-8-sig")
    print(f"[保存] {args.output} ({len(out)} 只)")
    summarize(out)


if __name__ == "__main__":
    main()
