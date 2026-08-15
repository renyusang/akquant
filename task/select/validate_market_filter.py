"""大盘环境过滤全池回测验证。

基线 vs 大盘过滤(弱市禁止新开仓) 全池对比:
  收益/夏普/回撤/交易数/胜率
用法:
  python validate_market_filter.py                      # HS300 全池
  python validate_market_filter.py --symbols 002594,600519  # 小样本调试
输出:
  market_filter_validation.csv
"""

import argparse
import contextlib
import io
import logging
import multiprocessing as mp
import os
import sys

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
    atr_adaptive_exit_enabled=True,
    exit_atr_factor=1.0,
    single_position_pct=1.0,
    max_positions=5,
)


def build_market_state_map() -> dict:
    """沪深300 大盘状态 → {date_str: 强/弱/过渡}。"""
    from data_utils import fetch_hs300
    hs = fetch_hs300("20200101", "20260814")
    ma20 = hs.rolling(20).mean()
    above = hs > ma20
    up = ma20 > ma20.shift(1)
    m = {}
    for d in hs.index:
        ds = str(d)[:10]
        if above.loc[d] and up.loc[d]:
            m[ds] = "强"
        elif not above.loc[d] and not up.loc[d]:
            m[ds] = "弱"
        else:
            m[ds] = "过渡"
    return m


def _backtest_one(args):
    symbol, cfg_name, cfg, market_map = args
    from backtest import run_kalman_backtest

    path = os.path.join(CATALOG_DIR, symbol, "data.parquet")
    if not os.path.exists(path):
        return {"symbol": symbol, "ok": False, "cfg": cfg_name, "error": "无数据"}
    try:
        df = pd.read_parquet(path).reset_index()
        df["symbol"] = symbol
        if len(df) < 100:
            return {"symbol": symbol, "ok": False, "cfg": cfg_name,
                    "error": "数据不足"}
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_kalman_backtest(df, symbol=symbol,
                                    strategy_params=dict(cfg,
                                                         market_state_map=market_map),
                                    show_progress=False)
        m = r.metrics
        t = r.trades_df
        return {
            "symbol": symbol, "cfg": cfg_name, "ok": True,
            "ret": float(m.total_return_pct),
            "sharpe": float(m.sharpe_ratio),
            "mdd": float(m.max_drawdown_pct),
            "ntrades": int(len(t)),
            "winrate": float((t["return_pct"] > 0).mean() * 100)
            if len(t) else 0.0,
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "cfg": cfg_name, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="大盘过滤全池回测验证")
    parser.add_argument("--config", default=os.path.join(
        KALMAN_DIR, "stocks_hs300.yaml"))
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--output", default="market_filter_validation.csv")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",")]
    else:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        symbols = [str(s["symbol"]).zfill(6)
                   for s in cfg["watchlist"]["stocks"]]
    symbols = sorted(set(symbols))  # 配置含重复(12 只), 去重
    print(f"候选: {len(symbols)} 只(去重后)")

    market_map = build_market_state_map()
    configs = {
        "baseline": dict(BASE_CFG),
        "market_filter": dict(BASE_CFG, market_filter_enabled=True),
    }

    tasks = []
    for s in symbols:
        for name, cfg in configs.items():
            tasks.append((s, name, cfg, market_map))

    rows = []
    with mp.Pool(max(1, min(8, os.cpu_count() or 4))) as pool:
        for i, res in enumerate(pool.imap_unordered(_backtest_one, tasks,
                                                    chunksize=3)):
            rows.append(res)
            if (i + 1) % 120 == 0:
                print(f"  进度 {i + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    df = df[df["ok"]]
    out = df.pivot(index="symbol", columns="cfg",
                   values=["ret", "sharpe", "mdd", "ntrades", "winrate"])
    out.columns = [f"{m}_{c}" for m, c in out.columns]
    out.to_csv(args.output, encoding="utf-8-sig")
    print(f"结果已保存: {args.output} ({len(out)} 只)")

    print("\n=== 全池对比 (基线 vs 大盘过滤) ===")
    for metric, name in [("ret", "收益均值"), ("ret", "收益中位"),
                         ("sharpe", "夏普中位"), ("mdd", "回撤中位"),
                         ("ntrades", "交易中位")]:
        b = out[f"{metric}_baseline"].median() if metric != "ret" or True else out[f"{metric}_baseline"].median()
        # 均值/中位分别打印
    print(f"  收益均值: 基线 {out['ret_baseline'].mean():+.1f}% "
          f"vs 过滤 {out['ret_market_filter'].mean():+.1f}%")
    print(f"  收益中位: 基线 {out['ret_baseline'].median():+.1f}% "
          f"vs 过滤 {out['ret_market_filter'].median():+.1f}%")
    print(f"  夏普中位: 基线 {out['sharpe_baseline'].median():+.2f} "
          f"vs 过滤 {out['sharpe_market_filter'].median():+.2f}")
    print(f"  回撤中位: 基线 {out['mdd_baseline'].median():+.1f}% "
          f"vs 过滤 {out['mdd_market_filter'].median():+.1f}%")
    print(f"  交易中位: 基线 {out['ntrades_baseline'].median():.0f} "
          f"vs 过滤 {out['ntrades_market_filter'].median():.0f}")
    better = (out["ret_market_filter"] > out["ret_baseline"]).sum()
    print(f"  过滤优于基线: {better}/{len(out)} "
          f"({better / len(out):.0%})")
    print(f"  收益差: 均值 {(out['ret_market_filter'] - out['ret_baseline']).mean():+.2f}pp "
          f"中位 {(out['ret_market_filter'] - out['ret_baseline']).median():+.2f}pp")


if __name__ == "__main__":
    main()
