"""换手率验证(2026-10-19): 最小持仓期能不能把换手压下来?

背景: 实盘 2026-07-17~10-19 买卖额 407 万 / 本金 40 万 = **10.2 倍/季**
(年化约 40 倍), 单笔期望为负。候选手段 `min_hold_bars`(买入后 N 根内
抑制价格回归/速度反转卖出, 仅止损可卖)。

本脚本对组合级回测做同口径计量:
    换手(倍/季) = Σ成交额(买+卖) / 初始资金 / (交易日数/63)

用法:
    python study_turnover.py --pool stock --period full --values 0,5,10
    python study_turnover.py --pool etf   --period full --values 0,10

⚠️ 与 sweep_params.py 相同的两个约束: 走 _map_strategy_params、注入资金池
三参数(single_position_pct 在 stock/etf 段)。
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

PERIODS = {
    "is": ("20200101", "20240831"),
    "oos": ("20240901", "20260918"),
    "full": ("20200101", "20260918"),
}
_DATA = {}


def _init(data_map, cash, base):
    logging.disable(logging.CRITICAL)   # 静默策略日志(否则刷屏且拖慢)
    _DATA["map"], _DATA["cash"], _DATA["base"] = data_map, cash, base


def _run(overrides):
    from backtest import _run_pool
    r = _run_pool(_DATA["map"], list(_DATA["map"]), _DATA["cash"],
                  {**_DATA["base"], **overrides}, False, None)
    eq = r.equity_curve
    dr = eq.pct_change().dropna()
    o = r.orders_df
    f = o[o.status == "filled"]
    value = float(f["filled_value"].sum())
    days = len(eq)
    return {
        "ret": float(eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "sharpe": float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0,
        "mdd": float((eq / eq.cummax() - 1).min()) * 100,
        "buys": int((f.side.str.lower() == "buy").sum()),
        "sells": int((f.side.str.lower() == "sell").sum()),
        "turnover_q": value / _DATA["cash"] / max(days, 1) * 63,
        "days": days,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="min_hold_bars")
    ap.add_argument("--values", default="0,5,10")
    ap.add_argument("--period", default="full")
    ap.add_argument("--pool", default="stock", choices=["stock", "etf"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    import yaml
    from backtest import _map_strategy_params
    from data_utils import load_watchlist_data

    start, end = PERIODS[args.period]
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "stocks.yaml"), encoding="utf-8"))
    sm, em, _, _ = load_watchlist_data(cfg, start, end, quiet=True)
    dm = sm if args.pool == "stock" else em
    pool = cfg[args.pool]
    cash = float(pool["initial_cash"])
    base = {**_map_strategy_params(cfg.get("strategy", {})),
            "initial_cash": cash,
            "max_positions": int(pool["max_positions"]),
            "single_position_pct": float(pool["single_position_pct"])}
    values = [int(v) for v in args.values.split(",")]

    print(f"扫描 {args.param}={values} [{args.pool} {len(dm)} 只 | "
          f"{args.period} {start}~{end} | {args.workers} 进程]", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(dm, cash, base)) as p:
        res = p.map(_run, [{args.param: v} for v in values])
    print(f"\n  {'min_hold':>8} {'收益%':>8} {'夏普':>6} {'回撤%':>7} "
          f"{'买单':>5} {'卖单':>5} {'换手倍/季':>9}")
    for v, m in zip(values, res):
        print(f"  {v:>8d} {m['ret']:>8.1f} {m['sharpe']:>6.2f} {m['mdd']:>7.1f} "
              f"{m['buys']:>5d} {m['sells']:>5d} {m['turnover_q']:>9.1f}")
    print(f"  ({time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    main()
