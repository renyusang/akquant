"""超参数敏感性扫描框架(2026-09-22 建立, 超参数研究四批的通用工具)。

对单个策略参数做组合级扫描, 默认按**样本内/样本外分割**输出(避免全样本
内挑参数): IS 2020-01~2024-08 调参, OOS 2024-09~至今 验证。

用法:
    python sweep_params.py --param entry_threshold --values 0.015,0.02,0.025
    python sweep_params.py --param downtrend_entry_threshold --values 0.03,0.04 --period full
    python sweep_params.py --param trailing_stop_pct --values 0.03,0.04,0.05 --pool etf

⚠️ 两个必须遵守的约束(踩坑记录, 2026-09-21):
1. **必须走 `_map_strategy_params`**: yaml 名与策略参数名不同
   (trend_bear_pct→trend_bear_position_pct / downtrend_entry→
   downtrend_entry_threshold / trend_confirm_bars→trend_filter_confirm_bars),
   直接把 yaml 名传给 `_run_pool` 会被 KalmanStrategy 当未知 kwarg **静默忽略**
   —— 表现为"所有取值结果完全相同"。
2. **必须注入资金池三参数**(initial_cash / max_positions / single_position_pct):
   `single_position_pct` 在 `stocks.yaml` 的 stock 段而非 strategy 段, 漏掉会
   落到策略默认 **1.0 = 单只满仓**, 整批结果失真。
本脚本内置自检: 所有取值结果相同 → 直接报错退出(而非静默给出错误结论)。

判读约定(仓库门槛, 见 kalman.md):
- 收益改善 **且** 夏普不恶化 才考虑采纳; 同时看**年度分解**(是否单年依赖)
  与 **ETF 池交叉验证**(跨池不一致时不采纳), 组合口径按股票 30 万 : ETF 10 万。
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

PERIODS = {
    "is": ("20200101", "20240831"),      # 样本内(调参)
    "oos": ("20240901", "20260918"),     # 样本外(验证)
    "full": ("20200101", "20260918"),    # 全区间
}
_DATA = {}


def _init(data_map, cash, base):
    _DATA["map"], _DATA["cash"], _DATA["base"] = data_map, cash, base


def _run(overrides):
    from backtest import _run_pool
    import numpy as np
    r = _run_pool(_DATA["map"], list(_DATA["map"]), _DATA["cash"],
                  {**_DATA["base"], **overrides}, False, None)
    eq = r.equity_curve
    dr = eq.pct_change().dropna()
    o = r.orders_df
    f = o[o.status == "filled"]
    return {
        "ret": float(eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "sharpe": float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0,
        "mdd": float((eq / eq.cummax() - 1).min()) * 100,
        "buys": int(len(f[f.side.str.lower() == "buy"])),
    }


def sweep(name, values, period, pool="stock", workers=6, config="stocks.yaml"):
    import yaml
    from backtest import _map_strategy_params
    from data_utils import load_watchlist_data

    start, end = PERIODS[period]
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    sm, em, _, _ = load_watchlist_data(cfg, start, end, quiet=True)
    dm = sm if pool == "stock" else em
    pool_cfg = cfg[pool]
    cash = float(pool_cfg["initial_cash"])
    # ⚠️ 约束1+2: 走 _map_strategy_params 并注入资金池三参数
    base = {**_map_strategy_params(cfg.get("strategy", {})),
            "initial_cash": cash,
            "max_positions": int(pool_cfg["max_positions"]),
            "single_position_pct": float(pool_cfg["single_position_pct"])}
    print(f"\n{'=' * 74}\n  扫描 {name} = {values}  [{pool} 池 {len(dm)} 只 | "
          f"{period} {start}~{end} | 生产值见 stocks.yaml]\n{'=' * 74}", flush=True)

    t0 = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(dm, cash, base)) as p:
        res = p.map(_run, [{name: v} for v in values])
    # 自检: 取值结果全同 = 参数覆盖未生效(名字写错/未注入池参数)
    if len({m["ret"] for m in res}) == 1 and len(values) > 1:
        print("  ❌ 所有取值结果完全相同 → 参数覆盖未生效!\n"
              "     检查: 是否用了 yaml 名而非策略参数名? 资金池参数是否注入?",
              flush=True)
        sys.exit(1)

    print(f"  {'值':>12} {'收益%':>9} {'夏普':>7} {'回撤%':>9} {'买单':>6}", flush=True)
    for v, m in zip(values, res):
        print(f"  {v:>12g} {m['ret']:>9.1f} {m['sharpe']:>7.2f} "
              f"{m['mdd']:>9.1f} {m['buys']:>6}", flush=True)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="组合级超参数敏感性扫描")
    ap.add_argument("--param", required=True, help="策略参数名(非 yaml 名, 见文件头说明)")
    ap.add_argument("--values", required=True, help="逗号分隔的取值")
    ap.add_argument("--period", default="is,oos", help="is / oos / full, 逗号分隔")
    ap.add_argument("--pool", default="stock", choices=["stock", "etf"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--config", default=os.path.join(TASK_DIR, "stocks.yaml"))
    args = ap.parse_args()
    values = [float(v) for v in args.values.split(",")]
    for period in args.period.split(","):
        sweep(args.param, values, period.strip(), args.pool, args.workers, args.config)


if __name__ == "__main__":
    main()
