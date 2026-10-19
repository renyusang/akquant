"""连续换池回测(2026-10-20): 持仓跨换池延续, 生产参数不变(min_hold=0)。

与分段拼接的区别: 整个区间只跑一次回测, 引擎始终持有并管理已有仓位;
pool_schedule 只限制**新建仓**的候选集, 已持仓标的照常加仓/卖出。
因此换池不会强制清仓, 也不会每段重置为现金。

用法:
    python pool_schedule_backtest.py --cadences quarterly,semiannual,annual,never
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KALMAN = os.path.join(HERE, "..", "kalman")
CATALOG = os.path.join(HERE, ".catalog_pool")
FALLBACK = os.path.join(KALMAN, ".catalog")
sys.path.insert(0, HERE)
sys.path.insert(0, KALMAN)
START, END = "2021-04-01", "2026-10-19"

CFG = None


def _init():
    global CFG
    logging.disable(logging.CRITICAL)
    import yaml
    CFG = yaml.safe_load(open(os.path.join(KALMAN, "stocks.yaml"), encoding="utf-8"))


def _read(sym):
    for root in (CATALOG, FALLBACK):
        f = os.path.join(root, sym, "data.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            return d.sort_index()
    return None


def _eligible(date):
    import pool_rule as pr
    d = pr.features(date)
    d = d[~d.symbol.isin(pr.load_red_lines())]
    d = d[d.amount20 >= pr.MIN_AMOUNT]
    d = d[d.vol <= d.vol.quantile(pr.VOL_PCT)]
    return set(d.symbol)


def build_sticky(cadence):
    targets = json.load(open(os.path.join(HERE, "pool_rule40_quarters.json"),
                             encoding="utf-8"))
    all_dates = sorted(targets)
    if cadence == "quarterly":
        dates = all_dates
    elif cadence == "semiannual":
        dates = all_dates[::2]
    elif cadence == "annual":
        dates = all_dates[::4]
    else:
        dates = all_dates[:1]
    prev, out = [], {}
    for dt in dates:
        elig = _eligible(dt)
        keep = [s for s in prev if s in elig][:40]
        add = [s for s in targets[dt] if s not in keep][:40 - len(keep)]
        prev = keep + add
        out[dt] = prev
    return out


def _run(cadence):
    import pool_schedule_backtest as _m
    from backtest import _run_pool, _map_strategy_params
    sched = build_sticky(cadence)
    union = sorted({s for v in sched.values() for s in v})
    dm = {}
    for s in union:
        d = _read(s)
        if d is None:
            continue
        w = d.loc[START:END]
        if len(w) >= 100:
            dm[s] = w.reset_index().rename(columns={"index": "date"})
    pool = CFG["stock"]
    p = {**_map_strategy_params(CFG.get("strategy", {})),
         "initial_cash": float(pool["initial_cash"]),
         "max_positions": int(pool["max_positions"]),
         "single_position_pct": float(pool["single_position_pct"]),
         "pool_schedule": sched}
    r = _run_pool(dm, list(dm), float(p["initial_cash"]), p, False, None)
    eq = r.equity_curve
    eq = eq.set_index("date")["equity"] if isinstance(eq, pd.DataFrame) else eq
    ret = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    f = r.orders_df
    f = f[f.status == "filled"]
    return {"cadence": cadence, "universe": len(dm), "rebalances": len(sched),
            "total": total, "ann": (1 + total) ** (252 / len(ret)) - 1,
            "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0,
            "mdd": (eq / eq.cummax() - 1).min(),
            "buys": int((f.side.str.lower() == "buy").sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="quarterly,semiannual,annual,never")
    args = ap.parse_args()
    names = args.cadences.split(",")
    with mp.Pool(min(4, len(names)), initializer=_init) as p:
        rows = p.map(_run, names)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pool_schedule_cadence.csv"), index=False,
              encoding="utf-8-sig")
    show = df.assign(累计=(df.total * 100).round(1), 年化=(df.ann * 100).round(1),
                     夏普=df.sharpe.round(2), 回撤=(df.mdd * 100).round(1))
    print(show[["cadence", "universe", "rebalances", "累计", "年化", "夏普", "回撤", "buys"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
