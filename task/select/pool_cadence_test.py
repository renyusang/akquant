"""换池周期对策略收益的影响(2026-10-20)。

新方案 = rule40 粘性池 + min_hold=20，只改"多久重选一次":
quarterly(3M) / semiannual(6M) / annual(12M) / never(2021-03 建池不动)。

做法: 每个周期用"保留合格在池 + 规则补位"得到池子, 段内用生产参数回测
(窗口前 250 根预热, min_hold=20), 拼接段收益。
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
END = "2026-10-19"
CFG = None
MIN_HOLD = 20


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


def _window(sym, start, end):
    d = _read(sym)
    if d is None:
        return None
    w = d.loc[start:end]
    return w.reset_index().rename(columns={"index": "date"}) if len(w) >= 15 else None


def _prime(sym, before, bars=250):
    d = _read(sym)
    if d is None:
        return None
    w = d.loc[:before].tail(bars)
    return w if len(w) >= 60 else None


def _base():
    from backtest import _map_strategy_params
    pool = CFG["stock"]
    return {**_map_strategy_params(CFG.get("strategy", {})),
            "initial_cash": float(pool["initial_cash"]),
            "max_positions": int(pool["max_positions"]),
            "single_position_pct": float(pool["single_position_pct"]),
            "min_hold_bars": MIN_HOLD}


def _eligible(date):
    import pool_rule as pr
    d = pr.features(date)
    d = d[~d.symbol.isin(pr.load_red_lines())]
    d = d[d.amount20 >= pr.MIN_AMOUNT]
    d = d[d.vol <= d.vol.quantile(pr.VOL_PCT)]
    return set(d.symbol)


def build_sticky(cadence):
    """返回 [(date, members)] 与平均换手。"""
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
    prev, out, churn = [], [], []
    for dt in dates:
        elig = _eligible(dt)
        keep = [s for s in prev if s in elig][:40]
        add = [s for s in targets[dt] if s not in keep][:40 - len(keep)]
        members = keep + add
        if prev:
            churn.append(len(set(prev) - set(members)))
        out.append((dt, members))
        prev = members
    return out, (np.mean(churn) if churn else 0.0)


def _job(args):
    start, end, members = args
    from backtest import _run_pool
    dm = {s: df for s in members if (df := _window(s, start, end)) is not None}
    if len(dm) < 5:
        return None
    prime = {s: _prime(s, (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
             for s in dm}
    prime = {k: v for k, v in prime.items() if v is not None}
    p = _base()
    p["prime_enabled"] = True
    p["prime_data"] = prime
    r = _run_pool(dm, list(dm), float(p["initial_cash"]), p, False, None)
    eq = r.equity_curve
    if isinstance(eq, pd.DataFrame):
        eq = eq.set_index("date")["equity"]
    return eq


def run_cadence(cadence):
    sched, churn = build_sticky(cadence)
    jobs = []
    for i, (dt, members) in enumerate(sched):
        start = (pd.Timestamp(dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = ((pd.Timestamp(sched[i + 1][0]) if i + 1 < len(sched)
                else pd.Timestamp(END)).strftime("%Y-%m-%d"))
        jobs.append((start, end, members))
    with mp.Pool(6, initializer=_init) as p:
        eqs = p.map(_job, jobs)
    eq = pd.concat([e for e in eqs if e is not None]).sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    ret = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (1 + total) ** (252 / len(ret)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    return {"cadence": cadence, "segments": len(jobs), "total": total,
            "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "churn": churn, "days": len(ret)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="quarterly,semiannual,annual")
    args = ap.parse_args()
    rows = []
    for c in args.cadences.split(","):
        r = run_cadence(c)
        rows.append(r)
        print(f"{c:11s} 段数{r['segments']:>3} 累计{r['total']*100:>8.1f}% "
              f"年化{r['ann']*100:>6.1f}% 夏普{r['sharpe']:>5.2f} "
              f"回撤{r['mdd']*100:>6.1f}% 换手{r['churn']:>5.1f}/次", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "pool_cadence.csv"),
                              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
