"""股票池网格研究(2026-10-20): 风格配比 × 池子规模, 连续换池(持仓延续)。

- 风格: 逆向/趋势/热门 三桶配额; 池子规模 20/30/40/60;
- 换池: 季度; 粘性(保留合格在池) + 换池时把末期持仓并入池子(策略内置);
- 回测: 连续一次跑完(2021-04~2026-10), 生产参数(min_hold=0), 无前视。

用法: python pool_grid_study.py [--workers 5]
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


def select(d, quota, top):
    """按配额选池(局部实现: 主题 cap 随规模缩放)。"""
    cap_theme = max(4, int(round(top * 0.30)))
    picked, tn, inn = [], {}, {}
    taken = {k: 0 for k in quota}
    for b in ["contrarian", "trend", "hot"]:
        for _, r in d[d.bucket == b].sort_values("score", ascending=False).iterrows():
            if taken[b] >= quota[b] or len(picked) >= top:
                break
            t, i = r.theme, r.industry
            if tn.get(t, 0) >= cap_theme or inn.get(i, 0) >= 3:
                continue
            picked.append(r.symbol)
            taken[b] += 1
            tn[t] = tn.get(t, 0) + 1
            inn[i] = inn.get(i, 0) + 1
    if len(picked) < top:
        for _, r in d.sort_values("score", ascending=False).iterrows():
            if len(picked) >= top:
                break
            if r.symbol in picked:
                continue
            t, i = r.theme, r.industry
            if tn.get(t, 0) >= cap_theme or inn.get(i, 0) >= 3:
                continue
            picked.append(r.symbol)
            tn[t] = tn.get(t, 0) + 1
            inn[i] = inn.get(i, 0) + 1
    return d[d.symbol.isin(picked)]


def build_schedule(panel, quota, top):
    import pool_variants as pv
    prev, sched = [], {}
    for dt, g in panel.groupby("date"):
        d = pv.eligible(g.copy())
        elig = set(d.symbol)
        keep = [s for s in prev if s in elig][:top]
        target = select(d[~d.symbol.isin(keep)], quota, top - len(keep))
        members = keep + list(target.symbol)
        members = members[:top]
        sched[dt] = members
        prev = members
    return sched


def _read(sym):
    for root in (CATALOG, FALLBACK):
        f = os.path.join(root, sym, "data.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            return d.sort_index()
    return None


def _job(args):
    name, quota, top, sched = args
    from backtest import _run_pool, _map_strategy_params
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
    return {"variant": name, "size": top, "universe": len(dm),
            "total": total, "ann": (1 + total) ** (252 / len(ret)) - 1,
            "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0,
            "mdd": (eq / eq.cummax() - 1).min(),
            "buys": int((f.side.str.lower() == "buy").sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    panel = pd.read_parquet(os.path.join(HERE, "pool_panel.parquet"))
    variants = []
    ratios = [("R0 逆向16/趋势24/热门0", (16, 24, 0)),
              ("R1 14/20/6", (14, 20, 6)),
              ("R2 12/18/10", (12, 18, 10)),
              ("R3 10/16/14", (10, 16, 14)),
              ("R4 8/14/18", (8, 14, 18)),
              ("R5 0/32/8", (0, 32, 8)),
              ("R6 20/20/0", (20, 20, 0))]
    for name, q in ratios:
        variants.append((name, dict(zip(["contrarian", "trend", "hot"], q)), 40))
    for n in [20, 30, 60]:
        c = int(round(0.4 * n))
        variants.append((f"S{n} 规模{n}(40/60/0)", {"contrarian": c,
                         "trend": n - c, "hot": 0}, n))
    jobs = []
    for name, q, top in variants:
        sched = build_schedule(panel, q, top)
        jobs.append((name, q, top, sched))
        print(f"  {name}: {len(sched)} 期, 宇宙 "
              f"{len({s for v in sched.values() for s in v})}", flush=True)
    with mp.Pool(args.workers, initializer=_init) as p:
        rows = p.map(_job, jobs)
    df = pd.DataFrame(rows).sort_values("variant")
    df.to_csv(os.path.join(HERE, "pool_grid_study.csv"), index=False,
              encoding="utf-8-sig")
    show = df.assign(累计=(df.total * 100).round(1), 年化=(df.ann * 100).round(1),
                     夏普=df.sharpe.round(2), 回撤=(df.mdd * 100).round(1))
    print(show[["variant", "size", "universe", "累计", "年化", "夏普", "回撤", "buys"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
