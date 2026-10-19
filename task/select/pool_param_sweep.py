"""规则池上的参数复核(2026-10-20, P2)。

背景: 现有参数结论(entry 2% / downtrend 4% / trailing 4% / min_hold 0)都是在
"当前 25 只池"(2026-08 选出, 含选股后视)上得到的。本脚本在**无前视 rule40 池**
上重跑同一套季度换池口径, 看结论是否可迁移。

用法:
    python pool_param_sweep.py --build          # 预计算 22 季 rule40 池(约 3 分钟)
    python pool_param_sweep.py --run --workers 6
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
POOLS = os.path.join(HERE, "pool_rule40_quarters.json")
sys.path.insert(0, HERE)
sys.path.insert(0, KALMAN)

DATES = [d.strftime("%Y-%m-%d") for d in
         pd.date_range("2021-03-31", "2026-06-30", freq="QE")]
SWEEP = {
    "entry_threshold": [0.015, 0.02, 0.025, 0.03],
    "downtrend_entry_threshold": [0.03, 0.04, 0.05],
    "trailing_stop_pct": [0.03, 0.04, 0.05],
    "min_hold_bars": [0, 10, 20, 30, 40],
}
CFG = None


def _init():
    global CFG
    logging.disable(logging.CRITICAL)
    import yaml
    CFG = yaml.safe_load(open(os.path.join(KALMAN, "stocks.yaml"), encoding="utf-8"))


def build_pools():
    import pool_rule as pr
    out = {}
    for d in DATES:
        sel, _ = pr.build_pool(d, top=40)
        out[d] = list(sel.symbol)
        print(f"  {d}: {len(out[d])} 只", flush=True)
    with open(POOLS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"写入 {POOLS}")


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


def _base_params():
    from backtest import _map_strategy_params
    pool = CFG["stock"]
    return {**_map_strategy_params(CFG.get("strategy", {})),
            "initial_cash": float(pool["initial_cash"]),
            "max_positions": int(pool["max_positions"]),
            "single_position_pct": float(pool["single_position_pct"])}


def _job(args):
    param, value, qend, qstart, qnext, syms = args
    from backtest import _run_pool
    dm = {s: df for s in syms if (df := _window(s, qstart, qnext)) is not None}
    if len(dm) < 5:
        return {"param": param, "value": value, "date": qend, "ret": np.nan}
    prime = {s: _prime(s, (pd.Timestamp(qstart) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
             for s in dm}
    prime = {k: v for k, v in prime.items() if v is not None}
    p = _base_params()
    p["prime_enabled"] = True
    p["prime_data"] = prime
    p[param] = value
    r = _run_pool(dm, list(dm), float(p["initial_cash"]), p, False, None)
    eq = r.equity_curve
    return {"param": param, "value": value, "date": qend,
            "ret": float(eq.iloc[-1] / eq.iloc[0] - 1)}


def run(workers):
    pools = json.load(open(POOLS, encoding="utf-8"))
    jobs = []
    for i, qend in enumerate(DATES):
        qstart = (pd.Timestamp(qend) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        qnext = (pd.Timestamp(DATES[i + 1]) if i + 1 < len(DATES)
                 else pd.Timestamp("2026-09-30")).strftime("%Y-%m-%d")
        for param, values in SWEEP.items():
            for v in values:
                jobs.append((param, v, qend, qstart, qnext, pools[qend]))
    print(f"扫参: {len(jobs)} 次回测, {workers} 进程", flush=True)
    with mp.Pool(workers, initializer=_init) as p:
        rows = p.map(_job, jobs, chunksize=2)
    df = pd.DataFrame(rows).dropna(subset=["ret"])
    df.to_csv(os.path.join(HERE, "pool_param_sweep.csv"), index=False,
              encoding="utf-8-sig")
    prod = {"entry_threshold": 0.02, "downtrend_entry_threshold": 0.04,
            "trailing_stop_pct": 0.04, "min_hold_bars": 0}
    for param, g in df.groupby("param"):
        print(f"\n{param}  (生产值 {prod[param]})")
        print(f"  {'值':>8} {'累计':>9} {'年化':>8} {'季度夏普':>8} {'回撤':>8}")
        for v, h in g.groupby("value"):
            h = h.sort_values("date")
            eq = (1 + h.ret).cumprod()
            tot = eq.iloc[-1] - 1
            ann = (1 + tot) ** (4 / len(h)) - 1
            sh = h.ret.mean() / h.ret.std() * np.sqrt(4) if h.ret.std() > 0 else 0
            mark = "  ← 生产" if abs(v - prod[param]) < 1e-9 else ""
            print(f"  {v:>8g} {tot*100:>8.1f}% {ann*100:>7.1f}% {sh:>8.2f} "
                  f"{(eq/eq.cummax()-1).min()*100:>7.1f}%{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.build:
        build_pools()
    if args.run:
        run(args.workers)


if __name__ == "__main__":
    main()
