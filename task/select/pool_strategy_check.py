"""策略层验证(2026-10-20): 选池规则在策略回测里是否恶化。

A. quarterly: 2021Q2~2026Q2 每季末选池 → 下一季用生产参数回测(窗口前 250 根预热,
   消除冷启动), 拼接季度收益。臂: rule40 / random40 / current25(事后选择, 仅参考)。
B. fixed: 2021-04-01~2026-10-19 固定池连续回测。臂同 A(2021-03-31 选池, 无前视;
   current25 仍是事后池, 只作参考)。

用法:
    python pool_strategy_check.py --mode quarterly --workers 6
    python pool_strategy_check.py --mode fixed --workers 3
"""

import argparse
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

CFG = None


def _init():
    global CFG
    logging.disable(logging.CRITICAL)
    import yaml
    CFG = yaml.safe_load(open(os.path.join(KALMAN, "stocks.yaml"), encoding="utf-8"))


def _params(prime=None):
    from backtest import _map_strategy_params
    pool = CFG["stock"]
    p = {**_map_strategy_params(CFG.get("strategy", {})),
         "initial_cash": float(pool["initial_cash"]),
         "max_positions": int(pool["max_positions"]),
         "single_position_pct": float(pool["single_position_pct"])}
    if prime is not None:
        p["prime_enabled"] = True
        p["prime_data"] = prime
    return p


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


def _arm_pool(arm, date):
    if arm == "current25":
        import yaml
        cfg = yaml.safe_load(open(os.path.join(KALMAN, "stocks.yaml"), encoding="utf-8"))
        return [str(s["symbol"]).zfill(6) for s in cfg["watchlist"]["stocks"]]
    import pool_rule as pr
    if arm == "rule40":
        return list(pr.build_pool(date, top=40)[0].symbol)
    if arm == "random40":
        d = pr.features(date)
        d = d[~d.symbol.isin(pr.load_red_lines())]
        d = d[d.amount20 >= pr.MIN_AMOUNT]
        d = d[d.vol <= d.vol.quantile(pr.VOL_PCT)]
        key = int(date[:4]) * 100 + int(date[5:7])
        return list(d.sample(min(40, len(d)), random_state=key).symbol)
    raise ValueError(arm)


def _quarter_job(args):
    arm, qend, qstart, qnext = args
    from backtest import _run_pool
    syms = _arm_pool(arm, qend)
    dm = {s: df for s in syms if (df := _window(s, qstart, qnext)) is not None}
    if len(dm) < 5:
        return {"arm": arm, "date": qend, "ret": np.nan, "n": len(dm), "buys": 0}
    prime = {s: _prime(s, (pd.Timestamp(qstart) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
             for s in dm}
    prime = {k: v for k, v in prime.items() if v is not None}
    r = _run_pool(dm, list(dm), float(_params()["initial_cash"]),
                  _params(prime), False, None)
    eq = r.equity_curve
    f = r.orders_df
    f = f[f.status == "filled"]
    return {"arm": arm, "date": qend, "ret": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "n": len(dm), "buys": int((f.side.str.lower() == "buy").sum())}


def _fixed_job(args):
    arm, date = args
    from backtest import _run_pool
    syms = _arm_pool(arm, date)
    dm = {s: df for s in syms if (df := _window(s, "2021-04-01", "2026-10-19")) is not None}
    r = _run_pool(dm, list(dm), float(_params()["initial_cash"]),
                  _params(), False, None)
    eq = r.equity_curve
    dr = eq.pct_change().dropna()
    f = r.orders_df
    f = f[f.status == "filled"]
    return {"arm": arm, "n": len(dm), "ret": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "sharpe": float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0,
            "mdd": float((eq / eq.cummax() - 1).min()),
            "buys": int((f.side.str.lower() == "buy").sum())}


def run_quarterly(workers):
    qs = list(pd.date_range("2021-03-31", "2026-06-30", freq="QE"))
    jobs = []
    for i, q in enumerate(qs):
        qend = q.strftime("%Y-%m-%d")
        qstart = (q + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        nxt = (qs[i + 1] if i + 1 < len(qs) else q + pd.offsets.QuarterEnd())
        for arm in ("rule40", "random40", "current25"):
            jobs.append((arm, qend, qstart, nxt.strftime("%Y-%m-%d")))
    print(f"季度换池: {len(qs)} 季 × 3 臂 = {len(jobs)} 次, {workers} 进程", flush=True)
    with mp.Pool(workers, initializer=_init) as p:
        rows = p.map(_quarter_job, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pool_strategy_quarterly.csv"), index=False,
              encoding="utf-8-sig")
    print(f"\n  {'arm':>9} {'季数':>4} {'累计':>9} {'年化':>8} {'季度夏普':>8} "
          f"{'回撤':>8} {'均买单':>6}")
    for arm, g in df.groupby("arm"):
        g = g.dropna(subset=["ret"]).sort_values("date")
        eq = (1 + g.ret).cumprod()
        tot = eq.iloc[-1] - 1
        ann = (1 + tot) ** (4 / len(g)) - 1
        print(f"  {arm:>9} {len(g):>4} {tot*100:>8.1f}% {ann*100:>7.1f}% "
              f"{g.ret.mean()/g.ret.std()*np.sqrt(4):>8.2f} "
              f"{(eq/eq.cummax()-1).min()*100:>7.1f}% {g.buys.mean():>6.1f}")


def run_fixed(workers):
    print(f"固定池连续回测: 3 臂, {workers} 进程", flush=True)
    with mp.Pool(workers, initializer=_init) as p:
        rows = p.map(_fixed_job, [("rule40", "2021-03-31"),
                                  ("random40", "2021-03-31"),
                                  ("current25", "2021-03-31")])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pool_strategy_fixed.csv"), index=False,
              encoding="utf-8-sig")
    print(df.assign(ret=lambda x: (x.ret * 100).round(1),
                    sharpe=lambda x: x.sharpe.round(2),
                    mdd=lambda x: (x.mdd * 100).round(1)).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quarterly",
                    choices=["quarterly", "fixed", "both"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.mode in ("quarterly", "both"):
        run_quarterly(args.workers)
    if args.mode in ("fixed", "both"):
        run_fixed(min(args.workers, 3))


if __name__ == "__main__":
    main()
