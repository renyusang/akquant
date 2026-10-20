"""验证"换池后旧持仓占用名额多久" (2026-10-20)。

对 M-3M / Q-3M 两种方案各跑一次连续回测(保留订单), 重建每日持仓, 统计:
- 平均持仓只数 / 其中"已离池但仍持有"(legacy) 的只数
- 离池到卖出的天数(中位/均值), 以及从未卖出的比例
- 买入中"新建仓"与"加仓"的比例
"""

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
START, END = "2022-04-01", "2026-10-19"
QUOTA = {"contrarian": 14, "trend": 20, "hot": 6}
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


def _schedule(freq, look):
    import pool_freq_lookback as g
    panel = pd.read_parquet(os.path.join(HERE, "pool_monthly_panel.parquet"))
    panel = panel[(panel.date >= "2022-04-01") & (panel.date <= "2026-09-30")]
    dates = sorted(panel.date.unique())
    return g.build(panel, freq, look, dates)[0]


def _run(args):
    freq, look = args
    from backtest import _run_pool, _map_strategy_params
    sched = _schedule(freq, look)
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
    f = r.orders_df
    f = f[f.status == "filled"].copy()
    f["t"] = pd.to_datetime(f["updated_at"]).dt.tz_localize(None).dt.normalize()
    f["side"] = f.side.str.lower()
    return {"freq": freq, "look": look, "sched": sched,
            "fills": f[["t", "symbol", "side", "filled_quantity"]].to_dict("records")}


def _active_pool(sched, day):
    keys = sorted(sched)
    i = np.searchsorted(keys, day.strftime("%Y-%m-%d"), side="right") - 1
    return set(sched[keys[i]]) if i >= 0 else set()


def analyze(res):
    fills = sorted(res["fills"], key=lambda r: r["t"])
    pos, held_since = {}, {}
    days = pd.date_range("2022-04-04", "2026-10-19", freq="B")
    legacy_counts, held_counts, inpool_counts = [], [], []
    left_events, closed_days = [], []
    buys_new, buys_add = 0, 0
    prev_held = set()
    for day in days:
        todays = [r for r in fills if r["t"] == day]
        for r in todays:
            sym, q = r["symbol"], float(r["filled_quantity"])
            if r["side"] == "buy":
                if sym in pos:
                    buys_add += 1
                else:
                    buys_new += 1
                    held_since[sym] = day
                pos[sym] = pos.get(sym, 0) + q
            else:
                pos[sym] = pos.get(sym, 0) - q
                if pos.get(sym, 0) <= 1e-6:
                    pos.pop(sym, None)
                    if sym in held_since:
                        closed_days.append((day - held_since.pop(sym)).days)
        pool = _active_pool(res["sched"], day)
        held = set(pos)
        # 离池事件: 昨天在池且持有, 今天不在池
        if day > days[0]:
            prev_pool = _active_pool(res["sched"], day - pd.Timedelta(days=1))
            for s in held & prev_pool:
                if s not in pool:
                    left_events.append((s, day))
        held_counts.append(len(held))
        inpool_counts.append(len(held & pool))
        legacy_counts.append(len(held) - len(held & pool))
    # 离池到卖出的天数
    free_days = []
    for s, d0 in left_events:
        nxt = [r["t"] for r in fills if r["symbol"] == s and r["side"] == "sell" and r["t"] > d0]
        if nxt:
            free_days.append((min(nxt) - d0).days)
    return {
        "avg_held": np.mean(held_counts), "avg_inpool": np.mean(inpool_counts),
        "avg_legacy": np.mean(legacy_counts), "max_legacy": max(legacy_counts),
        "left_events": len(left_events),
        "free_median": np.median(free_days) if free_days else np.nan,
        "free_mean": np.mean(free_days) if free_days else np.nan,
        "never_free": len(left_events) - len(free_days),
        "hold_median": np.median(closed_days) if closed_days else np.nan,
        "buys_new": buys_new, "buys_add": buys_add,
    }


def main():
    with mp.Pool(2, initializer=_init) as p:
        res = p.map(_run, [("monthly", 3), ("quarterly", 3)])
    rows = []
    for r in res:
        m = analyze(r)
        m.update({"variant": f"{'M' if r['freq']=='monthly' else 'Q'}-{r['look']}M"})
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pool_slot_diagnostic.csv"), index=False,
              encoding="utf-8-sig")
    print(df.assign(**{c: df[c].round(2) for c in
                       ["avg_held", "avg_inpool", "avg_legacy", "free_median", "free_mean"]})
          .to_string(index=False))


if __name__ == "__main__":
    main()
