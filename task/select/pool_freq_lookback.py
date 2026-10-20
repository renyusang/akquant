"""换池频率 × 选股参考窗口 的二维研究(2026-10-20)。

- 频率: 月度 / 季度(3/6/9/12 月末)
- 参考窗口: 3 / 6 / 12 / 24 个月(用于分桶的涨幅)
- 固定: 池子 40 只、R1 配比 14/20/6、主题≤30%、申万≤3、粘性、持仓并入
- 回测: 连续(2022-04 ~ 2026-10), 生产参数

用法: python pool_freq_lookback.py [--workers 4]
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
START, END = "2022-04-01", "2026-10-19"
QUOTA = {"contrarian": 14, "trend": 20, "hot": 6}
TOP = 40
CFG = None


def _init():
    global CFG
    logging.disable(logging.CRITICAL)
    import yaml
    CFG = yaml.safe_load(open(os.path.join(KALMAN, "stocks.yaml"), encoding="utf-8"))


def _eligible(g, col):
    d = g[(~g.red) & (g.amount20 >= 1e8)].copy()
    d = d[d.vol <= d.vol.quantile(0.90)]
    d = d.dropna(subset=[col])
    d["pct"] = d[col].rank(pct=True) * 100
    d["bucket"] = np.select([d.pct < 40, d.pct < 80],
                            ["contrarian", "trend"], "hot")
    d["score"] = np.where(d.bucket == "contrarian", -d.vol,
                          d[col] / d.vol.replace(0, np.nan))
    return d


def _select(d, top):
    cap_theme = max(4, int(round(top * 0.30)))
    picked, tn, inn = [], {}, {}
    taken = {k: 0 for k in QUOTA}
    for b in ["contrarian", "trend", "hot"]:
        for _, r in d[d.bucket == b].sort_values("score", ascending=False).iterrows():
            if taken[b] >= QUOTA[b] or len(picked) >= top:
                break
            t, i = r.theme, r.industry
            if tn.get(t, 0) >= cap_theme or inn.get(i, 0) >= 3:
                continue
            picked.append(r.symbol); taken[b] += 1
            tn[t] = tn.get(t, 0) + 1; inn[i] = inn.get(i, 0) + 1
    for _, r in d.sort_values("score", ascending=False).iterrows():
        if len(picked) >= top:
            break
        if r.symbol in picked:
            continue
        t, i = r.theme, r.industry
        if tn.get(t, 0) >= cap_theme or inn.get(i, 0) >= 3:
            continue
        picked.append(r.symbol); tn[t] = tn.get(t, 0) + 1; inn[i] = inn.get(i, 0) + 1
    return d[d.symbol.isin(picked)]


def build(panel, freq, look, dates_all):
    if freq == "monthly":
        dates = dates_all
    else:
        dates = [d for d in dates_all if pd.Timestamp(d).month in (3, 6, 9, 12)]
    prev, sched, churn = [], {}, []
    for dt in dates:
        g = panel[panel.date == dt]
        d = _eligible(g, f"ret{look}")
        elig = set(d.symbol)
        keep = [s for s in prev if s in elig][:TOP]
        target = _select(d[~d.symbol.isin(keep)], TOP - len(keep))
        members = (keep + list(target.symbol))[:TOP]
        if prev:
            churn.append(len(set(prev) - set(members)))
        sched[dt] = members
        prev = members
    return sched, (np.mean(churn) if churn else 0.0)


def _read(sym):
    for root in (CATALOG, FALLBACK):
        f = os.path.join(root, sym, "data.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            return d.sort_index()
    return None


def _job(args):
    name, sched = args
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
    return {"variant": name, "universe": len(dm), "total": total,
            "ann": (1 + total) ** (252 / len(ret)) - 1,
            "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0,
            "mdd": (eq / eq.cummax() - 1).min(),
            "buys": int((f.side.str.lower() == "buy").sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    panel = pd.read_parquet(os.path.join(HERE, "pool_monthly_panel.parquet"))
    dates_all = sorted(d for d in panel.date.unique() if d >= "2022-04-01")
    panel = panel[panel.date >= "2022-04-01"]
    jobs, meta = [], {}
    for freq in ["monthly", "quarterly"]:
        for look in [3, 6, 12, 24]:
            name = f"{'M' if freq=='monthly' else 'Q'}-{look}M"
            sched, churn = build(panel, freq, look, dates_all)
            # 等权选择层: 周期内按月末复利(季度=3 个月), 池子固定不再平衡
            keys = list(sched)
            seg = []
            for i, dt in enumerate(keys):
                end = keys[i + 1] if i + 1 < len(keys) else "9999"
                months = [d for d in dates_all if dt <= d < end]
                r = 1.0
                for m in months:
                    x = panel[(panel.date == m) & (panel.symbol.isin(sched[dt]))].fwd
                    if len(x) and x.notna().any():
                        r *= (1 + float(x.mean()))
                seg.append(r - 1)
            seg = np.array(seg)
            ew = np.prod(1 + seg) - 1
            meta[name] = {"periods": len(sched), "churn": churn,
                          "ew_total": ew,
                          "ew_ann": (1 + ew) ** ((12 if freq == "monthly" else 4)
                                                 / max(len(seg), 1)) - 1,
                          "ew_worst": seg.min()}
            jobs.append((name, sched))
            print(f"  {name}: {len(sched)} 期, 宇宙 "
                  f"{len({s for v in sched.values() for s in v})}, 换手 {churn:.1f}/期",
                  flush=True)
    with mp.Pool(args.workers, initializer=_init) as p:
        rows = p.map(_job, jobs)
    for r in rows:
        r.update(meta[r["variant"]])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pool_freq_lookback.csv"), index=False,
              encoding="utf-8-sig")
    show = df.assign(等权累计=(df.ew_total*100).round(1), 等权年化=(df.ew_ann*100).round(1),
                     策略累计=(df.total*100).round(1), 策略年化=(df.ann*100).round(1),
                     夏普=df.sharpe.round(2), 回撤=(df.mdd*100).round(1),
                     换手=df.churn.round(1))
    print(show[["variant","periods","churn","等权累计","等权年化","策略累计","策略年化","夏普","回撤"]].to_string(index=False))


if __name__ == "__main__":
    main()
