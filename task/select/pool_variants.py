"""股票池方案对比(2026-10-19): 在 panel 上秒级回放不同选池规则。

用法: python pool_variants.py
输出: 各方案 22 期等权前向收益的累计/年化/回撤/超额, 以及崩盘季(2026-07~10)表现。
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THEME_CAP, IND_CAP = 8, 3


def eligible(d):
    d = d[~d.red]
    d = d[d.amount20 >= 1e8]
    q = d.vol.quantile(0.90)
    d = d[d.vol <= q].copy()
    d["pct"] = d.ret12.rank(pct=True) * 100
    d["bucket"] = np.select([d.pct < 40, d.pct < 80],
                            ["contrarian", "trend"], "hot")
    d["score"] = np.where(d.bucket == "contrarian", -d.vol,
                          d.ret12 / d.vol.replace(0, np.nan))
    return d


def select(d, quota, caps=True, top=25):
    picked, tn, inn = [], {}, {}
    taken = {k: 0 for k in quota}
    order = ["contrarian", "trend", "hot"]
    for b in order:
        for _, r in d[d.bucket == b].sort_values("score", ascending=False).iterrows():
            if taken[b] >= quota[b] or len(picked) >= top:
                break
            t, i = r.theme, r.industry
            if caps and (tn.get(t, 0) >= THEME_CAP or inn.get(i, 0) >= IND_CAP):
                continue
            picked.append(r.symbol); taken[b] += 1
            tn[t] = tn.get(t, 0) + 1; inn[i] = inn.get(i, 0) + 1
    if len(picked) < top:
        for _, r in d.sort_values("score", ascending=False).iterrows():
            if len(picked) >= top:
                break
            if r.symbol in picked:
                continue
            t, i = r.theme, r.industry
            if caps and (tn.get(t, 0) >= THEME_CAP or inn.get(i, 0) >= IND_CAP):
                continue
            picked.append(r.symbol)
            tn[t] = tn.get(t, 0) + 1; inn[i] = inn.get(i, 0) + 1
    return d[d.symbol.isin(picked)]


def run(panel, name, fn):
    per = []
    for dt, g in panel.groupby("date"):
        d = eligible(g.copy())
        if d.empty:
            continue
        sel = fn(d)
        if sel is None or sel.empty:
            continue
        per.append({"date": dt, "ret": sel.fwd.mean(), "n": len(sel),
                    "hit": (sel.fwd > 0).mean(),
                    "bench": d.fwd.mean()})
    p = pd.DataFrame(per).sort_values("date")
    eq = (1 + p.ret).cumprod()
    total = eq.iloc[-1] - 1
    years = len(p) / 4
    ann = (1 + total) ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = p.ret.mean() / p.ret.std() * np.sqrt(4)
    crash = p[p.date == "2026-06-30"].ret
    return {"方案": name, "累计%": total * 100, "年化%": ann * 100,
            "夏普": sharpe, "回撤%": mdd * 100,
            "胜率%": (p.ret > 0).mean() * 100,
            "超额/季%": (p.ret - p.bench).mean() * 100,
            "崩盘季%": (crash.iloc[0] * 100) if len(crash) else np.nan,
            "均持仓": p.n.mean()}


def main():
    panel = pd.read_parquet(os.path.join(HERE, "pool_panel.parquet"))
    fns = {
        "全样本(合格池)": lambda d: d,
        "随机25": lambda d: d.sample(min(25, len(d)), random_state=42),
        "最热25": lambda d: d.nlargest(25, "ret12"),
        "只有趋势桶(40-80)": lambda d: select(d, {"contrarian": 0, "trend": 25, "hot": 0}),
        "趋势+热门 0/20/5": lambda d: select(d, {"contrarian": 0, "trend": 20, "hot": 5}),
        "趋势+热门 0/25/0": lambda d: select(d, {"contrarian": 0, "trend": 25, "hot": 0}),
        "规则 4/16/5": lambda d: select(d, {"contrarian": 4, "trend": 16, "hot": 5}),
        "规则 8/12/5": lambda d: select(d, {"contrarian": 8, "trend": 12, "hot": 5}),
        "规则 8/12/5 无cap": lambda d: select(d, {"contrarian": 8, "trend": 12, "hot": 5}, caps=False),
        "规则 10/15/0": lambda d: select(d, {"contrarian": 10, "trend": 15, "hot": 0}),
        "随机40": lambda d: d.sample(min(40, len(d)), random_state=42),
        "规则40 13/19/8": lambda d: select(d, {"contrarian": 13, "trend": 19, "hot": 8}, top=40),
        "规则40 16/24/0": lambda d: select(d, {"contrarian": 16, "trend": 24, "hot": 0}, top=40),
        "随机60": lambda d: d.sample(min(60, len(d)), random_state=42),
        "规则60 24/36/0": lambda d: select(d, {"contrarian": 24, "trend": 36, "hot": 0}, top=60),
    }
    rows = [run(panel, k, v) for k, v in fns.items()]
    out = pd.DataFrame(rows).round(2)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "pool_variants.csv"), index=False,
               encoding="utf-8-sig")


if __name__ == "__main__":
    main()
