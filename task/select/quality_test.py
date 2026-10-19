"""质量因子验证(2026-10-20): rule40 加/不加质量排除。

用法: python quality_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_variants as pv
import quality_filter


def run(panel, use_quality):
    per, excluded = [], []
    for dt, g in panel.groupby("date"):
        d = pv.eligible(g.copy())
        bad = quality_filter.bad_symbols(dt) if use_quality else {}
        if use_quality:
            excluded.append(d.symbol.isin(bad).sum())
            d = d[~d.symbol.isin(bad)]
        sel = pv.select(d, {"contrarian": 16, "trend": 24, "hot": 0}, top=40)
        if len(sel) == 0:
            continue
        per.append({"date": dt, "ret": sel.fwd.mean(), "bench": d.fwd.mean()})
    p = pd.DataFrame(per).sort_values("date")
    eq = (1 + p.ret).cumprod()
    tot = eq.iloc[-1] - 1
    return {
        "累计%": tot * 100, "年化%": ((1 + tot) ** (4 / len(p)) - 1) * 100,
        "夏普": p.ret.mean() / p.ret.std() * np.sqrt(4),
        "回撤%": (eq / eq.cummax() - 1).min() * 100,
        "崩盘季%": p[p.date == "2026-06-30"].ret.iloc[0] * 100,
        "平均排除": np.mean(excluded) if excluded else 0,
    }


def main():
    panel = pd.read_parquet(os.path.join(HERE, "pool_panel.parquet"))
    rows = []
    for flag, name in [(False, "rule40"), (True, "rule40+质量")]:
        m = run(panel, flag)
        rows.append({"方案": name, **m})
    out = pd.DataFrame(rows).round(2)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "quality_test.csv"), index=False,
               encoding="utf-8-sig")


if __name__ == "__main__":
    main()
