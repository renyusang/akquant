"""换池周期的"纯选池"对照(2026-10-20): 等权持有, 不含策略/交易。

对每个周期: 粘性池在周期内固定, 用季度面板的 fwd 复利到下一个重选日。
用于把"换池周期"的影响与"策略交易"分开。
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_variants as pv


def main():
    panel = pd.read_parquet(os.path.join(HERE, "pool_panel.parquet"))
    targets = json.load(open(os.path.join(HERE, "pool_rule40_quarters.json"),
                             encoding="utf-8"))
    all_dates = sorted(targets)
    groups = {d: g for d, g in panel.groupby("date")}
    rows = []
    for cadence, step in [("quarterly", 1), ("semiannual", 2),
                          ("annual", 4), ("never", 99)]:
        dates = all_dates[::step] if step < 99 else all_dates[:1]
        prev, segs = [], []
        for i, dt in enumerate(dates):
            d = pv.eligible(groups[dt].copy())
            elig = set(d.symbol)
            keep = [s for s in prev if s in elig][:40]
            add = [s for s in targets[dt] if s not in keep][:40 - len(keep)]
            members = keep + add
            end = dates[i + 1] if i + 1 < len(dates) else None
            seg_ret = 1.0
            for q in all_dates:
                if q < dt or (end is not None and q >= end):
                    continue
                g = groups[q]
                m = g[g.symbol.isin(members)]
                if len(m):
                    seg_ret *= (1 + m.fwd.mean())
            segs.append(seg_ret - 1)
            prev = members
        eq = np.prod([1 + r for r in segs]) - 1
        years = sum(1 for _ in all_dates) / 4
        ann = (1 + eq) ** (1 / years) - 1
        rows.append({"cadence": cadence, "segments": len(segs),
                     "total": eq, "ann": ann,
                     "worst_seg": min(segs)})
    out = pd.DataFrame(rows)
    out["total%"] = (out.total * 100).round(1)
    out["ann%"] = (out.ann * 100).round(1)
    out["worst%"] = (out.worst_seg * 100).round(1)
    print(out[["cadence", "segments", "total%", "ann%", "worst%"]].to_string(index=False))
    out.to_csv(os.path.join(HERE, "pool_cadence_equalweight.csv"), index=False,
               encoding="utf-8-sig")


if __name__ == "__main__":
    main()
