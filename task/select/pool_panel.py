"""构建股票池研究面板(2026-10-19): 每季度末的个券特征 + 下一季度收益。

输出 `pool_panel.parquet`: date,symbol,ret12,vol,amount20,pct,bucket,fwd,theme,industry
一次读取 828 只, 供 pool_variants.py 做秒级方案对比。
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KALMAN = os.path.join(HERE, "..", "kalman")
CATALOG = os.path.join(HERE, ".catalog_pool")
FALLBACK = os.path.join(KALMAN, ".catalog")
sys.path.insert(0, HERE)

import pool_rule as pr  # noqa: E402

DATES = pd.date_range("2021-03-31", "2026-06-30", freq="QE")


def main():
    ind = pr.load_industry()
    red = pr.load_red_lines()
    rows = []
    n = 0
    for sym in pr.load_universe():
        f = os.path.join(CATALOG, sym, "data.parquet")
        if not os.path.exists(f):
            f = os.path.join(FALLBACK, sym, "data.parquet")
        if not os.path.exists(f):
            continue
        try:
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()
        except Exception:
            continue
        n += 1
        theme, industry = ind.get(sym, ("", ""))
        close = d["close"].astype(float)
        vol_series = d["volume"].astype(float)
        for i, dt in enumerate(DATES):
            pre = close.loc[:dt]
            if len(pre) < 250:
                continue
            nxt_end = (DATES[i + 1] if i + 1 < len(DATES)
                       else pd.Timestamp("2026-10-19"))
            fwd = close.loc[dt:nxt_end]
            if len(fwd) < 10:
                continue
            c = pre.values[-252:]
            rows.append({
                "date": dt.strftime("%Y-%m-%d"), "symbol": sym,
                "ret12": float(c[-1] / c[0] - 1),
                "vol": float(np.diff(np.log(c)).std() * np.sqrt(252)),
                "amount20": float((close * vol_series).loc[:dt].tail(20).mean()),
                "fwd": float(fwd.iloc[-1] / fwd.iloc[0] - 1),
                "theme": theme or "未映射", "industry": industry or "未映射",
                "red": sym in red,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(HERE, "pool_panel.parquet"), index=False)
    print(f"面板: {len(df)} 行, {df.symbol.nunique()} 只, {df.date.nunique()} 期, "
          f"读取 {n} 只")


if __name__ == "__main__":
    main()
