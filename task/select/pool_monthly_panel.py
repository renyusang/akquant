"""月度选池面板(2026-10-20): 月末截面 × 多窗口涨幅, 供"换池频率×参考窗口"研究。

输出 `pool_monthly_panel.parquet`:
date, symbol, ret3, ret6, ret12, ret24, vol, amount20, fwd, theme, industry, red
- fwd = 到下一个月末的收益(最后一个截面到 2026-10-19)
- retL = 过去 L 个交易日收益(3/6/12/24 个月)

用法: python pool_monthly_panel.py
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

WINDOWS = {"ret3": 63, "ret6": 126, "ret12": 252, "ret24": 504}
END = pd.Timestamp("2026-10-19")


def _read(sym):
    for root in (CATALOG, FALLBACK):
        f = os.path.join(root, sym, "data.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            return d.sort_index()
    return None


def main():
    ref = _read("600519")
    days = ref.loc["2021-01-01":].index
    month_ends = days.to_series().groupby(days.to_period("M")).max().values
    dates = [pd.Timestamp(d) for d in month_ends]
    print(f"月末截面 {len(dates)} 个: {dates[0].date()} ~ {dates[-1].date()}")
    ind = pr.load_industry()
    red = pr.load_red_lines()
    rows = []
    n = 0
    for sym, name in pr.load_universe().items():
        d = _read(sym)
        if d is None:
            continue
        n += 1
        theme, industry = ind.get(sym, ("", ""))
        close, vol = d["close"].astype(float), d["volume"].astype(float)
        for i, dt in enumerate(dates):
            pre = close.loc[:dt]
            if len(pre) < 300:
                continue
            row = {"date": dt.strftime("%Y-%m-%d"), "symbol": sym,
                   "vol": float(np.diff(np.log(pre.values[-252:])).std() * np.sqrt(252)),
                   "amount20": float((close * vol).loc[:dt].tail(20).mean()),
                   "theme": theme or "未映射", "industry": industry or "未映射",
                   "red": sym in red}
            for k, L in WINDOWS.items():
                row[k] = (float(pre.iloc[-1] / pre.iloc[-1 - L] - 1)
                          if len(pre) > L else np.nan)
            nxt = dates[i + 1] if i + 1 < len(dates) else END
            fwd = close.loc[dt:nxt]
            row["fwd"] = float(fwd.iloc[-1] / fwd.iloc[0] - 1) if len(fwd) >= 5 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(HERE, "pool_monthly_panel.parquet"), index=False)
    print(f"面板 {len(df)} 行, {df.symbol.nunique()} 只, {df.date.nunique()} 期")


if __name__ == "__main__":
    main()
