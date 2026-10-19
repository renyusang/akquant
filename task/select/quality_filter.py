"""质量因子(2026-10-20): 连续亏损 / 经营现金流为负 / ROE 为负 → 排除。

数据: quality_panel.parquet(build_quality_panel.py 生成, 同花顺财务摘要)。
时点: 只用 "报告期 + 90 天" 之前已披露的报告(保守处理公告滞后)。

用法:
    python quality_filter.py --date 2026-06-30          # 列出不合格标的
    from quality_filter import bad_symbols; bad_symbols("2026-06-30")
"""

import argparse
import os
from datetime import timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "quality_panel.parquet")
LAG_DAYS = 120   # 年报法定 4-30 披露; 统一 120 天最保守


_CACHE = None
_BAD_CACHE = {}


def load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not os.path.exists(PANEL):
        return None
    df = pd.read_parquet(PANEL)
    df["report_date"] = pd.to_datetime(df["report_date"])
    _CACHE = df.sort_values(["symbol", "report_date"])
    return _CACHE


def bad_symbols(date: str, panel=None):
    """返回 {(symbol, reason)}: 连续两年亏损 / 最新年报经营现金流为负 / ROE<0。"""
    if panel is None and date in _BAD_CACHE:
        return _BAD_CACHE[date]
    q = panel if panel is not None else load()
    if q is None or q.empty:
        return {}
    asof = pd.Timestamp(date) - timedelta(days=LAG_DAYS)
    q = q[q.report_date <= asof]
    out = {}
    for sym, g in q.groupby("symbol"):
        annual = g[g.report_date.dt.month == 12]
        if len(annual) >= 2:
            last2 = annual.tail(2)
            if (last2.net_profit < 0).all():
                out[sym] = "连续两年亏损"
                continue
        if len(annual):
            last = annual.iloc[-1]
            if pd.notna(last.ocf_ps) and last.ocf_ps < 0:
                out[sym] = "年报经营现金流为负"
                continue
            if pd.notna(last.roe) and last.roe < 0:
                out[sym] = "年报ROE为负"
    if panel is None:
        _BAD_CACHE[date] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    bad = bad_symbols(args.date)
    print(f"{args.date}: 质量不合格 {len(bad)} 只")
    for s, r in list(bad.items())[:30]:
        print(f"  {s} {r}")
    if len(bad) > 30:
        print(f"  ... 共 {len(bad)} 只")


if __name__ == "__main__":
    main()
