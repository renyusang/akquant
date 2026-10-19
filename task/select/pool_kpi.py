"""池子月度 KPI(2026-10-20): 风格暴露 / 主题集中度 / 离散度 / 换手。

用法:
    python pool_kpi.py --date 2026-09-30
    python pool_kpi.py --date 2026-09-30 --write   # 追加 pool_kpi.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STATE = os.path.join(HERE, "pool_state.json")
HISTORY = os.path.join(HERE, "pool_history.csv")
CATALOG = os.path.join(HERE, ".catalog_pool")
FALLBACK = os.path.join(HERE, "..", "kalman", ".catalog")


def _read(sym):
    for root in (CATALOG, FALLBACK):
        f = os.path.join(root, sym, "data.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            return d.sort_index()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    import pool_rule as pr

    state = json.load(open(STATE, encoding="utf-8"))
    members = state["members"]
    d = pr.features(args.date)
    p = d[d.symbol.isin(members)].copy()

    p["pct"] = p.ret12.rank(pct=True) * 100
    p["bucket"] = np.select([p.pct < 40, p.pct < 80],
                            ["contrarian", "trend"], "hot")
    n = len(p)
    theme_share = p.theme.value_counts(normalize=True)
    ind_max = p.industry.value_counts().max()
    # 离散度: 12M 收益截面标准差 + 近 60 日平均两两相关
    disp = float(p.ret12.std())
    closes = {}
    for s in p.symbol:
        x = _read(s)
        if x is not None:
            closes[s] = x["close"].tail(61)
    corr = np.nan
    if len(closes) >= 5:
        mat = pd.DataFrame(closes).pct_change().dropna()
        c = mat.corr().values
        corr = float(c[np.triu_indices_from(c, 1)].mean())
    # 换手: 最近一次评审
    churn = {"add": 0, "remove": 0, "date": "-"}
    if os.path.exists(HISTORY):
        h = pd.read_csv(HISTORY, dtype={"symbol": str})
        last = h.date.max()
        churn = {"date": last,
                 "add": int(((h.date == last) & (h.action == "add")).sum()),
                 "remove": int(((h.date == last) & (h.action == "remove")).sum())}

    out = {
        "date": args.date, "n": n,
        "contrarian": int((p.bucket == "contrarian").sum()),
        "trend": int((p.bucket == "trend").sum()),
        "hot": int((p.bucket == "hot").sum()),
        "ret12_med": round(float(p.ret12.median()) * 100, 1),
        "vol_med": round(float(p.vol.median()) * 100, 1),
        "theme_max": f"{theme_share.index[0]} {theme_share.iloc[0]*100:.0f}%",
        "theme_cap_ok": bool(theme_share.iloc[0] <= 0.30),
        "industry_max": int(ind_max), "industry_cap_ok": bool(ind_max <= 3),
        "dispersion": round(disp * 100, 1), "avg_corr": round(corr, 2)
        if not np.isnan(corr) else None,
        "last_review": churn["date"], "add": churn["add"], "remove": churn["remove"],
    }
    print(f"\n池 KPI · {args.date}（{n} 只）")
    print(f"  风格桶: 逆向 {out['contrarian']} / 趋势 {out['trend']} / 热门 {out['hot']}")
    print(f"  12M 涨幅中位 {out['ret12_med']}% | 波动中位 {out['vol_med']}% | "
          f"截面离散度 {out['dispersion']}pp | 平均两两相关 {out['avg_corr']}")
    print(f"  最大主题: {out['theme_max']} {'✅' if out['theme_cap_ok'] else '❌ 超30%'} | "
          f"申万单行业最大 {out['industry_max']} {'✅' if out['industry_cap_ok'] else '❌ 超3'}")
    print(f"  最近评审 {out['last_review']}: 调入 {out['add']} / 调出 {out['remove']}")
    if args.write:
        df = pd.DataFrame([out])
        path = os.path.join(HERE, "pool_kpi.csv")
        df.to_csv(path, mode="a", header=not os.path.exists(path),
                  index=False, encoding="utf-8-sig")
        print(f"追加 {path}")


if __name__ == "__main__":
    main()
