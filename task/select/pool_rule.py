"""选池规则(2026-10-19, P1): 资格过滤 → 风格配额 → 主题/行业上限。

设计依据: `stock_pool_methodology.md`
- 不按策略回测收益排序(无前向有效性);
- 按 12M 相对涨幅分桶, 默认 40 只: 逆向16 / 趋势24 / 热门0(可开到≤20%);
- 桶内: 逆向低波优先; 趋势/热门 用 12M收益/波动(风险调整动量);
- 主题(中证门类)≤30%, 细分(申万门类)≤3; 输出 40 只 + 备选 10 只。

用法:
    python pool_rule.py --date 2026-06-30
    python pool_rule.py --date 2026-06-30 --out pool_20260630.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KALMAN = os.path.join(HERE, "..", "kalman")
CATALOG = os.path.join(HERE, ".catalog_pool")
FALLBACK_CATALOG = os.path.join(KALMAN, ".catalog")


def catalog_path(sym: str) -> str:
    """优先用刷新后的候选池 catalog(2026-10-19), 缺失回退旧 catalog。"""
    p = os.path.join(CATALOG, sym, "data.parquet")
    return p if os.path.exists(p) else os.path.join(FALLBACK_CATALOG, sym, "data.parquet")

QUOTA_RATIO = {"contrarian": 0.40, "trend": 0.60, "hot": 0.0}  # 默认不含最热桶(崩盘保险);
# 想要部分动量暴露可把 hot 调到 0.20(实测 40 只 13/19/8: +51.9%/-11.6%)


def quota_for(top: int):
    q = {k: int(round(v * top)) for k, v in QUOTA_RATIO.items()}
    q["trend"] += top - sum(q.values())
    return q
THEME_CAP = 0.30      # 中证门类 ≤30% (40 只 → 12 只)
IND_CAP = 3           # 申万门类 ≤3
MIN_AMOUNT = 1e8      # 20 日日均成交额
_FEAT_CACHE = {}
VOL_PCT = 0.90        # 波动分位上限


def load_universe():
    cands = pd.read_csv(os.path.join(HERE, "candidates_full.csv"),
                        dtype={"code": str})
    cands["symbol"] = cands.code.str.zfill(6)
    return dict(zip(cands.symbol, cands.name))


def load_industry():
    """返回 {symbol: (theme, industry)}; 优先 cninfo, 回退 Sina 板块。"""
    out = {}
    p = os.path.join(HERE, "industry_map_cninfo.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, dtype={"symbol": str})
        for _, r in d.iterrows():
            if pd.notna(r.get("theme")) or pd.notna(r.get("industry")):
                out[str(r.symbol).zfill(6)] = (
                    str(r.get("theme") or ""), str(r.get("industry") or ""))
    p2 = os.path.join(HERE, "industry_map.csv")
    if os.path.exists(p2):
        d = pd.read_csv(p2, dtype={"symbol": str})
        for _, r in d.iterrows():
            s = str(r.symbol).zfill(6)
            if s not in out:
                out[s] = (str(r.get("sector") or ""), str(r.get("sector") or ""))
    return out


def load_red_lines():
    p = os.path.join(HERE, "risk_flags.csv")
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p, dtype={"code": str})
    d.columns = [c.strip() for c in d.columns]
    sev = d.get("severity", pd.Series(dtype=str)).astype(str).str.lower()
    return set(d.loc[sev == "red", "code"].str.zfill(6))


def features(date: str):
    """截至 date(含) 的个券特征(按 date 缓存, walk-forward 复用)。"""
    if date in _FEAT_CACHE:
        return _FEAT_CACHE[date]
    uni = load_universe()
    rows = []
    for sym, name in uni.items():
        f = catalog_path(sym)
        if not os.path.exists(f):
            continue
        try:
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            d = d.sort_index().loc[:date]
            if len(d) < 250:
                continue
            c = d["close"].values[-252:]
            ret12 = c[-1] / c[0] - 1
            vol = float(np.diff(np.log(c)).std() * np.sqrt(252))
            amt = float((d["close"] * d["volume"]).tail(20).mean())
            rows.append({"symbol": sym, "name": name, "ret12": ret12,
                         "vol": vol, "amount20": amt, "bars": len(d)})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    ind = load_industry()
    df["theme"] = [ind.get(s, ("", ""))[0] for s in df.symbol]
    df["industry"] = [ind.get(s, ("", ""))[1] for s in df.symbol]
    _FEAT_CACHE[date] = df
    return df


def build_pool(date: str, top: int = 40, reserve: int = 10, verbose=False):
    """返回 (selected_df, reserve_df)。"""
    d = features(date)
    if d.empty:
        return d, d
    red = load_red_lines()
    d = d[~d.symbol.isin(red)].copy()
    d = d[d.amount20 >= MIN_AMOUNT]
    d = d[d.vol <= d.vol.quantile(VOL_PCT)]
    d["pct"] = d.ret12.rank(pct=True) * 100
    d["bucket"] = np.select(
        [d.pct < 40, d.pct < 80], ["contrarian", "trend"], "hot")
    d["score"] = np.where(d.bucket == "contrarian",
                          -d.vol,                       # 逆向: 低波优先
                          d.ret12 / d.vol.replace(0, np.nan))
    d = d.sort_values(["bucket", "score"], ascending=[True, False])

    theme_cap_n = max(4, int(round(top * THEME_CAP)))
    quota = quota_for(top)
    picked, res, theme_n, ind_n = [], [], {}, {}
    taken = {k: 0 for k in quota}
    # 先按各桶配额取, 配额不足时由后续"补位"逻辑填充
    for bucket in ["contrarian", "trend", "hot"]:
        for _, r in d[d.bucket == bucket].iterrows():
            if taken[bucket] >= quota[bucket] or len(picked) >= top:
                break
            if any(p == r.symbol for p in picked):
                continue
            t, i = r.theme or "未映射", r.industry or "未映射"
            if theme_n.get(t, 0) >= theme_cap_n or ind_n.get(i, 0) >= IND_CAP:
                continue
            picked.append(r.symbol)
            taken[bucket] += 1
            theme_n[t] = theme_n.get(t, 0) + 1
            ind_n[i] = ind_n.get(i, 0) + 1
    # 剩余名额: 全局按分数补(同一 cap)
    if len(picked) < top:
        for _, r in d.sort_values("score", ascending=False).iterrows():
            if len(picked) >= top:
                break
            if r.symbol in picked:
                continue
            t, i = r.theme or "未映射", r.industry or "未映射"
            if theme_n.get(t, 0) >= theme_cap_n or ind_n.get(i, 0) >= IND_CAP:
                continue
            picked.append(r.symbol)
            theme_n[t] = theme_n.get(t, 0) + 1
            ind_n[i] = ind_n.get(i, 0) + 1
    # 备选: 继续按同规则往下取
    for _, r in d.sort_values("score", ascending=False).iterrows():
        if len(res) >= reserve:
            break
        if r.symbol in picked:
            continue
        t, i = r.theme or "未映射", r.industry or "未映射"
        if theme_n.get(t, 0) >= theme_cap_n or ind_n.get(i, 0) >= IND_CAP:
            continue
        res.append(r.symbol)
        theme_n[t] = theme_n.get(t, 0) + 1
        ind_n[i] = ind_n.get(i, 0) + 1

    sel = d[d.symbol.isin(picked)].copy()
    sel["rank"] = sel.symbol.map({s: i + 1 for i, s in enumerate(picked)})
    sel = sel.sort_values("rank")
    rs = d[d.symbol.isin(res)].copy()
    rs["rank"] = rs.symbol.map({s: i + 1 for i, s in enumerate(res)})
    rs = rs.sort_values("rank")
    if verbose:
        cols = ["rank", "symbol", "name", "bucket", "pct", "ret12", "vol",
                "amount20", "industry", "theme"]
        print(f"\n{date} 选池 {len(sel)} 只 (候选 {len(d)} 只)")
        print(sel[cols].assign(ret12=lambda x: (x.ret12 * 100).round(0),
                               vol=lambda x: (x.vol * 100).round(0),
                               amount20=lambda x: (x.amount20 / 1e8).round(1),
                               pct=lambda x: x.pct.round(0)
                               ).to_string(index=False))
        print("桶分布:", sel.bucket.value_counts().to_dict())
        print("主题分布:", sel.theme.value_counts().head(5).to_dict())
    return sel, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--reserve", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sel, rs = build_pool(args.date, args.top, args.reserve, verbose=True)
    if args.out:
        sel.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"写入 {args.out}")
    print(f"备选 {len(rs)} 只: {', '.join(rs.symbol + ' ' + rs.name)}")


if __name__ == "__main__":
    main()
