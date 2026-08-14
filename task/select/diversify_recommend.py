"""推荐分散工具: 补充行业字段 + 单行业 cap + TOP N 输出 watchlist 建议。

全池筛选(screen_pool.py)产出 97 只推荐, 行业扎堆 AI 算力主线
(光模块/PCB/半导体同涨同跌风险高)。本工具:
  1. akshare 批量查推荐标的行业(stock_individual_info_em)
  2. 单行业 cap(默认 3 只, 按验证期收益取行业内 TOP)
  3. 分散后按验证期收益取 TOP N(默认 20) → watchlist 建议列表

用法:
  python diversify_recommend.py --input screen_pool_full_result.csv
  python diversify_recommend.py --cap 3 --top-n 20 --output recommended_watchlist.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalman"))


def fetch_industry(symbol: str) -> str:
    """akshare 查单只股票行业(东财 F10)。失败返回 '未知'。"""
    import akshare as ak

    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        row = df[df["item"] == "行业"]
        if len(row):
            return str(row["value"].iloc[0])
    except Exception:
        pass
    return "未知"


def add_industries(df: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """批量补充行业字段(串行, 每只 ~1 秒)。"""
    d = df.copy()
    inds = []
    for i, sym in enumerate(d["symbol"], 1):
        ind = fetch_industry(str(sym).zfill(6))
        inds.append(ind)
        if not quiet and i % 20 == 0:
            print(f"  行业查询 {i}/{len(d)}", flush=True)
        time.sleep(0.3)  # 限流保护
    d["industry"] = inds
    return d


def diversify(df: pd.DataFrame, cap: int = 3) -> pd.DataFrame:
    """单行业 cap: 行业内按验证期收益降序取前 cap 只。"""
    d = df[df["recommended"]].copy()
    d = d.sort_values("out_ret", ascending=False)
    out = d.groupby("industry", dropna=False).head(cap)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="推荐分散: 行业字段 + 单行业 cap + TOP N")
    parser.add_argument("--input", default="screen_pool_full_result.csv")
    parser.add_argument("--cap", type=int, default=3,
                        help="单行业最多保留几只(默认 3)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="分散后取验证期收益 TOP N(默认 20)")
    parser.add_argument("--output", default="recommended_watchlist.csv")
    parser.add_argument("--no-cache", action="store_true",
                        help="不使用行业缓存 CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"symbol": str})
    rec = df[df["recommended"]]
    print(f"推荐标的: {len(rec)} 只, 开始补充行业字段...")

    # 行业缓存(避免重复查询, 失败时降级)
    cache_path = os.path.join(os.path.dirname(__file__),
                              "industry_cache.csv")
    cache: dict[str, str] = {}
    if os.path.exists(cache_path) and not args.no_cache:
        cdf = pd.read_csv(cache_path, dtype={"symbol": str})
        cache = dict(zip(cdf["symbol"], cdf["industry"]))

    inds = []
    missing = []
    for sym in rec["symbol"]:
        s = str(sym).zfill(6)
        if s in cache:
            inds.append(cache[s])
        else:
            inds.append(None)
            missing.append(s)
    if missing:
        print(f"缓存命中 {len(rec) - len(missing)}/{len(rec)}, "
              f"待查 {len(missing)} 只")
        from akshare import stock_individual_info_em  # noqa: F401
        import akshare as ak

        mi = 0
        for i, (sym, ind) in enumerate(zip(rec["symbol"], inds)):
            if ind is None:
                s = str(sym).zfill(6)
                try:
                    info = ak.stock_individual_info_em(symbol=s)
                    row = info[info["item"] == "行业"]
                    inds[i] = str(row["value"].iloc[0]) if len(row) else "未知"
                except Exception:
                    inds[i] = "未知"
                cache[s] = inds[i]
                mi += 1
                if mi % 20 == 0:
                    print(f"  行业查询 {mi}/{len(missing)}", flush=True)
                time.sleep(0.3)
        # 更新缓存
        pd.DataFrame({"symbol": list(cache.keys()),
                      "industry": list(cache.values())}).to_csv(
            cache_path, index=False, encoding="utf-8-sig")

    rec = rec.copy()
    rec["industry"] = inds

    div = diversify(rec, args.cap)
    top = div.sort_values("out_ret", ascending=False).head(args.top_n)
    top = top.reset_index(drop=True)

    print(f"\n{'=' * 78}")
    print(f"  行业分散后推荐 (cap={args.cap}) → TOP {args.top_n}")
    print("=" * 78)
    print(f"{'代码':<8}{'名称':<10}{'行业':<10}{'筛选期':<10}{'验证期':<10}"
          f"{'夏普':<7}{'回撤':<8}来源")
    for _, r in top.iterrows():
        src = "✅已有" if r["in_watchlist"] else "🆕"
        risk = "⚠️" if str(r.get("risk_note", "")) != "nan" and str(
            r.get("risk_note", "")) else ""
        print(f"{r['symbol']:<8}{str(r['name'])[:10]:<10}"
              f"{str(r['industry'])[:10]:<10}{r['in_ret']:>+7.1f}%  "
              f"{r['out_ret']:>+7.1f}%  {r['out_sharpe']:>5.2f}   "
              f"{r['out_mdd']:>5.1f}%   {src}{risk}")

    # 行业分布
    print(f"\n行业分布 (分散后 {len(div)} 只):")
    dist = div["industry"].value_counts()
    for ind, n in dist.items():
        print(f"  {ind}: {n}")

    top.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n建议列表已保存: {args.output} ({len(top)} 只)")


if __name__ == "__main__":
    main()
