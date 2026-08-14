"""候选池生成器: akshare 免费数据源(默认) + 妙想导出 CSV 兼容。

用途: 为 screen_pool.py 生成 candidates CSV。妙想 MCP 是付费服务,
akshare 提供免费替代——指数成分/全市场行情/ST 列表均可本地实现,
唯一缺口是公司违规事件(风险分级), 妙想保留给该低频高价值场景。

用法:
  # akshare 生成候选池(默认四个指数)
  python pool_provider.py --output candidates_ak.csv

  # 自定义指数/过滤条件
  python pool_provider.py --indexes 000300,000905,399673,000688 \
      --min-market-cap 100 --min-turnover 1 --output candidates_ak.csv

  # 合并妙想导出的 CSV(动量池等)
  python pool_provider.py --miaoxiang-file candidates_full.csv \
      --output candidates_ak.csv

输出 CSV 列: code, name, source (source: 指数代码/妙想/momentum 等)
"""

import argparse
import os
import sys

import pandas as pd

# =============================================================================
# 指数成分(akshare)
# =============================================================================
# 中证指数官网代码(优先) / 新浪代码(创业板50 等深交所指数)
CSINDEX_CODES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000688": "科创50",
}
SINA_CODES = {
    "399673": "创业板50",
}


def fetch_index_constituents(index_code: str) -> list[tuple[str, str]]:
    """拉取指数成分股列表 [(code, name)], 失败抛异常。

    中证指数官网接口优先(index_stock_cons_csindex),
    深交所指数(如创业板50)回退新浪(index_stock_cons)。
    """
    import akshare as ak

    if index_code in CSINDEX_CODES:
        df = ak.index_stock_cons_csindex(symbol=index_code)
        # 列含"指数代码"(全是指数自身, 如 000300)——必须优先"成分券代码"
        code_col = next((c for c in df.columns if "成分券代码" in c), None) \
            or next((c for c in df.columns if "代码" in c), None)
        name_col = next((c for c in df.columns if "成分券名称" in c), None) \
            or next((c for c in df.columns if "名称" in c), code_col)
        return [(str(r[code_col]).zfill(6), str(r[name_col]))
                for _, r in df.iterrows()]
    # 新浪接口
    df = ak.index_stock_cons(symbol=index_code)
    code_col = next((c for c in df.columns if "代码" in c), df.columns[0])
    name_col = next((c for c in df.columns if "名称" in c), df.columns[1])
    return [(str(r[code_col]).zfill(6), str(r[name_col]))
            for _, r in df.iterrows()]


def fetch_spot_data() -> pd.DataFrame:
    """全市场实时行情: 东财优先(含市值/成交额), 失败回退新浪(无市值列)。

    返回列(东财): 代码/名称/最新价/涨跌幅/总市值/成交额/...
    """
    import akshare as ak

    def _norm_code(s):
        # 新浪代码带 sh/sz/bj 前缀 → 统一 6 位纯数字
        return str(s).replace("sh", "").replace("sz", "").replace("bj",
                                                                   "").zfill(6)

    try:
        df = ak.stock_zh_a_spot_em()
        df["代码"] = df["代码"].map(_norm_code)
        return df
    except Exception:
        # 新浪: 无市值/成交额列, 市值过滤降级为跳过
        df = ak.stock_zh_a_spot()
        df["代码"] = df["代码"].map(_norm_code)
        return df


def fetch_st_list() -> set:
    """ST/*ST 股票代码集合(风险排除)。失败返回空集(不阻塞)。"""
    import akshare as ak

    try:
        df = ak.stock_zh_a_st_em()
        col = next((c for c in df.columns if "代码" in c), df.columns[0])
        return set(str(x).zfill(6) for x in df[col])
    except Exception:
        return set()


# =============================================================================
# 候选池构建
# =============================================================================
def build_ak_pool(indexes: list[str], min_market_cap: float = 100.0,
                  min_turnover: float = 1.0, exclude_st: bool = True,
                  quiet: bool = False) -> list[dict]:
    """akshare 构建候选池: 指数成分 ∪ 过滤(市值/成交额/ST)。

    返回 [{symbol, name, asset_type, source}], 按指数顺序去重。
    min_market_cap/min_turnover 单位: 亿元。
    """
    rows: dict[str, dict] = {}

    # 指数成分
    for idx in indexes:
        label = CSINDEX_CODES.get(idx, SINA_CODES.get(idx, idx))
        try:
            members = fetch_index_constituents(idx)
            if not quiet:
                print(f"  {label}({idx}): {len(members)} 只")
            for code, name in members:
                if code not in rows:
                    rows[code] = {
                        "symbol": code, "name": name,
                        "asset_type": "etf" if code.startswith(("51", "15",
                                                                "58", "56"))
                        else "stock",
                        "source": label,
                    }
        except Exception as e:
            if not quiet:
                print(f"  {label}({idx}): ❌ 拉取失败 {str(e)[:60]}")

    if not rows:
        return []

    # 全市场行情(市值/成交额过滤)
    try:
        spot = fetch_spot_data()
        cap_col = next((c for c in spot.columns if "市值" in c), None)
        amt_col = next((c for c in spot.columns if "成交额" in c), None)
        spot_map = spot.set_index("代码")
        filtered = []
        for r in rows.values():
            s = r["symbol"]
            if s not in spot_map.index:
                continue
            row = spot_map.loc[s]
            if cap_col is not None and min_market_cap > 0:
                cap = float(row[cap_col])
                if cap < min_market_cap * 1e8:
                    continue
            if amt_col is not None and min_turnover > 0:
                amt = float(row[amt_col])
                if amt < min_turnover * 1e8:
                    continue
            filtered.append(r)
        if not quiet:
            print(f"  市值>={min_market_cap:.0f}亿/成交额>={min_turnover:.0f}亿"
                  f"过滤后: {len(filtered)} 只")
        rows = {r["symbol"]: r for r in filtered}
    except Exception as e:
        if not quiet:
            print(f"  ⚠️ 行情过滤失败(市值/成交额未应用): {str(e)[:60]}")

    # ST 排除
    if exclude_st:
        st = fetch_st_list()
        if st:
            before = len(rows)
            rows = {c: r for c, r in rows.items() if c not in st}
            if not quiet:
                print(f"  ST 排除: {before - len(rows)} 只")

    return list(rows.values())


def load_miaoxiang_csv(path: str) -> list[dict]:
    """读取妙想导出的候选 CSV(code,name 列), 与 akshare 池合并。"""
    df = pd.read_csv(path, dtype={"code": str})
    rows = []
    for _, r in df.iterrows():
        code = str(r["code"]).strip().zfill(6)
        if code and code != "nan":
            rows.append({
                "symbol": code,
                "name": str(r.get("name", code)).strip(),
                "asset_type": "etf" if code.startswith(("51", "15", "58",
                                                        "56")) else "stock",
                "source": str(r.get("source", "妙想")),
            })
    return rows


def merge_pools(*pools: list[dict]) -> list[dict]:
    """多池合并去重(先到先得保留 source)。"""
    seen, out = set(), []
    for pool in pools:
        for r in pool:
            if r["symbol"] not in seen:
                seen.add(r["symbol"])
                out.append(r)
    return out


def save_candidates(rows: list[dict], path: str) -> None:
    """保存候选池 CSV(code,name,source)——code 列名对齐 screen_pool.py。"""
    out = [{"code": r["symbol"], "name": r["name"], "source": r["source"]}
           for r in rows]
    pd.DataFrame(out).to_csv(path, index=False, encoding="utf-8-sig")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="候选池生成器: akshare 免费源 + 妙想 CSV 兼容")
    parser.add_argument("--indexes",
                        default="000300,000905,399673,000688",
                        help="指数代码逗号分隔(默认四个核心指数)")
    parser.add_argument("--min-market-cap", type=float, default=100.0,
                        help="最小总市值(亿元), 0 关闭")
    parser.add_argument("--min-turnover", type=float, default=1.0,
                        help="最小成交额(亿元), 0 关闭")
    parser.add_argument("--no-st-filter", action="store_true",
                        help="不过滤 ST")
    parser.add_argument("--miaoxiang-file", default=None,
                        help="妙想导出的候选 CSV 合并(动量池等)")
    parser.add_argument("--output", default="candidates_ak.csv",
                        help="输出 CSV 路径")
    args = parser.parse_args()

    indexes = [i.strip() for i in args.indexes.split(",") if i.strip()]
    print(f"akshare 拉取指数成分: {indexes}")
    pool = build_ak_pool(
        indexes, min_market_cap=args.min_market_cap,
        min_turnover=args.min_turnover, exclude_st=not args.no_st_filter)

    if args.miaoxiang_file and os.path.exists(args.miaoxiang_file):
        mx = load_miaoxiang_csv(args.miaoxiang_file)
        print(f"妙想 CSV: {len(mx)} 只")
        pool = merge_pools(pool, mx)

    if not pool:
        print("错误: 候选池为空")
        sys.exit(1)
    save_candidates(pool, args.output)
    print(f"候选池(去重后): {len(pool)} 只 → {args.output}")
    print(f"  股票 {sum(1 for r in pool if r['asset_type']=='stock')} 只 + "
          f"ETF {sum(1 for r in pool if r['asset_type']=='etf')} 只")


if __name__ == "__main__":
    main()
