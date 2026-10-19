"""构建候选池行业映射(2026-10-19): 巨潮/中证/申万 行业分类。

Sina 板块只覆盖 2990 只(候选覆盖 63%), 改用 cninfo 的个股行业变更记录:
每只取最新一条"中证行业分类"(门类=粗主题)与"申万行业分类"(门类=中类行业)。
输出 `industry_map_cninfo.csv`: symbol,name,theme,industry,asof。

用法: python build_industry_map_cninfo.py
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "industry_map_cninfo.csv")
STD_THEME = "中证行业分类标准"
STD_IND = "申银万国行业分类标准"


def fetch_one(sym: str, name: str):
    import akshare as ak
    try:
        d = ak.stock_industry_change_cninfo(symbol=sym, start_date="20000101",
                                            end_date="20261231")
        if d is None or d.empty:
            return None
        row = {"symbol": sym, "name": name}
        for std, key in [(STD_THEME, "theme"), (STD_IND, "industry")]:
            sub = d[d["分类标准"].astype(str) == std]
            if len(sub):
                last = sub.sort_values("变更日期").iloc[-1]
                row[key] = str(last.get("行业门类") or "")
                row["asof"] = str(last.get("变更日期") or "")
        return row
    except Exception:
        return None


def main():
    cands = pd.read_csv(os.path.join(HERE, "candidates_full.csv"),
                        dtype={"code": str})
    cands["symbol"] = cands.code.str.zfill(6)
    items = list(zip(cands.symbol, cands.name))
    print(f"候选 {len(items)} 只, 3 线程抓取...")
    rows, fail = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_one, s, n): s for s, n in items}
        for i, f in enumerate(futs):
            r = f.result()
            if r is None:
                fail.append(futs[f])
            else:
                rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(items)}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n写入 {OUT}: {len(df)}/{len(items)} 只 (失败 {len(fail)}), "
          f"耗时 {time.time()-t0:.0f}s")
    if len(df):
        print("中证门类 top:", df.theme.value_counts().head(6).to_dict())
        print("申万门类 top:", df.industry.value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
