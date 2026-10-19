"""构建全市场行业映射(2026-10-19): Sina 板块 → symbol/name/sector + 市值/换手。

东财行业接口在本机被重置, 改用 Sina 板块(49 个, 粒度适中, 适合"主题 cap")。
输出 `industry_map.csv`: symbol,name,sector,mktcap,nmc,turnover。
用法: python build_industry_map.py
"""

import os
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "industry_map.csv")


def main():
    import akshare as ak
    sectors = ak.stock_sector_spot()
    sectors = sectors[sectors["板块"].astype(str).str.len() > 0]
    print(f"板块 {len(sectors)} 个")
    rows, fail = [], []
    for i, r in sectors.iterrows():
        label, name = r["label"], r["板块"]
        try:
            d = ak.stock_sector_detail(sector=label)
            if d is None or d.empty:
                fail.append(name)
                continue
            d = d.copy()
            d["sector"] = name
            d["code"] = d["code"].astype(str).str.zfill(6)
            keep = [c for c in ["code", "name", "sector", "mktcap", "nmc",
                                "turnoverratio"] if c in d.columns]
            rows.append(d[keep])
            print(f"  [{i+1}/{len(sectors)}] {name}: {len(d)}")
        except Exception as e:
            fail.append(name)
            print(f"  [{i+1}/{len(sectors)}] {name}: FAIL {type(e).__name__}")
        time.sleep(0.3)
    df = pd.concat(rows, ignore_index=True).drop_duplicates("code")
    df = df.rename(columns={"code": "symbol", "turnoverratio": "turnover"})
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n写入 {OUT}: {len(df)} 只, {df.sector.nunique()} 个板块"
          + (f"; 失败 {fail}" if fail else ""))


if __name__ == "__main__":
    main()
