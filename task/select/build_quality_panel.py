"""构建质量因子面板(2026-10-20): 同花顺财务摘要 → 季度财务指标。

输出 `quality_panel.parquet`: symbol,report_date,net_profit,np_yoy,roe,debt_ratio,ocf_ps
仅保留 2020-12-31 之后的报告期(够 walk-forward 用)。
用法: python build_quality_panel.py [--workers 4]
"""

import argparse
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

socket.setdefaulttimeout(20)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "quality_panel.parquet")
COLS = {"净利润": "net_profit", "净利润同比增长率": "np_yoy",
        "净资产收益率": "roe", "资产负债率": "debt_ratio",
        "每股经营现金流": "ocf_ps"}


def num(x):
    if x is False or x is None:
        return np.nan
    s = str(x).strip()
    if s in ("", "--", "None", "nan", "False"):
        return np.nan
    mult = 1.0
    if s.endswith("万亿"):
        mult, s = 1e12, s[:-2]
    elif s.endswith("亿"):
        mult, s = 1e8, s[:-1]
    elif s.endswith("万"):
        mult, s = 1e4, s[:-1]
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        v = float(s.replace(",", "")) * mult
    except ValueError:
        return np.nan
    return v


def fetch(sym):
    import akshare as ak
    try:
        d = ak.stock_financial_abstract_ths(symbol=sym, indicator="按报告期")
        if d is None or d.empty:
            return None
        rows = []
        for _, r in d.iterrows():
            rd = str(r.get("报告期", ""))[:10]
            if rd < "2020-12-31":
                continue
            row = {"symbol": sym, "report_date": rd}
            for src, dst in COLS.items():
                row[dst] = num(r.get(src))
            rows.append(row)
        return rows
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cands = pd.read_csv(os.path.join(HERE, "candidates_full.csv"),
                        dtype={"code": str})
    syms = [str(s).zfill(6) for s in cands.code]
    out, fail = [], []
    t0 = time.time()

    def work(s):
        r = fetch(s)
        return s, r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (s, r) in enumerate(ex.map(work, syms)):
            if r:
                out.extend(r)
            else:
                fail.append(s)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(syms)}  ok={len(set(x['symbol'] for x in out))} "
                      f"fail={len(fail)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    df.to_parquet(OUT, index=False)
    print(f"\n写入 {OUT}: {len(df)} 行, {df.symbol.nunique()}/{len(syms)} 只, "
          f"失败 {len(fail)}, 耗时 {time.time()-t0:.0f}s")
    if len(df):
        print(df.report_date.value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
