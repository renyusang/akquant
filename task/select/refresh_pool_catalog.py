"""刷新候选池行情到独立 catalog(2026-10-19)。

背景: `.catalog` 末根日期不一致(519 只到 08-13 / 286 只到 07-24 / 49 只到 09-17),
"未来收益"窗口因股而异 → 股票池研究会被系统性污染。
本脚本把 828 只候选全历史(qfq)刷到 `.catalog_pool/`, **不动** `.catalog`
(避免影响既有回测基线)。

用法: python refresh_pool_catalog.py [--workers 6] [--force]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KALMAN = os.path.join(HERE, "..", "kalman")
sys.path.insert(0, KALMAN)

ROOT = os.path.join(HERE, ".catalog_pool")
END = "20261019"
MIN_LAST = "2026-10-17"


def _need(path: str, force: bool) -> bool:
    if force or not os.path.exists(path):
        return True
    try:
        d = pd.read_parquet(path)
        return str(pd.to_datetime(d.index).max().date()) < MIN_LAST
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from data_utils import download_data, preprocess_data, get_catalog
    cat = get_catalog(ROOT)
    cands = pd.read_csv(os.path.join(HERE, "candidates_full.csv"),
                        dtype={"code": str})
    syms = [str(s).zfill(6) for s in cands.code]
    os.makedirs(ROOT, exist_ok=True)

    def work(sym):
        path = os.path.join(ROOT, sym, "data.parquet")
        if not _need(path, args.force):
            return sym, "skip", 0
        for attempt in range(2):
            try:
                t0 = time.time()
                df = preprocess_data(download_data(
                    sym, asset_type="stock", start_date="20200101",
                    end_date=END, adjust="qfq"))
                if df is None or df.empty:
                    raise ValueError("empty")
                cat.write(sym, df)
                return sym, "ok", time.time() - t0
            except Exception as e:
                if attempt == 1:
                    return sym, f"fail:{type(e).__name__}", 0
                time.sleep(1)
        return sym, "fail", 0

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "fail": 0}
    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (sym, st, dt) in enumerate(ex.map(work, syms)):
            if st == "ok":
                stats["ok"] += 1
            elif st == "skip":
                stats["skip"] += 1
            else:
                stats["fail"] += 1
                fails.append((sym, st))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(syms)}  ok={stats['ok']} skip={stats['skip']} "
                      f"fail={stats['fail']}  {time.time()-t0:.0f}s", flush=True)
    print(f"\n完成: {stats} | 耗时 {time.time()-t0:.0f}s")
    if fails:
        print("失败:", fails[:20])


if __name__ == "__main__":
    main()
