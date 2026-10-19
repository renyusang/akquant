"""股票池诊断(2026-10-19): 选池统计量的前向有效性 / 风格暴露 / 拥挤度。

回答三个问题:
1. 用"策略回测收益"选股, 对未来有预测力吗?      → section ic
2. 股票池的风格暴露与 2026Q3 回撤是什么关系?      → section style
3. 当前 25 只池长什么样, 换一种规则会怎样?        → section pool

用法:
    python pool_diagnostics.py                 # 全部
    python pool_diagnostics.py --section ic
数据: screen_pool_full_result.csv + ../kalman/.catalog + ../kalman/stocks.yaml
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KALMAN = os.path.join(HERE, "..", "kalman")
CATALOG = os.path.join(KALMAN, ".catalog")
sys.path.insert(0, HERE)


def _spearman(a, b, n_perm=1000, seed=0):
    r1, r2 = pd.Series(a).rank(), pd.Series(b).rank()
    rho = r1.corr(r2)
    rng = np.random.default_rng(seed)
    perm = [r1.corr(pd.Series(rng.permutation(r2.values), index=r2.index))
            for _ in range(n_perm)]
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1)
    return rho, p


def _panel(start="2025-07-01", end="2026-06-30", fwd_start="2026-07-01",
           fwd_end="2026-10-19"):
    rows = []
    for f in glob.glob(os.path.join(CATALOG, "*", "data.parquet")):
        sym = os.path.basename(os.path.dirname(f))
        if sym.startswith(("51", "15", "58", "56")):
            continue
        try:
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()["close"]
            pre = d.loc[start:end]
            fwd = d.loc[fwd_start:fwd_end]
            if len(pre) < 200 or len(fwd) < 20:
                continue
            rows.append((sym, pre.iloc[-1] / pre.iloc[0] - 1,
                         np.diff(np.log(pre.values)).std() * np.sqrt(252),
                         fwd.iloc[-1] / fwd.iloc[0] - 1))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["symbol", "ret12", "vol", "fwd"])


def section_ic():
    print("\n[1] 用策略回测收益选股, 有前向预测力吗? (screen_pool_full_result.csv)")
    d = pd.read_csv(os.path.join(HERE, "screen_pool_full_result.csv")).dropna(
        subset=["in_ret", "out_ret"])
    rho, p = _spearman(d.in_ret, d.out_ret)
    print(f"  n={len(d)} | 筛选期(2020-2024)收益中位 {d.in_ret.median():+.1f}% | "
          f"验证期(2025-2026)中位 {d.out_ret.median():+.1f}%")
    print(f"  Spearman(筛选期, 验证期) = {rho:+.3f} (置换 p={p:.3f})")
    d["q"] = pd.qcut(d.in_ret, 5, labels=False, duplicates="drop")
    t = d.groupby("q").agg(n=("out_ret", "size"), in_med=("in_ret", "median"),
                           oos_med=("out_ret", "median"),
                           pos=("out_ret", lambda s: (s > 0).mean() * 100))
    print("  按筛选期收益分 5 组 → 验证期收益:")
    print(t.round(1).to_string().replace("\n", "\n  "))
    top_is = set(d.nlargest(25, "in_ret").symbol)
    top_oos = set(d.nlargest(25, "out_ret").symbol)
    print(f"  筛选期 top25 与 验证期 top25 的重合: {len(top_is & top_oos)} 只")
    print("  ⚠️ recommended 条件含 out_ret>0 → 验证期被用进选择, 不是真样本外")


def section_style():
    print("\n[2] 风格暴露 vs 2026-07~10 回撤 (538 只, 2026-06-30 截面)")
    u = _panel()
    u["qm"] = pd.qcut(u.ret12, 5, labels=False)
    u["qv"] = pd.qcut(u.vol, 5, labels=False)
    print("  按过去 12 个月涨幅分组 → 未来收益:")
    print(u.groupby("qm").agg(n=("fwd", "size"), pre12=("ret12", "median"),
                              fwd_mean=("fwd", "mean"), fwd_med=("fwd", "median"),
                              pos=("fwd", lambda s: (s > 0).mean() * 100)
                              ).round(3).to_string().replace("\n", "\n  "))
    rho, p = _spearman(u.ret12, u.fwd)
    print(f"  IC(12M 涨幅, 未来收益) = {rho:+.3f} (p={p:.3f})")

    print("\n  但注意: 这是单次崩盘。2021-2026 共 21 个季度的面板:")
    dates = pd.date_range("2021-03-31", "2026-06-30", freq="QE")
    close = {}
    for f in glob.glob(os.path.join(CATALOG, "*", "data.parquet")):
        sym = os.path.basename(os.path.dirname(f))
        if sym.startswith(("51", "15", "58", "56")):
            continue
        try:
            d = pd.read_parquet(f)
            d.index = pd.to_datetime(d.index)
            close[sym] = d.sort_index()["close"]
        except Exception:
            pass
    recs = []
    for dt in dates:
        rows = []
        for s, px in close.items():
            try:
                pre = px.loc[:dt].iloc[-252:]
                nxt = px.loc[dt:].iloc[:63]
                if len(pre) < 200 or len(nxt) < 40:
                    continue
                rows.append((s, pre.iloc[-1] / pre.iloc[0] - 1,
                             np.diff(np.log(pre.values)).std() * np.sqrt(252),
                             nxt.iloc[-1] / nxt.iloc[0] - 1))
            except Exception:
                pass
        if len(rows) < 100:
            continue
        t = pd.DataFrame(rows, columns=["s", "r12", "vol", "fwd"])
        t["qm"] = pd.qcut(t.r12, 5, labels=False)
        t["qv"] = pd.qcut(t.vol, 5, labels=False)
        recs.append({"date": dt,
                     **{f"mom{q}": t[t.qm == q].fwd.mean() for q in range(5)},
                     **{f"vol{q}": t[t.qv == q].fwd.mean() for q in range(5)}})
    pnl = pd.DataFrame(recs).set_index("date") * 100
    print(f"    {'组':>4} {'12M涨幅':>9} {'波动':>9}")
    for q in range(5):
        print(f"    Q{q:<3} {pnl[f'mom{q}'].mean():>+9.2f} {pnl[f'vol{q}'].mean():>+9.2f}")
    print(f"    Q4-Q0 {pnl.mom4.mean()-pnl.mom0.mean():>+8.2f} "
          f"{(pnl.vol4.mean()-pnl.vol0.mean()):>+9.2f}  (单位 pp/季)")
    print(f"    Q4-Q0 为负的季度: {(pnl.mom4 < pnl.mom0).sum()}/{len(pnl)} "
          f"(动量在多数季度仍占优, 崩盘季度反转)")


def section_pool():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(KALMAN, "stocks.yaml"), encoding="utf-8"))
    pool = [str(s["symbol"]).zfill(6) for s in cfg["watchlist"]["stocks"]]
    rows = []
    for s in pool:
        f = os.path.join(CATALOG, s, "data.parquet")
        d = pd.read_parquet(f)
        d.index = pd.to_datetime(d.index)
        d = d.sort_index()["close"]
        pre = d.loc["2025-07-01":"2026-06-30"]
        fwd = d.loc["2026-07-01":"2026-10-19"]
        rows.append((s, pre.iloc[-1] / pre.iloc[0] - 1,
                     np.diff(np.log(pre.values)).std() * np.sqrt(252),
                     fwd.iloc[-1] / fwd.iloc[0] - 1))
    p = pd.DataFrame(rows, columns=["symbol", "ret12", "vol", "fwd"])
    print(f"\n[3] 当前 25 只池 (2026-06-30 截面)")
    print(f"  过去 12M 涨幅中位 {p.ret12.median()*100:+.0f}% | 波动中位 {p.vol.median()*100:.0f}%")
    print(f"  2026-07~10: 中位 {p.fwd.median()*100:+.1f}% | 正收益 {(p.fwd>0).sum()}/{len(p)}")

    u = _panel()
    u["qm"] = pd.qcut(u.ret12, 5, labels=False)
    u["qv"] = pd.qcut(u.vol, 5, labels=False)
    def pick(n, qs):
        return u[u.qm.isin(qs)].nsmallest(n, "vol")
    alt = pd.concat([pick(6, [0, 1]), pick(10, [1, 2, 3]), pick(9, [2, 3])]
                    ).drop_duplicates("symbol").head(25)
    rng = np.random.default_rng(42)
    sims = [u.sample(25, random_state=i).fwd.mean() for i in range(500)]
    print("\n  对照(同一时间窗, 等权、不含成本):")
    print(f"    当前池             {p.fwd.mean()*100:+7.1f}%")
    print(f"    全样本(538)        {u.fwd.mean()*100:+7.1f}%")
    print(f"    最热五分之一        {u[u.qm==4].fwd.mean()*100:+7.1f}%")
    print(f"    风格均衡25(低波)    {alt.fwd.mean()*100:+7.1f}%  (正收益 {(alt.fwd>0).sum()}/{len(alt)})")
    print(f"    随机25 中位         {np.median(sims)*100:+7.1f}%  "
          f"(5-95 分位 {np.percentile(sims,5)*100:+.1f}~{np.percentile(sims,95)*100:+.1f}%)")
    print("  ⚠️ 单期对照, 不能当收益承诺; 用途是说明'拥挤度'的尾部风险量级")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["all", "ic", "style", "pool"])
    args = ap.parse_args()
    if args.section in ("all", "ic"):
        section_ic()
    if args.section in ("all", "style"):
        section_style()
    if args.section in ("all", "pool"):
        section_pool()


if __name__ == "__main__":
    main()
