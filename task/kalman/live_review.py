"""实盘评估复现脚本(2026-10-19)。

从本地状态(trades.csv / execution_log.csv / positions.json / .cache)重建
逐日权益, 输出与 `live_review_2026Q3.md` 同口径的指标。

口径:
- 收益 = 时间加权(TWR): 账户在 2026-08-21 股票池 20万→30万, 按入金日剔除
  资本跳变; 全程 40 万口径的结果一并给出(A/B 两行)。
- 已实现 = trades.csv 的 pnl 之和; 浮动 = (最新价-成本)×股数。
- 换手 = Σ成交额(买+卖)/本金; 季度化按 63 个交易日。
- 基准用 akshare 指数日线, 窗口与账户一致。

用法:
    python live_review.py            # 打印全部指标
    python live_review.py --json     # 机器可读
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

CACHE = os.path.join(TASK_DIR, ".cache")
START, END = "2026-07-17", "2026-10-19"


def _trading_days():
    import akshare as ak
    d = ak.stock_zh_index_daily(symbol="sh000300")
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].loc[START:END].index


def load_state():
    from portfolio import calc_fee
    el = pd.read_csv(os.path.join(TASK_DIR, "execution_log.csv"), dtype={"symbol": str})
    el["symbol"] = el["symbol"].str.zfill(6)
    ex = el[el.status == "executed"].copy()
    ex["d"] = pd.to_datetime(ex["exec_date"])
    buys = ex[ex.action == "buy"].copy()
    buys["fee"] = [calc_fee(int(s), float(p), "buy")
                   for s, p in zip(buys.shares, buys.exec_price)]
    tr = pd.read_csv(os.path.join(TASK_DIR, "trades.csv"))
    tr["symbol"] = tr["symbol"].astype(str).str.zfill(6)
    tr["d"] = pd.to_datetime(tr["exit_date"])
    return buys, tr


def daily_equity(cal, buys, tr):
    prices = {}
    for f in os.listdir(CACHE):
        if f.endswith(".parquet"):
            try:
                prices[f[:-8]] = pd.read_parquet(os.path.join(CACHE, f)).set_index("date")["close"]
            except Exception:
                pass
    bev = {d: g for d, g in buys.groupby("d")}
    sev = {d: g for d, g in tr.groupby("d")}
    pos, rows = {}, []
    for d in cal:
        for _, r in bev.get(d, pd.DataFrame()).iterrows():
            p = pos.setdefault(r.symbol, [0.0, 0.0])
            p[0] += r.shares
            p[1] += r.shares * r.exec_price + r.fee
        for _, r in sev.get(d, pd.DataFrame()).iterrows():
            p = pos.get(r.symbol)
            if p:
                avg = p[1] / p[0]
                p[0] -= r.shares
                p[1] = max(0.0, p[1] - avg * r.shares)
                if p[0] <= 1e-9:
                    pos.pop(r.symbol, None)
        mkt = sum(sh * float(prices[s].loc[:d].iloc[-1])
                  for s, (sh, ct) in pos.items() if s in prices)
        rows.append({"date": d, "mkt": mkt,
                     "cost": sum(ct for _, ct in pos.values()),
                     "realized": tr[tr.d <= d].pnl.sum()})
    return pd.DataFrame(rows).set_index("date")


def metrics(df, mode):
    cap = pd.Series(
        np.where(df.index < pd.Timestamp("2026-08-21"), 300000.0, 400000.0)
        if mode == "B" else 400000.0, index=df.index)
    eq = cap + df.realized + (df.mkt - df.cost)
    ret = (eq.pct_change() - cap.diff().fillna(0) / eq.shift(1)).dropna()
    twr = float((1 + ret).prod() - 1)
    vol = float(ret.std() * np.sqrt(252))
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "mode": mode, "final": float(eq.iloc[-1]), "twr": twr,
        "ann": float((1 + twr) ** (252 / len(ret)) - 1), "vol": vol,
        "sharpe": float((ret.mean() * 252 - 0.015) / vol) if vol else 0.0,
        "mdd": dd, "exposure": float((df.mkt / eq).mean()),
        "peak": float(eq.max()),
        "peak_date": str(eq.idxmax().date()),
        "ret_series": ret,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cal = _trading_days()
    buys, tr = load_state()
    df = daily_equity(cal, buys, tr)
    A, B = metrics(df, "A"), metrics(df, "B")

    # 基准
    import akshare as ak
    bench = {}
    for name, sym in [("沪深300", "sh000300"), ("中证500", "sh000905"),
                      ("中证1000", "sh000852"), ("创业板指", "sz399006"),
                      ("科创50", "sh000688"), ("中证银行", "sz399986"),
                      ("中证主要消费", "sh000932")]:
        d = ak.stock_zh_index_daily(symbol=sym)
        d["date"] = pd.to_datetime(d["date"])
        s = d.set_index("date")["close"].loc[START:END]
        bench[name] = {"ret": float(s.iloc[-1] / s.iloc[0] - 1),
                       "mdd": float((s / s.cummax() - 1).min())}

    # 交易统计
    p = tr.pnl.values
    fees = float(tr.fee.fillna(0).sum()
                 + sum(__import__("portfolio").calc_fee(int(s), float(pr), "buy")
                       for s, pr in zip(buys.shares, buys.exec_price)))
    buy_val = float((buys.shares * buys.exec_price).sum())
    sell_val = float((tr.shares * tr.exit_price).sum())
    stats = {
        "trades": int(len(tr)), "wins": int((p > 0).sum()),
        "avg_win": float(p[p > 0].mean()) if (p > 0).any() else 0.0,
        "avg_loss": float(p[p < 0].mean()) if (p < 0).any() else 0.0,
        "expectancy": float(p.mean()), "t": float(p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))),
        "fees": fees, "turnover_q": (buy_val + sell_val) / 400000 / (len(cal) / 63),
        "hold_median": float((pd.to_datetime(tr.exit_date) - pd.to_datetime(tr.entry_date)).dt.days.median()),
    }
    out = {"span": [START, END, len(cal)], "A": A, "B": B,
           "bench": bench, "stats": stats}
    if args.json:
        out.pop("ret_series", None)
        for m in ("A", "B"):
            out[m].pop("ret_series", None)
        print(json.dumps(out, ensure_ascii=False, indent=2, default=float))
        return

    print(f"区间 {START} ~ {END} ({len(cal)} 个交易日)")
    for m in (A, B):
        print(f"[{m['mode']}] 期末 ¥{m['final']:,.0f} | TWR {m['twr']*100:+.2f}% | "
              f"年化 {m['ann']*100:+.1f}% | 波动 {m['vol']*100:.1f}% | "
              f"Sharpe {m['sharpe']:.2f} | MDD {m['mdd']*100:.2f}% | "
              f"平均仓位 {m['exposure']*100:.1f}% | 峰值 {m['peak']:,.0f}({m['peak_date']})")
    print("\n基准:")
    for k, v in bench.items():
        print(f"  {k:10s} {v['ret']*100:+7.2f}% | MDD {v['mdd']*100:6.2f}%")
    print(f"\n交易: {stats['trades']} 笔 | 胜 {stats['wins']} | "
          f"均盈 {stats['avg_win']:+.0f} | 均亏 {stats['avg_loss']:+.0f} | "
          f"期望 {stats['expectancy']:+.0f}(t={stats['t']:.2f})")
    print(f"费用 ¥{stats['fees']:,.0f} | 换手 {stats['turnover_q']:.1f} 倍/季 | "
          f"持仓中位 {stats['hold_median']:.0f} 天")


if __name__ == "__main__":
    main()
