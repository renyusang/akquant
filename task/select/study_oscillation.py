"""震荡期识别实证研究。

对 HS300 全池跑基线 SignalEngine,收集每个买入信号,按信号日的
ADX / 效率比ER / 布林带宽 / EMA间距 / NATR 分桶,统计信号后
10/20 根实际收益——回答:
  1. 震荡期买入是否平均亏损?
  2. 哪个指标最能预警震荡期(信号密度高 + 未来收益差)?

用法:
  python study_oscillation.py              # 全池
  python study_oscillation.py --symbols 002594,000001  # 调试
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalman"))

from signal_engine import SignalEngine, compute_adx

KALMAN_DIR = os.path.join(os.path.dirname(__file__), "..", "kalman")
CATALOG_DIR = os.path.join(KALMAN_DIR, ".catalog")

ENGINE_PARAMS = dict(
    kalman_q_price=1e-4, kalman_q_vel=1e-5, kalman_r=1e-2,
    entry_threshold=0.02, exit_threshold=0.005, stop_loss_pct=0.05,
    use_price_signal=True, use_velocity_signal=True,
    trend_filter_enabled=True, trend_confirm_bars=1,
    trend_bear_pct=0.30, downtrend_entry=0.03,
)


# ---- 指标序列(向量化) ----

def _ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def er_series(close, n=10):
    """Kaufman 效率比: |净位移| / 路径长度。"""
    out = np.full(len(close), np.nan)
    diff = np.abs(np.diff(close))
    for i in range(n, len(close)):
        net = abs(close[i] - close[i - n])
        tot = diff[i - n : i].sum()
        out[i] = net / tot if tot > 0 else 0.0
    return out


def bb_bandwidth(close, period=20, nbdev=2.0):
    """布林带带宽: (upper-lower)/middle。"""
    out = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        w = close[i - period + 1 : i + 1]
        mid = w.mean()
        out[i] = (2 * nbdev * w.std(ddof=0)) / mid if mid > 0 else np.nan
    return out


def ema_spread(close, fast=20, slow=60):
    """EMA 归一化间距: (ema_fast - ema_slow) / ema_slow。"""
    f, s = _ema(close, fast), _ema(close, slow)
    return np.abs((f - s) / s)


def natr_series(high, low, close, period=14):
    """NATR: ATR/close(Wilder 平滑)。"""
    n = len(close)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    out = np.full(n, np.nan)
    if n <= period:
        return out
    atr = np.full(n - 1, np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n - 1):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    out[1:] = atr / close[1:]
    return out


# ---- 单只股票信号扫描 ----

def scan_symbol(symbol):
    """跑基线 SignalEngine,收集买入信号及其指标/未来收益。"""
    path = os.path.join(CATALOG_DIR, symbol, "data.parquet")
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    if len(df) < 100:
        return []
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    # 指标序列
    adx = compute_adx(h, l, c, 14)
    er = er_series(c, 10)
    bw = bb_bandwidth(c, 20)
    emasp = ema_spread(c, 20, 60)
    natr = natr_series(h, l, c, 14)

    # 未来收益(T+1 开盘近似 close, 10/20 根)
    fwd10 = np.full(n, np.nan)
    fwd20 = np.full(n, np.nan)
    for i in range(n - 20):
        fwd10[i] = c[i + 10] / c[i + 1] - 1
        fwd20[i] = c[i + 20] / c[i + 1] - 1

    # 基线信号引擎(模拟持仓状态机, 记录交易对: 持仓时长 + 往返收益)
    engine = SignalEngine(**ENGINE_PARAMS)
    has_pos = False
    buy_i = -1
    buy_close = 0.0
    buy_rows = []
    for i in range(n):
        ma20_cur = float(c[max(0, i - 19) : i + 1].mean())
        ma20_prev = float(c[max(0, i - 20) : i].mean())
        engine.set_position(has_pos, 0.0)
        r = engine.update(c[i], ma20_cur, ma20_prev, h[i], l[i])
        if r["signal"] == "buy" and not has_pos:
            has_pos = True
            buy_i = i
            buy_close = c[i]
        elif r["signal"] == "sell" and has_pos:
            # T+1 开盘成交近似: 卖出价 ≈ 信号日次日收盘
            sell_price = c[i + 1] if i + 1 < n else c[i]
            hold = i - buy_i
            buy_rows.append(
                {
                    "symbol": symbol,
                    "i": buy_i,
                    "adx": adx[buy_i],
                    "er": er[buy_i],
                    "bw": bw[buy_i],
                    "ema_spread": emasp[buy_i],
                    "natr": natr[buy_i],
                    "hold_bars": hold,
                    "round_trip": sell_price / buy_close - 1.0,
                }
            )
            has_pos = False
    return buy_rows


def _worker(symbol):
    try:
        return scan_symbol(symbol)
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--output", default="oscillation_signals.csv")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        import yaml

        with open(os.path.join(KALMAN_DIR, "stocks_hs300.yaml")) as f:
            cfg = yaml.safe_load(f)
        symbols = list(dict.fromkeys(
            s["symbol"] for s in cfg.get("watchlist", {}).get("stocks", [])
        ))

    print(f"扫描 {len(symbols)} 只股票的买入信号...")
    all_rows = []
    with mp.Pool(max(1, min(8, os.cpu_count() or 4))) as pool:
        for rows in pool.imap_unordered(_worker, symbols, chunksize=4):
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(os.path.dirname(__file__), args.output),
              index=False, encoding="utf-8-sig")
    print(f"[保存] {args.output}: {len(df)} 个买入信号\n")

    # ---- 分桶统计(交易对视角) ----
    n = len(df)
    if not n:
        print("无信号!")
        return

    def bucket_report(name, col, edges, labels):
        print(f"\n=== 按 {name} 分桶(完整交易对: 买入→卖出) ===")
        print(f"{'桶':<18} {'交易数':>6} {'往返均值':>9} {'往返中位':>8} "
              f"{'亏损率':>7} {'≤5根快速':>9} {'持仓中位':>7}")
        binned = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)
        for lab in labels:
            m = binned == lab
            if not m.sum():
                continue
            rt = df.loc[m, "round_trip"].dropna()
            if not len(rt):
                continue
            h = df.loc[m, "hold_bars"].dropna()
            fast = (h <= 5).mean() * 100 if len(h) else 0.0
            print(
                f"{lab:<18} {m.sum():>6} {rt.mean()*100:>+9.2f}% {rt.median()*100:>+8.2f}% "
                f"{(rt < 0).mean()*100:>6.1f}% {fast:>8.1f}% {h.median():>7.0f}"
            )

    # 全量基准
    rt = df["round_trip"].dropna()
    h = df["hold_bars"].dropna()
    print(f"全部交易对: {len(rt)} 个, 往返均值 {rt.mean()*100:+.2f}%, "
          f"中位 {rt.median()*100:+.2f}%, 亏损率 {(rt<0).mean()*100:.1f}%, "
          f"≤5根快速反转 {(h<=5).mean()*100:.1f}%, 持仓中位 {h.median():.0f}根")

    bucket_report("ADX(14)", "adx", [-1, 20, 25, 200],
                  ["ADX<20(震荡)", "20-25(过渡)", "ADX>=25(趋势)"])
    bucket_report("效率比ER(10)", "er", [-0.01, 0.3, 0.5, 1.01],
                  ["ER<0.3(震荡)", "0.3-0.5", "ER>=0.5(趋势)"])
    bucket_report("布林带宽", "bw", [-1, 0.08, 0.15, 1.0],
                  ["带宽<8%(挤压)", "8-15%", ">15%(扩张)"])
    bucket_report("EMA间距", "ema_spread", [-1, 0.02, 0.05, 1.0],
                  ["间距<2%", "2-5%", ">5%(发散)"])
    bucket_report("NATR(14)", "natr", [-1, 0.015, 0.03, 1.0],
                  ["NATR<1.5%(低波)", "1.5-3%", ">3%(高波)"])

    # 快速反转交易(≤5根)的往返收益 vs 慢交易
    fast = df[df["hold_bars"] <= 5]
    slow = df[df["hold_bars"] > 5]
    if len(fast) and len(slow):
        print(f"\n=== 快速反转(≤5根) vs 正常持仓 ===")
        print(f"  快速: {len(fast)} 笔({len(fast)/n*100:.1f}%), 往返均值 "
              f"{fast['round_trip'].mean()*100:+.3f}%, 亏损率 "
              f"{(fast['round_trip']<0).mean()*100:.1f}%")
        print(f"  正常: {len(slow)} 笔, 往返均值 {slow['round_trip'].mean()*100:+.3f}%, "
              f"亏损率 {(slow['round_trip']<0).mean()*100:.1f}%")
        print(f"  快速交易贡献的亏损占总亏损比例: "
              f"{fast['round_trip'].clip(upper=0).sum() / df['round_trip'].clip(upper=0).sum()*100:.1f}%")


if __name__ == "__main__":
    main()
