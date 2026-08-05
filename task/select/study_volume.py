"""量价确认象限实证研究。

对 HS300 全池跑基线 SignalEngine, 收集每个买入信号, 计算信号日的
量价指标(量比/MFI/OBV方向/量能趋势), 分桶统计信号后 20 根收益与
交易对质量(hold_bars/round_trip)——回答:
  1. 放量/缩量信号的质量差异?
  2. 资金动量(MFI)能否区分信号质量?
  3. 哪个量价指标最能预警坏信号?

用法:
  python study_volume.py
"""

import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalman"))

from signal_engine import SignalEngine

KALMAN_DIR = os.path.join(os.path.dirname(__file__), "..", "kalman")
CATALOG_DIR = os.path.join(KALMAN_DIR, ".catalog")

ENGINE_PARAMS = dict(
    kalman_q_price=1e-4, kalman_q_vel=1e-5, kalman_r=1e-2,
    entry_threshold=0.02, exit_threshold=0.005, stop_loss_pct=0.05,
    use_price_signal=True, use_velocity_signal=True,
    trend_filter_enabled=True, trend_confirm_bars=1,
    trend_bear_pct=0.30, downtrend_entry=0.03,
)


# ---- 量价指标(向量化) ----

def vol_ratio_series(volume, n=5):
    """量比 = 当日量 / 前 n 日均量。"""
    out = np.full(len(volume), np.nan)
    for i in range(n, len(volume)):
        base = volume[i - n : i].mean()
        out[i] = volume[i] / base if base > 0 else np.nan
    return out


def mfi_series(high, low, close, volume, period=14):
    """Money Flow Index: 典型价×成交量 的资金流入占比。"""
    n = len(close)
    tp = (high + low + close) / 3.0
    mf = tp * volume
    out = np.full(n, np.nan)
    for i in range(period, n):
        pos = neg = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos += mf[j]
            elif tp[j] < tp[j - 1]:
                neg += mf[j]
        out[i] = 100.0 if neg == 0 else 100.0 - 100.0 / (1.0 + pos / neg)
    return out


def obv_direction(close, volume, lookback=20):
    """OBV 方向: obv[i] > obv[i-lookback] 记为 1, 否则 0。"""
    n = len(close)
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        out[i] = 1.0 if obv[i] > obv[i - lookback] else 0.0
    return out


def vol_trend_series(volume, fast=20, lookback=10):
    """量能趋势: vol_ma_fast[i] > vol_ma_fast[i-lookback] 记为 1。"""
    n = len(volume)
    out = np.full(n, np.nan)
    for i in range(fast + lookback, n):
        a = volume[i - fast + 1 : i + 1].mean()
        b = volume[i - fast + 1 - lookback : i + 1 - lookback].mean()
        out[i] = 1.0 if a > b else 0.0
    return out


# ---- 单只股票扫描 ----

def scan_symbol(symbol):
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
    v = df["volume"].to_numpy(float)
    n = len(c)

    vr = vol_ratio_series(v, 5)
    mfi = mfi_series(h, l, c, v, 14)
    obv = obv_direction(c, v, 20)
    vt = vol_trend_series(v, 20, 10)

    fwd20 = np.full(n, np.nan)
    for i in range(n - 20):
        fwd20[i] = c[i + 20] / c[i + 1] - 1

    engine = SignalEngine(**ENGINE_PARAMS)
    has_pos = False
    buy_i = -1
    buy_close = 0.0
    rows = []
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
            sell_price = c[i + 1] if i + 1 < n else c[i]
            rows.append({
                "symbol": symbol,
                "i": buy_i,
                "vol_ratio": vr[buy_i],
                "mfi": mfi[buy_i],
                "obv_up": obv[buy_i],
                "vol_trend": vt[buy_i],
                "fwd20": fwd20[buy_i],
                "hold_bars": i - buy_i,
                "round_trip": sell_price / buy_close - 1.0,
            })
            has_pos = False
    return rows


def _worker(symbol):
    try:
        return scan_symbol(symbol)
    except Exception:
        return []


def main():
    import yaml

    with open(os.path.join(KALMAN_DIR, "stocks_hs300.yaml")) as f:
        cfg = yaml.safe_load(f)
    symbols = list(dict.fromkeys(
        s["symbol"] for s in cfg.get("watchlist", {}).get("stocks", [])
    ))
    print(f"扫描 {len(symbols)} 只股票的买入信号(量价指标)...")
    all_rows = []
    with mp.Pool(max(1, min(8, os.cpu_count() or 4))) as pool:
        for rows in pool.imap_unordered(_worker, symbols, chunksize=4):
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(os.path.dirname(__file__), "volume_signals.csv"),
              index=False, encoding="utf-8-sig")
    print(f"[保存] volume_signals.csv: {len(df)} 个交易对\n")

    n = len(df)
    rt = df["round_trip"].dropna()
    print(f"全部交易对: {len(rt)} 个, 往返均值 {rt.mean()*100:+.2f}%, "
          f"亏损率 {(rt<0).mean()*100:.1f}%, ≤5根快速 {(df['hold_bars']<=5).mean()*100:.1f}%")

    def bucket_report(name, col, edges, labels):
        print(f"\n=== 按 {name} 分桶 ===")
        print(f"{'桶':<14} {'交易数':>6} {'往返均值':>9} {'亏损率':>7} "
              f"{'≤5根':>6} {'20根均':>8}")
        binned = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)
        for lab in labels:
            m = binned == lab
            if not m.sum():
                continue
            r = df.loc[m, "round_trip"].dropna()
            f20 = df.loc[m, "fwd20"].dropna()
            print(
                f"{lab:<14} {m.sum():>6} {r.mean()*100:>+9.2f}% "
                f"{(r<0).mean()*100:>6.1f}% {(df.loc[m,'hold_bars']<=5).mean()*100:>5.1f}% "
                f"{f20.mean()*100:>+8.2f}%"
            )

    bucket_report("量比(vol/5日均量)", "vol_ratio", [-1, 0.8, 1.5, 100],
                  ["<0.8(缩量)", "0.8-1.5(平量)", ">1.5(放量)"])
    bucket_report("MFI(14)", "mfi", [-1, 30, 50, 70, 101],
                  ["<30", "30-50", "50-70", ">70"])
    bucket_report("OBV方向(20日)", "obv_up", [-0.5, 0.5, 1.5],
                  ["OBV↓", "OBV↑"])
    bucket_report("量能趋势(20日)", "vol_trend", [-0.5, 0.5, 1.5],
                  ["量能↓", "量能↑"])


if __name__ == "__main__":
    main()
