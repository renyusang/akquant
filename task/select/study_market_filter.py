"""大盘环境过滤研究(信号级): 大盘弱市时买入信号质量是否更差。

流程:
  1. 沪深300 定义大盘状态(强: close>MA20且MA20↑ / 弱: close<MA20且MA20↓ / 过渡)
  2. 对候选池每只股票逐 bar 喂 SignalEngine, 收集买入信号
  3. 按信号日大盘状态分桶, 比较信号后 20 根收益/亏损率/快速反转率

用法:
  python study_market_filter.py                    # HS300 全池
  python study_market_filter.py --symbols 002594,600519
  python study_market_filter.py --config ../../kalman/stocks.yaml  # 自定义池
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalman"))

from data_utils import fetch_hs300  # noqa: E402

KALMAN_DIR = os.path.join(os.path.dirname(__file__), "..", "kalman")
CATALOG_DIR = os.path.join(KALMAN_DIR, ".catalog")

# 与实盘一致的策略参数
ENGINE_PARAMS = dict(
    kalman_q_price=0.0001, kalman_q_vel=0.00001, kalman_r=0.01,
    entry_threshold=0.02, exit_threshold=0.005, stop_loss_pct=0.05,
    use_price_signal=True, use_velocity_signal=True,
    trend_filter_enabled=True, trend_confirm_bars=1,
    trend_bear_position_pct=0.3, downtrend_entry_threshold=0.03,
    atr_adaptive_exit_enabled=True, exit_atr_factor=1.0,
    sar_exit_enabled=False,
)

HORIZON = 20  # 信号后观察根数


def build_market_state() -> pd.Series:
    """沪深300 大盘状态序列: date → 强/弱/过渡。"""
    hs = fetch_hs300("20200101", "20260814")
    ma20 = hs.rolling(20).mean()
    st = pd.DataFrame({"close": hs, "ma20": ma20})
    above = st["close"] > st["ma20"]
    up = st["ma20"] > st["ma20"].shift(1)
    state = pd.Series("过渡", index=st.index)
    state[above & up] = "强"
    state[~above & ~up] = "弱"
    return state


def scan_symbol(symbol: str, market_state: pd.Series) -> list[dict]:
    """逐 bar 喂 SignalEngine, 收集买入信号并按大盘状态标注后验收益。"""
    from signal_engine import SignalEngine

    path = os.path.join(CATALOG_DIR, symbol, "data.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    if len(df) < 120:
        return []

    engine = SignalEngine(**ENGINE_PARAMS)
    engine.set_position(False, 0.0)
    closes = df["close"].astype(float).values
    dates = df["date"].dt.strftime("%Y-%m-%d")
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    vols = df["volume"].astype(float).values

    rows = []
    ma20_prev = float(df["close"].iloc[:20].mean())
    for i in range(40, len(df)):
        ma20_cur = float(closes[i - 19:i + 1].mean())
        res = engine.update(closes[i], ma20_cur, ma20_prev,
                            high=highs[i], low=lows[i], volume=vols[i])
        ma20_prev = ma20_cur
        if res.get("signal") == "buy":
            d = dates.iloc[i]
            st = market_state.get(d, "过渡")
            # 信号后 HORIZON 根收益
            if i + HORIZON < len(df):
                fwd = (closes[i + HORIZON] / closes[i] - 1) * 100
            else:
                fwd = np.nan
            rows.append({"symbol": symbol, "date": d, "market": st,
                         "fwd_ret": fwd, "hold": HORIZON})
    return rows


def _worker(args):
    symbol, market_state = args
    try:
        return scan_symbol(symbol, market_state)
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="大盘环境过滤信号级研究")
    parser.add_argument("--config", default=os.path.join(
        KALMAN_DIR, "stocks_hs300.yaml"), help="候选池配置")
    parser.add_argument("--symbols", default=None, help="指定标的(调试)")
    parser.add_argument("--output", default="market_filter_signals.csv")
    args = parser.parse_args()

    # 标的列表
    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",")]
    else:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        symbols = [str(s["symbol"]).zfill(6)
                   for s in cfg["watchlist"]["stocks"]]
    print(f"候选: {len(symbols)} 只")

    market_state = build_market_state()
    print(f"大盘状态: 强 {sum(market_state=='强')} / "
          f"弱 {sum(market_state=='弱')} / 过渡 {sum(market_state=='过渡')} 天")

    print("扫描买入信号...")
    with mp.Pool(max(1, min(8, os.cpu_count() or 4))) as pool:
        results = []
        for i, rows in enumerate(pool.imap_unordered(
                _worker, [(s, market_state) for s in symbols], chunksize=4)):
            results.extend(rows)
            if (i + 1) % 60 == 0:
                print(f"  进度 {i + 1}/{len(symbols)}", flush=True)
    df = pd.DataFrame(results)
    if df.empty:
        print("无信号")
        return
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"买入信号: {len(df)} 个 → {args.output}")

    # 分桶统计
    d = df.dropna(subset=["fwd_ret"])
    print("\n=== 买入信号后 20 根收益 × 大盘状态 ===")
    for st in ["强", "过渡", "弱"]:
        sub = d[d["market"] == st]
        if not len(sub):
            continue
        fast = sub[sub["fwd_ret"] < -2]  # 快速亏损
        print(f"{st:<4} 信号 {len(sub):>5} 个 | 均值 {sub['fwd_ret'].mean():>+6.2f}% "
              f"| 中位 {sub['fwd_ret'].median():>+6.2f}% "
              f"| 亏损率 {(sub['fwd_ret'] < 0).mean():.1%} "
              f"| 亏损>2%占比 {len(fast)/len(sub):.1%}")
    # 均值差显著性(简单 t 检验)
    strong = d[d["market"] == "强"]["fwd_ret"].dropna()
    weak = d[d["market"] == "弱"]["fwd_ret"].dropna()
    if len(strong) > 10 and len(weak) > 10:
        from scipy import stats
        t, p = stats.ttest_ind(strong, weak)
        print(f"\n强 vs 弱 均值差: {strong.mean() - weak.mean():+.2f}pp "
              f"(t={t:.2f}, p={p:.4f})")


if __name__ == "__main__":
    main()
