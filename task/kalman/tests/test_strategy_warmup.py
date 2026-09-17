"""Tests for 预热语义(2026-09-17 研究): per_symbol_warmup_bars / prime_enabled。

背景: 框架 `warmup_bars` 是**跨标的全局事件计数**(25 只池里 40 个事件
≈ 1 个交易日), 因此每只标的的冷启动期仍会被评估; 而实盘 `daily_signal`
用 `process_history(全历史)` 预热后才评估当日, 从不评估无历史的标的。

实测结论(见 kalman.md):
- `process_history` 预热 ≡ 逐 bar 增量 update(状态/信号逐根一致)
- 简单在早期禁交易(`per_symbol_warmup_bars`)大幅有害(-83pp)——
  那是"少了交易机会", 不是"去掉了假信号"
- 用**窗口前真实历史**预热(`prime_enabled` + prime_data)结果不降反升
  (+2.1pp / 夏普 +0.09)
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.disable(logging.CRITICAL)

from akquant import run_backtest                                    # noqa: E402
from akquant.backtest.engine import make_fill_policy                # noqa: E402

from signal_engine import SignalEngine                              # noqa: E402
from strategy import KalmanStrategy                                 # noqa: E402

N_BARS = 80
BREAKOUT_BAR = 20   # ⚠️ 必须 > 框架全局门闸: 突破日事件数
                    # (breakout_bar × n_symbols) 需 > warmup_bars(40),
                    # 否则 on_bar 根本不被调用(全局计数器的直接体现)


def _pool(breakout_bar=BREAKOUT_BAR, n_symbols=3, flat_after=True):
    """突破发生在第 BREAKOUT_BAR 根(×3 只 = 60 事件 > 框架全局门闸 40)。

    flat_after=True: 突破后保持平盘 → 只有一次入场机会, 断言可确定
    (否则后续还会"回归卖出 → 再突破再买"多轮)。
    """
    dates = pd.date_range("2026-01-01", periods=N_BARS, freq="B")
    out = {}
    for i in range(n_symbols):
        sym = f"{600000 + i:06d}"
        close = np.full(N_BARS, 100.0)
        close[breakout_bar] = 105.0
        close[breakout_bar + 1:] = 105.0 if flat_after else 105.0 * 1.002
        out[sym] = pd.DataFrame({
            "date": dates, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": [1_000_000] * N_BARS, "symbol": [sym] * N_BARS,
        })
    return out


def _run(data_map, **overrides):
    fill_policy = make_fill_policy(
        price_basis="open", temporal="next_event", bar_offset=1)
    strategy = KalmanStrategy(
        initial_cash=1000000, entry_threshold=0.02, exit_threshold=0.005,
        stop_loss_pct=0.05, trend_filter_enabled=False,
        single_position_pct=0.1, max_positions=5, **overrides)
    return run_backtest(
        data=data_map, strategy=strategy, symbols=list(data_map),
        initial_cash=1000000, t_plus_one=True, fill_policy=fill_policy,
        lot_size={s: 100 for s in data_map}, commission_rate=0.0003,
        stamp_tax_rate=0.001, transfer_fee_rate=0.00001,
        min_commission=5.0, show_progress=False)


def _buys(result):
    """买入订单(策略会"卖出→再突破→再买"多轮, 断言请用标的数见 _bought)。"""
    o = result.orders_df
    if o is None or o.empty:
        return o
    f = o[o["status"] == "filled"]
    return f[f["side"].astype(str).str.lower() == "buy"]


def _bought(result):
    """实际买入过的标的集合(唯一)。"""
    b = _buys(result)
    if b is None or b.empty:
        return set()
    return {str(s).zfill(6) for s in b["symbol"].unique()}


class TestPerSymbolWarmup:
    """per_symbol_warmup_bars: 按标的自身 bar 数抑制交易(引擎照常喂)。"""

    def test_disabled_by_default(self):
        """默认 0 = 行为不变: 突破照常成交。"""
        r = _run(_pool(), use_velocity_signal=False)
        assert len(_bought(r)) == 3

    def test_suppresses_early_trades(self):
        """预热 20 根 > 突破时点的标的 bar 数 → 早期突破被抑制。

        关掉速度信号: 否则平盘期还会产生"速度反转"型入场(与本测试无关)。
        """
        r = _run(_pool(), per_symbol_warmup_bars=20,
                 use_velocity_signal=False)
        assert len(_bought(r)) == 0

    def test_late_breakout_still_trades(self):
        """突破发生在预热之后 → 照常成交(只抑制冷启动期)。"""
        r = _run(_pool(breakout_bar=40), per_symbol_warmup_bars=20)
        assert len(_bought(r)) == 3

    def test_engine_still_fed_during_warmup(self):
        """预热期只是不交易, 引擎仍被喂(否则预热结束状态是冷的)。"""
        r = _run(_pool(breakout_bar=40), per_symbol_warmup_bars=20)
        st = r.strategy
        # 框架全局门闸会跳过最初 ~13 根(3 只池), 断言"喂到门闸之后"
        assert st._symbol_bars.get("600000", 0) >= 40


class TestPrime:
    """prime_enabled: 用窗口前历史在首次 on_bar 预热引擎。"""

    @staticmethod
    def _prime_df(n=80, base=50.0):
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        close = np.linspace(base, base * 1.5, n)          # 明显上升趋势
        return pd.DataFrame({
            "date": dates, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": [1_000_000] * n,
        })

    def test_disabled_by_default(self, monkeypatch):
        called = []
        monkeypatch.setattr(SignalEngine, "process_history",
                            lambda self, df: called.append(len(df)))
        r = _run(_pool(), prime_data={"600000": self._prime_df()})
        assert called == [] and len(_bought(r)) == 3

    def test_primes_engine_with_provided_history(self, monkeypatch):
        called = []
        monkeypatch.setattr(SignalEngine, "process_history",
                            lambda self, df: called.append(len(df)))
        _run(_pool(), prime_enabled=True,
             prime_data={"600000": self._prime_df(n=80)})
        assert called == [80], f"process_history 未被调用或参数错误: {called}"

    def test_prime_affects_engine_state(self):
        """预热必须真的改变引擎状态(否则只是空调用)。"""
        df = self._prime_df()
        eng_a = SignalEngine(trend_filter_enabled=False)
        eng_b = SignalEngine(trend_filter_enabled=False)
        eng_b.process_history(df)
        eng_a.update(105.0, 100.0, 100.0, 106.0, 104.0, 1e6)
        eng_b.update(105.0, 100.0, 100.0, 106.0, 104.0, 1e6)
        assert abs(float(eng_a.filtered_price) - float(eng_b.filtered_price)) > 1e-6

    def test_missing_symbol_in_prime_data_is_safe(self):
        """prime_data 缺该标的 → 静默跳过, 不影响交易。"""
        r = _run(_pool(), prime_enabled=True, prime_data={})
        assert len(_bought(r)) == 3
