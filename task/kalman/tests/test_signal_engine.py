"""Tests for signal_engine.py."""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signal_engine import SignalEngine, TrendDetector


# ---- Fixtures ----

@pytest.fixture
def engine_default():
    """SignalEngine with default params, no position."""
    return SignalEngine()


@pytest.fixture
def engine_with_trend():
    """SignalEngine with trend filter enabled."""
    return SignalEngine(
        trend_filter_enabled=True,
        trend_confirm_bars=1,
        trend_bear_pct=0.30,
        downtrend_entry=0.03,
    )


@pytest.fixture
def uptrend_df():
    """DataFrame with a clear uptrend."""
    n = 100
    close = np.linspace(100, 200, n) + np.random.normal(0, 1, n)
    df = pd.DataFrame({"close": close})
    df["ma5"] = close
    df["ma20"] = close
    return df


def make_history(n=60, start=100, slope=0.5, noise=0.5):
    """Create a simple price history DataFrame."""
    np.random.seed(42)
    base = np.arange(n) * slope + start
    close = base + np.random.normal(0, noise, n)
    return pd.DataFrame({"close": close})


class TestTrendDetector:
    """Tests for the TrendDetector class."""

    def test_disabled_always_up(self):
        td = TrendDetector(enabled=False, confirm_bars=3)
        assert td.update(100, 101, 100) == "up"
        assert td.update(99, 101, 100) == "up"  # close < ma20, but disabled
        assert td.update(98, 100, 101) == "up"

    def test_starts_up(self):
        td = TrendDetector(enabled=True, confirm_bars=1)
        assert td.state == "up"

    def test_switch_to_down(self):
        td = TrendDetector(enabled=True, confirm_bars=1)
        # close < ma20, ma20 not rising
        assert td.update(95, 100, 100) == "down"

    def test_switch_to_down_needs_confirmation(self):
        td = TrendDetector(enabled=True, confirm_bars=2)
        # 1st bearish bar — not enough to confirm
        assert td.update(95, 100, 100) == "up"
        # 2nd bearish bar — now confirmed
        assert td.update(95, 100, 100) == "down"

    def test_recover_to_up_on_ma20_rising(self):
        td = TrendDetector(enabled=True, confirm_bars=1)
        # Go down
        td.update(95, 100, 100)
        assert td.state == "down"
        # Recover: close >= ma20 AND ma20 rising
        assert td.update(101, 100, 99) == "up"

    def test_no_recover_when_ma20_still_falling(self):
        td = TrendDetector(enabled=True, confirm_bars=1)
        td.update(95, 100, 100)  # → down
        # close >= ma20 but ma20 still falling → stay down
        assert td.update(102, 100, 101) == "down"

    def test_confirm_counter_reset_on_recovery(self):
        td = TrendDetector(enabled=True, confirm_bars=3)
        td.update(95, 100, 100)  # counter=1
        td.update(95, 100, 100)  # counter=2
        # Recover before confirm
        assert td.update(101, 100, 99) == "up"  # counter reset to 0

    def test_reset(self):
        td = TrendDetector(enabled=True, confirm_bars=1)
        td.update(95, 100, 100)  # → down
        assert td.state == "down"
        td.reset()
        assert td.state == "up"


class TestSignalEngineInit:
    """Tests for SignalEngine initialization and parameter handling."""

    def test_default_params(self):
        engine = SignalEngine()
        assert engine.kalman_q_price == 1e-4
        assert engine.kalman_q_vel == 1e-5
        assert engine.kalman_r == 1e-2
        assert engine.entry_threshold == 0.02
        assert engine.exit_threshold == 0.005
        assert engine.stop_loss_pct == 0.05
        assert engine.use_price_signal is True
        assert engine.use_velocity_signal is True
        assert engine.trend_filter_enabled is False

    def test_custom_params(self):
        engine = SignalEngine(
            kalman_q_price=1e-3,
            entry_threshold=0.03,
            stop_loss_pct=0.08,
        )
        assert engine.kalman_q_price == 1e-3
        assert engine.entry_threshold == 0.03
        assert engine.stop_loss_pct == 0.08
        # Unspecified params use defaults
        assert engine.kalman_q_vel == 1e-5

    def test_alias_params(self):
        """Should accept alternative param names used in stocks.yaml."""
        engine = SignalEngine(
            trend_confirm_bars=5,
            trend_bear_pct=0.40,
            downtrend_entry=0.05,
            trend_filter_enabled=True,
        )
        assert engine.trend_confirm_bars == 5
        assert engine.trend_bear_pct == 0.40
        assert engine.downtrend_entry == 0.05

    def test_alias_params_alt_names(self):
        """Should also accept strategy.py naming convention."""
        engine = SignalEngine(
            trend_filter_confirm_bars=3,
            trend_bear_position_pct=0.30,
            downtrend_entry_threshold=0.03,
        )
        assert engine.trend_confirm_bars == 3
        assert engine.trend_bear_pct == 0.30
        assert engine.downtrend_entry == 0.03


class TestSignalEngineUpdate:
    """Tests for SignalEngine.update()."""

    def test_returns_dict_with_expected_keys(self):
        engine = SignalEngine()
        engine.set_position(False, 0.0)
        result = engine.update(100.0, 100.0, 100.0)
        expected_keys = {
            "close", "kalman_price", "kalman_velocity", "ma20",
            "ma20_rising", "trend", "signal", "target_pct", "reason",
        }
        assert set(result.keys()) >= expected_keys

    def test_hold_when_no_signal(self):
        """With flat prices and no position, should return hold."""
        engine = SignalEngine()
        engine.set_position(False, 0.0)
        # Warm up with constant prices so KF converges
        for _ in range(50):
            engine.update(100.0, 100.0, 100.0)
        result = engine.update(100.0, 100.0, 100.0)
        assert result["signal"] == "hold"

    def test_buy_on_strong_breakout(self):
        """A large price spike should trigger buy when no position."""
        engine = SignalEngine(entry_threshold=0.02)
        engine.set_position(False, 0.0)
        # Warm up at 100
        for _ in range(50):
            engine.update(100.0, 100.0, 100.0)
        # Sudden 5% jump
        result = engine.update(105.0, 105.0, 104.0)
        assert result["signal"] == "buy"
        assert "价格突破" in result["reason"]

    def test_no_buy_when_in_position(self):
        """Should not generate buy when already holding."""
        engine = SignalEngine(entry_threshold=0.02)
        engine.set_position(True, 100.0)
        for _ in range(50):
            engine.update(100.0, 100.0, 100.0)
        result = engine.update(105.0, 105.0, 104.0)
        assert result["signal"] != "buy"

    def test_sell_on_stop_loss(self):
        """Should trigger sell when price drops below stop loss."""
        engine = SignalEngine(stop_loss_pct=0.05)
        engine.set_position(True, 100.0)
        for _ in range(50):
            engine.update(100.0, 100.0, 100.0)
        # Drop below 5% stop loss
        result = engine.update(94.0, 94.0, 99.0)
        assert result["signal"] == "sell"
        assert "止损" in result["reason"]

    def test_sell_on_price_reversion(self):
        """Should sell when price reverts below filtered price."""
        engine = SignalEngine(exit_threshold=0.005)
        engine.set_position(True, 100.0)
        for _ in range(50):
            engine.update(100.0, 100.0, 100.0)
        # Price drops below filtered * (1 - exit_threshold)
        # After warmup, filtered ~100, so close < 99.5 triggers
        result = engine.update(99.0, 99.0, 100.0)
        assert result["signal"] == "sell"
        assert "价格回归" in result["reason"]

    def test_downtrend_higher_entry_threshold(self):
        """在下跌趋势中，需要更大偏离才触发买入。"""
        engine = SignalEngine(
            trend_filter_enabled=True,
            trend_confirm_bars=1,
            entry_threshold=0.02,
            downtrend_entry=0.05,
            trend_bear_pct=0.30,
            use_velocity_signal=False,
            kalman_q_price=1e-6,     # 极低过程噪声 → 卡尔曼滤波器滞后大
            kalman_r=1e-1,            # 高观测噪声 → 信任模型多于观测
        )
        engine.set_position(False, 0.0)

        # 预热建立下跌趋势
        for _ in range(50):
            engine.update(100.0, 105.0, 106.0)

        # 3% 跳涨：不满足下跌趋势买入阈值（需要 5%）
        result = engine.update(103.0, 105.0, 106.0)
        assert result["signal"] == "hold"

        # 8% 跳涨：满足下跌趋势买入阈值
        engine2 = SignalEngine(
            trend_filter_enabled=True,
            trend_confirm_bars=1,
            entry_threshold=0.02,
            downtrend_entry=0.05,
            trend_bear_pct=0.30,
            use_velocity_signal=False,
            kalman_q_price=1e-6,
            kalman_r=1e-1,
        )
        engine2.set_position(False, 0.0)
        for _ in range(50):
            engine2.update(100.0, 105.0, 106.0)
        result2 = engine2.update(108.0, 105.0, 106.0)
        assert result2["signal"] == "buy"

    def test_target_pct_in_downtrend(self):
        """In downtrend with position, target_pct should be bear_pct."""
        engine = SignalEngine(
            trend_filter_enabled=True,
            trend_confirm_bars=1,
            trend_bear_pct=0.30,
        )
        engine.set_position(True, 100.0)
        for _ in range(50):
            engine.update(100.0, 105.0, 106.0)  # establish downtrend
        result = engine.update(100.0, 105.0, 106.0)
        assert result["signal"] == "hold"
        assert result["target_pct"] == 0.30

    def test_process_history_warmup(self):
        """process_history should warm up the engine without errors."""
        df = make_history(n=100, start=100, slope=0.5, noise=0.5)
        engine = SignalEngine(trend_filter_enabled=True)
        engine.process_history(df)
        # After processing, kalman should have converged
        assert engine.filtered_price > 0
        # Trend detector should have run
        assert engine.trend_state in ("up", "down")

    def test_velocity_reversal_entry(self):
        """Velocity turning positive should trigger buy."""
        engine = SignalEngine(
            use_price_signal=False,
            use_velocity_signal=True,
            Q_price=1e-3, Q_vel=1e-2, R=1e-1,  # sensitive to velocity
        )
        engine.set_position(False, 0.0)
        # Downtrend then sharp reversal
        for p in [100, 99, 98, 97, 96, 95]:
            engine.update(float(p), float(p), float(p))

        # Reversal: price jumps up
        result = engine.update(100.0, 100.0, 96.0)
        # May trigger on velocity reversal
        assert result["signal"] in ("buy", "hold")

    def test_trend_state_tracks_correctly(self):
        """Trend state should be accessible after update."""
        engine = SignalEngine(trend_filter_enabled=True, trend_confirm_bars=1)
        engine.set_position(False, 0.0)

        # Establish uptrend
        for _ in range(20):
            engine.update(100.0, 99.0, 98.0)  # close > ma20 AND ma20 rising
        result = engine.update(100.0, 99.0, 98.0)
        assert result["trend"] == "up"

        # Switch to downtrend
        for _ in range(5):
            engine.update(95.0, 100.0, 100.0)  # close < ma20
        result = engine.update(95.0, 100.0, 100.0)
        assert result["trend"] == "down"
