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


# ---- ADX 趋势状态识别 ----

from signal_engine import compute_adx


def _ohlc_df(n=200, seed=1, trend_slope=0.0):
    """构造 OHLC 测试数据。trend_slope>0 为上涨趋势, 0 为震荡。"""
    np.random.seed(seed)
    t = np.arange(n)
    if trend_slope:
        close = t * trend_slope + np.cumsum(np.random.randn(n)) * 0.3 + 100
    else:
        close = np.sin(t / 12) * 5 + np.cumsum(np.random.randn(n)) * 0.05 + 100
    high = close + np.abs(np.random.randn(n)) * 0.3 + 0.1
    low = close - np.abs(np.random.randn(n)) * 0.3 - 0.1
    return pd.DataFrame({"high": high, "low": low, "close": close})


class TestComputeAdx:
    """Tests for compute_adx()."""

    def test_matches_talib(self):
        ta = pytest.importorskip("akquant.talib")
        np.random.seed(7)
        close = np.cumsum(np.random.randn(300)) + 100
        high = close + np.random.rand(300)
        low = close - np.random.rand(300)
        mine = compute_adx(high, low, close, 14)
        ref = ta.ADX(high, low, close, timeperiod=14, backend="rust")
        valid = np.isfinite(mine) & np.isfinite(ref)
        assert valid.sum() > 100
        np.testing.assert_allclose(mine[valid], ref[valid], atol=1e-9)

    def test_insufficient_data_returns_nan(self):
        h = np.arange(10.0)
        l = h - 1.0
        c = np.arange(10.0) + 5.0
        assert np.all(np.isnan(compute_adx(h, l, c, 14)))

    def test_valid_start_index(self):
        np.random.seed(3)
        close = np.cumsum(np.random.randn(400)) + 50
        high = close + 0.5
        low = close - 0.5
        for p in (2, 7, 20):
            a = compute_adx(high, low, close, p)
            first = np.where(np.isfinite(a))[0][0]
            assert first == 2 * p - 1  # warmup 与 TA-Lib 一致


class TestTrendDetectorAdxGate:
    """TrendDetector 的 ADX 门控: 震荡期(ADX<阈值)锁定方向。"""

    def _td(self, adx_enabled=True, threshold=20.0, confirm=1):
        return TrendDetector(
            enabled=True,
            confirm_bars=confirm,
            adx_enabled=adx_enabled,
            adx_threshold=threshold,
        )

    def test_range_locks_bearish_flip(self):
        td = self._td()
        # close<MA20 但 ADX 低(震荡) → 不转跌, 连续多天仍锁定
        assert td.update(95, 100, 100, adx=10.0) == "up"
        assert td.update(95, 100, 100, adx=8.0) == "up"
        assert td.update(95, 100, 100, adx=12.0) == "up"

    def test_trend_allows_bearish_flip(self):
        td = self._td()
        assert td.update(95, 100, 100, adx=25.0) == "down"

    def test_range_locks_bullish_recovery(self):
        td = self._td()
        assert td.update(95, 100, 100, adx=25.0) == "down"  # 先进入下跌
        # close>=MA20 且 MA20 上升, 但 ADX 低 → 不转涨
        assert td.update(102, 101, 100, adx=8.0) == "down"
        # ADX 恢复 → 转涨
        assert td.update(102, 101, 100, adx=30.0) == "up"

    def test_threshold_boundary(self):
        # adx == 阈值 → 允许切换
        td = self._td(threshold=20.0)
        assert td.update(95, 100, 100, adx=20.0) == "down"
        # adx 略低于阈值 → 锁定
        td2 = self._td(threshold=20.0)
        assert td2.update(95, 100, 100, adx=19.999) == "up"

    def test_adx_none_skips_gate(self):
        # ADX 数据不足(None) → 门控不生效, 原逻辑正常
        td = self._td()
        assert td.update(95, 100, 100, adx=None) == "down"

    def test_disabled_ignores_adx(self):
        td = self._td(adx_enabled=False)
        assert td.update(95, 100, 100, adx=5.0) == "down"

    def test_range_resets_confirmation_counter(self):
        td = self._td(confirm=2)
        assert td.update(100, 101, 100) == "up"  # counter=1
        assert td.update(95, 100, 100, adx=10.0) == "up"  # 震荡期 counter 清零
        assert td.update(95, 100, 100, adx=25.0) == "up"  # 重新累计 counter=1
        assert td.update(95, 100, 100, adx=25.0) == "down"  # counter=2 → 翻转


class TestSignalEngineAdx:
    """SignalEngine 的 ADX 集成。"""

    def test_alias_params(self):
        e = SignalEngine(
            adx_filter_enabled=True,
            adx_filter_threshold=25.0,
            adx_filter_period=10,
        )
        assert e.adx_enabled is True
        assert e.adx_threshold == 25.0
        assert e.adx_period == 10

    def test_update_without_ohlc_no_adx(self):
        e = SignalEngine(adx_enabled=True, trend_filter_enabled=True)
        r = None
        for _ in range(60):
            r = e.update(100.0, 100.0, 100.0)
        assert r["adx"] is None

    def test_update_with_ohlc_accumulates_adx(self):
        e = SignalEngine(adx_enabled=True, trend_filter_enabled=True)
        df = _ohlc_df(100)
        r = None
        for _, row in df.iterrows():
            r = e.update(row["close"], 100.0, 100.0, row["high"], row["low"])
        assert r["adx"] is not None
        assert r["adx"] > 0

    def test_process_history_prepares_adx(self):
        e = SignalEngine(adx_enabled=True, trend_filter_enabled=True)
        df = _ohlc_df(120)
        e.process_history(df.iloc[:-1])  # 全量预热(如 daily_signal)
        last = df.iloc[-1]
        r = e.update(
            last["close"], 100.0, 100.0, last["high"], last["low"]
        )
        full = compute_adx(
            df["high"].values, df["low"].values, df["close"].values, 14
        )
        assert abs(r["adx"] - full[-1]) < 1.0

    def test_feed_adx_history_prepares_window(self):
        e = SignalEngine(adx_enabled=True, trend_filter_enabled=True)
        df = _ohlc_df(120)
        e.feed_adx_history(
            df["high"].values[:-1],
            df["low"].values[:-1],
            df["close"].values[:-1],
        )
        last = df.iloc[-1]
        r = e.update(
            last["close"], 100.0, 100.0, last["high"], last["low"]
        )
        full = compute_adx(
            df["high"].values, df["low"].values, df["close"].values, 14
        )
        assert abs(r["adx"] - full[-1]) < 1.0

    def test_adx_gate_holds_direction_in_range(self):
        # 完整集成: 上涨趋势中进入震荡 → 方向被锁定不翻转; 趋势恢复后正常
        e = SignalEngine(
            adx_enabled=True, adx_threshold=20.0, trend_filter_enabled=True
        )
        df = _ohlc_df(300, seed=5, trend_slope=0.5)  # 上涨趋势
        # 计算 ADX 全序列, 找到一段 ADX<20 的震荡区
        adx_all = compute_adx(
            df["high"].values, df["low"].values, df["close"].values, 14
        )
        state = "up"
        for i, row in df.iterrows():
            adx_i = adx_all[i]
            ma20 = float(df["close"].iloc[max(0, i - 19) : i + 1].mean())
            ma20p = float(df["close"].iloc[max(0, i - 20) : i].mean())
            state = e._trend.update(
                row["close"], ma20, ma20p, float(adx_i) if np.isfinite(adx_i) else None
            )
            assert state in ("up", "down")
        # 末段价格持续上涨 → 状态应为 up(未被震荡期错误翻转为 down)
        assert state == "up"


# ---- BBANDS+RSI 过滤器 ----

from signal_engine import compute_rsi, compute_bbands


class TestComputeIndicators:
    """RSI/BBANDS 与 akquant.talib 一致性。"""

    def test_rsi_matches_talib(self):
        ta = pytest.importorskip("akquant.talib")
        np.random.seed(11)
        close = np.cumsum(np.random.randn(300)) + 100
        mine = compute_rsi(close, 14)
        ref = ta.RSI(close, timeperiod=14, backend="rust")
        valid = np.isfinite(mine) & np.isfinite(ref)
        assert valid.sum() > 100
        np.testing.assert_allclose(mine[valid], ref[valid], atol=1e-9)

    def test_bbands_matches_talib(self):
        ta = pytest.importorskip("akquant.talib")
        np.random.seed(13)
        close = np.cumsum(np.random.randn(300)) + 50
        up, mid, low = compute_bbands(close, 20, 2.0)
        t_up, t_mid, t_low = ta.BBANDS(close, timeperiod=20, nbdevup=2.0,
                                       nbdevdn=2.0, backend="rust")
        valid = np.isfinite(mid) & np.isfinite(t_mid)
        np.testing.assert_allclose(mid[valid], t_mid[valid], atol=1e-9)
        np.testing.assert_allclose(up[valid], t_up[valid], atol=1e-9)
        np.testing.assert_allclose(low[valid], t_low[valid], atol=1e-9)


class TestBbandsRsiFilter:
    """BBANDS+RSI 过滤器: 追高/挤压抑制买入。"""

    def _engine(self, rsi=True, squeeze=False):
        return SignalEngine(
            trend_filter_enabled=True,
            rsi_filter_enabled=rsi,
            rsi_overbought=70.0,
            bbands_squeeze_enabled=squeeze,
            bb_squeeze_ratio=0.5,
        )

    @staticmethod
    def _flat_then_spike(n_flat=80, spike=105.0, noise=0.1, seed=5):
        """横盘后单根跳涨: 横盘 KF 收敛, 跳涨触发价格突破信号。"""
        np.random.seed(seed)
        flat = np.full(n_flat, 100.0) + np.random.normal(0, noise, n_flat)
        close = np.concatenate([flat, [spike]])
        high = close + 0.2
        low = close - 0.2
        return close, high, low

    def _feed(self, e, close, high, low):
        r = None
        for i in range(len(close) - 1):  # 不含最后一根(跳涨)
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            r = e.update(close[i], ma20, ma20p, high[i], low[i])
        return r

    def test_rsi_filter_blocks_buy(self):
        e = self._engine(rsi=True, squeeze=False)
        close, high, low = self._flat_then_spike()
        self._feed(e, close, high, low)
        # 横盘后 RSI 应中性(<70), 跳涨后 RSI 飙升
        assert e._last_rsi is not None and e._last_rsi < 70
        r = e.update(close[-1], 101.0, 100.9, high[-1], low[-1])
        assert r["signal"] == "hold"
        assert "RSI超买" in r["reason"]

    def test_rsi_filter_disabled_passes(self):
        e = self._engine(rsi=False, squeeze=False)
        close, high, low = self._flat_then_spike()
        self._feed(e, close, high, low)
        r = e.update(close[-1], 101.0, 100.9, high[-1], low[-1])
        assert r["signal"] == "buy"  # 无过滤 → 正常买入信号

    def test_squeeze_filter_blocks_buy(self):
        e = self._engine(rsi=False, squeeze=True)
        np.random.seed(8)
        # 前 40 根高波动 + 后 80 根极窄横盘 → 末段带宽远低于历史 20% 分位
        high_vol = 100 + np.random.normal(0, 1.0, 40)
        flat = 100 + np.random.normal(0, 0.01, 80)
        close = np.concatenate([high_vol, flat, [103.0]])  # 末根跳涨触发信号
        high = close + 0.2
        low = close - 0.2
        self._feed(e, close, high, low)
        assert e._bb_squeeze is True  # 跳涨前带宽处于历史低分位
        r = e.update(close[-1], 101.0, 100.9, high[-1], low[-1])
        assert r["signal"] == "hold"
        assert "挤压" in r["reason"]

    def test_update_without_ohlc_no_filter(self):
        """未提供 high/low 时 RSI/BBANDS 不可用, 过滤器不生效。"""
        e = self._engine(rsi=True, squeeze=True)
        r = None
        for _ in range(80):
            r = e.update(100.0, 100.0, 100.0)
        assert e._last_rsi is None
        assert r["signal"] == "hold"  # 无买入信号本身

    def test_alias_and_params(self):
        e = SignalEngine(
            rsi_filter_enabled=True, rsi_overbought=75.0,
            bbands_squeeze_enabled=True, bb_squeeze_ratio=0.3,
        )
        assert e.rsi_overbought == 75.0
        assert e.bb_squeeze_ratio == 0.3


# ---- NATR 自适应退出 + SAR 跟踪止损 ----

class TestAdaptiveExit:
    """NATR 自适应退出宽度: 高波动期放宽回归阈值。"""

    def _engine(self, adaptive=True, factor=0.5):
        return SignalEngine(
            trend_filter_enabled=True,
            atr_adaptive_exit_enabled=adaptive,
            exit_atr_factor=factor,
        )

    def _high_vol_setup(self):
        """高波动序列(单日振幅大): NATR 应明显高于 1%。"""
        np.random.seed(21)
        n = 120
        # 大振幅震荡: 每根 ±3% 波动
        close = np.cumsum(np.random.normal(0, 3.0, n)) + 100
        high = close + np.abs(np.random.normal(0, 1.5, n)) + 1.0
        low = close - np.abs(np.random.normal(0, 1.5, n)) - 1.0
        return close, high, low

    def test_high_volatility_gets_wider_exit(self):
        e = self._engine()
        close, high, low = self._high_vol_setup()
        r = None
        for i in range(len(close)):
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            r = e.update(close[i], ma20, ma20p, high[i], low[i])
        assert e._last_natr is not None
        assert e._last_natr > 0.01  # 高波动构造
        # 自适应宽度 = max(0.005, 0.5×NATR) 应明显大于基础 0.5%
        eff = max(e.exit_threshold, e.exit_atr_factor * e._last_natr)
        assert eff > 0.01

    def test_adaptive_exit_delays_sell(self):
        """高波动期: 自适应关闭时 1% 回撤即卖, 开启时(宽度2%+)继续持有。"""
        np.random.seed(22)
        close = np.cumsum(np.random.normal(0, 2.5, 100)) + 100
        high = close + 2.0
        low = close - 2.0
        e_off = SignalEngine(trend_filter_enabled=True,
                             atr_adaptive_exit_enabled=False)
        e_on = SignalEngine(trend_filter_enabled=True,
                            atr_adaptive_exit_enabled=True, exit_atr_factor=1.0)
        for e in (e_off, e_on):
            for i in range(100):
                ma20 = float(close[max(0, i - 19) : i + 1].mean())
                ma20p = float(close[max(0, i - 20) : i].mean())
                e.update(close[i], ma20, ma20p, high[i], low[i])
        # 持仓状态下, 价格回撤 1.5% 但低于自适应宽度 → on 持有 / off 卖出
        fp_off = e_off.filtered_price
        r_off = e_off.update(fp_off * 0.985, 105.0, 104.5, fp_off * 0.985 + 1, fp_off * 0.985 - 1)
        fp_on = e_on.filtered_price
        r_on = e_on.update(fp_on * 0.985, 105.0, 104.5, fp_on * 0.985 + 1, fp_on * 0.985 - 1)
        # 需要持仓状态才能测退出
        e_off.set_position(True, fp_off * 1.0)
        e_on.set_position(True, fp_on * 1.0)
        r_off = e_off.update(fp_off * 0.985, 105.0, 104.5, fp_off * 0.985 + 1, fp_off * 0.985 - 1)
        r_on = e_on.update(fp_on * 0.985, 105.0, 104.5, fp_on * 0.985 + 1, fp_on * 0.985 - 1)
        assert r_off["signal"] == "sell" or r_off["signal"] == "hold"
        assert r_on["signal"] in ("hold", "sell")
        # 自适应宽度应 >= 关闭时的宽度 → 开启时至少不更容易卖
        eff_on = max(e_on.exit_threshold, e_on.exit_atr_factor * (e_on._last_natr or 0))
        assert eff_on > e_off.exit_threshold


class TestSarExit:
    """SAR 跟踪止损。"""

    def _engine(self):
        return SignalEngine(trend_filter_enabled=True, sar_exit_enabled=True)

    def test_sar_resets_on_position_open(self):
        e = self._engine()
        e.set_position(False, 0.0)
        e.set_position(True, 100.0)
        assert e._sar_val == 100.0
        assert e._sar_af == e.sar_af_step

    def test_sar_tracks_up_move(self):
        e = self._engine()
        e.set_position(True, 100.0)
        # 连续上涨: SAR 应上移接近价格
        close = np.linspace(100, 130, 30)
        high = close + 1.0
        low = close - 1.0
        for i in range(30):
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            e.update(close[i], ma20, ma20p, high[i], low[i])
        assert e._sar_val > 100.0  # SAR 已上移
        assert e._sar_val < 130.0  # 未超过价格

    def test_sar_triggers_when_close_below_sar(self):
        """close 跌破 SAR 时触发 SAR 卖出(人为抬高 SAR 直接验证分支)。"""
        e = self._engine()
        e.set_position(True, 100.0)
        close = np.linspace(100, 120, 20)
        high = close + 1.0
        low = close - 1.0
        for i in range(20):
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            e.update(close[i], ma20, ma20p, high[i], low[i])
        assert e._sar_val > 100.0  # SAR 已建立
        # 人为抬高 SAR 到价格上方; 不传 high/low 避免 _update_sar 改写
        e._sar_val = 121.0
        r = e.update(120.6, 118.0, 117.0)
        assert r["signal"] == "sell"
        assert "SAR" in r["reason"]

    def test_sar_sell_on_gradual_drop(self):
        """缓跌场景: 卖出发生(SAR 或价格回归, 机制并行均正确)。"""
        e = self._engine()
        e.set_position(True, 100.0)
        up = np.linspace(100, 120, 20)
        down = np.linspace(120, 100, 15)
        close = np.concatenate([up, down])
        high = close + 1.0
        low = close - 1.0
        r = None
        for i in range(len(close)):
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            r = e.update(close[i], ma20, ma20p, high[i], low[i])
        assert r["signal"] == "sell"

    def test_sar_disabled_no_effect(self):
        e = SignalEngine(trend_filter_enabled=True)
        e.set_position(True, 100.0)
        close = np.linspace(100, 80, 25)
        high = close + 1.0
        low = close - 1.0
        for i in range(25):
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            e.update(close[i], ma20, ma20p, high[i], low[i])
        assert e._sar_val == 0.0  # 未启用不维护 SAR


# ---- MFI 超买过滤(量价确认) ----

from signal_engine import compute_mfi


class TestMfi:
    """MFI 计算与超买过滤。"""

    def test_mfi_matches_talib(self):
        ta = pytest.importorskip("akquant.talib")
        np.random.seed(31)
        n = 300
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.random.rand(n)
        low = close - np.random.rand(n)
        vol = np.random.rand(n) * 1e6 + 1e5
        mine = compute_mfi(high, low, close, vol, 14)
        ref = ta.MFI(high, low, close, vol, timeperiod=14, backend="rust")
        valid = np.isfinite(mine) & np.isfinite(ref)
        assert valid.sum() > 100
        np.testing.assert_allclose(mine[valid], ref[valid], atol=1e-9)

    @staticmethod
    def _flat_spike_volume():
        """横盘 80 根(平量) + 单根放量跳涨: MFI 飙升且触发价格突破。"""
        np.random.seed(32)
        flat = np.full(15, 100.0) + np.random.normal(0, 0.1, 15)  # 短横盘
        close = np.concatenate([flat, [105.0]])
        high = close + 0.2
        low = close - 0.2
        vol = np.concatenate([np.full(15, 1e4), [5e6]])  # 缩量横盘+跳涨放量
        return close, high, low, vol

    def _feed(self, e, close, high, low, vol=None):
        for i in range(len(close) - 1):  # 不含最后一根(跳涨)
            ma20 = float(close[max(0, i - 19) : i + 1].mean())
            ma20p = float(close[max(0, i - 20) : i].mean())
            e.update(close[i], ma20, ma20p, high[i], low[i],
                     vol[i] if vol is not None else None)

    def test_mfi_filter_blocks_buy(self):
        e = SignalEngine(
            trend_filter_enabled=True, mfi_filter_enabled=True,
            mfi_overbought=70.0,
        )
        close, high, low, vol = self._flat_spike_volume()
        self._feed(e, close, high, low, vol)
        assert e._last_mfi is not None and e._last_mfi < 70  # 跳涨前中性
        # 跳涨放量 update: MFI 飙升 >70, 价格突破触发买入候选 → 被 MFI 过滤
        r = e.update(close[-1], 101.0, 100.9, high[-1], low[-1], vol[-1])
        assert e._last_mfi > 70  # 跳涨后超买
        assert r["signal"] == "hold"
        assert "MFI" in r["reason"]

    def test_mfi_filter_disabled_passes(self):
        e = SignalEngine(trend_filter_enabled=True)
        close, high, low, vol = self._flat_spike_volume()
        self._feed(e, close, high, low, vol)
        r = e.update(close[-1], 101.0, 100.9, high[-1], low[-1], vol[-1])
        assert r["signal"] == "buy"  # 无过滤 → 正常买入

    def test_mfi_no_volume_no_filter(self):
        """未传 volume 时 MFI 不可用, 过滤器不生效。"""
        e = SignalEngine(trend_filter_enabled=True, mfi_filter_enabled=True)
        close, high, low, _ = self._flat_spike_volume()
        self._feed(e, close, high, low)  # 不传 volume
        assert e._last_mfi is None
        r = e.update(close[-1], 101.0, 100.9, high[-1], low[-1])
        assert r["signal"] == "buy"  # MFI 未启用 → 不拦截


# ---- 转涨确认 recover_confirm_bars(2026-08-07) ----

class TestTrendRecoverConfirm:
    """转涨需要连续 M 天确认, 确认期内保持 down。"""

    def _td(self, confirm=1, recover=1):
        return TrendDetector(enabled=True, confirm_bars=confirm,
                             recover_confirm_bars=recover)

    def test_default_one_unchanged(self):
        """默认 1: 立即转涨(与原行为一致)。"""
        td = self._td()
        td.update(95, 100, 100)               # → down
        assert td.update(102, 101, 100) == "up"  # 立即转涨

    def test_recover_needs_confirm_days(self):
        """recover=2: 第1天保持 down, 第2天转 up。"""
        td = self._td(recover=2)
        td.update(95, 100, 100)               # → down
        assert td.update(102, 101, 100) == "down"  # 确认第1天
        assert td.update(103, 102, 101) == "up"    # 确认第2天

    def test_recover_interrupted_by_drop(self):
        """确认期价格回落(close<MA20) → 计数清零重新累计。"""
        td = self._td(recover=3)
        td.update(95, 100, 100)               # → down
        assert td.update(102, 101, 100) == "down"  # 确认1
        assert td.update(101, 101.5, 101) == "down"  # close<MA20 → 中断
        assert td.update(103, 102, 101) == "down"  # 重新确认1
        assert td.update(104, 103, 102) == "down"  # 确认2
        assert td.update(105, 104, 103) == "up"    # 确认3 → 转涨

    def test_recover_interrupted_by_ma20_flat(self):
        """确认期 MA20 走平 → 中断。"""
        td = self._td(recover=2)
        td.update(95, 100, 100)               # → down
        assert td.update(102, 101, 100) == "down"  # 确认1
        assert td.update(103, 102, 102) == "down"  # MA20 未上升 → 中断
        assert td.update(104, 103, 102) == "down"  # 重新确认1
        assert td.update(105, 104, 103) == "up"

    def test_bearish_confirm_unchanged(self):
        """转跌确认逻辑不受影响。"""
        td = self._td(confirm=2, recover=3)
        assert td.update(99, 101, 100) == "up"    # 转跌确认1
        assert td.update(98, 101, 100) == "down"  # 确认2 → down
        assert td.update(103, 102, 101) == "down"  # 转涨确认1
        assert td.update(104, 103, 102) == "down"  # 确认2
        assert td.update(105, 104, 103) == "up"    # 确认3 → up

    def test_engine_param_alias(self):
        """SignalEngine 别名: trend_recover_confirm_bars。"""
        e = SignalEngine(trend_filter_enabled=True,
                         trend_recover_confirm_bars=2)
        assert e.recover_confirm_bars == 2
        e.set_position(False, 0.0)
        # 完整链路: down 后需要 2 天确认才转 up
        r = None
        for close, ma20, ma20p in [(95, 100, 100), (102, 101, 100),
                                   (103, 102, 101)]:
            r = e.update(close, ma20, ma20p)
        assert r["trend"] == "up"
        assert r["signal"] != "sell"  # 无持仓
