"""Tests for kalman_filter.py."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalman_filter import KalmanFilter2D, kalman_smooth, generate_signals


class TestKalmanFilter2D:
    """Unit tests for the 2-state Kalman filter."""

    def test_initial_update(self):
        """First update should return the observation as filtered price with zero velocity."""
        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)
        fp, vel = kf.update(100.0)
        assert fp == 100.0
        assert vel == 0.0

    def test_convergence_to_constant(self):
        """With constant input, the filter should converge to the constant with near-zero velocity."""
        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)
        constant = 50.0
        prices = [constant] * 100
        for p in prices:
            kf.update(p)
        # Filtered price should be very close to the constant
        assert abs(kf.get_filtered_price() - constant) < 0.01
        # Velocity should be near zero
        assert abs(kf.get_velocity()) < 0.01

    def test_tracks_linear_trend(self):
        """The filter should track a linear price trend with positive velocity."""
        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)
        # Linear uptrend: +0.5 per step
        prices = [100.0 + 0.5 * i for i in range(100)]
        for p in prices:
            kf.update(p)
        # Velocity should be positive and close to 0.5
        vel = kf.get_velocity()
        assert vel > 0.3

    def test_noise_reduction(self):
        """Filtered price should have lower variance than noisy observations."""
        np.random.seed(42)
        true_price = 100.0
        noisy = true_price + np.random.normal(0, 2.0, 200)

        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-1)
        filtered = []
        for p in noisy:
            fp, _ = kf.update(float(p))
            filtered.append(fp)

        filtered = np.array(filtered[20:])  # skip warmup
        noisy = noisy[20:]

        # Filtered should have lower variance
        assert np.std(filtered) < np.std(noisy)

    def test_velocity_sign_tracks_direction(self):
        """Velocity should be positive in uptrend and negative after reversal."""
        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2, initial_price=100.0)

        # Uptrend
        for p in [100, 101, 102, 103, 104, 105]:
            kf.update(float(p))
        assert kf.get_velocity() > 0, "Velocity should be positive in uptrend"

        # Downtrend
        for p in [104, 103, 102, 101, 100, 99, 98]:
            kf.update(float(p))
        assert kf.get_velocity() < 0, "Velocity should be negative in downtrend"

    def test_reset(self):
        """After reset, the filter should behave like a fresh instance."""
        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)
        for p in [100, 101, 102, 103]:
            kf.update(float(p))

        kf.reset(initial_price=50.0)
        fp, vel = kf.update(50.0)
        assert fp == 50.0
        assert vel == 0.0

    def test_multiple_resets(self):
        """Multiple resets should work correctly."""
        kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)

        # First pass
        for p in [10, 11, 12]:
            kf.update(float(p))
        assert kf.get_filtered_price() != 10.0

        kf.reset(initial_price=20.0)
        fp, _ = kf.update(20.0)
        assert fp == 20.0

        # Second pass
        for p in [20, 19, 18]:
            kf.update(float(p))

        kf.reset(initial_price=30.0)
        fp, _ = kf.update(30.0)
        assert fp == 30.0


class TestKalmanSmooth:
    """Tests for the batch kalman_smooth function."""

    def test_returns_same_length(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = kalman_smooth(prices)
        assert len(result) == len(prices)

    def test_smooths_noise(self):
        np.random.seed(123)
        true = np.linspace(100, 200, 500)
        noisy = true + np.random.normal(0, 5.0, 500)
        smoothed = kalman_smooth(noisy, Q_price=1e-4, Q_vel=1e-6, R=0.5)

        # Smoothed should be closer to true than noisy
        mse_noisy = np.mean((noisy - true) ** 2)
        mse_smoothed = np.mean((smoothed - true) ** 2)
        assert mse_smoothed < mse_noisy


class TestGenerateSignals:
    """Tests for the standalone generate_signals function."""

    def test_returns_integer_array(self):
        prices = np.array([100.0] * 50)
        signals = generate_signals(prices)
        assert signals.dtype == int
        assert len(signals) == len(prices)

    def test_first_signal_is_zero(self):
        """First bar should always have signal 0 (no prior state)."""
        prices = np.array([100.0, 105.0, 110.0])  # sharp uptrend
        signals = generate_signals(prices, entry_threshold=0.02)
        assert signals[0] == 0

    def test_entry_on_price_breakout(self):
        """A large enough price jump should trigger a buy signal."""
        prices = np.array([100.0] * 20 + [110.0])  # 10% jump
        signals = generate_signals(prices, entry_threshold=0.02)
        buy_signals = signals[signals == 1]
        assert len(buy_signals) > 0, "Should have at least one buy signal on breakout"
