"""
公共信号引擎模块。

提供趋势检测和买卖信号评估的统一实现，消除 strategy.py / daily_signal.py /
portfolio_backtest.py 中三份独立的重复逻辑。

用法:
    engine = SignalEngine(
        kalman_q_price=1e-4,
        kalman_q_vel=1e-5,
        kalman_r=1e-2,
        entry_threshold=0.02,
        exit_threshold=0.005,
        stop_loss_pct=0.05,
        trend_filter_enabled=True,
        trend_confirm_bars=1,
        trend_bear_pct=0.30,
        downtrend_entry=0.03,
    )
    engine.process_history(df)          # 预热
    engine.set_position(False, 0.0)     # 设置持仓状态
    result = engine.update(close, ma20_cur, ma20_prev)
    # → {"signal": "hold", "target_pct": 0.95, "reason": "...", ...}
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from kalman_filter import KalmanFilter2D


class TrendDetector:
    """基于 close vs MA20 + MA20 方向的趋势检测器。

    规则:
        下跌确认: close < MA20 连续 N 天 → 转跌
        上涨恢复: close >= MA20 且 MA20 向上 → 立即转涨
        MA20 向下时即使 close >= MA20 也不切换为上涨
    """

    def __init__(self, enabled: bool = True, confirm_bars: int = 1) -> None:
        self._enabled = enabled
        self._confirm = max(1, int(confirm_bars))
        self._state: str = "up"
        self._counter: int = 0

    @property
    def state(self) -> str:
        return self._state

    def update(self, close: float, ma20_cur: float, ma20_prev: float) -> str:
        """更新趋势状态并返回 ("up" | "down")。"""
        if not self._enabled:
            self._state = "up"
            return self._state

        is_bearish = close < ma20_cur
        ma20_rising = ma20_cur > ma20_prev

        if is_bearish:
            self._counter += 1
            if self._counter >= self._confirm and self._state == "up":
                self._state = "down"
        elif ma20_rising:
            self._counter = 0
            if self._state == "down":
                self._state = "up"
        else:
            # close >= MA20 但 MA20 仍向下 → 维持下跌状态
            self._counter = max(self._counter, self._confirm)

        return self._state

    def reset(self) -> None:
        self._state = "up"
        self._counter = 0


class SignalEngine:
    """统一的信号评估引擎。

    参数（支持两种命名约定）:
        - kalman_q_price, kalman_q_vel, kalman_r: 卡尔曼滤波器噪声参数
        - entry_threshold: 上涨买入偏离阈值
        - exit_threshold: 卖出偏离阈值
        - stop_loss_pct: 止损比例
        - use_price_signal: 是否使用价格偏离信号
        - use_velocity_signal: 是否使用速度反转信号
        - trend_filter_enabled: 是否启用趋势过滤
        - trend_confirm_bars / trend_filter_confirm_bars: 趋势确认天数
        - trend_bear_pct / trend_bear_position_pct: 下跌仓位比例
        - downtrend_entry / downtrend_entry_threshold: 下跌买入阈值
    """

    def __init__(self, **params: Any) -> None:
        # ---- 参数解析（兼容两种命名约定） ----
        def _p(key: str, default: Any, *aliases: str) -> Any:
            for k in (key,) + aliases:
                if k in params:
                    return params[k]
            return default

        self.kalman_q_price = float(_p("kalman_q_price", 1e-4))
        self.kalman_q_vel = float(_p("kalman_q_vel", 1e-5))
        self.kalman_r = float(_p("kalman_r", 1e-2))
        self.entry_threshold = float(_p("entry_threshold", 0.02))
        self.exit_threshold = float(_p("exit_threshold", 0.005))
        self.stop_loss_pct = float(_p("stop_loss_pct", 0.05))
        self.use_price_signal = bool(_p("use_price_signal", True))
        self.use_velocity_signal = bool(_p("use_velocity_signal", True))
        self.trend_filter_enabled = bool(_p("trend_filter_enabled", False))
        self.trend_confirm_bars = int(
            _p("trend_confirm_bars", 1, "trend_filter_confirm_bars")
        )
        self.trend_bear_pct = float(
            _p("trend_bear_pct", 0.30, "trend_bear_position_pct")
        )
        self.downtrend_entry = float(
            _p("downtrend_entry", 0.03, "downtrend_entry_threshold")
        )

        # ---- 卡尔曼滤波器 ----
        self._kf = KalmanFilter2D(
            Q_price=self.kalman_q_price,
            Q_vel=self.kalman_q_vel,
            R=self.kalman_r,
        )

        # ---- 趋势检测器 ----
        self._trend = TrendDetector(
            enabled=self.trend_filter_enabled,
            confirm_bars=self.trend_confirm_bars,
        )

        # ---- 内部状态 ----
        self._prev_velocity: float = 0.0
        self._prev_vel_stored: float = 0.0  # 上一次 update 前的速度
        self._has_position: bool = False
        self._entry_price: float = 0.0
        self._bars_processed: int = 0

        # 存储 params dict 供外部访问
        self.params = params

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------
    @property
    def trend_state(self) -> str:
        return self._trend.state

    @property
    def filtered_price(self) -> float:
        return self._kf.get_filtered_price()

    @property
    def velocity(self) -> float:
        return self._kf.get_velocity()

    @property
    def prev_velocity(self) -> float:
        return self._prev_vel_stored

    # ------------------------------------------------------------------
    # 持仓状态
    # ------------------------------------------------------------------
    def set_position(self, has_position: bool, entry_price: float = 0.0) -> None:
        """设置当前持仓状态（由外部调用方管理）。"""
        self._has_position = has_position
        self._entry_price = entry_price

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------
    def process_history(self, df: pd.DataFrame) -> None:
        """用历史数据预热卡尔曼滤波器和趋势检测器。

        遍历整个 DataFrame，逐 bar 更新卡尔曼滤波器，
        并在有足够数据时更新趋势状态。
        """
        if df.empty:
            return

        closes = df["close"].values
        n = len(closes)

        for i in range(n):
            close = float(closes[i])
            self._kf.update(close)
            self._bars_processed += 1

            # 趋势预热（需要至少 21 根 bar）
            if n >= 21 and i >= 20:
                ma20_cur = float(closes[i - 19 : i + 1].mean())
                ma20_prev = float(closes[i - 20 : i].mean())
                self._trend.update(close, ma20_cur, ma20_prev)

            # 更新上一步速度
            if i >= 1:
                self._prev_velocity = self._kf.get_velocity()

    # ------------------------------------------------------------------
    # 核心更新
    # ------------------------------------------------------------------
    def update(
        self, close: float, ma20_cur: float, ma20_prev: float
    ) -> Dict[str, Any]:
        """更新卡尔曼滤波器和趋势，评估当前信号。

        返回:
            {
                "close": float,
                "kalman_price": float,
                "kalman_velocity": float,
                "ma20": float,
                "ma20_rising": bool,
                "trend": str,           # "up" | "down"
                "signal": str,          # "buy" | "sell" | "hold"
                "target_pct": float,
                "reason": str,
            }
        """
        close = float(close)
        ma20_cur = float(ma20_cur)
        ma20_prev = float(ma20_prev)

        # 更新卡尔曼
        filtered, velocity = self._kf.update(close)
        self._prev_vel_stored = self._prev_velocity

        # 更新趋势
        self._trend.update(close, ma20_cur, ma20_prev)

        # 评估信号
        signal, target_pct, reason = self._evaluate(
            close, filtered, velocity, self._prev_velocity
        )

        # 保存速度状态（用于下次调用）
        self._prev_velocity = velocity
        self._bars_processed += 1

        return {
            "close": close,
            "kalman_price": filtered,
            "kalman_velocity": velocity,
            "ma20": ma20_cur,
            "ma20_rising": ma20_cur > ma20_prev,
            "trend": self._trend.state,
            "signal": signal,
            "target_pct": target_pct,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # 信号评估
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        close: float,
        filtered: float,
        velocity: float,
        prev_velocity: float,
    ) -> Tuple[str, float, str]:
        """评估买卖信号。

        返回 (signal, target_pct, reason)。
        """
        # 趋势决定仓位
        if self.trend_filter_enabled and self._trend.state == "down":
            target_pct = self.trend_bear_pct
        else:
            target_pct = 0.95

        if self._has_position:
            return self._evaluate_exit(
                close, filtered, velocity, prev_velocity, target_pct
            )
        else:
            return self._evaluate_entry(
                close, filtered, velocity, prev_velocity, target_pct
            )

    def _evaluate_entry(
        self,
        close: float,
        filtered: float,
        velocity: float,
        prev_velocity: float,
        target_pct: float,
    ) -> Tuple[str, float, str]:
        """评估买入信号。"""
        # 下跌趋势中需要更严格的阈值
        if self.trend_filter_enabled and self._trend.state == "down":
            entry_threshold = self.downtrend_entry
            extra = f"下跌趋势(+{entry_threshold * 100:.0f}%阈值)"
        else:
            entry_threshold = self.entry_threshold
            extra = ""

        reasons = []

        if self.use_price_signal:
            if close > filtered * (1.0 + entry_threshold):
                reasons.append(
                    f"价格突破(偏离{(close / filtered - 1) * 100:.1f}%)"
                )

        if self.use_velocity_signal:
            if velocity > 0.0 and prev_velocity <= 0.0:
                reasons.append("速度反转(转多)")

        if reasons:
            reason_str = " + ".join(reasons)
            if extra:
                reason_str += f" | {extra}"
            return "buy", target_pct, reason_str

        # 无信号
        trend_info = f"趋势={self._trend.state}"
        if extra:
            trend_info += f" | {extra}"
        return "hold", 0.0, f"等待信号 | {trend_info}"

    def _evaluate_exit(
        self,
        close: float,
        filtered: float,
        velocity: float,
        prev_velocity: float,
        target_pct: float,
    ) -> Tuple[str, float, str]:
        """评估卖出信号。"""
        # 止损（优先级最高）
        if self._entry_price > 0:
            pnl = close / self._entry_price - 1.0
            if pnl < -self.stop_loss_pct:
                return "sell", 0.0, f"止损({pnl * 100:.1f}%)"

        # 价格回归
        if self.use_price_signal:
            if close < filtered * (1.0 - self.exit_threshold):
                return (
                    "sell",
                    0.0,
                    f"价格回归(偏离{(close / filtered - 1) * 100:.1f}%)",
                )

        # 速度反转
        if self.use_velocity_signal:
            if velocity < 0.0 and prev_velocity >= 0.0:
                return "sell", 0.0, "速度反转(转空)"

        # 持有
        return "hold", target_pct, f"持仓中 | 趋势={self._trend.state}"
