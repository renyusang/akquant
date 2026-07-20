"""
卡尔曼滤波辅助交易策略。

基于 2 状态卡尔曼滤波器（价格 + 速度/趋势）生成买卖信号。

信号逻辑:
    买入:
        1. 价格突破: 收盘价 > 卡尔曼估计价 × (1 + entry_threshold)
        2. 速度反转: 卡尔曼速度由 ≤0 变为 >0（趋势转多）
    卖出:
        1. 止损:     (现价 / 入场价 - 1) < -stop_loss_pct
        2. 价格回归: 收盘价 < 卡尔曼估计价 × (1 - exit_threshold)
        3. 速度反转: 卡尔曼速度由 ≥0 变为 <0（趋势转空）
"""

from typing import Any, Dict

from akquant import Bar, FloatParam, ParamModel, Strategy

from kalman_filter import KalmanFilter2D


class KalmanParams(ParamModel):
    """卡尔曼策略参数模型。

    用于 run_grid_search 参数校验和策略实例化。
    """

    kalman_q_price: float = FloatParam(
        1e-4, ge=1e-6, le=1.0, title="价格过程噪声"
    )
    kalman_q_vel: float = FloatParam(
        1e-5, ge=1e-7, le=1.0, title="速度过程噪声"
    )
    kalman_r: float = FloatParam(
        1e-2, ge=1e-5, le=1.0, title="观测噪声"
    )
    entry_threshold: float = FloatParam(
        0.02, ge=0.001, le=0.20, title="买入价格偏离阈值"
    )
    exit_threshold: float = FloatParam(
        0.005, ge=0.0, le=0.10, title="卖出价格偏离阈值"
    )
    stop_loss_pct: float = FloatParam(
        0.05, ge=0.01, le=0.30, title="止损比例"
    )
    use_price_signal: bool = True
    use_velocity_signal: bool = True
    trend_filter_enabled: bool = False
    trend_filter_confirm_bars: int = 3
    trend_bear_position_pct: float = 0.30
    downtrend_entry_threshold: float = 0.03


class KalmanStrategy(Strategy):
    """卡尔曼滤波辅助交易策略。

    参数:
        kalman_q_price:     价格过程噪声 (默认 1e-4)，越大→越不信任模型
        kalman_q_vel:       速度过程噪声 (默认 1e-5)
        kalman_r:           观测噪声 (默认 1e-2)，越大→越不信任观测值
        entry_threshold:    买入价格偏离阈值 (默认 0.02 = 2%)
        exit_threshold:     卖出价格偏离阈值 (默认 0.005 = 0.5%)
        stop_loss_pct:      止损比例 (默认 0.05 = 5%)
        use_price_signal:   是否使用价格偏离信号 (默认 True)
        use_velocity_signal: 是否使用速度反转信号 (默认 True)
    """

    PARAM_MODEL = KalmanParams

    # 预热期：给卡尔曼滤波器足够的收敛时间
    warmup_period = 40

    # ---- 可调参数（通过构造函数传入） ----
    kalman_q_price: float
    kalman_q_vel: float
    kalman_r: float
    entry_threshold: float
    exit_threshold: float
    stop_loss_pct: float
    use_price_signal: bool
    use_velocity_signal: bool
    trend_filter_enabled: bool
    trend_filter_confirm_bars: int
    trend_bear_position_pct: float
    downtrend_entry_threshold: float

    def __init__(
        self,
        kalman_q_price: float = 1e-4,
        kalman_q_vel: float = 1e-5,
        kalman_r: float = 1e-2,
        entry_threshold: float = 0.02,
        exit_threshold: float = 0.005,
        stop_loss_pct: float = 0.05,
        use_price_signal: bool = True,
        use_velocity_signal: bool = True,
        trend_filter_enabled: bool = False,
        trend_filter_confirm_bars: int = 3,
        trend_bear_position_pct: float = 0.30,
        downtrend_entry_threshold: float = 0.03,
    ) -> None:
        """初始化策略。"""
        super().__init__()

        # ---- 卡尔曼滤波器参数 ----
        self.kalman_q_price = float(kalman_q_price)
        self.kalman_q_vel = float(kalman_q_vel)
        self.kalman_r = float(kalman_r)

        # ---- 信号阈值 ----
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.stop_loss_pct = float(stop_loss_pct)

        # ---- 信号开关 ----
        self.use_price_signal = bool(use_price_signal)
        self.use_velocity_signal = bool(use_velocity_signal)

        # ---- 趋势过滤 ----
        self.trend_filter_enabled = bool(trend_filter_enabled)
        self.trend_filter_confirm_bars = max(1, int(trend_filter_confirm_bars))
        self.trend_bear_position_pct = float(trend_bear_position_pct)
        self.downtrend_entry_threshold = float(downtrend_entry_threshold)

        # ---- 状态存储 ----
        self._kf_instances: Dict[str, KalmanFilter2D] = {}
        self._prev_velocities: Dict[str, float] = {}
        self._entry_prices: Dict[str, float] = {}
        self._trade_count: int = 0

        # 趋势过滤状态
        self._trend_state: str = "up"       # "up" | "down"
        self._trend_counter: int = 0         # 连续计数（确认用）

    # ------------------------------------------------------------------
    # 公开属性（回测后可通过 result.strategy 访问）
    # ------------------------------------------------------------------
    @property
    def trade_count(self) -> int:
        """总交易次数。"""
        return self._trade_count

    # ------------------------------------------------------------------
    # 卡尔曼滤波器管理
    # ------------------------------------------------------------------
    def _get_kalman(self, symbol: str) -> KalmanFilter2D:
        """获取或创建指定标的的卡尔曼滤波器实例。"""
        if symbol not in self._kf_instances:
            self._kf_instances[symbol] = KalmanFilter2D(
                Q_price=self.kalman_q_price,
                Q_vel=self.kalman_q_vel,
                R=self.kalman_r,
            )
        return self._kf_instances[symbol]

    # ------------------------------------------------------------------
    # 核心策略逻辑
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        """处理每根 K 线。"""
        symbol: str = bar.symbol
        close_price: float = bar.close

        # 1. 更新卡尔曼滤波器
        kf = self._get_kalman(symbol)
        filtered_price, velocity = kf.update(close_price)

        # 获取上一步速度（首次时为 0）
        prev_velocity = self._prev_velocities.get(symbol, 0.0)

        # 2. 趋势过滤——决定仓位比例
        target_pct = 0.95  # 默认满仓
        if self.trend_filter_enabled:
            self._update_trend(close_price, bar)
            if self._trend_state == "down":
                target_pct = self.trend_bear_position_pct  # 下跌趋势中减仓

        # 3. 获取当前持仓
        pos = float(self.get_position(symbol))

        # 4. 交易逻辑
        if pos == 0:
            should_buy = self._evaluate_entry(
                close_price, filtered_price, velocity, prev_velocity,
                in_downtrend=(self.trend_filter_enabled and self._trend_state == "down"),
            )
            if should_buy:
                self.order_target_percent(symbol=symbol, target_percent=target_pct)
                self._entry_prices[symbol] = close_price
                self._trade_count += 1
                self.log(
                    f"[买入] {bar.timestamp_str} | "
                    f"价格={close_price:.2f} | "
                    f"卡尔曼估计={filtered_price:.2f} | "
                    f"速度={velocity:.6f} | "
                    f"偏离={(close_price / filtered_price - 1) * 100:.2f}%"
                    + (f" | 仓位={target_pct*100:.0f}% 趋势={self._trend_state}"
                       if self.trend_filter_enabled else "")
                )

        elif pos > 0:
            entry_price = self._entry_prices.get(symbol, close_price)
            should_sell = self._evaluate_exit(
                close_price, filtered_price, velocity, prev_velocity,
                entry_price, symbol
            )
            if should_sell:
                self.close_position(symbol)
                self._trade_count += 1
                pnl_pct = (close_price / entry_price - 1) * 100
                self.log(
                    f"[卖出] {bar.timestamp_str} | "
                    f"价格={close_price:.2f} | "
                    f"入场={entry_price:.2f} | "
                    f"收益={pnl_pct:.2f}% | "
                    f"卡尔曼估计={filtered_price:.2f} | "
                    f"速度={velocity:.6f}"
                )
                self._entry_prices.pop(symbol, None)

        # 6. 保存速度状态
        self._prev_velocities[symbol] = velocity

    # ------------------------------------------------------------------
    # 趋势检测
    # ------------------------------------------------------------------
    def _update_trend(self, close: float, bar: Bar) -> None:
        """基于 close vs MA20 + MA20方向 检测趋势。

        规则:
            下跌: close < MA20（连续 N 天确认）
            上涨: close >= MA20 且 MA20 向上（立即切换）
            MA20 向下时即使 close >= MA20 也不切换为上涨
        """
        try:
            ma20_vals = self.get_history(21, bar.symbol, "close")
            if len(ma20_vals) < 21:
                return
            ma20_cur = float(ma20_vals[-20:].mean())
            ma20_prev = float(ma20_vals[:20].mean())
        except Exception:
            return

        is_bearish = close < ma20_cur
        ma20_rising = ma20_cur > ma20_prev
        confirm = self.trend_filter_confirm_bars

        if is_bearish:
            # 下跌信号：累计确认
            self._trend_counter += 1
            if self._trend_counter >= confirm and self._trend_state == "up":
                self._trend_state = "down"
                self.log(
                    f"[趋势转跌] {bar.timestamp_str} | "
                    f"收盘={close:.2f} | MA20={ma20_cur:.2f}"
                    + (f"↓" if not ma20_rising else "") +
                    f" | 连续{confirm}天确认"
                )
        else:
            # 上涨信号
            if ma20_rising:
                # MA20 向上 + close >= MA20 → 确认上涨，立即切换
                self._trend_counter = 0
                if self._trend_state == "down":
                    self._trend_state = "up"
                    self.log(
                        f"[趋势转涨] {bar.timestamp_str} | "
                        f"收盘={close:.2f} | MA20={ma20_cur:.2f}↑ | "
                        f"立即恢复"
                    )
            else:
                # close >= MA20 但 MA20 仍向下 → 维持下跌状态
                self._trend_counter = max(self._trend_counter, confirm)

    # ------------------------------------------------------------------
    # 信号评估
    # ------------------------------------------------------------------
    def _evaluate_entry(
        self,
        close: float,
        filtered: float,
        velocity: float,
        prev_velocity: float,
        in_downtrend: bool = False,
    ) -> bool:
        """评估买入信号。

        上涨趋势: 使用正常 entry_threshold
        下跌趋势: 使用更严格的 downtrend_entry_threshold。
                  只有价格大幅偏离卡尔曼估计时才买入（预示反转），
                  避免在下跌趋势中被普通反弹骗进去。
        """
        threshold = (
            self.downtrend_entry_threshold if in_downtrend
            else self.entry_threshold
        )

        if self.use_price_signal:
            if close > filtered * (1.0 + threshold):
                return True

        if self.use_velocity_signal:
            if velocity > 0.0 and prev_velocity <= 0.0:
                return True

        return False

    def _evaluate_exit(
        self,
        close: float,
        filtered: float,
        velocity: float,
        prev_velocity: float,
        entry_price: float,
        symbol: str,
    ) -> bool:
        """评估卖出信号。"""
        # 止损
        if entry_price > 0:
            pnl_pct = close / entry_price - 1.0
            if pnl_pct < -self.stop_loss_pct:
                self.log(f"[止损] {symbol} 亏损={pnl_pct * 100:.2f}%")
                return True

        # 价格回归信号
        if self.use_price_signal:
            if close < filtered * (1.0 - self.exit_threshold):
                return True

        # 速度反转信号
        if self.use_velocity_signal:
            if velocity < 0.0 and prev_velocity >= 0.0:
                return True

        return False


# ----------------------------------------------------------------------
# 工厂函数
# ----------------------------------------------------------------------
def create_kalman_strategy(**kwargs: Any) -> KalmanStrategy:
    """创建卡尔曼策略实例的工厂函数。

    用法:
        strategy = create_kalman_strategy(
            entry_threshold=0.03,
            stop_loss_pct=0.08,
        )
    """
    return KalmanStrategy(**kwargs)
