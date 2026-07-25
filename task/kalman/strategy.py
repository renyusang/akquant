"""
卡尔曼滤波辅助交易策略。

基于 2 状态卡尔曼滤波器（价格 + 速度/趋势）生成买卖信号。
信号评估逻辑委托给 signal_engine.SignalEngine。

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

from akquant import Bar, BoolParam, FloatParam, IntParam, Strategy

from signal_engine import SignalEngine


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

    # 预热期:run_backtest 读 warmup_bars(非 warmup_period),前 N bar 不调 on_bar
    warmup_bars = 40
    warmup_period = 40  # 兼容(部分接口读 warmup_period)

    # === 内联参数声明(0.3.20:替代 PARAM_MODEL + __init__ 参数) ===
    kalman_q_price = FloatParam(1e-4, ge=1e-6, le=1.0, title="价格过程噪声")
    kalman_q_vel = FloatParam(1e-5, ge=1e-7, le=1.0, title="速度过程噪声")
    kalman_r = FloatParam(1e-2, ge=1e-5, le=1.0, title="观测噪声")
    entry_threshold = FloatParam(0.02, ge=0.001, le=0.20, title="买入价格偏离阈值")
    exit_threshold = FloatParam(0.005, ge=0.0, le=0.10, title="卖出价格偏离阈值")
    stop_loss_pct = FloatParam(0.05, ge=0.01, le=0.30, title="止损比例")
    use_price_signal = BoolParam(True, title="是否使用价格偏离信号")
    use_velocity_signal = BoolParam(True, title="是否使用速度反转信号")
    trend_filter_enabled = BoolParam(False, title="是否启用趋势过滤")
    trend_filter_confirm_bars = IntParam(3, ge=1, le=20, title="趋势确认天数")
    trend_bear_position_pct = FloatParam(
        0.30, ge=0.0, le=1.0, title="下跌仓位比例"
    )
    downtrend_entry_threshold = FloatParam(
        0.03, ge=0.0, le=0.20, title="下跌买入阈值"
    )
    single_position_pct = FloatParam(
        1.0, ge=0.0, le=1.0, title="单只仓位上限比例(组合回测用0.20)"
    )
    max_positions = IntParam(0, ge=0, le=100, title="池内最多持仓数(0=不限)")
    initial_cash = FloatParam(100000.0, ge=0, title="初始资金")

    # 0.3.20 引擎要求的方法(Python 基类缺失,需子类提供空实现)
    def _flush_pending_order_events(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __init__(self, **kwargs: Any) -> None:
        """初始化策略(参数由内联字段自动注入,kwargs 接收覆盖值)。"""
        super().__init__()

        # 从 kwargs 或类体内联字段 default 获取参数值(覆盖类体 descriptor)
        p = lambda n: kwargs.get(n, getattr(type(self), n).default)
        self.kalman_q_price = float(p("kalman_q_price"))
        self.kalman_q_vel = float(p("kalman_q_vel"))
        self.kalman_r = float(p("kalman_r"))
        self.entry_threshold = float(p("entry_threshold"))
        self.exit_threshold = float(p("exit_threshold"))
        self.stop_loss_pct = float(p("stop_loss_pct"))
        self.use_price_signal = bool(p("use_price_signal"))
        self.use_velocity_signal = bool(p("use_velocity_signal"))
        self.trend_filter_enabled = bool(p("trend_filter_enabled"))
        self.trend_filter_confirm_bars = max(1, int(p("trend_filter_confirm_bars")))
        self.trend_bear_position_pct = float(p("trend_bear_position_pct"))
        self.downtrend_entry_threshold = float(p("downtrend_entry_threshold"))
        self.single_position_pct = float(p("single_position_pct"))
        self.max_positions = int(p("max_positions"))
        self.initial_cash = float(p("initial_cash"))

        # ---- SignalEngine 实例（按 symbol 管理） ----
        self._engines: Dict[str, SignalEngine] = {}
        self._entry_prices: Dict[str, float] = {}
        # 延迟下单:T日信号存 pending,T+1 on_bar 检查一字涨跌停后下单(CurrentOpen T+1 撮合)
        self._pending_buys: Dict[str, tuple] = {}    # {symbol: (signal_close, target_pct)}
        self._pending_sells: Dict[str, float] = {}   # {symbol: signal_close}
        self._trade_count: int = 0

    # ------------------------------------------------------------------
    # 公开属性（回测后可通过 result.strategy 访问）
    # ------------------------------------------------------------------
    @property
    def trade_count(self) -> int:
        """总交易次数。"""
        return self._trade_count

    # ------------------------------------------------------------------
    # SignalEngine 管理
    # ------------------------------------------------------------------
    def _get_engine(self, symbol: str) -> SignalEngine:
        """获取或创建指定标的的 SignalEngine 实例。"""
        if symbol not in self._engines:
            self._engines[symbol] = SignalEngine(
                kalman_q_price=self.kalman_q_price,
                kalman_q_vel=self.kalman_q_vel,
                kalman_r=self.kalman_r,
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
                stop_loss_pct=self.stop_loss_pct,
                use_price_signal=self.use_price_signal,
                use_velocity_signal=self.use_velocity_signal,
                trend_filter_enabled=self.trend_filter_enabled,
                trend_confirm_bars=self.trend_filter_confirm_bars,
                trend_bear_pct=self.trend_bear_position_pct,
                downtrend_entry=self.downtrend_entry_threshold,
            )
        return self._engines[symbol]

    # ------------------------------------------------------------------
    # 涨跌停判断(对齐实盘 orders.py,板块识别)
    # ------------------------------------------------------------------
    @staticmethod
    def _limit_pct(symbol: str) -> float:
        """涨跌幅:科创板/创业板 20%,主板 10%。"""
        return 0.20 if symbol.startswith(("688", "300", "301")) else 0.10

    def _get_prev_close(self, symbol: str, fallback: float) -> float:
        """获取前一日收盘价(get_history(2) 的第一个元素)。"""
        try:
            hist = self.get_history(2, symbol, "close")
            if len(hist) >= 2:
                return float(hist[0])
        except Exception:
            pass
        return fallback

    # ------------------------------------------------------------------
    # 核心策略逻辑
    # 注:AKQuant 0.2.22 强制 fill_policy(open) → bar_offset=1(NextOpen),且 NextOpen
    # pending 订单在 T+1 on_bar 无法 cancel(get_open_orders 查不到),无法精确拦截
    # T+1 一字涨跌停。改用 T日 close 涨跌停近似(涨停不买/跌停不卖,避免追涨杀跌)。
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        """处理每根 K 线。"""
        symbol: str = bar.symbol
        close_price: float = bar.close

        engine = self._get_engine(symbol)
        engine.set_position(
            float(self.get_position(symbol)) > 0,
            self._entry_prices.get(symbol, 0.0),
        )

        # 计算 MA20
        ma20_cur, ma20_prev = self._calc_ma20(symbol, close_price)

        # 评估信号
        result = engine.update(close_price, ma20_cur, ma20_prev)

        # 交易执行
        pos = float(self.get_position(symbol))

        # T日 close 涨跌停近似保护(涨停不买/跌停不卖)
        prev_close = self._get_prev_close(symbol, close_price)
        pct = self._limit_pct(symbol)
        limit_up = prev_close * (1 + pct) if prev_close > 0 else 0
        limit_down = prev_close * (1 - pct) if prev_close > 0 else 0

        if pos == 0:
            if result["signal"] == "buy":
                if limit_up and close_price >= limit_up:
                    self.log(
                        f"[涨停跳过买入] {bar.timestamp_iso} {symbol} "
                        f"close¥{close_price:.2f}≥涨停¥{limit_up:.2f}"
                    )
                    return
                if self.max_positions > 0:
                    held = sum(
                        1
                        for s, v in self.get_positions().items()
                        if s != symbol and abs(float(v)) > 0
                    )
                    if held >= self.max_positions:
                        return
                target_pct = result["target_pct"] * self.single_position_pct
                # 按 initial_cash 固定金额下单(对齐 portfolio_backtest,无复利效应)
                self.order_target_value(
                    symbol=symbol,
                    target_value=self.initial_cash * target_pct,
                )
                self._entry_prices[symbol] = close_price
                self._trade_count += 1
                self.log(
                    f"[买入] {bar.timestamp_iso} | "
                    f"价格={close_price:.2f} | "
                    f"卡尔曼估计={result['kalman_price']:.2f} | "
                    f"速度={result['kalman_velocity']:.6f} | "
                    f"偏离={(close_price / result['kalman_price'] - 1) * 100:.2f}%"
                    + (
                        f" | 仓位={target_pct * 100:.0f}% 趋势={result['trend']}"
                        if self.trend_filter_enabled
                        else ""
                    )
                )

        elif pos > 0:
            if result["signal"] == "sell":
                if limit_down and close_price <= limit_down:
                    self.log(
                        f"[跌停跳过卖出] {bar.timestamp_iso} {symbol} "
                        f"close¥{close_price:.2f}≤跌停¥{limit_down:.2f}"
                    )
                    return
                entry_price = self._entry_prices.get(symbol, close_price)
                self.close_position(symbol)
                self._trade_count += 1
                pnl_pct = (close_price / entry_price - 1) * 100
                self.log(
                    f"[卖出] {bar.timestamp_iso} | "
                    f"价格={close_price:.2f} | "
                    f"入场={entry_price:.2f} | "
                    f"收益={pnl_pct:.2f}% | "
                    f"卡尔曼估计={result['kalman_price']:.2f} | "
                    f"速度={result['kalman_velocity']:.6f}"
                )
                self._entry_prices.pop(symbol, None)

    # ------------------------------------------------------------------
    # MA20 计算
    # ------------------------------------------------------------------
    def _calc_ma20(self, symbol: str, fallback: float) -> tuple:
        """计算当前和前一天的 MA20。

        返回 (ma20_cur, ma20_prev),如果数据不足则返回 (fallback, fallback)。
        """
        try:
            ma20_vals = self.get_history(21, symbol, "close")
            if len(ma20_vals) >= 21:
                ma20_cur = float(ma20_vals[-20:].mean())
                ma20_prev = float(ma20_vals[:20].mean())
                return ma20_cur, ma20_prev
        except Exception:
            pass
        return fallback, fallback


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
