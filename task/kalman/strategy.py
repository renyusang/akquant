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
    market_filter_enabled = BoolParam(
        False, title="大盘环境过滤(弱市禁止新开仓, 不影响持仓退出)")
    use_velocity_signal = BoolParam(True, title="是否使用速度反转信号")
    trend_filter_enabled = BoolParam(False, title="是否启用趋势过滤")
    trend_filter_confirm_bars = IntParam(3, ge=1, le=20, title="趋势确认天数")
    trend_recover_confirm_bars = IntParam(
        1, ge=1, le=10, title="转涨确认天数(连续M天站上MA20且MA20向上才转涨, 防死区仓位跳变)"
    )
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
    adx_filter_enabled = BoolParam(False, title="是否启用ADX趋势状态过滤")
    adx_filter_threshold = FloatParam(
        20.0, ge=1.0, le=60.0, title="ADX趋势阈值(低于视为震荡锁定方向)"
    )
    adx_filter_period = IntParam(14, ge=2, le=60, title="ADX周期")
    min_hold_bars = IntParam(
        0, ge=0, le=60, title="最小持仓K线数(0=禁用。买入后N根内抑制"
        "价格回归/速度反转卖出,仅止损可卖出,避免震荡期高频磨损)"
    )
    rsi_filter_enabled = BoolParam(False, title="RSI追高过滤(超买抑制买入)")
    rsi_overbought = FloatParam(70.0, ge=50.0, le=95.0, title="RSI超买阈值")
    bbands_squeeze_enabled = BoolParam(False, title="布林带挤压过滤(蓄势期不买)")
    bb_squeeze_ratio = FloatParam(
        0.50, ge=0.10, le=1.00, title="带宽挤压阈值=历史均值×该比值(低于视为蓄势)"
    )
    atr_adaptive_exit_enabled = BoolParam(
        False, title="NATR自适应退出宽度(高波动期放宽回归阈值避免被洗出)"
    )
    exit_atr_factor = FloatParam(
        0.50, ge=0.1, le=3.0, title="退出NATR缩放系数"
    )
    sar_exit_enabled = BoolParam(False, title="SAR跟踪止损(持仓期跌破SAR卖出)")
    sar_af_step = FloatParam(0.02, ge=0.005, le=0.1, title="SAR加速因子步长")
    sar_af_max = FloatParam(0.20, ge=0.05, le=1.0, title="SAR加速因子上限")
    mfi_filter_enabled = BoolParam(
        False, title="MFI超买过滤(量价确认: 资金超买追高抑制买入)"
    )
    mfi_overbought = FloatParam(70.0, ge=50.0, le=95.0, title="MFI超买阈值")
    mfi_period = IntParam(14, ge=5, le=30, title="MFI周期")

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
        self.market_filter_enabled = bool(p("market_filter_enabled"))
        # 大盘状态映射 {date_str: "强"/"弱"/"过渡"}, 由回测层传入
        self.market_state_map = dict(kwargs.get("market_state_map") or {})
        self.use_velocity_signal = bool(p("use_velocity_signal"))
        self.trend_filter_enabled = bool(p("trend_filter_enabled"))
        self.trend_filter_confirm_bars = max(1, int(p("trend_filter_confirm_bars")))
        self.trend_recover_confirm_bars = max(1, int(p("trend_recover_confirm_bars")))
        self.trend_bear_position_pct = float(p("trend_bear_position_pct"))
        self.downtrend_entry_threshold = float(p("downtrend_entry_threshold"))
        self.single_position_pct = float(p("single_position_pct"))
        self.max_positions = int(p("max_positions"))
        self.initial_cash = float(p("initial_cash"))
        self.adx_filter_enabled = bool(p("adx_filter_enabled"))
        self.adx_filter_threshold = float(p("adx_filter_threshold"))
        self.adx_filter_period = max(2, int(p("adx_filter_period")))
        self.min_hold_bars = max(0, int(p("min_hold_bars")))
        self.rsi_filter_enabled = bool(p("rsi_filter_enabled"))
        self.rsi_overbought = float(p("rsi_overbought"))
        self.bbands_squeeze_enabled = bool(p("bbands_squeeze_enabled"))
        self.bb_squeeze_ratio = float(p("bb_squeeze_ratio"))
        self.atr_adaptive_exit_enabled = bool(p("atr_adaptive_exit_enabled"))
        self.exit_atr_factor = float(p("exit_atr_factor"))
        self.sar_exit_enabled = bool(p("sar_exit_enabled"))
        self.sar_af_step = float(p("sar_af_step"))
        self.sar_af_max = float(p("sar_af_max"))
        self.mfi_filter_enabled = bool(p("mfi_filter_enabled"))
        self.mfi_overbought = float(p("mfi_overbought"))
        self.mfi_period = max(5, int(p("mfi_period")))

        # ---- SignalEngine 实例（按 symbol 管理） ----
        self._engines: Dict[str, SignalEngine] = {}
        self._entry_prices: Dict[str, float] = {}
        # 已下单未成交的买入登记(2026-08-24 修复超买):
        #   {symbol: (signal_close, order_bar)} — 用于失败超时释放名额。
        # 原漏洞: 引擎按事件流调度(下一事件即撮合), get_positions() 的
        # ctx.positions 快照在多 symbol 回测下会 stale(返回空), 同 bar 多
        # 标的买入全部放行 → T+1 全部成交, 实际持仓远超 max_positions
        # (组合回测曾出现单日 19 只入场)。实盘 daily_signal 有两阶段下单
        # 含 pending 名额检查, 回测侧对齐。
        self._pending_buys: Dict[str, tuple] = {}    # {symbol: (signal_close, order_bar)}
        self._pending_sells: Dict[str, float] = {}   # {symbol: signal_close}
        # 自维护持仓标的数(含已下单未成交): 不依赖引擎快照(会 stale)
        self._opened_count: int = 0
        self._trade_count: int = 0
        # 最小持仓周期: 记录每标的的入场 bar 序号(加仓不重置)
        self._bar_count: int = 0
        self._entry_bars: Dict[str, int] = {}

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
            engine = SignalEngine(
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
                recover_confirm_bars=self.trend_recover_confirm_bars,
                trend_bear_pct=self.trend_bear_position_pct,
                downtrend_entry=self.downtrend_entry_threshold,
                adx_filter_enabled=self.adx_filter_enabled,
                adx_filter_threshold=self.adx_filter_threshold,
                adx_filter_period=self.adx_filter_period,
                rsi_filter_enabled=self.rsi_filter_enabled,
                rsi_overbought=self.rsi_overbought,
                bbands_squeeze_enabled=self.bbands_squeeze_enabled,
                bb_squeeze_ratio=self.bb_squeeze_ratio,
                atr_adaptive_exit_enabled=self.atr_adaptive_exit_enabled,
                exit_atr_factor=self.exit_atr_factor,
                sar_exit_enabled=self.sar_exit_enabled,
                sar_af_step=self.sar_af_step,
                sar_af_max=self.sar_af_max,
                mfi_filter_enabled=self.mfi_filter_enabled,
                mfi_overbought=self.mfi_overbought,
                mfi_period=self.mfi_period,
            )
            # 用历史数据预热指标窗口(不含当前 bar,当前 bar 由 update 增量喂入)
            if (
                self.adx_filter_enabled
                or self.rsi_filter_enabled
                or self.bbands_squeeze_enabled
                or self.mfi_filter_enabled
            ):
                self._warmup_adx(symbol, engine)
            self._engines[symbol] = engine
        return self._engines[symbol]

    def _warmup_adx(self, symbol: str, engine: SignalEngine) -> None:
        """用 get_history 预热 ADX 窗口(回测首根 on_bar 即获得正确 ADX)。"""
        try:
            n = self.adx_filter_period * 2 + 2
            hist_h = self.get_history(n + 1, symbol, "high")
            hist_l = self.get_history(n + 1, symbol, "low")
            hist_c = self.get_history(n + 1, symbol, "close")
            # 排除最后一根(当前 bar,由 update 增量喂入)
            if len(hist_h) >= 2 and len(hist_l) >= 2 and len(hist_c) >= 2:
                hist_v = None
                if self.mfi_filter_enabled:
                    try:
                        hist_v = self.get_history(n + 1, symbol, "volume")
                    except Exception:
                        hist_v = None
                engine.feed_adx_history(
                    hist_h[:-1], hist_l[:-1], hist_c[:-1],
                    volume=hist_v[:-1] if hist_v is not None else None,
                )
        except Exception:
            pass

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
        self._bar_count += 1

        self._clean_pending_buys(symbol)
        self._clean_pending_sells(symbol)
        engine = self._get_engine(symbol)
        engine.set_position(
            float(self.get_position(symbol)) > 0,
            self._entry_prices.get(symbol, 0.0),
        )

        # 计算 MA20
        ma20_cur, ma20_prev = self._calc_ma20(symbol, close_price)

        # 评估信号(传 high/low 供 ADX 计算)
        result = engine.update(
            close_price, ma20_cur, ma20_prev, high=bar.high, low=bar.low,
            volume=getattr(bar, "volume", None),
        )

        # 交易执行
        pos = float(self.get_position(symbol))

        # T日 close 涨跌停近似保护(涨停不买/跌停不卖)
        prev_close = self._get_prev_close(symbol, close_price)
        pct = self._limit_pct(symbol)
        limit_up = prev_close * (1 + pct) if prev_close > 0 else 0
        limit_down = prev_close * (1 - pct) if prev_close > 0 else 0

        if pos == 0:
            if result["signal"] == "buy":
                # 大盘环境过滤(2026-08-14 研究): 弱市禁止新开仓,
                # 持仓中的补仓/卖出不受影响
                if self.market_filter_enabled:
                    # bar.timestamp 为 UTC 纳秒 → 北京时间日期
                    from datetime import datetime, timedelta, timezone
                    _ts = datetime.fromtimestamp(bar.timestamp / 1e9,
                                                 tz=timezone.utc)
                    d = (_ts + timedelta(hours=8)).strftime("%Y-%m-%d")
                    if self.market_state_map.get(d) == "弱":
                        self.log(
                            f"[大盘弱市跳过买入] {bar.timestamp_iso} "
                            f"{symbol} close¥{close_price:.2f}"
                        )
                        return
                if limit_up and close_price >= limit_up:
                    self.log(
                        f"[涨停跳过买入] {bar.timestamp_iso} {symbol} "
                        f"close¥{close_price:.2f}≥涨停¥{limit_up:.2f}"
                    )
                    return
                # 修复(2026-08-24): 自维护持仓计数(含已下单未成交)检查名额。
                # 不用 get_positions()——ctx.positions 快照在多 symbol 回测下
                # stale(返回空导致超买漏洞); _opened_count 由下单/卖出维护。
                if self.max_positions > 0 and self._opened_count >= self.max_positions:
                    return
                target_pct = result["target_pct"] * self.single_position_pct
                # 按 initial_cash 固定金额下单(对齐 portfolio_backtest,无复利效应)
                self.order_target_value(
                    symbol=symbol,
                    target_value=self.initial_cash * target_pct,
                )
                self._entry_prices[symbol] = close_price
                self._entry_bars[symbol] = self._bar_count
                self._pending_buys[symbol] = (close_price, self._bar_count)
                self._opened_count += 1
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
                # 最小持仓周期: 买入后 N 根内抑制价格回归/速度反转卖出,
                # 仅止损可卖出(止损优先级最高,防止小回撤被洗出形成高频磨损)
                if self.min_hold_bars > 0:
                    held = self._bar_count - self._entry_bars.get(
                        symbol, self._bar_count
                    )
                    is_stop_loss = "止损" in str(result["reason"])
                    if held < self.min_hold_bars and not is_stop_loss:
                        self.log(
                            f"[持仓保护] {bar.timestamp_iso} {symbol} "
                            f"持仓{held}根<{self.min_hold_bars}根, 抑制卖出"
                            f"({result['reason']})"
                        )
                        return
                entry_price = self._entry_prices.get(symbol, close_price)
                self.close_position(symbol)
                # 修复(2026-08-24): 卖出下单不立即释放名额——卖出订单 T+1
                # 才成交, 若被拒持仓保留; 立即 -1 会让名额提前释放 → 同日
                # 新买入 → 持仓净增(曾致最大持仓 10)。改为登记 _pending_sells,
                # 成交后(持仓消失)由 _clean_pending_sells 释放名额。
                self._pending_sells[symbol] = close_price
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
            else:
                # 趋势翻转补仓: 下跌→上涨,目标仓位从30%→95%
                target_pct = result["target_pct"] * self.single_position_pct
                target_value = self.initial_cash * target_pct
                current_value = pos * close_price
                if target_value > current_value * 1.05:
                    self.order_target_value(symbol=symbol, target_value=target_value)
                    self._entry_prices[symbol] = close_price
                    self._trade_count += 1
                    self.log(
                        f"[加仓] {bar.timestamp_iso} | "
                        f"当前¥{current_value:,.0f}→目标¥{target_value:,.0f} | "
                        f"趋势→{result['trend']} | "
                        f"卡尔曼估计={result['kalman_price']:.2f}"
                    )

    # ------------------------------------------------------------------
    # 待成交买入清理(2026-08-24 超买修复配套)
    # ------------------------------------------------------------------
    def _clean_pending_buys(self, symbol: str) -> None:
        """清理某标的的待成交买入登记(2026-08-24 超买修复配套)。

        - 已成交: get_position(symbol) > 0 → 移除登记(_opened_count 不变,
          下单时已计入名额, 成交后该标的占用名额是事实)
        - 被拒(资金不足等): get_open_orders(symbol) 已不含该订单且无持仓
          → 移除登记并 _opened_count -= 1 释放名额
        注意: 不能用 bar 窗口超时判断——多 symbol 回测下引擎按事件流调度,
        同时间戳内 10 个 symbol 轮转会跨多个 bar 序号, 短窗口会误判
        "超时"释放名额导致超买重现。get_open_orders 查引擎实时订单状态,
        无此问题。
        """
        if symbol not in self._pending_buys:
            return
        if float(self.get_position(symbol)) > 0:
            del self._pending_buys[symbol]          # 已成交
        elif not self.get_open_orders(symbol):
            del self._pending_buys[symbol]          # 订单消失且无持仓 = 被拒
            self._opened_count = max(0, self._opened_count - 1)

    def _clean_pending_sells(self, symbol: str) -> None:
        """待成交卖出登记维护(2026-08-24/26 超买修复配套)。

        - 成交: 持仓消失 → 移除登记并释放名额(_opened_count -1)
        - 被拒(订单消失且持仓保留, 如数据缺口): 移除登记但**不释放名额**
          (该标的仍占名额)——防被拒订单使名额被永久占用(2026-08-26 补充;
          名额检查无预释放, 卖出成交才释放, 此处兜底防挂账)
        """
        if symbol not in self._pending_sells:
            return
        if float(self.get_position(symbol)) <= 0:
            del self._pending_sells[symbol]         # 成交 → 释放名额
            self._opened_count = max(0, self._opened_count - 1)
        elif not self.get_open_orders(symbol):
            del self._pending_sells[symbol]         # 被拒 → 移除登记(名额保留)

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
