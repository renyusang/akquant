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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kalman_filter import KalmanFilter2D


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI(14) Wilder 平滑, 与 TA-Lib 语义对齐。

    返回与输入等长的数组(前 period 根为 NaN)。
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    if n < period + 1:
        return np.full(n, np.nan)
    diff = np.diff(c)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    out = np.full(n, np.nan)
    avg_g = gain[:period].mean()
    avg_l = loss[:period].mean()
    out[period] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gain[i - 1]) / period
        avg_l = (avg_l * (period - 1) + loss[i - 1]) / period
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def compute_mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """MFI(Money Flow Index, 资金流量指数), 与 TA-Lib 语义对齐。

    典型价 × 成交量 的资金流入占比, 衡量资金动量。
    返回与输入等长的数组(前 period 根为 NaN)。
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    tp = (h + l + c) / 3.0
    mf = tp * v
    for i in range(period, n):
        pos = neg = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos += mf[j]
            elif tp[j] < tp[j - 1]:
                neg += mf[j]
        out[i] = 100.0 if neg == 0 else 100.0 - 100.0 / (1.0 + pos / neg)
    return out


def compute_bbands(close: np.ndarray, period: int = 20, ndev: float = 2.0):
    """布林带 (middle, upper, lower), 与 TA-Lib 语义对齐。

    返回与输入等长的三元组(前 period-1 根为 NaN)。
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    mid = np.full(n, np.nan)
    up = np.full(n, np.nan)
    low = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = c[i - period + 1 : i + 1]
        m = w.mean()
        sd = w.std(ddof=0)
        mid[i] = m
        up[i] = m + ndev * sd
        low[i] = m - ndev * sd
    return up, mid, low


def compute_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """计算 ADX（平均趋向指数，Wilder 平滑，对齐 TA-Lib 语义）。

    返回与输入等长的数组（前 2*period-1 根为 NaN），arr[i] 表示
    用 [0..i] 区间计算的 ADX 值。纯 numpy 实现，不依赖 akquant。
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = len(c)
    if n < 2 * period or period < 1:
        return np.full(n, np.nan)

    # +DM/-DM/TR（diff 后长度 n-1，索引 i 对应 [i, i+1] 区间）
    up = np.diff(h)
    dn = -np.diff(l)
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )

    def _wilder(x: np.ndarray, p: int) -> np.ndarray:
        """Wilder 平滑: seed = 前 p 个有效值的均值,之后 (prev*(p-1)+cur)/p。"""
        out = np.full(len(x), np.nan)
        first = np.where(np.isfinite(x))[0]
        if len(first) < p:
            return out
        s = first[0]
        out[s + p - 1] = x[s : s + p].mean()
        for i in range(s + p, len(x)):
            out[i] = (out[i - 1] * (p - 1) + x[i]) / p
        return out

    atr = _wilder(tr, period)
    pdi = 100.0 * _wilder(plus_dm, period) / atr
    mdi = 100.0 * _wilder(minus_dm, period) / atr
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = _wilder(dx, period)
    # 有效值从 index 2*period-1 开始，与输入对齐
    return np.concatenate([np.full(1, np.nan), adx]) if len(adx) == n - 1 else adx


class TrendDetector:
    """基于 close vs MA20 + MA20 方向的趋势检测器。

    规则:
        下跌确认: close < MA20 连续 N 天 → 转跌
        上涨恢复: close >= MA20 且 MA20 向上 → 立即转涨
        MA20 向下时即使 close >= MA20 也不切换为上涨
        ADX 门控(可选): ADX < 阈值时视为震荡期,锁定当前状态,
            不允许方向翻转(防止震荡期 MA20 频繁翻转导致信号抖动)
    """

    def __init__(
        self,
        enabled: bool = True,
        confirm_bars: int = 1,
        adx_enabled: bool = False,
        adx_threshold: float = 20.0,
    ) -> None:
        self._enabled = enabled
        self._confirm = max(1, int(confirm_bars))
        self._adx_enabled = adx_enabled
        self._adx_threshold = float(adx_threshold)
        self._state: str = "up"
        self._counter: int = 0

    @property
    def state(self) -> str:
        return self._state

    def update(
        self,
        close: float,
        ma20_cur: float,
        ma20_prev: float,
        adx: Optional[float] = None,
    ) -> str:
        """更新趋势状态并返回 ("up" | "down")。

        adx: 当前 ADX 值。adx_enabled 且 adx < 阈值时锁定状态。
             adx 为 None(数据不足)时门控不生效。
        """
        if not self._enabled:
            self._state = "up"
            return self._state

        # ADX 门控: 震荡期(ADX < 阈值)锁定状态, 确认计数清零
        if self._adx_enabled and adx is not None and adx < self._adx_threshold:
            self._counter = 0
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
        - adx_enabled / adx_filter_enabled: 是否启用 ADX 趋势状态门控
        - adx_threshold / adx_filter_threshold: ADX 趋势阈值(低于视为震荡)
        - adx_period / adx_filter_period: ADX 周期(默认 14)
        - rsi_filter_enabled: RSI 追高过滤(超买抑制买入)
        - rsi_overbought: RSI 超买阈值(默认 70)
        - rsi_period: RSI 周期(默认 14)
        - bbands_squeeze_enabled: 布林带挤压过滤(带宽显著低于历史均值抑制买入)
        - bb_period / bb_ndev: 布林带参数(默认 20 / 2.0)
        - bb_squeeze_ratio: 挤压判定阈值 = 历史均值 × 该比值(默认 0.5)
        - atr_adaptive_exit_enabled: NATR 自适应退出宽度
          (退出阈值 = max(exit_threshold, exit_atr_factor × NATR),
           高波动期放宽避免被洗出——实证:快速反转亏损随波动单调加深)
        - exit_atr_factor: NATR 缩放系数(默认 0.5)
        - sar_exit_enabled: SAR 跟踪止损(持仓期 close < SAR 卖出)
        - sar_af_step / sar_af_max: SAR 加速因子(默认 0.02 / 0.2)
        - mfi_filter_enabled: MFI 超买过滤(量价确认: 资金超买抑制买入)
        - mfi_overbought: MFI 超买阈值(默认 70)
        - mfi_period: MFI 周期(默认 14)
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
        self.adx_enabled = bool(_p("adx_enabled", False, "adx_filter_enabled"))
        self.adx_threshold = float(
            _p("adx_threshold", 20.0, "adx_filter_threshold")
        )
        self.adx_period = int(_p("adx_period", 14, "adx_filter_period"))
        self.rsi_filter_enabled = bool(_p("rsi_filter_enabled", False))
        self.rsi_overbought = float(_p("rsi_overbought", 70.0))
        self.rsi_period = int(_p("rsi_period", 14))
        self.bbands_squeeze_enabled = bool(_p("bbands_squeeze_enabled", False))
        self.bb_period = int(_p("bb_period", 20))
        self.bb_ndev = float(_p("bb_ndev", 2.0))
        self.bb_squeeze_ratio = float(_p("bb_squeeze_ratio", 0.5))
        self.atr_adaptive_exit_enabled = bool(
            _p("atr_adaptive_exit_enabled", False)
        )
        self.exit_atr_factor = float(_p("exit_atr_factor", 0.5))
        self.sar_exit_enabled = bool(_p("sar_exit_enabled", False))
        self.sar_af_step = float(_p("sar_af_step", 0.02))
        self.sar_af_max = float(_p("sar_af_max", 0.2))
        self.mfi_filter_enabled = bool(_p("mfi_filter_enabled", False))
        self.mfi_overbought = float(_p("mfi_overbought", 70.0))
        self.mfi_period = int(_p("mfi_period", 14))

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
            adx_enabled=self.adx_enabled,
            adx_threshold=self.adx_threshold,
        )

        # ---- 内部状态 ----
        self._prev_velocity: float = 0.0
        self._prev_vel_stored: float = 0.0  # 上一次 update 前的速度
        self._has_position: bool = False
        self._entry_price: float = 0.0
        self._bars_processed: int = 0

        # ---- ADX 滑动窗口 ----
        # Wilder 递归平滑衰减慢,窗口过短时截断重算误差大(实测30根平均误差8),
        # 120 根后误差 < 0.1,对阈值判定(20/25)无影响
        self._adx_window_len = max(2 * self.adx_period + 2, 120)
        self._adx_h: List[float] = []
        self._adx_l: List[float] = []
        self._adx_c: List[float] = []
        self._last_adx: Optional[float] = None

        # ---- RSI / BBANDS 窗口(供过滤器使用) ----
        self._rsi_window_len = max(self.rsi_period * 4, 120)
        self._bb_window_len = max(self.bb_period * 6, 120)
        self._rsi_c: List[float] = []
        self._bb_c: List[float] = []
        self._last_rsi: Optional[float] = None
        self._last_bandwidth: Optional[float] = None
        self._bb_squeeze: bool = False

        # ---- NATR / SAR 状态(自适应退出) ----
        self._last_natr: Optional[float] = None
        self._sar_val: float = 0.0
        self._sar_ep: float = 0.0
        self._sar_af: float = 0.02
        self._prev_low: Optional[float] = None

        # ---- MFI 状态(量价确认) ----
        self._mfi_v: List[float] = []
        self._last_mfi: Optional[float] = None

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
        """设置当前持仓状态（由外部调用方管理）。

        开仓时重置 SAR 跟踪止损状态(从入场价起算)。
        """
        if self.sar_exit_enabled and has_position and not self._has_position:
            self._sar_val = entry_price
            self._sar_ep = entry_price
            self._sar_af = self.sar_af_step
            self._prev_low = None
        self._has_position = has_position
        self._entry_price = entry_price

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------
    def process_history(self, df: pd.DataFrame) -> None:
        """用历史数据预热卡尔曼滤波器和趋势检测器。

        遍历整个 DataFrame，逐 bar 更新卡尔曼滤波器，
        并在有足够数据时更新趋势状态。
        若 df 含 high/low 列且启用 ADX，同时预热 ADX 门控。
        """
        if df.empty:
            return

        closes = df["close"].values
        n = len(closes)

        # 预热 ADX 序列（df 含 high/low 时）
        adx_seq: Optional[np.ndarray] = None
        has_ohlc = "high" in df.columns and "low" in df.columns
        if self.adx_enabled and has_ohlc:
            adx_seq = compute_adx(
                df["high"].values, df["low"].values, closes, self.adx_period
            )
            # 同步窗口: 保留最后 _adx_window_len 根
            w = self._adx_window_len
            self._adx_h = list(df["high"].values[-w:])
            self._adx_l = list(df["low"].values[-w:])
            self._adx_c = list(closes[-w:])
            fin = np.isfinite(adx_seq)
            if fin.any():
                idx = np.where(fin)[0][-1]
                self._last_adx = float(adx_seq[idx])
        # 预热 RSI/BBANDS 窗口(独立于 adx_enabled, 过滤器启用时即生效)
        if has_ohlc and (self.rsi_filter_enabled or self.bbands_squeeze_enabled):
            self._rsi_c = list(closes[-self._rsi_window_len:])
            self._bb_c = list(closes[-self._bb_window_len:])
            self._refresh_rsi_bbands()
        # 预热 MFI 窗口(df 含 volume 且 MFI 过滤器启用时)
        if (
            self.mfi_filter_enabled
            and has_ohlc
            and "volume" in df.columns
        ):
            w = self._adx_window_len
            self._adx_h = list(df["high"].values[-w:])
            self._adx_l = list(df["low"].values[-w:])
            self._adx_c = list(closes[-w:])
            self._mfi_v = list(df["volume"].values[-w:])
            self._refresh_mfi()

        for i in range(n):
            close = float(closes[i])
            self._kf.update(close)
            self._bars_processed += 1

            # 趋势预热（需要至少 21 根 bar）
            if n >= 21 and i >= 20:
                ma20_cur = float(closes[i - 19 : i + 1].mean())
                ma20_prev = float(closes[i - 20 : i].mean())
                adx_i = None
                if adx_seq is not None:
                    v = adx_seq[i]
                    adx_i = float(v) if np.isfinite(v) else None
                self._trend.update(close, ma20_cur, ma20_prev, adx_i)

            # 更新上一步速度
            if i >= 1:
                self._prev_velocity = self._kf.get_velocity()

    # ------------------------------------------------------------------
    # ADX 窗口维护
    # ------------------------------------------------------------------
    def feed_adx_history(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
    ) -> None:
        """喂入一段历史 OHLCV 序列（不含当前 bar），预热指标窗口。

        volume 提供时同步预热 MFI(量价确认)窗口。
        """
        h = [float(x) for x in high]
        l = [float(x) for x in low]
        c = [float(x) for x in close]
        if len(h) != len(l) or len(h) != len(c) or not h:
            return
        w = self._adx_window_len
        self._adx_h = (self._adx_h + h)[-w:]
        self._adx_l = (self._adx_l + l)[-w:]
        self._adx_c = (self._adx_c + c)[-w:]
        adx = compute_adx(
            np.asarray(self._adx_h),
            np.asarray(self._adx_l),
            np.asarray(self._adx_c),
            self.adx_period,
        )
        fin = np.isfinite(adx)
        self._last_adx = float(adx[fin][-1]) if fin.any() else None
        self._refresh_natr()
        # RSI/BBANDS 窗口同步
        self._rsi_c = (self._rsi_c + c)[-self._rsi_window_len:]
        self._bb_c = (self._bb_c + c)[-self._bb_window_len:]
        self._refresh_rsi_bbands()
        # MFI 窗口同步(volume 提供时)
        if volume is not None and len(volume) == len(c):
            v = [float(x) for x in volume]
            self._mfi_v = (self._mfi_v + v)[-w:]
            self._refresh_mfi()

    def _feed_adx_bar(
        self, high: float, low: float, close: float, volume: Optional[float] = None
    ) -> None:
        """追加一根 bar 到 ADX 窗口并更新当前 ADX 值。"""
        w = self._adx_window_len
        self._adx_h.append(float(high))
        self._adx_l.append(float(low))
        self._adx_c.append(float(close))
        if len(self._adx_h) > w:
            self._adx_h = self._adx_h[-w:]
            self._adx_l = self._adx_l[-w:]
            self._adx_c = self._adx_c[-w:]
        adx = compute_adx(
            np.asarray(self._adx_h),
            np.asarray(self._adx_l),
            np.asarray(self._adx_c),
            self.adx_period,
        )
        fin = np.isfinite(adx)
        self._last_adx = float(adx[fin][-1]) if fin.any() else None
        self._refresh_natr()
        # RSI/BBANDS 窗口同步
        self._rsi_c = (self._rsi_c + [float(close)])[-self._rsi_window_len:]
        self._bb_c = (self._bb_c + [float(close)])[-self._bb_window_len:]
        self._refresh_rsi_bbands()
        # MFI 窗口同步(volume 提供时)
        if volume is not None:
            self._mfi_v = (self._mfi_v + [float(volume)])[-w:]
            self._refresh_mfi()

    def _refresh_mfi(self) -> None:
        """基于窗口重算 MFI(需要 high/low/close/volume 窗口齐全)。"""
        n = len(self._adx_c)
        if (
            n >= self.mfi_period + 1
            and len(self._mfi_v) == n
            and all(v > 0 for v in self._mfi_v)
        ):
            mfi = compute_mfi(
                np.asarray(self._adx_h),
                np.asarray(self._adx_l),
                np.asarray(self._adx_c),
                np.asarray(self._mfi_v),
                self.mfi_period,
            )
            if np.isfinite(mfi[-1]):
                self._last_mfi = float(mfi[-1])

    def _refresh_natr(self) -> None:
        """基于 ADX 窗口计算 NATR = ATR(period)/close(Wilder 平滑)。"""
        n = len(self._adx_c)
        if n < 2 * self.adx_period or self.adx_period < 1:
            return
        hh = np.asarray(self._adx_h)
        ll = np.asarray(self._adx_l)
        cc = np.asarray(self._adx_c)
        tr = np.maximum(
            hh[1:] - ll[1:],
            np.maximum(np.abs(hh[1:] - cc[:-1]), np.abs(ll[1:] - cc[:-1])),
        )
        p = self.adx_period
        atr = tr[:p].mean()
        for i in range(p, len(tr)):
            atr = (atr * (p - 1) + tr[i]) / p
        if cc[-1] > 0:
            self._last_natr = float(atr / cc[-1])

    def _update_sar(self, high: float, low: float) -> None:
        """持仓期 SAR 跟踪止损更新(多头)。

        SAR = SAR_prev + AF × (EP - SAR_prev); 新高时 EP 上移、AF 递增;
        SAR 不高于最近两根 bar 的最低点(防追价)。
        """
        if not self._has_position or self._sar_val <= 0:
            return
        if high > self._sar_ep:
            self._sar_ep = high
            self._sar_af = min(
                self._sar_af + self.sar_af_step, self.sar_af_max
            )
        self._sar_val = self._sar_val + self._sar_af * (
            self._sar_ep - self._sar_val
        )
        # 约束: 不超过最近两根 bar 低点(经典 SAR 保护)
        lo2 = min(low, self._prev_low) if self._prev_low else low
        if self._sar_val > lo2:
            self._sar_val = lo2
        self._prev_low = low

    def _refresh_rsi_bbands(self) -> None:
        """基于窗口重算 RSI 与布林带宽(截断误差随窗口增大衰减,120根后可忽略)。

        挤压判定使用**不含当前根**的历史窗口——挤压是"信号前"的状态,
        跳涨当天带宽会瞬间扩大,用历史带宽判定才符合"蓄势期"语义。
        """
        if len(self._rsi_c) >= self.rsi_period + 1:
            rsi = compute_rsi(np.asarray(self._rsi_c), self.rsi_period)
            if np.isfinite(rsi[-1]):
                self._last_rsi = float(rsi[-1])
        self._bb_squeeze = False
        if len(self._bb_c) >= self.bb_period + 1:
            hist = np.asarray(self._bb_c[:-1])  # 不含当前根
            if len(hist) >= self.bb_period:
                up, mid, low = compute_bbands(hist, self.bb_period, self.bb_ndev)
                bw = (up - low) / mid
                valid = bw[np.isfinite(bw)]
                if len(valid) >= self.bb_period:
                    self._last_bandwidth = float(valid[-1])
                    # 挤压判定: 当前带宽 < 历史均值 × ratio
                    # (相对均值比, 避免分位数在平稳窄段上的自引用临界)
                    mean_bw = float(np.nanmean(valid))
                    self._bb_squeeze = bool(
                        mean_bw > 0
                        and valid[-1] <= mean_bw * self.bb_squeeze_ratio
                    )

    # ------------------------------------------------------------------
    # 核心更新
    # ------------------------------------------------------------------
    def update(
        self,
        close: float,
        ma20_cur: float,
        ma20_prev: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> Dict[str, Any]:
        """更新卡尔曼滤波器和趋势，评估当前信号。

        high/low: 当前 bar 的最高/最低价。提供时更新 ADX 窗口；
                 未提供(或数据不足)时 ADX 门控不生效。
        volume: 当前 bar 成交量。提供时更新 MFI 窗口(量价确认)。

        返回:
            {
                "close": float,
                "kalman_price": float,
                "kalman_velocity": float,
                "ma20": float,
                "ma20_rising": bool,
                "trend": str,           # "up" | "down"
                "adx": float | None,    # 当前 ADX 值(数据不足为 None)
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

        # 更新 ADX 窗口(high/low 提供时)
        if high is not None and low is not None:
            self._feed_adx_bar(high, low, close, volume)
            # SAR 跟踪止损(持仓期)
            if self.sar_exit_enabled:
                self._update_sar(high, low)

        # 更新趋势(ADX 门控: adx_enabled 且值有效时生效)
        adx = self._last_adx if self.adx_enabled else None
        self._trend.update(close, ma20_cur, ma20_prev, adx)

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
            "adx": adx,
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
            # ---- BBANDS+RSI 过滤器(追高/挤压抑制买入) ----
            if self.rsi_filter_enabled and self._last_rsi is not None:
                if self._last_rsi > self.rsi_overbought:
                    return (
                        "hold",
                        0.0,
                        f"RSI超买过滤(RSI={self._last_rsi:.0f}>{self.rsi_overbought:.0f}) | "
                        f"趋势={self._trend.state}",
                    )
            if self.bbands_squeeze_enabled and self._bb_squeeze:
                return (
                    "hold",
                    0.0,
                    f"带宽挤压过滤(蓄势期) | 趋势={self._trend.state}",
                )
            # ---- MFI 超买过滤(量价确认: 资金超买追高抑制买入) ----
            # 实证: MFI>70 时买入信号快速反转率 71.6%(vs <30 时 16.7%)
            if self.mfi_filter_enabled and self._last_mfi is not None:
                if self._last_mfi > self.mfi_overbought:
                    return (
                        "hold",
                        0.0,
                        f"MFI超买过滤(MFI={self._last_mfi:.0f}>{self.mfi_overbought:.0f}) | "
                        f"趋势={self._trend.state}",
                    )
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

        # SAR 跟踪止损(动态保护, 高于价格回归优先级)
        if self.sar_exit_enabled and self._sar_val > 0:
            if close < self._sar_val:
                return (
                    "sell",
                    0.0,
                    f"SAR跟踪止损(close={close:.2f}<SAR={self._sar_val:.2f})",
                )

        # 价格回归(自适应退出宽度: 高波动期放宽阈值避免被洗出)
        if self.use_price_signal:
            eff_exit = self.exit_threshold
            if self.atr_adaptive_exit_enabled and self._last_natr is not None:
                eff_exit = max(eff_exit, self.exit_atr_factor * self._last_natr)
            if close < filtered * (1.0 - eff_exit):
                if eff_exit != self.exit_threshold:
                    return (
                        "sell",
                        0.0,
                        f"价格回归(偏离{(close / filtered - 1) * 100:.1f}%, "
                        f"自适应宽度{eff_exit * 100:.1f}%)",
                    )
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
