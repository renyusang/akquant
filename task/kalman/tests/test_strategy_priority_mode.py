"""Tests for KalmanStrategy.priority_mode 名额优先级(2026-09-17 研究)。

背景: 回测侧名额分配原本是纯先到先得(引擎同 timestamp 的 bar 到达顺序 =
symbols 插入顺序), 而实盘 daily_signal 是"补仓优先 + 池内偏离度降序"。
本组测试锁定新增的四模式语义:

    arrival          旧路径(不延迟)= 冻结基线, 逐笔不变
    arrival_deferred 到达序但收盘统一分配(隔离"下单时点"效应)
    deviation        偏离度降序(实盘口径)
    uptrend_first    上升趋势优先, 再按偏离降序

关键不变量:
- 排序只在名额/资金不足时产生差异(名额足够时各模式逐笔一致)
- 收盘下单仍成交于**次日开盘**(T+1 语义不因延迟而漂移)
- 缓冲按交易日过期 + flush 幂等(防隔日补单/重复下单)
- 整手买不起的候选不占名额(幽灵名额, 对齐实盘 buy_qty>0 守卫)
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

from strategy import KalmanStrategy                                 # noqa: E402

N_BARS = 60
BREAKOUT_BAR = 20


def _make_ranked_pool(n_symbols=10, base=0.03, step=0.005, price=100.0):
    """n_symbols 只同日突破, 但**跳幅递增** → 偏离度与插入顺序相反。

    既有 test_strategy_max_positions._make_pool_df 用完全同序列池(全部标的
    偏离度相等), 无法区分 arrival 与 deviation —— 本池专为区分而建:
    i 越大偏离越大 → arrival 取前 5 只(600000..600004), deviation 取后 5 只。
    """
    dates = pd.date_range("2026-01-01", periods=N_BARS, freq="B")
    out = {}
    for i in range(n_symbols):
        sym = f"{600000 + i:06d}"
        close = np.full(N_BARS, price, dtype=float)
        close[BREAKOUT_BAR] = price * (1.0 + base + step * i)
        close[BREAKOUT_BAR + 1:] = close[BREAKOUT_BAR] * 1.001   # 缓涨不触发卖出
        out[sym] = pd.DataFrame({
            "date": dates,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": [1_000_000] * N_BARS,
            "symbol": [sym] * N_BARS,
        })
    return out


def _run_pool(data_map, max_positions=5, priority_mode="arrival",
              single_position_pct=0.1, trend_filter_enabled=False):
    fill_policy = make_fill_policy(
        price_basis="open", temporal="next_event", bar_offset=1)
    strategy = KalmanStrategy(
        initial_cash=1000000,
        entry_threshold=0.02,
        exit_threshold=0.005,
        stop_loss_pct=0.05,
        trend_filter_enabled=trend_filter_enabled,
        single_position_pct=single_position_pct,
        max_positions=max_positions,
        priority_mode=priority_mode,
    )
    return run_backtest(
        data=data_map, strategy=strategy,
        symbols=list(data_map.keys()),
        initial_cash=1000000,
        t_plus_one=True,
        fill_policy=fill_policy,
        lot_size={s: 100 for s in data_map},
        commission_rate=0.0003, stamp_tax_rate=0.001,
        transfer_fee_rate=0.00001, min_commission=5.0,
        show_progress=False,
    )


def _bought_symbols(result):
    o = result.orders_df
    f = o[o["status"] == "filled"]
    return list(f[f["side"].str.lower() == "buy"]["symbol"].unique())


class TestArrivalBaseline:
    """arrival = 冻结基线语义(先到先得, 且不进入新代码路径)。"""

    def test_arrival_keeps_first_come_first_served(self):
        data_map = _make_ranked_pool()
        result = _run_pool(data_map, priority_mode="arrival")
        bought = _bought_symbols(result)
        assert bought == [f"{600000 + i:06d}" for i in range(5)], (
            f"arrival 应按插入序取前 5 只, 实际 {bought}")

    def test_arrival_never_touches_candidate_buffer(self):
        """结构不变式: arrival 路径不得留下缓冲痕迹(证明逐笔等价)。"""
        data_map = _make_ranked_pool()
        result = _run_pool(data_map, priority_mode="arrival")
        st = result.strategy
        assert st._entry_candidates == {}
        assert st._candidate_day is None


class TestDeviationMode:
    """deviation = 实盘口径(池内偏离度降序)。"""

    def test_deviation_picks_highest_deviation(self):
        data_map = _make_ranked_pool()
        result = _run_pool(data_map, priority_mode="deviation")
        bought = _bought_symbols(result)
        # 跳幅随 i 递增 → 偏离最大的 5 只是最后 5 只
        assert bought == [f"{600000 + i:06d}" for i in range(5, 10)], (
            f"deviation 应按偏离降序取后 5 只, 实际 {bought}")

    def test_slot_limit_still_holds(self):
        data_map = _make_ranked_pool()
        result = _run_pool(data_map, priority_mode="deviation")
        pos = result.positions
        assert (pos > 0).sum(axis=1).max() <= 5


class TestArrivalDeferred:
    """arrival_deferred = 到达序 + 收盘统一分配(隔离"下单时点"效应的对照臂)。

    ⚠️ 首版实现曾让该模式落到 deviation 分支(排序键写错) → 与 deviation
    逐字节相同、A/B 归因失效。本组测试专门钉死"它必须保持到达序"。
    """

    def test_arrival_deferred_keeps_arrival_order(self):
        data_map = _make_ranked_pool()
        result = _run_pool(data_map, priority_mode="arrival_deferred")
        bought = _bought_symbols(result)
        assert bought == [f"{600000 + i:06d}" for i in range(5)], (
            f"arrival_deferred 应按到达序取前 5 只, 实际 {bought}")

    def test_arrival_deferred_differs_from_deviation(self):
        """偏离度与到达序相反的池上, 两模式必须选出不同集合。"""
        data_map = _make_ranked_pool()
        a = _bought_symbols(_run_pool(data_map, priority_mode="arrival_deferred"))
        d = _bought_symbols(_run_pool(data_map, priority_mode="deviation"))
        assert a != d, "arrival_deferred 与 deviation 选了同一组 → 排序键未生效"

    def test_arrival_deferred_matches_arrival_selection(self):
        """对照臂与旧路径在"选谁"上应一致(差异只在时点/幽灵名额)。"""
        data_map = _make_ranked_pool()
        a = _bought_symbols(_run_pool(data_map, priority_mode="arrival"))
        b = _bought_symbols(_run_pool(data_map, priority_mode="arrival_deferred"))
        assert a == b


class TestNoContention:
    """名额充足时排序不产生差异(需求: 只在争抢时才起作用)。"""

    def test_same_fills_when_slots_abundant(self):
        data_map = _make_ranked_pool(n_symbols=3)
        a = _run_pool(data_map, max_positions=5, priority_mode="arrival")
        d = _run_pool(data_map, max_positions=5, priority_mode="deviation")
        assert _bought_symbols(a) == _bought_symbols(d)

        oa = a.orders_df.drop(columns=["id"], errors="ignore")
        od = d.orders_df.drop(columns=["id"], errors="ignore")
        cols = [c for c in oa.columns if c in od.columns]
        assert oa[cols].reset_index(drop=True).equals(
            od[cols].reset_index(drop=True)), "无争抢时逐笔应完全一致"


class TestDeferredTiming:
    """收盘延迟下单不得改变 T+1 成交语义。"""

    def test_fill_is_next_day_in_both_paths(self):
        data_map = _make_ranked_pool(n_symbols=6)
        for mode in ("arrival", "deviation"):
            result = _run_pool(data_map, priority_mode=mode)
            o = result.orders_df
            o = o[o["status"] == "filled"]
            assert len(o) > 0
            created = pd.to_datetime(o["created_at"]).dt.normalize()
            updated = pd.to_datetime(o["updated_at"]).dt.normalize()
            gap = (updated - created).dt.days
            # 合成交易日历为工作日: 周五→周一 = 3 天
            assert set(gap.unique()) <= {1, 3}, (
                f"{mode} 成交日与下单日间隔异常: {sorted(set(gap))}")


class TestCandidateBufferUnit:
    """缓冲语义单元测试(过期丢弃 + flush 幂等 + 已持仓丢弃)。"""

    def _mk(self, **kw):
        s = KalmanStrategy(initial_cash=100000, single_position_pct=0.2,
                           max_positions=5, priority_mode="deviation", **kw)
        s._placed = []

        def _spy(sym, entry):          # 复刻真实 _place_entry 的名额记账
            s._placed.append(sym)
            s._opened_count += 1
            s._trade_count += 1

        s._place_entry = _spy
        s.get_position = lambda sym: 0.0
        return s

    @staticmethod
    def _entry(dev=0.05, trend="up"):
        return {"close": 10.0, "target_pct": 0.19, "bar_count": 1,
                "ts_iso": "t", "kalman_price": 9.5, "kalman_velocity": 0.0,
                "trend": trend, "dev": dev}

    def test_buffer_expires_on_new_day(self):
        s = self._mk()
        s._queue_entry_candidate("600000", self._entry(), "2026-01-05")
        assert "600000" in s._entry_candidates
        s._queue_entry_candidate("600001", self._entry(), "2026-01-06")
        assert list(s._entry_candidates) == ["600001"], "跨日未过期丢弃"
        assert s._candidate_day == "2026-01-06"

    def test_flush_is_idempotent(self):
        s = self._mk()
        s._queue_entry_candidate("600000", self._entry(), "2026-01-05")
        s.on_after_trading("2026-01-05", 0)
        s.on_after_trading("2026-01-05", 0)      # shutdown 钩子补发
        assert s._placed == ["600000"], "flush 不幂等 → 重复下单"

    def test_flush_skips_already_held(self):
        s = self._mk()
        s.get_position = lambda sym: 100.0       # 收盘前已持仓
        s._queue_entry_candidate("600000", self._entry(), "2026-01-05")
        s.on_after_trading("2026-01-05", 0)
        assert s._placed == []

    def test_flush_respects_max_positions(self):
        s = self._mk()
        s._opened_count = 5
        s._queue_entry_candidate("600000", self._entry(), "2026-01-05")
        s.on_after_trading("2026-01-05", 0)
        assert s._placed == []

    def test_uptrend_first_beats_deviation(self):
        """趋势优先压过偏离度(纯单元, 钉死两个排序键)。"""
        s = self._mk()
        s.priority_mode = "uptrend_first"
        s.max_positions = 1
        s._entry_candidates = {
            "AAA": self._entry(dev=0.09, trend="down"),
            "BBB": self._entry(dev=0.03, trend="up"),
        }
        s.on_after_trading("2026-01-05", 0)
        assert s._placed == ["BBB"]

        s2 = self._mk()
        s2.max_positions = 1
        s2._entry_candidates = {
            "AAA": self._entry(dev=0.09, trend="down"),
            "BBB": self._entry(dev=0.03, trend="up"),
        }
        s2.on_after_trading("2026-01-05", 0)
        assert s2._placed == ["AAA"], "deviation 模式应取偏离最大者"


class TestGhostSlotGuard:
    """整手买不起的候选不得占名额(对齐实盘 buy_qty>0 守卫)。"""

    def _make_pool_with_expensive(self, n=6, expensive_price=20000.0):
        dates = pd.date_range("2026-01-01", periods=N_BARS, freq="B")
        out = {}
        for i in range(n):
            sym = f"{600000 + i:06d}"
            price = expensive_price if i == 0 else 100.0
            close = np.full(N_BARS, price, dtype=float)
            close[BREAKOUT_BAR] = price * 1.05
            close[BREAKOUT_BAR + 1:] = close[BREAKOUT_BAR] * 1.001
            out[sym] = pd.DataFrame({
                "date": dates,
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "volume": [1_000_000] * N_BARS,
                "symbol": [sym] * N_BARS,
            })
        return out

    def test_unaffordable_does_not_consume_slot(self):
        """600000 单价 2 万 → 1 手(100股)20 万 > 预算, 不得占名额。"""
        data_map = self._make_pool_with_expensive()
        result = _run_pool(data_map, max_positions=5,
                           priority_mode="deviation",
                           single_position_pct=0.1)
        bought = _bought_symbols(result)
        assert "600000" not in bought, "买不起 1 手的候选被下单"
        assert len(bought) == 5, f"名额应给可买候选, 实际只成交 {len(bought)} 只"
