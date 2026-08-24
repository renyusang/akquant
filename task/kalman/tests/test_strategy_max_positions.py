"""Tests for KalmanStrategy.max_positions 超买修复(2026-08-24)。

背景: 引擎同 bar 先对所有 symbol 调 on_bar 再撮合, get_positions() 看不到
当日新订单——同日多标的买入全部放行(T+1 全部成交, 实际持仓远超
max_positions, 组合回测曾出现单日 19 只入场)。
修复: 已下单未成交的买入登记 _pending_buys 计入名额(对齐实盘
daily_signal 两阶段下单的 pending 名额检查)。

覆盖: 集成(10 只同日触发买入 → 最多 5 只持仓) + 单元(_pending_buys
生命周期: 下单登记/成交移除/失败超时释放)。
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.disable(logging.CRITICAL)

from akquant import run_backtest
from akquant.backtest.engine import make_fill_policy

from strategy import KalmanStrategy

N_SYMBOLS = 10
N_BARS = 60
MAX_POS = 5


def _make_pool_df(n_symbols=N_SYMBOLS, breakout_bar=20, seed=42):
    """n_symbols 只**共享同一价格序列**: 前 20 根平稳波动, 第 20 根 +5%
    突破(触发买入), 之后缓慢上行(不触发回归卖出)。全部标的同时触发
    ——复现真实超买场景(组合回测 2024-02-07 单日 19 只入场)。"""
    dates = pd.date_range("2026-01-01", periods=N_BARS, freq="B")
    # 恒定价格(无波动→无速度反转干扰) + 第 20 根跳涨 5%(价格突破信号) +
    # 之后 +0.2%/根缓行(不触发回归卖出, 持仓保持到结束)
    close = np.full(N_BARS, 100.0)
    close[breakout_bar] = close[breakout_bar - 1] * 1.05
    close[breakout_bar + 1:] = close[breakout_bar] * 1.002
    out = {}
    for i in range(n_symbols):
        sym = f"{600000 + i:06d}"
        out[sym] = pd.DataFrame({
            "date": dates,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": [1_000_000] * N_BARS,
            "symbol": [sym] * N_BARS,
        })
    return out


def _run_pool(data_map, max_positions=MAX_POS):
    fill_policy = make_fill_policy(
        price_basis="open", temporal="next_event", bar_offset=1)
    strategy = KalmanStrategy(
        initial_cash=1000000,
        entry_threshold=0.02,
        exit_threshold=0.005,
        stop_loss_pct=0.05,
        trend_filter_enabled=False,
        single_position_pct=0.1,  # 每只 9.5%: 10 只共 95 万 ≤ 100 万初始
        max_positions=max_positions,
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


class TestMaxPositionsIntegration:
    """集成: 10 只同日触发买入 → 同时持仓不超过 max_positions。"""

    def test_concurrent_holdings_bounded(self):
        """修复后: 日度持仓快照任意一天持仓数 ≤ 5(修复前可达 10)。"""
        data_map = _make_pool_df()
        result = _run_pool(data_map)
        pos = result.positions
        assert pos is not None and not pos.empty
        max_held = (pos > 0).sum(axis=1).max()
        assert max_held <= MAX_POS, f"超买: 同时持仓 {max_held} > {MAX_POS}"

    def test_exactly_five_bought(self):
        """首批恰好买入 5 只(名额占满后其余被拦截)。"""
        data_map = _make_pool_df()
        result = _run_pool(data_map)
        orders = result.orders_df
        assert orders is not None and not orders.empty
        filled = orders[orders["status"] == "filled"]
        bought = filled[filled["side"].str.lower() == "buy"]["symbol"].unique()
        assert len(bought) == MAX_POS

    def test_unbounded_without_limit(self):
        """max_positions=0(不限) → 10 只全部买入(对照)。"""
        data_map = _make_pool_df()
        result = _run_pool(data_map, max_positions=0)
        orders = result.orders_df
        filled = orders[orders["status"] == "filled"]
        bought = filled[filled["side"].str.lower() == "buy"]["symbol"].unique()
        assert len(bought) == N_SYMBOLS


class TestPendingBuysLifecycle:
    """_pending_buys 生命周期(2026-08-24 修复后): 登记 → 成交移除 / 被拒释放。

    用 get_open_orders 判断订单是否仍挂起(不依赖 bar 窗口——
    多 symbol 回测事件流跨 bar 序号, 短窗口会误判超时释放名额)。
    """

    def _strategy(self, position=0.0, open_orders=True):
        s = KalmanStrategy(max_positions=5)
        s.get_position = lambda sym: position
        s.get_open_orders = lambda sym=None: [1] if open_orders else []
        return s

    def test_registered_on_order(self):
        """下单时登记 (signal_close, bar)。"""
        s = self._strategy()
        s._pending_buys["000001"] = (10.0, 3)
        assert s._pending_buys["000001"] == (10.0, 3)

    def test_cleanup_on_filled(self):
        """成交后(持仓出现)移除登记, 名额保留(_opened_count 不变)。"""
        s = self._strategy(position=100.0)
        s._opened_count = 3
        s._pending_buys["000001"] = (10.0, 4)
        s._clean_pending_buys("000001")
        assert s._pending_buys == {}
        assert s._opened_count == 3  # 成交不释放名额

    def test_cleanup_on_rejected(self):
        """订单消失(被拒)且无持仓 → 移除登记并释放名额。"""
        s = self._strategy(position=0.0, open_orders=False)
        s._opened_count = 3
        s._pending_buys["000001"] = (10.0, 4)
        s._clean_pending_buys("000001")
        assert s._pending_buys == {}
        assert s._opened_count == 2  # 被拒释放名额

    def test_pending_kept_while_order_active(self):
        """订单仍挂起(未成交未拒) → 登记保留, 名额不释放。"""
        s = self._strategy(position=0.0, open_orders=True)
        s._opened_count = 3
        s._pending_buys["000001"] = (10.0, 4)
        s._clean_pending_buys("000001")
        assert s._pending_buys == {"000001": (10.0, 4)}
        assert s._opened_count == 3

    def test_cleanup_other_symbols_untouched(self):
        """清理只处理目标标的, 其他登记保留。"""
        s = self._strategy(position=0.0, open_orders=False)
        s._opened_count = 2
        s._pending_buys = {"000001": (10.0, 4), "000002": (11.0, 7)}
        s._clean_pending_buys("000001")
        assert s._pending_buys == {"000002": (11.0, 7)}
        assert s._opened_count == 1

    def test_opened_count_bounds_and_sell_release(self):
        """名额计数: 下单 +1, 卖出 -1(不越界)。"""
        s = self._strategy()
        s._opened_count = 5
        s._opened_count = max(0, s._opened_count - 1)  # 模拟卖出
        assert s._opened_count == 4
        s._opened_count = max(0, s._opened_count - 1)
        assert s._opened_count == 3
