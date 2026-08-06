"""Tests for 无限仓位系统: 归一化收益 + 首笔买入价 + 最小单位。"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unlimited_report as ur


class TestLotSize:
    def test_star_board_200(self):
        assert ur._lot_size("688041") == 200
        assert ur._lot_size("688256") == 200

    def test_others_100(self):
        assert ur._lot_size("002594") == 100
        assert ur._lot_size("300308") == 100
        assert ur._lot_size("512500") == 100


class TestFirstBuyPrice:
    def test_from_elog_executed(self):
        elog = pd.DataFrame([
            {"symbol": "002594", "action": "buy", "status": "pending",
             "exec_price": ""},
            {"symbol": "002594", "action": "buy", "status": "executed",
             "exec_price": 95.2},
            {"symbol": "002594", "action": "buy", "status": "executed",
             "exec_price": 100.5},
        ])
        assert ur._first_buy_price("002594", elog) == 95.2  # 首笔

    def test_no_executed_returns_zero(self):
        elog = pd.DataFrame([
            {"symbol": "002594", "action": "buy", "status": "pending",
             "exec_price": ""},
        ])
        assert ur._first_buy_price("002594", elog) == 0.0


class TestNormalizedPnl:
    """归一化收益 = 总利润 / (3 × 最小单位 × 首笔买入价)。"""

    def _elog(self):
        return pd.DataFrame([
            {"symbol": "002594", "action": "buy", "status": "executed",
             "exec_price": 100.0},
        ])

    def test_held_position_floating(self):
        """持仓中: 总利润 = 浮动盈亏。"""
        positions = {"002594": {"name": "比亚迪", "shares": 300,
                                "avg_cost": 100.0}}
        trades = pd.DataFrame(columns=["symbol", "pnl"])
        elog = self._elog()
        prices = {"002594": 110.0}
        total, norm, realized, floating = ur._normalized_pnl(
            "002594", positions, trades, elog, prices)
        # 总资金 = 3×100×100 = 30000; 浮动 = (110-100)×300 = 3000
        assert total == pytest.approx(3000.0)
        assert norm == pytest.approx(0.10)
        assert realized == pytest.approx(0.0)
        assert floating == pytest.approx(3000.0)

    def test_realized_plus_floating(self):
        """已清仓(已实现) + 无持仓。"""
        positions = {}
        trades = pd.DataFrame([
            {"symbol": "002594", "pnl": -1500.0},
            {"symbol": "002594", "pnl": 500.0},
        ])
        elog = self._elog()
        total, norm, realized, floating = ur._normalized_pnl(
            "002594", positions, trades, elog, {})
        # 总利润 = -1500+500 = -1000; 总资金 30000 → -3.33%
        assert total == pytest.approx(-1000.0)
        assert norm == pytest.approx(-1000.0 / 30000.0)
        assert realized == pytest.approx(-1000.0)

    def test_no_first_buy_price_norm_zero(self):
        """无成交记录: 总资金为 0, 归一化收益为 0。"""
        positions = {}
        trades = pd.DataFrame(columns=["symbol", "pnl"])
        elog = pd.DataFrame(columns=["symbol", "action", "status",
                                     "exec_price"])
        total, norm, realized, floating = ur._normalized_pnl(
            "002594", positions, trades, elog, {})
        assert total == 0.0
        assert norm == 0.0

    def test_star_board_capital_uses_lot200(self):
        """科创板: 总资金 = 3 × 200 × 首笔价。"""
        positions = {"688041": {"name": "海光信息", "shares": 600,
                                "avg_cost": 100.0}}
        trades = pd.DataFrame(columns=["symbol", "pnl"])
        elog = pd.DataFrame([
            {"symbol": "688041", "action": "buy", "status": "executed",
             "exec_price": 100.0},
        ])
        prices = {"688041": 110.0}
        total, norm, _, _ = ur._normalized_pnl(
            "688041", positions, trades, elog, prices)
        # 总资金 = 3×200×100 = 60000; 浮动 = 10×600 = 6000 → 10%
        assert total == pytest.approx(6000.0)
        assert norm == pytest.approx(0.10)
