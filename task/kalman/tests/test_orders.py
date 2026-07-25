"""Tests for orders.py."""

import os
import json
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orders


class TestPendingOrders:
    """Tests for pending order CRUD operations."""

    def test_load_empty(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))
            assert orders.load_pending() == []

    def test_add_order(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-01-01", 0.95)
            pending = orders.load_pending()
            assert len(pending) == 1
            assert pending[0]["symbol"] == "000001"
            assert pending[0]["action"] == "buy"
            assert pending[0]["shares"] == 100

    def test_add_duplicate_same_action(self, monkeypatch):
        """Same symbol + same action should NOT overwrite (keep earliest signal)."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-01-01", 0.95)
            orders.add_pending_order("000001", "测试", "buy", 200, 11.0,
                                     "2026-01-02", 0.95)
            pending = orders.load_pending()
            assert len(pending) == 1
            # Should keep the first signal date
            assert pending[0]["signal_date"] == "2026-01-01"
            assert pending[0]["shares"] == 100

    def test_add_opposite_action_replaces(self, monkeypatch):
        """Buy then sell on same symbol should replace (sell overrides buy)."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-01-01", 0.95)
            orders.add_pending_order("000001", "测试", "sell", 100, 12.0,
                                     "2026-01-02", 0.0)
            pending = orders.load_pending()
            assert len(pending) == 1
            assert pending[0]["action"] == "sell"

    def test_remove_order(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-01-01", 0.95)
            removed = orders.remove_pending("000001")
            assert removed["symbol"] == "000001"
            assert orders.load_pending() == []

    def test_remove_nonexistent(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))
            assert orders.remove_pending("999999") == {}

    def test_symbol_zfill(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("1", "测试", "buy", 100, 10.5,
                                     "2026-01-01", 0.95)
            pending = orders.load_pending()
            assert pending[0]["symbol"] == "000001"


class TestExecutePending:
    """Tests for order execution with limit checks."""

    def test_execute_tomorrow(self, monkeypatch):
        """Orders with signal_date == today should be deferred to tomorrow."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-07-24", 0.95)
            price_map = {"000001": 10.8}
            prev_close_map = {"000001": 10.5}
            executed = orders.execute_pending_orders(
                price_map, prev_close_map, today_str="2026-07-24"
            )
            # signal_date >= today → deferred, not executed
            assert len(executed) == 0

    def test_execute_yesterday_order(self, monkeypatch):
        """Orders from yesterday should execute today."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-07-23", 0.95)
            price_map = {"000001": 10.8}
            prev_close_map = {"000001": 10.5}
            executed = orders.execute_pending_orders(
                price_map, prev_close_map, today_str="2026-07-24"
            )
            assert len(executed) == 1
            assert executed[0]["exec_price"] == 10.8
            assert executed[0]["exec_date"] == "2026-07-24"

    def test_limit_up_blocked(self, monkeypatch):
        """Buy order should be blocked when price hits limit up."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("600001", "测试", "buy", 100, 10.0,
                                     "2026-07-23", 0.95)
            # prev_close = 10.0, limit_up = 11.0 (10% for main board)
            price_map = {"600001": 11.0}  # exactly at limit up
            prev_close_map = {"600001": 10.0}
            executed = orders.execute_pending_orders(
                price_map, prev_close_map, today_str="2026-07-24"
            )
            assert len(executed) == 0  # blocked by limit up

    def test_limit_up_not_quite(self, monkeypatch):
        """Buy order should execute if just below limit up."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("600001", "测试", "buy", 100, 10.0,
                                     "2026-07-23", 0.95)
            # limit_up = 11.0, 10.99 < 11.0 * 0.999 → executes
            price_map = {"600001": 10.98}
            prev_close_map = {"600001": 10.0}
            executed = orders.execute_pending_orders(
                price_map, prev_close_map, today_str="2026-07-24"
            )
            assert len(executed) == 1

    def test_no_open_price(self, monkeypatch):
        """Order should fail if no open price data."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "PENDING_FILE",
                                os.path.join(tmp, "pending.json"))

            orders.add_pending_order("000001", "测试", "buy", 100, 10.5,
                                     "2026-07-23", 0.95)
            executed = orders.execute_pending_orders(
                {}, {}, today_str="2026-07-24"
            )
            assert len(executed) == 0


class TestActualPrices:
    """Tests for actual_prices parameter (real fill price override)."""

    def _setup_pending(self, monkeypatch, tmp):
        monkeypatch.setattr(orders, "PENDING_FILE", os.path.join(tmp, "pending.json"))
        monkeypatch.setattr(orders, "ACTUAL_FILLS_FILE", os.path.join(tmp, "actual.json"))
        orders.save_pending([{
            "symbol": "000001", "name": "测试", "action": "buy",
            "shares": 100, "signal_price": 10.0, "signal_date": "2026-01-01",
            "target_pct": 0.95,
        }])

    def test_actual_price_overrides_open(self, monkeypatch):
        """actual_prices 应覆盖 open 假设,用真实成交价。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._setup_pending(monkeypatch, tmp)
            executed = orders.execute_pending_orders(
                price_map={"000001": 10.5},
                previous_close_map={"000001": 10.0},
                today_str="2026-01-02",
                actual_prices={"000001": 10.3},
            )
            assert len(executed) == 1
            assert executed[0]["exec_price"] == 10.3  # 实际价,非 open(10.5)

    def test_no_actual_price_uses_open(self, monkeypatch):
        """无 actual_prices 时用 open(向后兼容)。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._setup_pending(monkeypatch, tmp)
            executed = orders.execute_pending_orders(
                price_map={"000001": 10.5},
                previous_close_map={"000001": 10.0},
                today_str="2026-01-02",
            )
            assert len(executed) == 1
            assert executed[0]["exec_price"] == 10.5  # open

    def test_actual_price_skips_limit_check(self, monkeypatch):
        """actual_prices 提供时跳过涨跌停检查(人工已实际成交)。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._setup_pending(monkeypatch, tmp)
            # open=11.5 涨停(prev_close=10.0, limit_up=11.0),无 actual 应跳过
            executed_no_actual = orders.execute_pending_orders(
                price_map={"000001": 11.5},
                previous_close_map={"000001": 10.0},
                today_str="2026-01-02",
            )
            assert len(executed_no_actual) == 0  # 涨停跳过
            # 重新 setup(上次 save_pending 清空了)
            self._setup_pending(monkeypatch, tmp)
            executed_with_actual = orders.execute_pending_orders(
                price_map={"000001": 11.5},
                previous_close_map={"000001": 10.0},
                today_str="2026-01-02",
                actual_prices={"000001": 11.5},  # 人工实际成交
            )
            assert len(executed_with_actual) == 1  # actual 跳过涨跌停,执行
            assert executed_with_actual[0]["exec_price"] == 11.5


class TestActualFills:
    """Tests for actual_fills.json 读写(add_actual_fill/load/save)。"""

    def test_add_and_load_actual_fill(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "ACTUAL_FILLS_FILE", os.path.join(tmp, "actual.json"))
            orders.add_actual_fill("002594", "buy", 800, 95.20, "2026-07-24")
            fills = orders.load_actual_fills()
            assert "002594" in fills
            assert fills["002594"]["exec_price"] == 95.2
            assert fills["002594"]["action"] == "buy"

    def test_load_actual_fills_missing(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(orders, "ACTUAL_FILLS_FILE", os.path.join(tmp, "actual.json"))
            assert orders.load_actual_fills() == {}

    def test_load_actual_fills_corrupt(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            fpath = os.path.join(tmp, "actual.json")
            monkeypatch.setattr(orders, "ACTUAL_FILLS_FILE", fpath)
            with open(fpath, "w") as f:
                f.write("{corrupt json")
            assert orders.load_actual_fills() == {}  # 损坏返回空
