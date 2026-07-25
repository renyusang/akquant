"""Tests for portfolio.py."""

import os
import json
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import portfolio


class TestPositions:
    """Tests for position CRUD operations."""

    def test_load_empty(self, monkeypatch):
        """load_positions should return empty dict when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(portfolio, "POSITIONS_FILE",
                                os.path.join(tmp, "positions.json"))
            assert portfolio.load_positions() == {}

    def test_save_and_load(self, monkeypatch):
        """Save then load should return the same data."""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)

            data = {
                "000001": {"name": "测试", "shares": 100, "avg_cost": 10.5,
                           "first_buy_date": "2026-01-01"},
            }
            portfolio.save_positions(data)
            loaded = portfolio.load_positions()
            assert loaded == data

    def test_add_new_position(self, monkeypatch):
        """Adding a new position should create an entry."""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)

            portfolio.add_position("000001", "测试", 100, 10.5, "2026-01-01")
            pos = portfolio.get_position("000001")
            assert pos is not None
            assert pos["name"] == "测试"
            assert pos["shares"] == 100
            assert pos["avg_cost"] == 10.5
            assert pos["first_buy_date"] == "2026-01-01"

    def test_add_position_merges(self, monkeypatch):
        """Adding same symbol again should merge with weighted avg cost."""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)

            # First buy: 100 shares @ 10
            portfolio.add_position("000001", "测试", 100, 10.0, "2026-01-01")
            # Second buy: 200 shares @ 13
            portfolio.add_position("000001", "测试", 200, 13.0, "2026-01-02")

            pos = portfolio.get_position("000001")
            assert pos["shares"] == 300
            # Weighted avg: (100*10 + 200*13) / 300 = 3600/300 = 12
            assert pos["avg_cost"] == pytest.approx(12.0, abs=0.01)
            # first_buy_date stays as the original
            assert pos["first_buy_date"] == "2026-01-01"

    def test_remove_position(self, monkeypatch):
        """Removing a position should delete it and return the record."""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)

            portfolio.add_position("000001", "测试", 100, 10.0, "2026-01-01")
            removed = portfolio.remove_position("000001")
            assert removed["name"] == "测试"
            assert removed["shares"] == 100

            # Should be gone
            assert portfolio.get_position("000001") is None

    def test_remove_nonexistent(self, monkeypatch):
        """Removing a non-existent position should return empty dict."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(portfolio, "POSITIONS_FILE",
                                os.path.join(tmp, "positions.json"))
            assert portfolio.remove_position("999999") == {}

    def test_has_position(self, monkeypatch):
        """has_position should return True/False correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)

            assert not portfolio.has_position("000001")
            portfolio.add_position("000001", "测试", 100, 10.0, "2026-01-01")
            assert portfolio.has_position("000001")

    def test_symbol_zfill_normalization(self, monkeypatch):
        """Symbol codes should be normalized to 6 digits."""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)

            portfolio.add_position("1", "测试", 100, 10.0, "2026-01-01")
            pos = portfolio.get_position("000001")
            assert pos is not None
            assert pos["shares"] == 100


class TestTrades:
    """Tests for trade record operations."""

    def test_load_empty(self, monkeypatch):
        """load_trades should return empty DataFrame when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(portfolio, "TRADES_FILE",
                                os.path.join(tmp, "trades.csv"))
            trades = portfolio.load_trades()
            assert len(trades) == 0

    def test_record_trade(self, monkeypatch):
        """Recording a trade should append to trades.csv."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(portfolio, "TRADES_FILE",
                                os.path.join(tmp, "trades.csv"))

            trade = portfolio.record_trade(
                symbol="000001", name="测试",
                shares=100, entry_price=10.0, exit_price=12.0,
                entry_date="2026-01-01", exit_date="2026-02-01",
                reason="价格回归",
            )
            assert trade["pnl"] == pytest.approx(200.0)  # (12-10)*100
            assert trade["pnl_pct"] == pytest.approx(20.0)
            assert trade["symbol"] == "000001"

            trades = portfolio.load_trades()
            assert len(trades) == 1
            assert str(trades.iloc[0]["symbol"]).zfill(6) == "000001"

    def test_multiple_trades(self, monkeypatch):
        """Multiple trades should all be recorded."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(portfolio, "TRADES_FILE",
                                os.path.join(tmp, "trades.csv"))

            portfolio.record_trade("000001", "A", 100, 10, 12, "2026-01-01", "2026-02-01")
            portfolio.record_trade("000002", "B", 200, 20, 18, "2026-01-05", "2026-02-05")

            trades = portfolio.load_trades()
            assert len(trades) == 2

    def test_trade_summary(self, monkeypatch):
        """get_trade_summary should return correct aggregates."""
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(portfolio, "TRADES_FILE",
                                os.path.join(tmp, "trades.csv"))

            portfolio.record_trade("000001", "A", 100, 10, 12, "2026-01-01", "2026-02-01")  # +200
            portfolio.record_trade("000002", "B", 100, 20, 18, "2026-01-05", "2026-02-05")  # -200

            summary = portfolio.get_trade_summary()
            assert summary["count"] == 2
            assert summary["wins"] == 1
            assert summary["losses"] == 1
            assert summary["total_pnl"] == pytest.approx(0.0)


class TestAtomicWriteAndValidation:
    """Tests for atomic write and load validation(备份增强)。"""

    def test_save_atomic_no_tmp_left(self, monkeypatch):
        """原子写入后无临时文件残留。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)
            portfolio.save_positions({"000001": {"shares": 100, "name": "A",
                                                 "avg_cost": 10.0, "first_buy_date": "2026-01-01"}})
            assert os.path.exists(pos_file)
            assert not [f for f in os.listdir(tmp) if f.endswith(".tmp")]

    def test_load_corrupt_json_warns(self, monkeypatch, capsys):
        """损坏 JSON 告警 + 返回空(不静默丢失)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)
            with open(pos_file, "w") as f:
                f.write("{corrupt")
            result = portfolio.load_positions()
            assert result == {}
            captured = capsys.readouterr().out
            assert "读取失败" in captured or "结构异常" in captured

    def test_load_wrong_structure_warns(self, monkeypatch, capsys):
        """非字典结构告警。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos_file = os.path.join(tmp, "positions.json")
            monkeypatch.setattr(portfolio, "POSITIONS_FILE", pos_file)
            with open(pos_file, "w") as f:
                json.dump([1, 2, 3], f)  # 列表,非字典
            result = portfolio.load_positions()
            assert result == {}
            assert "结构异常" in capsys.readouterr().out

    def test_save_trades_atomic(self, monkeypatch):
        """save_trades 原子写入,无临时文件残留。"""
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            trades_file = os.path.join(tmp, "trades.csv")
            monkeypatch.setattr(portfolio, "TRADES_FILE", trades_file)
            df = pd.DataFrame([{"symbol": "000001", "pnl": 100}])
            portfolio.save_trades(df)
            assert os.path.exists(trades_file)
            assert not [f for f in os.listdir(tmp) if f.endswith(".tmp")]
