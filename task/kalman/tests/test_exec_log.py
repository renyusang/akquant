"""Tests for exec_log.py."""

import os
import tempfile

import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import exec_log


@pytest.fixture
def mock_exec_log(monkeypatch):
    """Fixture that redirects exec_log to a temp file."""
    tmp = tempfile.TemporaryDirectory()
    monkeypatch.setattr(exec_log, "EXEC_LOG",
                        os.path.join(tmp.name, "exec.csv"))
    yield
    tmp.cleanup()


class TestExecLog:
    """Tests for execution log operations."""

    def test_get_summary_empty(self, mock_exec_log):
        assert "无执行记录" in exec_log.get_summary()

    def test_log_pending(self, mock_exec_log):
        exec_log.log_pending(
            "000001", "测试", "buy", 0.95, 100,
            "价格突破", "2026-07-24",
        )
        df = exec_log._load()
        assert len(df) == 1
        assert df.iloc[0]["status"] == "pending"
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        assert df.iloc[0]["symbol"] == "000001"

    def test_log_pending_dedup(self, mock_exec_log):
        """Same symbol+date+action should deduplicate (keep latest)."""
        exec_log.log_pending(
            "000001", "测试", "buy", 0.95, 100,
            "价格突破", "2026-07-24",
        )
        exec_log.log_pending(
            "000001", "测试", "buy", 0.95, 200,
            "价格突破v2", "2026-07-24",
        )
        df = exec_log._load()
        assert len(df) == 1  # deduped
        assert int(df.iloc[0]["shares"]) == 200  # latest wins
        assert df.iloc[0]["signal_reason"] == "价格突破v2"

    def test_log_executed(self, mock_exec_log):
        exec_log.log_pending(
            "000001", "测试", "buy", 0.95, 100,
            "价格突破", "2026-07-23",
        )
        exec_log.log_executed("000001", "2026-07-23", 10.8, "2026-07-24")

        df = exec_log._load()
        assert df.iloc[0]["status"] == "executed"
        assert float(df.iloc[0]["exec_price"]) == 10.8
        assert df.iloc[0]["exec_date"] == "2026-07-24"

    def test_log_failed(self, mock_exec_log):
        exec_log.log_pending(
            "000001", "测试", "buy", 0.95, 100,
            "价格突破", "2026-07-23",
        )
        exec_log.log_failed("000001", "2026-07-23", "涨停")

        df = exec_log._load()
        assert df.iloc[0]["status"] == "failed"
        assert df.iloc[0]["reason"] == "涨停"

    def test_log_skipped(self, mock_exec_log):
        exec_log.log_skipped(
            "000001", "测试", "2026-07-24",
            "已达最大持仓数(5)", "价格突破(偏离3.2%)",
        )

        df = exec_log._load()
        assert df.iloc[0]["status"] == "skipped"
        assert df.iloc[0]["reason"] == "已达最大持仓数(5)"
        assert df.iloc[0]["signal_reason"] == "价格突破(偏离3.2%)"

    def test_get_pending_count(self, mock_exec_log):
        assert exec_log.get_pending_count() == 0

        exec_log.log_pending(
            "000001", "A", "buy", 0.95, 100, "r1", "2026-07-23",
        )
        exec_log.log_pending(
            "000002", "B", "buy", 0.95, 200, "r2", "2026-07-24",
        )
        assert exec_log.get_pending_count() == 2

        exec_log.log_executed("000001", "2026-07-23", 10.0, "2026-07-24")
        assert exec_log.get_pending_count() == 1

    def test_get_summary(self, mock_exec_log):
        exec_log.log_pending(
            "000001", "A", "buy", 0.95, 100, "r1", "2026-07-23",
        )
        exec_log.log_executed("000001", "2026-07-23", 10.0, "2026-07-24")
        exec_log.log_skipped("000002", "B", "2026-07-24", "超仓", "信号")

        summary = exec_log.get_summary()
        assert "已成交 1" in summary
        assert "已跳过 1" in summary

    def test_symbol_zfill(self, mock_exec_log):
        exec_log.log_pending(
            "1", "测试", "buy", 0.95, 100,
            "价格突破", "2026-07-24",
        )
        df = exec_log._load()
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        assert df.iloc[0]["symbol"] == "000001"
