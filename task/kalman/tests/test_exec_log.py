"""Tests for exec_log.py."""

import json
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


class TestLogExecutedAppend:
    """log_executed 无匹配记录时追加(2026-08-06 修复)。"""

    def test_append_when_no_pending_record(self, tmp_path, monkeypatch):
        """订单被回滚丢失日志后, 成交应追加 executed 记录。"""
        import exec_log
        monkeypatch.setattr(exec_log, "EXEC_LOG", str(tmp_path / "exec_log.csv"))
        # 只有一条 skipped 记录(长电 8-6 场景)
        pd.DataFrame([{
            "signal_date": "2026-08-05", "exec_date": "",
            "symbol": "600584", "name": "长电科技", "action": "buy",
            "target_pct": 0.06, "shares": 100, "signal_reason": "价格突破",
            "exec_price": "", "status": "skipped", "reason": "已达最大持仓数",
        }]).to_csv(tmp_path / "exec_log.csv", index=False, encoding="utf-8-sig")

        exec_log.log_executed("600584", "2026-08-05", 67.01, "2026-08-06")
        df = exec_log._load()
        ex = df[(df["symbol"] == "600584") & (df["status"] == "executed")]
        assert len(ex) == 1
        assert ex.iloc[0]["exec_price"] == 67.01
        assert ex.iloc[0]["shares"] == 100  # 从 skipped 记录复制

    def test_update_existing_pending_unchanged(self, tmp_path, monkeypatch):
        """有 pending 记录时仍走更新路径(不追加)。"""
        import exec_log
        monkeypatch.setattr(exec_log, "EXEC_LOG", str(tmp_path / "exec_log.csv"))
        pd.DataFrame([{
            "signal_date": "2026-08-05", "exec_date": "",
            "symbol": "600584", "name": "长电科技", "action": "buy",
            "target_pct": 0.06, "shares": 100, "signal_reason": "",
            "exec_price": "", "status": "pending", "reason": "",
        }]).to_csv(tmp_path / "exec_log.csv", index=False, encoding="utf-8-sig")

        exec_log.log_executed("600584", "2026-08-05", 67.01, "2026-08-06")
        df = exec_log._load()
        assert len(df[df["status"] == "executed"]) == 1
        assert len(df[df["status"] == "pending"]) == 0


class TestCleanupStalePending:
    """cleanup_stale_pending: 以 pending_orders.json 为准清理陈旧 pending(2026-08-28)。

    背景: 历史遗留 pending 行(已成交未更新状态/测试污染/拦截误标)虚高
    "待执行" 计数(实盘曾显示 27 而实际队列为空)。日志非权威,
    实际待执行以 pending_orders.json 为准。
    """

    def _setup(self, tmp_path, monkeypatch):
        import exec_log
        monkeypatch.setattr(exec_log, "EXEC_LOG", str(tmp_path / "exec_log.csv"))
        import orders
        monkeypatch.setattr(orders, "PENDING_FILE", str(tmp_path / "pending.json"))
        return exec_log

    def test_removes_stale_keeps_current(self, tmp_path, monkeypatch):
        """匹配当前订单的 pending 保留, 历史遗留/测试污染行删除。"""
        exec_log = self._setup(tmp_path, monkeypatch)
        json.dump([{"symbol": "002916", "name": "深南电路", "action": "buy",
                    "shares": 300, "signal_price": 362.4,
                    "signal_date": "2026-08-28", "target_pct": 0.19}],
                  open(tmp_path / "pending.json", "w"))
        pd.DataFrame([
            # 匹配当前待执行订单 → 保留
            {"signal_date": "2026-08-28", "exec_date": "", "symbol": "002916",
             "name": "深南电路", "action": "buy", "target_pct": 0.19, "shares": 300,
             "signal_reason": "速度反转(转多)", "exec_price": "", "status": "pending",
             "reason": ""},
            # 历史遗留(订单早已成交, 状态未更新) → 删除
            {"signal_date": "2026-07-24", "exec_date": "", "symbol": "688041",
             "name": "海光信息", "action": "sell", "target_pct": 0.0, "shares": 100,
             "signal_reason": "止损(-7.0%)", "exec_price": "", "status": "pending",
             "reason": "止损(-7.0%)"},
            # 测试污染行(空日期, reason=x) → 删除
            {"signal_date": "", "exec_date": "", "symbol": "000002",
             "name": "000002", "action": "buy", "target_pct": 0.19, "shares": 100,
             "signal_reason": "x", "exec_price": "", "status": "pending",
             "reason": ""},
        ]).to_csv(tmp_path / "exec_log.csv", index=False, encoding="utf-8-sig")

        assert exec_log.cleanup_stale_pending() == 2
        df = exec_log._load()
        pending = df[df["status"] == "pending"]
        assert len(pending) == 1
        assert pending.iloc[0]["symbol"] == "002916"

    def test_no_pending_rows_returns_zero(self, tmp_path, monkeypatch):
        """无 pending 行 → 返回 0, 文件不变。"""
        exec_log = self._setup(tmp_path, monkeypatch)
        open(tmp_path / "pending.json", "w").write("[]")
        pd.DataFrame([{
            "signal_date": "2026-08-05", "exec_date": "2026-08-06",
            "symbol": "600584", "name": "长电科技", "action": "buy",
            "target_pct": 0.06, "shares": 100, "signal_reason": "",
            "exec_price": 67.01, "status": "executed", "reason": "",
        }]).to_csv(tmp_path / "exec_log.csv", index=False, encoding="utf-8-sig")

        assert exec_log.cleanup_stale_pending() == 0
        assert len(exec_log._load()) == 1


class TestSummaryFailedWindow:
    """2026-09-29 修复: `未成交` 只统计近期失败行。

    症状: execution_log 的 failed 行是永久记录, 2026-07-27 长江电力那笔让摘要
    长期显示"未成交 1", 与"当次运行情况"无关地失真。
    """

    def _add_failed(self, symbol: str, signal_date: str) -> None:
        exec_log.log_pending(symbol, "测试", "sell", 0.0, 100, "价格回归", signal_date)
        exec_log.log_failed(symbol, signal_date, "无开盘价数据")

    def test_old_failure_not_counted(self, mock_exec_log):
        old = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        recent = (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        self._add_failed("600900", old)
        self._add_failed("002460", recent)
        s = exec_log.get_summary()
        assert f"未成交(近{exec_log.FAILED_WINDOW_DAYS}天) 1" in s

    def test_only_old_failures_shows_zero(self, mock_exec_log):
        old = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        self._add_failed("600900", old)
        assert f"未成交(近{exec_log.FAILED_WINDOW_DAYS}天) 0" in exec_log.get_summary()

    def test_window_is_configurable(self, mock_exec_log):
        d = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        self._add_failed("600900", d)
        assert "未成交(近365天) 1" in exec_log.get_summary(window_days=365)

    def test_summary_keeps_executed_and_skipped(self, mock_exec_log):
        """既有断言不回归: 已成交/已跳过 仍按全量统计。"""
        exec_log.log_pending("000001", "A", "buy", 0.95, 100, "r1", "2026-07-23")
        exec_log.log_executed("000001", "2026-07-23", 10.0, "2026-07-24")
        exec_log.log_skipped("000002", "B", "2026-07-24", "超仓", "信号")
        s = exec_log.get_summary()
        assert "已成交 1" in s
        assert "已跳过 1" in s


class TestCleanupWiredIntoDailyFlow:
    """cleanup_stale_pending 早在 2026-08-28 就有, 但从未被每日流程调用
    (残留 pending 行因此持续虚高"待执行"); 2026-09-29 接线并加护栏防再次脱钩。"""

    def test_daily_main_calls_cleanup(self):
        import inspect

        import daily_signal
        src = inspect.getsource(daily_signal.main)
        assert "cleanup_stale_pending()" in src, (
            "daily_signal.main 必须调用 exec_log.cleanup_stale_pending(), "
            "否则 execution_log 的残留 pending 行会再次累积")

    def test_cleanup_keeps_live_pending_drops_orphan(self, mock_exec_log, monkeypatch):
        """有 pending_orders.json 对应订单 → 保留; 孤儿行 → 清除。"""
        exec_log.log_pending("000001", "A", "buy", 0.95, 100, "r1", "2026-09-28")
        exec_log.log_pending("002460", "赣锋锂业", "sell", 0.0, 1000, "价格回归",
                             "2026-09-01")           # 孤儿(队列里没有)
        monkeypatch.setattr("orders.load_pending", lambda: [
            {"symbol": "000001", "signal_date": "2026-09-28", "action": "buy"},
        ])
        assert exec_log.cleanup_stale_pending() == 1
        df = exec_log._load()
        assert list(df["symbol"]) == ["000001"]
