"""Tests for live_report.py 权益曲线/月度收益。

背景(2026-08-13):
1. 权益曲线历史点原用"当前价"估算持仓市值——7 月等历史月份收益
   随每日价格漂移(7月 +0.4% 隔日变 -0.2%)。
   修复: 历史点用当日收盘价(price_history), 缺失回退当前价。
"""

import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import live_report


def _setup_env(monkeypatch, tmp_path, elog_rows):
    """指向临时目录并写入 execution_log.csv。"""
    monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
    if elog_rows:
        pd.DataFrame(elog_rows).to_csv(
            tmp_path / "execution_log.csv", index=False,
            encoding="utf-8-sig")


class TestEquityCurvePriceHistory:
    """历史点用当日收盘价, 不随当前价漂移。"""

    def test_historical_point_uses_day_close(self, monkeypatch, tmp_path):
        """7-20 买入成交: 当日收盘 10 元 vs 当前价 15 元 → 权益按 10 元算。"""
        _setup_env(monkeypatch, tmp_path, [{
            "signal_date": "2026-07-17", "exec_date": "2026-07-20",
            "symbol": "000001", "name": "测试", "action": "buy",
            "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
            "exec_price": 9.0, "status": "executed", "reason": "",
        }])
        trades = pd.DataFrame(columns=["exit_date", "symbol", "shares",
                                       "exit_price"])
        eq = live_report._build_equity_curve(
            trades, {}, {"000001": 15.0}, 100000.0,
            {"000001": {"2026-07-20": 10.0}})
        # 现金 91000 + 1000股×当日收盘10 = 101,000
        day20 = eq[eq["date"] == "2026-07-20"]["equity"].iloc[-1]
        assert day20 == pytest.approx(101000.0)
        # 若误用当前价会是 106,000
        assert day20 != pytest.approx(106000.0)

    def test_missing_history_falls_back_current(self, monkeypatch, tmp_path):
        """price_history 缺该日 → 回退当前价。"""
        _setup_env(monkeypatch, tmp_path, [{
            "signal_date": "2026-07-17", "exec_date": "2026-07-20",
            "symbol": "000001", "name": "测试", "action": "buy",
            "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
            "exec_price": 9.0, "status": "executed", "reason": "",
        }])
        trades = pd.DataFrame(columns=["exit_date", "symbol", "shares",
                                       "exit_price"])
        eq = live_report._build_equity_curve(
            trades, {}, {"000001": 15.0}, 100000.0, {})  # 无历史
        day20 = eq[eq["date"] == "2026-07-20"]["equity"].iloc[-1]
        assert day20 == pytest.approx(106000.0)  # 91000 + 1000×15

    def test_price_change_does_not_drift_history(self, monkeypatch, tmp_path):
        """当前价变化不影响历史权益点(修复核心)。"""
        _setup_env(monkeypatch, tmp_path, [{
            "signal_date": "2026-07-17", "exec_date": "2026-07-20",
            "symbol": "000001", "name": "测试", "action": "buy",
            "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
            "exec_price": 9.0, "status": "executed", "reason": "",
        }])
        trades = pd.DataFrame(columns=["exit_date", "symbol", "shares",
                                       "exit_price"])
        ph = {"000001": {"2026-07-20": 10.0}}
        eq1 = live_report._build_equity_curve(
            trades, {}, {"000001": 15.0}, 100000.0, ph)
        eq2 = live_report._build_equity_curve(
            trades, {}, {"000001": 30.0}, 100000.0, ph)  # 现价翻倍
        v1 = eq1[eq1["date"] == "2026-07-20"]["equity"].iloc[0]
        v2 = eq2[eq2["date"] == "2026-07-20"]["equity"].iloc[0]
        assert v1 == pytest.approx(v2)


class TestMonthlyReturns:
    """月度收益: 首月以初始资金为基准。"""

    def test_first_month_baseline_is_initial_cash(self):
        """7 月末权益 102,000 / 初始 100,000 → +2.0%。"""
        eq = pd.Series([100000.0, 102000.0],
                       index=pd.to_datetime(["2026-07-20", "2026-07-31"]))
        m = live_report._monthly_returns(eq, 100000.0)
        assert m.loc["2026-07-31", "pct"] == pytest.approx(2.0)
        assert m.loc["2026-07-31", "amount"] == pytest.approx(2000.0)
