"""Tests for validate.py 涨跌停误报修复 + state_check.py 卖出成交误报修复。

背景(2026-08-06):
1. 宏和科技主板涨停恰 10.0%(125.55→138.11, 取整后 +10.004%) 被误报为
   "涨跌幅异常"——修复: 收盘价 ≤ 涨停价(+1分容差) 时放行
2. 宁德时代经订单执行卖出成交后, state_check 误报"持仓消失但无卖出信号"
   ——修复: execution_log 的 executed sell 记录放行
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import state_check
import validate


def _write_cache(monkeypatch, tmp_path, closes, volume=None, symbol="603256"):
    """写缓存 parquet 并指向临时目录。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    n = len(closes)
    dates = pd.date_range("2026-06-01", periods=n, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": volume if volume is not None else [1000] * n,
        "symbol": [symbol] * n,
    })
    df.to_parquet(cache_dir / f"{symbol}.parquet")
    monkeypatch.setattr(validate, "CACHE_DIR", str(cache_dir))


class TestLimitUpFalsePositive:
    """涨跌停恰在限制位不应报错。"""

    def test_main_board_limit_up_passes(self, monkeypatch, tmp_path):
        """主板: 前收125.55 → 涨停 138.11(取整后 +10.004%) → 放行。"""
        _write_cache(monkeypatch, tmp_path, [125.55, 138.11])
        issues = validate.validate_data("603256", "宏和科技")
        assert all(i["check"] != "涨跌幅异常" for i in issues)

    def test_real_anomaly_still_reported(self, monkeypatch, tmp_path):
        """主板: +15% 远超涨停价 → 仍报错。"""
        _write_cache(monkeypatch, tmp_path, [125.55, 144.38])  # +15.0%
        issues = validate.validate_data("603256", "宏和科技")
        assert any(i["check"] == "涨跌幅异常" for i in issues)

    def test_gem_limit_up_passes(self, monkeypatch, tmp_path):
        """创业板: +20% 涨停放行。"""
        _write_cache(monkeypatch, tmp_path, [100.0, 120.0], symbol="300308")
        issues = validate.validate_data("300308", "中际旭创")
        assert all(i["check"] != "涨跌幅异常" for i in issues)

    def test_star_board_limit_down_passes(self, monkeypatch, tmp_path):
        """科创板: -20% 跌停放行。"""
        _write_cache(monkeypatch, tmp_path, [100.0, 80.0], symbol="688256")
        issues = validate.validate_data("688256", "寒武纪")
        assert all(i["check"] != "涨跌幅异常" for i in issues)


class TestExecutedSellFalsePositive:
    """订单执行卖出成交后 state_check 不应误报。"""

    def _setup(self, monkeypatch, tmp_path, prev_positions, prev_pending,
               curr_positions, curr_pending, elog_rows):
        snap = tmp_path / "state_snapshot.json"
        snap.write_text(json.dumps({
            "positions": prev_positions, "pending": prev_pending,
            "signals": {},
        }), encoding="utf-8")
        monkeypatch.setattr(state_check, "SNAPSHOT_FILE", str(snap))
        monkeypatch.setattr(state_check, "TASK_DIR", str(tmp_path))

        import orders
        import portfolio
        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps(curr_positions), encoding="utf-8")
        pend_file = tmp_path / "pending_orders.json"
        pend_file.write_text(json.dumps(curr_pending), encoding="utf-8")
        monkeypatch.setattr(portfolio, "POSITIONS_FILE", str(pos_file))
        monkeypatch.setattr(portfolio, "TRADES_FILE",
                            str(tmp_path / "trades.csv"))
        monkeypatch.setattr(orders, "PENDING_FILE", str(pend_file))

        if elog_rows:
            pd.DataFrame(elog_rows).to_csv(
                tmp_path / "execution_log.csv", index=False,
                encoding="utf-8-sig")

    def test_executed_sell_no_warning(self, monkeypatch, tmp_path):
        """宁德时代卖出已成交(execution_log executed) → 不误报。"""
        self._setup(
            monkeypatch, tmp_path,
            prev_positions={"300750": {"shares": 100, "avg_cost": 372.67}},
            prev_pending=[{"symbol": "300750", "signal_date": "2026-08-04",
                           "shares": 100}],
            curr_positions={},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-04", "symbol": "300750",
                        "action": "sell", "status": "executed",
                        "exec_price": 400.02}],
        )
        warnings = state_check.check_consistency()
        assert not any("300750" in w for w in warnings)

    def test_no_executed_record_still_warns(self, monkeypatch, tmp_path):
        """无任何卖出记录 → 仍警告(保留原检测能力)。"""
        self._setup(
            monkeypatch, tmp_path,
            prev_positions={"300750": {"shares": 100, "avg_cost": 372.67}},
            prev_pending=[],
            curr_positions={},
            curr_pending=[],
            elog_rows=[],
        )
        warnings = state_check.check_consistency()
        assert any("300750" in w and "持仓消失" in w for w in warnings)
