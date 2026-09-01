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


def _setup_state_check(monkeypatch, tmp_path, prev_positions, prev_pending,
                   curr_positions, curr_pending, elog_rows,
                   snapshot_ts="2026-08-10 22:00:00",
                   prev_signals=None, signals_rows=None):
    """构造 state_check 检查环境: 快照 + 持仓 + 待执行 + 执行日志。"""
    snap = tmp_path / "state_snapshot.json"
    snap.write_text(json.dumps({
        "timestamp": snapshot_ts,
        "positions": prev_positions, "pending": prev_pending,
        "signals": prev_signals or {},
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

    if signals_rows:
        pd.DataFrame(signals_rows).to_csv(
            tmp_path / "signals.csv", index=False,
            encoding="utf-8-sig")



class TestExecutedSellFalsePositive:
    """订单执行卖出成交后 state_check 不应误报。"""

    def test_executed_sell_no_warning(self, monkeypatch, tmp_path):
        """宁德时代卖出已成交(execution_log executed) → 不误报。"""
        _setup_state_check(
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
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"300750": {"shares": 100, "avg_cost": 372.67}},
            prev_pending=[],
            curr_positions={},
            curr_pending=[],
            elog_rows=[],
        )
        warnings = state_check.check_consistency()
        assert any("300750" in w and "持仓消失" in w for w in warnings)


class TestExecutedBuyFalsePositive:
    """订单执行买入成交导致股数增加时 state_check 不应误报(2026-08-11 修复)。

    修复背景: 昨日信号今日开盘成交(如赣锋锂业 200→700), 检查时 signals.csv
    最新信号为 buy 或已变 hold(满仓), 原逻辑对股数变化仅豁免卖出成交,
    买入成交一律误报"无对应信号"。
    """

    def test_executed_buy_no_warning(self, monkeypatch, tmp_path):
        """窗口内买入成交 500 股, 持仓 200→700 → 不误报。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"002460": {"shares": 200, "avg_cost": 50.0}},
            prev_pending=[],
            curr_positions={"002460": {"shares": 700, "avg_cost": 52.97}},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-10", "exec_date": "2026-08-11",
                        "symbol": "002460", "action": "buy",
                        "status": "executed", "shares": 500,
                        "exec_price": 54.14}],
        )
        warnings = state_check.check_consistency()
        assert not any("002460" in w for w in warnings)

    def test_executed_buy_share_mismatch_warns(self, monkeypatch, tmp_path):
        """买入 400 股却多出 500 股 → 净变化不吻合, 仍警告(防漏报)。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"002460": {"shares": 200, "avg_cost": 50.0}},
            prev_pending=[],
            curr_positions={"002460": {"shares": 700, "avg_cost": 52.97}},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-10", "exec_date": "2026-08-11",
                        "symbol": "002460", "action": "buy",
                        "status": "executed", "shares": 400,
                        "exec_price": 54.14}],
        )
        warnings = state_check.check_consistency()
        assert any("002460" in w and "股数变化" in w for w in warnings)

    def test_outside_window_execution_warns(self, monkeypatch, tmp_path):
        """成交在快照之前(exec_date < prev timestamp) → 不豁免, 仍警告。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"002460": {"shares": 200, "avg_cost": 50.0}},
            prev_pending=[],
            curr_positions={"002460": {"shares": 700, "avg_cost": 52.97}},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-04", "exec_date": "2026-08-05",
                        "symbol": "002460", "action": "buy",
                        "status": "executed", "shares": 500,
                        "exec_price": 50.1}],
        )
        warnings = state_check.check_consistency()
        assert any("002460" in w and "股数变化" in w for w in warnings)

    def test_partial_sell_reduces_shares_no_warning(self, monkeypatch, tmp_path):
        """窗口内部分卖出 500 股, 持仓 700→200 → 不误报。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"002460": {"shares": 700, "avg_cost": 52.97}},
            prev_pending=[],
            curr_positions={"002460": {"shares": 200, "avg_cost": 50.0}},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-10", "exec_date": "2026-08-11",
                        "symbol": "002460", "action": "sell",
                        "status": "executed", "shares": 500,
                        "exec_price": 55.0}],
        )
        warnings = state_check.check_consistency()
        assert not any("002460" in w for w in warnings)

    def test_no_exec_date_column_exempts(self, monkeypatch, tmp_path):
        """历史回填数据无 exec_date 列 → 保守视为窗口内, 不误报。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"002460": {"shares": 200, "avg_cost": 50.0}},
            prev_pending=[],
            curr_positions={"002460": {"shares": 700, "avg_cost": 52.97}},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-10",
                        "symbol": "002460", "action": "buy",
                        "status": "executed", "shares": 500}],
        )
        warnings = state_check.check_consistency()
        assert not any("002460" in w for w in warnings)

    def test_old_snapshot_without_timestamp_exempts(self, monkeypatch, tmp_path):
        """旧快照无 timestamp(快照日期无法判定) → 全量豁免, 不误报。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"002460": {"shares": 200, "avg_cost": 50.0}},
            prev_pending=[],
            curr_positions={"002460": {"shares": 700, "avg_cost": 52.97}},
            curr_pending=[],
            elog_rows=[{"signal_date": "2026-08-04", "exec_date": "2026-08-05",
                        "symbol": "002460", "action": "buy",
                        "status": "executed", "shares": 500,
                        "exec_price": 50.1}],
            snapshot_ts="",
        )
        warnings = state_check.check_consistency()
        assert not any("002460" in w for w in warnings)


class TestSameDayFillNotDoubleCounted:
    """快照当日成交不重复计入窗口(2026-08-28 修复)。

    修复背景: 688347 华虹宏力 200→600 误报"净成交 +600 vs 实际 +400"——
    8-27 开盘成交的 200 股已计入 8-27 22:59 快照, 窗口 exec_date >= 快照日
    又计一次 → 双重计数。窗口应为严格大于快照日期。
    """

    def test_fill_on_snapshot_date_not_counted(self, monkeypatch, tmp_path):
        """快照日成交(已含于快照) + 快照后成交 → 净变化吻合, 不误报。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"688347": {"shares": 200, "avg_cost": 244.09}},
            prev_pending=[],
            curr_positions={"688347": {"shares": 600, "avg_cost": 249.33}},
            curr_pending=[],
            elog_rows=[
                {"signal_date": "2026-08-26", "exec_date": "2026-08-27",
                 "symbol": "688347", "action": "buy", "status": "executed",
                 "shares": 200, "exec_price": 244.01},   # 已含于快照
                {"signal_date": "2026-08-27", "exec_date": "2026-08-28",
                 "symbol": "688347", "action": "buy", "status": "executed",
                 "shares": 400, "exec_price": 251.88},   # 快照后新成交
            ],
            snapshot_ts="2026-08-27 22:59:48",
        )
        warnings = state_check.check_consistency()
        assert not any("688347" in w for w in warnings)

    def test_only_snapshot_date_fill_no_share_change(self, monkeypatch, tmp_path):
        """仅快照日成交、股数未变 → 不误报(修复前 +200 vs 0 误报)。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"688347": {"shares": 200, "avg_cost": 244.09}},
            prev_pending=[],
            curr_positions={"688347": {"shares": 200, "avg_cost": 244.09}},
            curr_pending=[],
            elog_rows=[
                {"signal_date": "2026-08-26", "exec_date": "2026-08-27",
                 "symbol": "688347", "action": "buy", "status": "executed",
                 "shares": 200, "exec_price": 244.01},
            ],
            snapshot_ts="2026-08-27 22:59:48",
        )
        warnings = state_check.check_consistency()
        assert not any("688347" in w for w in warnings)

    def test_real_mismatch_still_warns(self, monkeypatch, tmp_path):
        """快照后净成交与股数变化仍不吻合 → 保留警告(防漏报)。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={"688347": {"shares": 200, "avg_cost": 244.09}},
            prev_pending=[],
            curr_positions={"688347": {"shares": 600, "avg_cost": 249.33}},
            curr_pending=[],
            elog_rows=[
                {"signal_date": "2026-08-27", "exec_date": "2026-08-28",
                 "symbol": "688347", "action": "buy", "status": "executed",
                 "shares": 300, "exec_price": 251.88},   # 只成交 300
            ],
            snapshot_ts="2026-08-27 22:59:48",
        )
        warnings = state_check.check_consistency()
        assert any("688347" in w and "股数变化" in w for w in warnings)


class TestRebuySignalWithPendingOrder:
    """重复买入信号 + 待执行订单 → 不误报(2026-08-28 修复)。

    背景: 同日多次运行(--no-save 回归/补跑)时, 快照记录的 buy 与当日
    signals.csv 最新 buy 是**同一条**信号, 原逻辑误报"昨日买入未执行"。
    实际信号已在 pending_orders.json 中排队等成交 → 以队列为事实来源。
    """

    def test_rebuy_with_pending_buy_no_warning(self, monkeypatch, tmp_path):
        """快照 buy + 当前 buy + 订单在队列 → 不误报。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={},
            prev_pending=[{"symbol": "002916", "signal_date": "2026-08-28",
                           "shares": 300}],
            curr_positions={},
            curr_pending=[{"symbol": "002916", "action": "buy",
                           "signal_date": "2026-08-28", "shares": 300}],
            elog_rows=[],
            snapshot_ts="2026-08-28 21:46:49",
            prev_signals={"002916": "buy"},
            signals_rows=[{"date": "2026-08-28", "symbol": "002916",
                           "signal": "buy"}],
        )
        warnings = state_check.check_consistency()
        assert not any("002916" in w and "再次买入" in w for w in warnings)

    def test_rebuy_without_pending_still_warns(self, monkeypatch, tmp_path):
        """快照 buy + 当前 buy + 无订单在队列 → 保留警告(防漏报)。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={},
            prev_pending=[],
            curr_positions={},
            curr_pending=[],
            elog_rows=[],
            snapshot_ts="2026-08-27 22:59:48",
            prev_signals={"002916": "buy"},
            signals_rows=[{"date": "2026-08-28", "symbol": "002916",
                           "signal": "buy"}],
        )
        warnings = state_check.check_consistency()
        assert any("002916" in w and "再次买入" in w for w in warnings)

    def test_pending_sell_does_not_exempt(self, monkeypatch, tmp_path):
        """队列中只有卖出订单 → 不豁免买入未执行警告。"""
        _setup_state_check(
            monkeypatch, tmp_path,
            prev_positions={},
            prev_pending=[],
            curr_positions={},
            curr_pending=[{"symbol": "002916", "action": "sell",
                           "signal_date": "2026-08-28", "shares": 300}],
            elog_rows=[],
            snapshot_ts="2026-08-27 22:59:48",
            prev_signals={"002916": "buy"},
            signals_rows=[{"date": "2026-08-28", "symbol": "002916",
                           "signal": "buy"}],
        )
        warnings = state_check.check_consistency()
        assert any("002916" in w and "再次买入" in w for w in warnings)
