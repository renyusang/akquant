"""Tests for 峰值跨日持久化 (peak_close) —— 2026-09-28 修复。

背景(见 `plan_profit_protection.md` §3, 含实盘证据):
实盘 `daily_signal` 每天为每个标的重建 `SignalEngine`, 峰值被 `set_position`
重置为持仓成本 → "峰值回撤止盈"退化为"-4% 成本止损", **完全不保护浮盈**;
而回测里引擎全程存活、峰值跨 bar 单调累积。两者是两个不同机制
(全区间组合回测: 股票池 +38.5pp 差异)。

本文件锁定修复链路:
`positions.json.peak_close`(台账) → `set_position(peak_price=...)`(恢复)
→ `update()` 抬高 → `_persist_peaks()` 回写; 并验证**实盘逐日重建引擎**
与**回测单引擎**在整条价格路径上给出**相同**的卖出日与规则。
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import daily_signal
import manage
import portfolio
from signal_engine import SignalEngine

# 只保留"峰值回撤止盈"一条规则, 隔离其它规则对结论的干扰
ONLY_TRAIL = dict(
    trailing_stop_pct=0.04,
    stop_loss_pct=0.0,
    use_price_signal=False,
    use_velocity_signal=False,
    trend_filter_enabled=False,
    target_profit_pct=0.0,
)


def _path() -> list:
    """价格路径: 横盘 → 涨到 130 → 回落到 124.8 之下(触发 4% 回撤)。"""
    return [100.0] * 5 + [110.0, 120.0, 130.0, 128.0, 124.0, 120.0, 118.0]


class TestEnginePeakRestore:
    """set_position 的 peak_price 参数与 peak_price 属性。"""

    def test_restore_peak_triggers_trailing(self):
        """恢复高峰值后, 今日收盘回撤 ≥ 阈值 → 触发峰值回撤止盈。"""
        e = SignalEngine(**ONLY_TRAIL)
        e.set_position(True, 100.0, peak_price=130.0)
        assert e.peak_price == pytest.approx(130.0)
        r = e.update(124.0, 124.0, 124.0)
        assert r["signal"] == "sell"
        assert "峰值回撤止盈" in r["reason"]

    def test_no_restore_means_cost_basis(self):
        """不恢复(现状实盘)时峰值 = 成本 → 仅当跌破成本 -4% 才卖, 浮盈全回吐。"""
        e = SignalEngine(**ONLY_TRAIL)
        e.set_position(True, 100.0)
        assert e.peak_price == pytest.approx(100.0)
        r = e.update(124.0, 124.0, 124.0)
        assert r["signal"] == "hold"           # 修复前: 该卖不卖(不保护浮盈)
        r2 = e.update(95.0, 95.0, 95.0)
        assert r2["signal"] == "sell"          # 只有跌到成本 -4% 才出(已亏损)

    def test_invalid_peak_falls_back_to_entry(self):
        """peak_price 为 0/None/负 → 回退入场价(保持原行为)。"""
        for bad in (0.0, None, -5.0):
            e = SignalEngine(**ONLY_TRAIL)
            e.set_position(True, 88.0, peak_price=bad)
            assert e.peak_price == pytest.approx(88.0)

    def test_peak_raises_with_update(self):
        """恢复后的峰值仍随新高抬升(只增不减)。"""
        e = SignalEngine(**ONLY_TRAIL)
        e.set_position(True, 100.0, peak_price=110.0)
        e.update(120.0, 120.0, 120.0)
        assert e.peak_price == pytest.approx(120.0)


class TestLiveBacktestParity:
    """核心: 逐日重建引擎(实盘) == 单引擎(回测), 卖出日与规则一致。"""

    def _backtest(self, path, cost=100.0):
        e = SignalEngine(**ONLY_TRAIL)
        e.set_position(True, cost)
        for i, c in enumerate(path):
            r = e.update(c, c, c)
            if r["signal"] == "sell":
                return i, r["reason"]
        return None, ""

    def _live(self, path, cost=100.0, persist=True):
        """模拟 daily_signal: 每天新建引擎; 峰值由 positions.json 台账恢复。"""
        ledger = cost
        for i, c in enumerate(path):
            e = SignalEngine(**ONLY_TRAIL)
            e.set_position(True, cost, peak_price=(ledger if persist else None))
            r = e.update(c, c, c)
            if r["signal"] == "sell":
                return i, r["reason"]
            ledger = max(ledger, c)            # 当日评估后回写(见 _persist_peaks)
        return None, ""

    def test_live_with_persist_matches_backtest(self):
        """修复后: 实盘(持久化峰值)与回测在同一天、同一条规则卖出。"""
        path = _path()
        bt_i, bt_reason = self._backtest(path)
        lv_i, lv_reason = self._live(path)
        assert bt_i is not None and lv_i is not None
        assert lv_i == bt_i
        assert lv_reason.split("(")[0] == bt_reason.split("(")[0]

    def test_live_without_persist_lags_behind(self):
        """修复前: 实盘因峰值每天重置, 同一路径上卖出显著更晚(或永不卖) —— 缺陷本身。"""
        path = _path()
        bt_i, _ = self._backtest(path)
        lv_i, _ = self._live(path, persist=False)
        assert lv_i is None or lv_i > bt_i


class TestPortfolioPeakLedger:
    """positions.json 的 peak_close 台账维护。"""

    @pytest.fixture(autouse=True)
    def _tmp_positions(self, monkeypatch):
        self.tmp = tempfile.TemporaryDirectory()
        monkeypatch.setattr(portfolio, "POSITIONS_FILE",
                            os.path.join(self.tmp.name, "positions.json"))
        yield
        self.tmp.cleanup()

    def test_new_position_peak_equals_cost(self):
        """新开仓: 峰值初值 = 成交均价(等价回测"峰值初值 = 入场价")。"""
        portfolio.add_position("000001", "测试", 100, 10.0)
        assert portfolio.load_positions()["000001"]["peak_close"] == 10.0

    def test_add_preserves_peak(self):
        """加仓**不重置**峰值(与回测一致: 仅 0→1 开仓时重置)。"""
        portfolio.add_position("000001", "测试", 100, 10.0)
        portfolio.update_peak("000001", 13.5)
        portfolio.add_position("000001", "测试", 100, 12.0)
        p = portfolio.load_positions()["000001"]
        assert p["peak_close"] == pytest.approx(13.5)
        assert p["shares"] == 200

    def test_legacy_position_without_peak_migrates_on_add(self):
        """旧文件无 peak_close: 加仓时以旧均价回退, 不报错也不丢字段。"""
        portfolio.save_positions({"000001": {
            "name": "测试", "shares": 100, "avg_cost": 10.0,
            "first_buy_date": "2026-01-01"}})
        portfolio.add_position("000001", "测试", 100, 12.0)
        assert portfolio.load_positions()["000001"]["peak_close"] > 0

    def test_update_peaks_only_increases_and_skips_nonholdings(self):
        """批量回写: 只增不减; 无持仓的标的被跳过。"""
        portfolio.add_position("000001", "测试", 100, 10.0)
        portfolio.update_peaks({"000001": 12.0, "999999": 5.0})
        assert portfolio.load_positions()["000001"]["peak_close"] == 12.0
        portfolio.update_peaks({"000001": 11.0})       # 低于现值 → 不改
        assert portfolio.load_positions()["000001"]["peak_close"] == 12.0
        assert "999999" not in portfolio.load_positions()

    def test_remove_position_drops_peak(self):
        """卖出后整条记录(含峰值)消失, 下一轮持仓重新起算。"""
        portfolio.add_position("000001", "测试", 100, 10.0)
        portfolio.update_peak("000001", 12.0)
        portfolio.remove_position("000001")
        assert portfolio.load_positions() == {}

    def test_split_scales_peak(self, monkeypatch):
        """除权(送转)后峰值必须同比例缩放, 否则会误判为深度回撤而立刻卖出。"""
        portfolio.add_position("000001", "测试", 100, 10.0)
        portfolio.update_peak("000001", 16.0)
        manage.cmd_split("000001", 2.0)                # 股价减半
        p = portfolio.load_positions()["000001"]
        assert p["avg_cost"] == pytest.approx(5.0)
        assert p["peak_close"] == pytest.approx(8.0)


class TestDailySignalWiring:
    """daily_signal 的回写与影子对照。"""

    @pytest.fixture(autouse=True)
    def _tmp_positions(self, monkeypatch):
        self.tmp = tempfile.TemporaryDirectory()
        monkeypatch.setattr(portfolio, "POSITIONS_FILE",
                            os.path.join(self.tmp.name, "positions.json"))
        yield
        self.tmp.cleanup()

    def test_persist_peaks_only_for_held_symbols(self):
        """只回写仍有持仓、且有有效峰值的标的。"""
        portfolio.add_position("000001", "甲", 100, 10.0)
        n = daily_signal._persist_peaks([
            {"symbol": "000001", "shares": 100, "peak_price": 11.5},
            {"symbol": "000002", "shares": 0, "peak_price": 9.9},      # 无持仓 → 跳过
            {"symbol": "000003", "shares": 100, "peak_price": 0},      # 无有效峰值 → 跳过
            {"symbol": "000004", "shares": 100, "error": "无数据", "peak_price": 9.0},
        ])
        assert n == 1
        assert portfolio.load_positions()["000001"]["peak_close"] == 11.5

    def test_shadow_flags_earlier_sell(self):
        """影子: 峰值 120 收盘 124... → 回撤 ≥4% 且未被止损抢占 → 标记"会提前卖"。"""
        s = daily_signal._peak_shadow(
            peak_prev=120.0, avg_cost=100.0, close=114.0,
            result={"signal": "hold", "reason": "持仓中"},
            params={"trailing_stop_pct": 0.04, "stop_loss_pct": 0.05})
        assert s["would_sell_new"] is True
        assert s["shadow_earlier"] is True
        assert s["dd_new_pct"] == pytest.approx(5.0)

    def test_shadow_respects_stop_priority(self):
        """止损(优先级 1)在新旧语义下相同: 已跌破 -5% 时峰值规则不改变结论。"""
        s = daily_signal._peak_shadow(
            peak_prev=120.0, avg_cost=100.0, close=94.0,
            result={"signal": "sell", "reason": "止损(-6.0%)"},
            params={"trailing_stop_pct": 0.04, "stop_loss_pct": 0.05})
        assert s["preempted_by_stop"] is True
        assert s["would_sell_new"] is False
        assert s["shadow_earlier"] is False

    def test_shadow_disabled_without_trailing(self):
        """未启用峰值回撤止盈 → 不产生影子记录。"""
        s = daily_signal._peak_shadow(
            peak_prev=120.0, avg_cost=100.0, close=110.0,
            result={"signal": "hold", "reason": ""},
            params={"trailing_stop_pct": 0.0, "stop_loss_pct": 0.05})
        assert s == {}


class TestShadowLog:
    """影子对照写盘(logs/peak_shadow.csv) —— 生产每日都会跑, 必须只追加不覆盖。"""

    @pytest.fixture(autouse=True)
    def _tmp_logs(self, monkeypatch):
        self.tmp = tempfile.TemporaryDirectory()
        monkeypatch.setattr(daily_signal, "LOG_DIR", self.tmp.name)
        yield
        self.tmp.cleanup()

    def _row(self, sym, peak_prev, close, date="2026-09-28"):
        s = daily_signal._peak_shadow(
            peak_prev=peak_prev, avg_cost=100.0, close=close,
            result={"signal": "hold", "reason": "持仓中"},
            params={"trailing_stop_pct": 0.04, "stop_loss_pct": 0.05})
        return {"symbol": sym, "name": "测试", "date": date,
                "shares": 100, "_shadow": s}

    def test_append_and_count(self):
        """首次写表头, 二次追加不重复表头; 返回"会提前卖"条数。"""
        path = os.path.join(self.tmp.name, "peak_shadow.csv")
        n1 = daily_signal._persist_peak_shadow(
            [self._row("000001", 120.0, 114.0, "2026-09-28")], "2026-09-28")  # 回撤5%→卖
        assert n1 == 1
        n2 = daily_signal._persist_peak_shadow(
            [self._row("000001", 120.0, 119.0, "2026-09-29")], "2026-09-29")  # 回撤0.8%→不卖
        assert n2 == 0
        # 代码列按项目约定用 dtype=str 读(否则前导零被吃掉, 同 execution_log.csv)
        df = pd.read_csv(path, dtype={"symbol": str})
        assert list(df["symbol"]) == ["000001", "000001"]
        assert list(df["date"]) == ["2026-09-28", "2026-09-29"]
        assert df["shadow_earlier"].tolist() == [True, False]
        assert len(df.columns) == 13

    def test_date_falls_back_to_argument(self):
        """结果行缺 date 时用传入的 date_str(与 signals.csv 同一天)。"""
        row = self._row("000001", 120.0, 114.0)
        row.pop("date")
        daily_signal._persist_peak_shadow([row], "2026-10-09")
        df = pd.read_csv(os.path.join(self.tmp.name, "peak_shadow.csv"),
                         dtype={"symbol": str})
        assert df["date"].iloc[0] == "2026-10-09"

    def test_empty_results_writes_nothing(self):
        """无影子记录(未启用)时不创建文件。"""
        assert daily_signal._persist_peak_shadow(
            [{"symbol": "000001", "shares": 100}], "2026-09-28") == 0
        assert not os.path.exists(os.path.join(self.tmp.name, "peak_shadow.csv"))
