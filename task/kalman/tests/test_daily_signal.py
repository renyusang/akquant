"""Tests for daily_signal.py 趋势翻转补仓逻辑。

覆盖: 补仓市值按市价计算(修复)、浮盈/浮亏场景、资金不足、
allocated_cash 累加、无持仓/无信号不补仓等边界。
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import daily_signal
import orders


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_engine_factory(signal: str, target_pct: float):
    """构造返回固定信号的 FakeEngine。"""

    class FakeEngine:
        def __init__(self, **kwargs):
            self._close = 0.0

        def set_position(self, has, price):
            pass

        def process_history(self, df):
            pass

        def update(self, close, ma20_cur, ma20_prev):
            self._close = close
            return {
                "signal": signal,
                "target_pct": target_pct,
                "kalman_price": close,
                "kalman_velocity": 0.0,
                "ma20": ma20_cur,
                "ma20_rising": True,
                "trend": "up",
                "close": close,
                "reason": "test",
            }

        @property
        def filtered_price(self):
            return self._close

    return FakeEngine


def _make_df(last_close: float) -> pd.DataFrame:
    """构造 50 行日线数据, 最后一天 close=last_close。"""
    dates = pd.date_range("2026-05-01", periods=50, freq="B")
    closes = [last_close] * 50  # 除最后一天外的价格(简化,MA20计算无影响)
    df = pd.DataFrame({
        "date": dates,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000] * 50, "symbol": ["300750"] * 50,
    })
    return df


def _make_config(cash: float = 100000.0, max_pct: float = 0.20,
                 max_pos: int = 5) -> dict:
    return {
        "strategy": {},
        "stock": {
            "initial_cash": cash,
            "max_positions": max_pos,
            "single_position_pct": max_pct,
        },
    }


def _setup_common(monkeypatch, tmp, signal="hold", target_pct=0.95,
                  last_close=100.0, pos=None, pool_pos=None):
    """公共 monkeypatch: 引擎/数据/持仓/日志。"""
    monkeypatch.setattr(daily_signal, "SignalEngine",
                        _make_engine_factory(signal, target_pct))
    monkeypatch.setattr(daily_signal, "download_with_cache",
                        lambda symbol, data_years, asset_type="stock": _make_df(last_close))
    monkeypatch.setattr(daily_signal, "get_position", lambda symbol: pos)
    monkeypatch.setattr(daily_signal, "load_positions",
                        lambda: pool_pos if pool_pos is not None else {})
    # 订单写入 tmp
    monkeypatch.setattr(orders, "PENDING_FILE",
                        os.path.join(tmp, "pending.json"))
    # 日志 spy
    calls = {"log_pending": [], "log_skipped": []}
    monkeypatch.setattr(daily_signal, "log_pending",
                        lambda *a, **kw: calls["log_pending"].append((a, kw)))
    monkeypatch.setattr(daily_signal, "log_skipped",
                        lambda *a, **kw: calls["log_skipped"].append((a, kw)))
    return calls


STOCK = {"symbol": "300750", "name": "宁德时代", "type": "stock"}


# ---------------------------------------------------------------------------
# 补仓: 市值按市价计算(本次修复)
# ---------------------------------------------------------------------------
class TestAddPositionMarketValue:
    """趋势翻转补仓的 current_value 应按市价计算。"""

    def test_no_add_when_market_value_meets_target(self, monkeypatch):
        """浮盈大 → 按市价已达标 → 不补仓(修复前按成本价会误补)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 50.0,  # 成本 5000
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            # 市价 150: 市值 15000; 目标 = 100000*0.95*0.2 = 19000
            # 修复前: 按成本 5000 → 补仓; 修复后: 按市价 15000 → 15000>19000*? 否
            # target(19000) vs current(15000): 15000*1.05=15750 < 19000 → 仍补仓?
            # 用更高市价确保达标: 市价 200 → 市值 20000 > 19000*1.05? 20000>19950 ✓
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=200.0, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            # 修复后: 市值 20000 ≥ 目标 19000 → 不补仓
            assert result["signal"] == "hold"
            assert calls["log_pending"] == []
            assert orders.load_pending() == []

    def test_add_when_market_value_below_target(self, monkeypatch):
        """浮亏/市值不足 → 补仓, 股数按差额/市价向下取整到整手。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,  # 成本 10000
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            # 市价 50: 市值 5000; 目标 = 100000*0.95*0.2 = 19000
            # add_value = 14000 → qty = int(14000/50/100)*100 = int(2.8)*100 = 200
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert result["signal"] == "buy"
            assert result["shares"] == 200
            pending = orders.load_pending()
            assert len(pending) == 1
            assert pending[0]["action"] == "buy"
            assert pending[0]["shares"] == 200
            assert pending[0]["signal_price"] == 50.0
            assert pending[0]["target_pct"] == pytest.approx(0.19)  # 0.95*0.2

    def test_add_value_uses_market_price_not_cost(self, monkeypatch):
        """核心: 补仓差额基于市价市值而非成本市值(整手取整可区分)。"""
        with tempfile.TemporaryDirectory() as tmp:
            # 成本 50(成本市值 5000), 市价 100(市价市值 10000)
            # cash=200000 → target = 200000*0.19 = 38000
            # 按市价: add_value=28000 → int(28000/100/100)=2 → 200股
            # 按成本: add_value=33000 → int(33000/100/100)=3 → 300股
            pos = {"shares": 100, "avg_cost": 50.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=100.0, pos=pos, pool_pos={"300750": pos},
            )
            daily_signal.evaluate_stock(STOCK, _make_config(cash=200000.0), 2,
                                        {"stock": 0.0})
            pending = orders.load_pending()
            assert len(pending) == 1
            assert pending[0]["shares"] == 200  # 市价算法; 若用成本会是 300

            # 第二部分: 市价已达标 → 不补; 成本价会误判为不足
            pos2 = {"shares": 100, "avg_cost": 50.0,
                    "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls2 = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=100.0, pos=pos2, pool_pos={"300750": pos2},
            )
            result2 = daily_signal.evaluate_stock(
                STOCK, _make_config(cash=50000.0), 2, {"stock": 0.0})
            # 按市价市值 10000 > 目标 9500 → 不补仓
            assert result2["signal"] == "hold"
            assert calls2["log_pending"] == []
            # pending 文件保持第一部分的订单数量(未新增)
            assert len(orders.load_pending()) == 1

    def test_no_add_when_just_above_threshold(self, monkeypatch):
        """边界: 市值在 1.05 阈值内 → 不补仓。"""
        with tempfile.TemporaryDirectory() as tmp:
            # 市值 = 目标/1.05 附近 → 不补
            pos = {"shares": 100, "avg_cost": 10.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            # target=19000, 需市值 > 19950 才不补? 否: 补仓条件 target > current*1.05
            # 即 current < target/1.05 = 18095 时补仓
            # 市值 18100 > 18095 → 不补仓
            last_close = 181.0  # 市值 18100
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=last_close, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert result["signal"] == "hold"
            assert calls["log_pending"] == []

    def test_no_add_position_when_no_position(self, monkeypatch):
        """回归: 无持仓 → 不进入补仓分支。"""
        with tempfile.TemporaryDirectory() as tmp:
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=100.0, pos=None, pool_pos={},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert result["signal"] == "hold"
            assert calls["log_pending"] == []

    def test_no_add_when_target_pct_zero(self, monkeypatch):
        """回归: target_pct=0(下跌趋势无信号) → 不补仓。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.0,
                last_close=50.0, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert result["signal"] == "hold"
            assert calls["log_pending"] == []

    def test_no_add_when_signal_is_buy(self, monkeypatch):
        """回归: 信号为 buy(新买入) → 走买入分支而非补仓。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="buy", target_pct=0.95,
                last_close=100.0, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            # buy 信号: 已有持仓时 evaluate_stock 会生成买入订单(加仓新单)
            assert result["signal"] == "buy"
            assert len(calls["log_pending"]) == 1


# ---------------------------------------------------------------------------
# 补仓: 资金与 allocated_cash
# ---------------------------------------------------------------------------
class TestAddPositionCash:
    """补仓的资金检查与承诺资金累加。"""

    def test_skip_when_insufficient_cash(self, monkeypatch):
        """池内资金不足 → 补仓被跳过。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            # 池内还有另一只占用 90000 资金
            other = {"shares": 1000, "avg_cost": 90.0,
                     "first_buy_date": "2026-01-01", "name": "其他"}
            pool_pos = {"300750": pos, "000001": other}
            # cash=100000, used=1000*90+100*100=100000 → remaining≈0
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos=pool_pos,
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(cash=100000.0), 2, {"stock": 0.0},
            )
            # 资金不足 → 不生成补仓订单
            assert result["signal"] == "hold"
            assert calls["log_pending"] == []
            assert orders.load_pending() == []

    def test_skip_when_allocated_cash_exhausted(self, monkeypatch):
        """本轮已承诺资金(allocated_cash)耗尽 → 跳过。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos={"300750": pos},
            )
            # allocated_cash 已承诺 95000, 池内无其他持仓
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(cash=100000.0), 2,
                {"stock": 95000.0},
            )
            assert result["signal"] == "hold"
            assert calls["log_pending"] == []

    def test_allocated_cash_accumulates(self, monkeypatch):
        """补仓成功后 allocated_cash 应累加承诺金额。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos={"300750": pos},
            )
            allocated = {"stock": 0.0}
            daily_signal.evaluate_stock(STOCK, _make_config(), 2, allocated)
            # add_qty=200 @ 50 = 10000
            assert allocated["stock"] == pytest.approx(10000.0)

    def test_kcb_lot_200_rounding(self, monkeypatch):
        """科创板(688)补仓按 200 股/手取整。"""
        with tempfile.TemporaryDirectory() as tmp:
            kcb_stock = {"symbol": "688041", "name": "海光信息", "type": "stock"}
            pos = {"shares": 200, "avg_cost": 200.0,
                   "first_buy_date": "2026-01-01", "name": "海光信息"}
            # 市价 50: 市值 10000; cash=300000 → target = 300000*0.19 = 57000
            # add_value = 47000 → qty = int(47000/50/200)*200 = int(4.7)*200 = 800
            # 若按 100 股/手: int(47000/50/100)*100 = 900 → 可区分
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos={"688041": pos},
            )
            daily_signal.evaluate_stock(kcb_stock, _make_config(cash=300000.0), 2,
                                        {"stock": 0.0})
            pending = orders.load_pending()
            assert len(pending) == 1
            assert pending[0]["shares"] == 800

    def test_reason_marks_trend_flip(self, monkeypatch):
        """补仓订单 reason 应标记'趋势翻转补仓'。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert "趋势翻转补仓" in result["reason"]
            assert "趋势翻转补仓" in calls["log_pending"][0][0][5]  # signal_reason

    def test_pending_order_signal_date_is_last_data_date(self, monkeypatch):
        """补仓订单 signal_date = 数据最后一天(次日 T+1 执行)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="hold", target_pct=0.95,
                last_close=50.0, pos=pos, pool_pos={"300750": pos},
            )
            daily_signal.evaluate_stock(STOCK, _make_config(), 2, {"stock": 0.0})
            pending = orders.load_pending()
            assert len(pending) == 1
            df = _make_df(50.0)
            expected_date = str(df["date"].iloc[-1])[:10]
            assert pending[0]["signal_date"] == expected_date
