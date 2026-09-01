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

        def update(self, close, ma20_cur, ma20_prev, high=None, low=None,
                   volume=None):
            self._close = close
            return {
                "signal": signal,
                "target_pct": target_pct,
                "kalman_price": close,
                "kalman_velocity": 0.0,
                "ma20": ma20_cur,
                "ma20_rising": True,
                "trend": "up",
                "adx": None,
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


# ---------------------------------------------------------------------------
# 卖出成交后自动补仓(2026-08-05 改进: 无条件调用, 池内空位判断)
# ---------------------------------------------------------------------------
class TestAutoFillAfterSell:
    """卖出成交空出仓位后, 用被跳过候选立即补仓。"""

    def _setup(self, monkeypatch, tmp_path, positions, pending):
        """初始化临时状态文件。

        修复(2026-08-21): 原 _auto_fill_pool 经 load_config() 读**生产**
        stocks.yaml——股票池现金从 20 万调至 30 万后买入量 100→200 股,
        测试断言随之失败。改为固定临时配置(20万池), 测试与生产配置解耦。
        """
        import json
        import portfolio

        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps(positions), encoding="utf-8")
        pend_file = tmp_path / "pending_orders.json"
        pend_file.write_text(json.dumps(pending), encoding="utf-8")

        monkeypatch.setattr(portfolio, "POSITIONS_FILE", str(pos_file))
        monkeypatch.setattr(portfolio, "TRADES_FILE",
                            str(tmp_path / "trades.csv"))
        monkeypatch.setattr(orders, "PENDING_FILE", str(pend_file))
        # 缓存目录指向不存在 → 跳过涨停检查
        monkeypatch.setattr(daily_signal, "CACHE_DIR",
                            str(tmp_path / "no_cache"))
        # 固定测试配置(股票池 20 万), 不随生产 stocks.yaml 漂移。
        # 注意: load_config 默认参数 path=CONFIG_FILE 在定义时已绑定,
        # patch 模块属性无效 → 直接 patch load_config 指向临时文件
        cfg_file = tmp_path / "stocks.yaml"
        cfg_file.write_text(
            "stock:\n  initial_cash: 200000\n  max_positions: 5\n"
            "  single_position_pct: 0.20\n"
            "etf:\n  initial_cash: 100000\n  max_positions: 5\n"
            "  single_position_pct: 0.20\n"
            "strategy: {}\n"
            "watchlist:\n  stocks: []\n  etfs: []\n", encoding="utf-8")

        def _fixed_config(path=None):
            import yaml
            with open(str(cfg_file), encoding="utf-8") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(daily_signal, "load_config", _fixed_config)
        # 日志隔离(2026-08-28): _auto_fill_pool 经 log_pending 写执行日志,
        # 不拦截会污染生产 execution_log.csv(301511 8-4 测试行反复出现)
        monkeypatch.setattr(daily_signal, "log_pending",
                            lambda *a, **k: None)
        monkeypatch.setattr(daily_signal, "log_skipped",
                            lambda *a, **k: None)
        return portfolio

    @staticmethod
    def _defu_skipped():
        """德福科技被跳过信号——与 evaluate_stock 真实输出一致:
        target_pct 被清零(显示用), 原始仓位保留在 _target_pct。"""
        return {
            "symbol": "301511", "name": "德福科技", "close": 68.71,
            "kalman_price": 58.66, "target_pct": 0.0, "_target_pct": 0.30,
            "trend": "down", "date": "2026-08-04",
            "reason": "价格突破(偏离17.1%)",
        }

    def _stock_positions(self, with_jushi=True):
        pos = {
            "001270": {"name": "铖昌科技", "shares": 100, "avg_cost": 92.0},
            "002192": {"name": "融捷股份", "shares": 100, "avg_cost": 62.86},
            "002460": {"name": "赣锋锂业", "shares": 200, "avg_cost": 50.0},
            "300750": {"name": "宁德时代", "shares": 100, "avg_cost": 372.67},
        }
        if with_jushi:
            pos["600176"] = {"name": "中国巨石", "shares": 300,
                             "avg_cost": 38.007}
        return pos

    def test_fill_after_sell_frees_slot(self, monkeypatch, tmp_path):
        """8-4 场景: 巨石卖出成交 → 股票池空位 → 德福立即补入。"""
        portfolio = self._setup(
            monkeypatch, tmp_path, self._stock_positions(True),
            [{"symbol": "600176", "action": "sell", "shares": 300}],
        )
        # 模拟执行卖出成交
        portfolio.remove_position("600176")
        positions_now = portfolio.load_positions()
        # 触发补仓(新逻辑: 无条件调用)
        config = daily_signal.load_config()
        filled = daily_signal._auto_fill_pool(
            [self._defu_skipped()], 10, positions_now, config,
            "stock", "股票", ("51", "15", "58", "56"), quiet=True,
        )
        assert filled == 1
        pending = orders.load_pending()
        assert any(o["symbol"] == "301511" and o["action"] == "buy"
                   for o in pending)

    def test_no_fill_when_pool_full(self, monkeypatch, tmp_path):
        """池满(5/5) → 无空位 → 不补。"""
        portfolio = self._setup(
            monkeypatch, tmp_path, self._stock_positions(True), [],
        )
        positions_now = portfolio.load_positions()  # 含巨石 = 5 只
        config = daily_signal.load_config()
        filled = daily_signal._auto_fill_pool(
            [self._defu_skipped()], 10, positions_now, config,
            "stock", "股票", ("51", "15", "58", "56"), quiet=True,
        )
        assert filled == 0
        assert not any(o.get("symbol") == "301511"
                       for o in orders.load_pending())

    def test_no_fill_without_skipped_candidates(self, monkeypatch, tmp_path):
        """无被跳过候选 → 不补。"""
        portfolio = self._setup(
            monkeypatch, tmp_path, self._stock_positions(True), [],
        )
        portfolio.remove_position("600176")
        config = daily_signal.load_config()
        filled = daily_signal._auto_fill_pool(
            [], 10, portfolio.load_positions(), config,
            "stock", "股票", ("51", "15", "58", "56"), quiet=True,
        )
        assert filled == 0

    def test_no_duplicate_when_pending_buy_exists(self, monkeypatch, tmp_path):
        """德福已有待买订单 → 不重复补(池内空位扣除待买)。"""
        portfolio = self._setup(
            monkeypatch, tmp_path, self._stock_positions(False),
            [{"symbol": "301511", "action": "buy", "shares": 100}],
        )
        config = daily_signal.load_config()
        filled = daily_signal._auto_fill_pool(
            [self._defu_skipped()], 10, portfolio.load_positions(), config,
            "stock", "股票", ("51", "15", "58", "56"), quiet=True,
        )
        assert filled == 0  # 待买已占位, 不重复下单
        buys = [o for o in orders.load_pending()
                if o.get("action") == "buy"]
        assert len(buys) == 1

    def test_fill_with_zeroed_target_pct(self, monkeypatch, tmp_path):
        """核心回归(2026-08-05): 跳过时 target_pct 清零但 _target_pct 保留
        → 补仓按原始仓位计算股数, 而非 0 股静默跳过。"""
        portfolio = self._setup(
            monkeypatch, tmp_path, self._stock_positions(True), [],
        )
        portfolio.remove_position("600176")
        config = daily_signal.load_config()
        # 旧 bug: 无 _target_pct 时 target_pct=0 → 0股跳过
        filled_old = daily_signal._auto_fill_pool(
            [self._defu_skipped()], 10, portfolio.load_positions(), config,
            "stock", "股票", ("51", "15", "58", "56"), quiet=True,
        )
        assert filled_old == 1  # 修复后按 _target_pct=0.30 → 100股
        pending = orders.load_pending()
        defu = next(o for o in pending if o["symbol"] == "301511")
        assert defu["shares"] == 100
        assert defu["target_pct"] == pytest.approx(0.06)  # 0.30×0.20


# ---------------------------------------------------------------------------
# 无限仓位模式 lot_based_position(2026-08-06): 数量=target_pct映射[1,3]手
# ---------------------------------------------------------------------------
class TestLotBasedPosition:
    """无资金限制的固定手数买入: 最小1手, 最大3手。"""

    def _lot_config(self, max_pct=0.20):
        cfg = _make_config(cash=1e9, max_pct=max_pct, max_pos=99999)
        cfg["strategy"] = {"lot_based_position": True}
        return cfg

    def test_buy_downtrend_one_lot(self, monkeypatch):
        """下跌趋势(0.30) → 1手 = 100股。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.30,
                          last_close=100.0)
            result = daily_signal.evaluate_stock(
                STOCK, self._lot_config(), 2, {"stock": 0.0})
            assert result["signal"] == "buy"
            assert result["shares"] == 100
            pending = orders.load_pending()
            assert len(pending) == 1
            assert pending[0]["shares"] == 100

    def test_buy_uptrend_three_lots(self, monkeypatch):
        """上涨趋势(0.95) → 3手 = 300股。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=100.0)
            result = daily_signal.evaluate_stock(
                STOCK, self._lot_config(), 2, {"stock": 0.0})
            assert result["shares"] == 300
            assert orders.load_pending()[0]["shares"] == 300

    def test_buy_star_board_lot_200(self, monkeypatch):
        """科创板(688, 最小200股): 下跌 → 1手 = 200股。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.30,
                          last_close=100.0)
            stock = {"symbol": "688041", "name": "海光信息", "type": "stock"}
            result = daily_signal.evaluate_stock(
                stock, self._lot_config(), 2, {"stock": 0.0})
            assert result["shares"] == 200

    def test_no_cash_or_slot_limit(self, monkeypatch):
        """无限仓位: 即使"满仓"(持仓超5只)仍可买入(无 max_pos 拦截)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pool_pos = {f"00000{i}": {"shares": 100, "avg_cost": 10.0,
                                      "first_buy_date": "2026-01-01",
                                      "name": f"S{i}"} for i in range(10)}
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=100.0, pool_pos=pool_pos)
            result = daily_signal.evaluate_stock(
                STOCK, self._lot_config(), 2, {"stock": 0.0})
            assert result["signal"] == "buy"  # 不被"已达最大持仓数"拦截
            assert result["shares"] == 300

    def test_add_to_target_lots(self, monkeypatch):
        """趋势翻转补仓: 持有1手(100股) → 补足3手(加200股)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            _setup_common(monkeypatch, tmp, signal="hold", target_pct=0.95,
                          last_close=100.0, pos=pos, pool_pos={"300750": pos})
            result = daily_signal.evaluate_stock(
                STOCK, self._lot_config(), 2, {"stock": 0.0})
            assert result["signal"] == "buy"  # 趋势翻转补仓
            assert result["shares"] == 200   # 补足 300-100

    def test_default_mode_unchanged(self, monkeypatch):
        """默认(无 lot_based_position): 原资金比例算法不变。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=100.0)
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(cash=100000.0, max_pct=0.20), 2,
                {"stock": 0.0})
            # 100000*0.95*0.20/100 = 190 → 100股(整手)
            assert result["shares"] == 100


# ---------------------------------------------------------------------------
# 待执行订单的名额占用(2026-08-12): 补仓单(已持仓)不占新增名额
# ---------------------------------------------------------------------------
class TestPendingSlotAccounting:
    """max_positions 名额: 新标的买入单占名额(防超买),
    已持仓标的的补仓单不占名额(成交后不新增持仓标的)。"""

    ETF_POOL = {
        "512400": {"name": "有色金属ETF", "shares": 11300, "avg_cost": 1.775},
        "159845": {"name": "中证1000ETF", "shares": 2000, "avg_cost": 2.93},
        "512660": {"name": "军工ETF", "shares": 5400, "avg_cost": 1.09},
        "515030": {"name": "新能源车ETF", "shares": 3700, "avg_cost": 1.66},
    }
    NEW_ETF = {"symbol": "510330", "name": "沪深300ETF", "type": "etf"}

    def _etf_config(self):
        cfg = _make_config(cash=100000.0, max_pct=0.20, max_pos=5)
        cfg["etf"] = {"initial_cash": 100000.0, "max_positions": 5,
                      "single_position_pct": 0.20}
        return cfg

    def _add_pending(self, symbol, name, action="buy", shares=5400):
        orders.add_pending_order(symbol, name, action, shares,
                                 signal_price=1.10, signal_date="2026-08-12")

    def test_refill_pending_not_block_new_entry(self, monkeypatch):
        """4只持仓 + 军工ETF补仓单(已持仓) → 新ETF买入放行。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=4.95, pool_pos=dict(self.ETF_POOL))
            self._add_pending("512660", "军工ETF", shares=600)  # 补仓单
            result = daily_signal.evaluate_stock(
                self.NEW_ETF, self._etf_config(), 2, {"etf": 0.0})
            assert result["signal"] == "buy", result.get("reason")

    def test_new_entry_pending_blocks_another(self, monkeypatch):
        """4只持仓 + 新标的买入单(半导体ETF) → 另一新ETF仍被拦截(防超买)。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=4.95, pool_pos=dict(self.ETF_POOL))
            self._add_pending("512480", "半导体ETF", shares=100)  # 新标的单
            result = daily_signal.evaluate_stock(
                self.NEW_ETF, self._etf_config(), 2, {"etf": 0.0})
            assert result["signal"] == "hold"
            assert "已达最大持仓数" in result.get("reason", "")

    def test_full_pool_blocks_new_entry(self, monkeypatch):
        """5只持仓满(无pending) → 新ETF仍被拦截(回归)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pool = dict(self.ETF_POOL)
            pool["515790"] = {"name": "光伏ETF", "shares": 24300,
                              "avg_cost": 0.816}
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=4.95, pool_pos=pool)
            result = daily_signal.evaluate_stock(
                self.NEW_ETF, self._etf_config(), 2, {"etf": 0.0})
            assert result["signal"] == "hold"
            assert "已达最大持仓数" in result.get("reason", "")


# ---------------------------------------------------------------------------
# 两阶段下单(2026-08-12): 补仓优先 + 新建按偏离度降序
# ---------------------------------------------------------------------------
class TestDeferredOrderPlacement:
    """_place_deferred_orders: 排序/名额/资金逻辑。"""

    STOCK_POOL = {
        "000001": {"name": "平安银行", "shares": 100, "avg_cost": 10.0},
        "000002": {"name": "万科A", "shares": 100, "avg_cost": 10.0},
        "000003": {"name": "标的C", "shares": 100, "avg_cost": 10.0},
        "000004": {"name": "标的D", "shares": 100, "avg_cost": 10.0},
    }

    def _cfg(self, max_pos=5, cash=200000.0):
        return _make_config(cash=cash, max_pct=0.20, max_pos=max_pos)

    def _candidate(self, sym, name, kind, dev, shares=100, price=10.0,
                   asset_type="stock"):
        return {
            "symbol": sym, "name": name, "close": price,
            "date": "2026-08-12",
            "_deferred": {
                "kind": kind, "symbol": sym, "name": name,
                "shares": shares, "price": price, "target_pct": 0.19,
                "reason": "test", "deviation": dev, "asset_type": asset_type,
            },
        }

    def test_new_by_deviation_priority(self, monkeypatch, tmp_path):
        """4持仓+1名额: 候选列表顺序与偏离度无关, 偏离9.9%应下单, 3.5%被拦。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, pool_pos=dict(self.STOCK_POOL))
            results = [
                self._candidate("000005", "低偏离", "new", dev=3.5),
                self._candidate("000006", "高偏离", "new", dev=9.9),
            ]
            allocated = {}
            daily_signal._place_deferred_orders(results, self._cfg(), allocated)
            assert results[1]["signal"] == "buy"      # 高偏离下单
            assert results[0]["signal"] == "hold"     # 低偏离被拦
            assert "已达最大持仓数" in results[0]["reason"]
            pend = orders.load_pending()
            assert len(pend) == 1
            assert pend[0]["symbol"] == "000006"

    def test_refill_priority_when_cash_limited(self, monkeypatch, tmp_path):
        """资金只够一单: 低偏离补仓优先消耗资金, 高偏离新建被资金不足拦截。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, pool_pos=dict(self.STOCK_POOL))
            results = [
                self._candidate("000006", "新建高偏离", "new", dev=9.9),
                self._candidate("000001", "补仓低偏离", "refill", dev=1.5),
            ]
            # 池现金 5000, 持仓市值 4000 → 剩余 1000, 只够一单
            daily_signal._place_deferred_orders(
                results, self._cfg(cash=5000.0), {})
            refill = next(r for r in results
                          if r["_deferred"]["kind"] == "refill")
            new = next(r for r in results
                       if r["_deferred"]["kind"] == "new")
            assert refill["signal"] == "buy"          # 补仓优先
            assert new["signal"] == "hold"
            assert "资金不足" in new["reason"]

    def test_refill_does_not_consume_slot(self, monkeypatch, tmp_path):
        """补仓不占名额: 4持仓 + 补仓 + 新建 → 两者都下单(名额5)。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, pool_pos=dict(self.STOCK_POOL))
            results = [
                self._candidate("000006", "新建", "new", dev=9.9),
                self._candidate("000001", "补仓", "refill", dev=1.5),
            ]
            daily_signal._place_deferred_orders(results, self._cfg(), {})
            assert all(r["signal"] == "buy" for r in results)
            assert len(orders.load_pending()) == 2

    def test_full_pool_blocks_new(self, monkeypatch, tmp_path):
        """5持仓满 + 1新建 → 名额拦截。"""
        with tempfile.TemporaryDirectory() as tmp:
            pool = dict(self.STOCK_POOL)
            pool["000007"] = {"name": "标的E", "shares": 100, "avg_cost": 10.0}
            _setup_common(monkeypatch, tmp, pool_pos=pool)
            results = [self._candidate("000008", "新建", "new", dev=5.0)]
            daily_signal._place_deferred_orders(results, self._cfg(), {})
            assert results[0]["signal"] == "hold"
            assert "已达最大持仓数" in results[0]["reason"]

    def test_insufficient_cash_blocks(self, monkeypatch, tmp_path):
        """现金不足: 大额候选 → 资金不足拦截, 无订单产生。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, pool_pos=dict(self.STOCK_POOL))
            results = [self._candidate("000008", "新建", "new", dev=5.0,
                                       shares=100, price=100.0)]
            daily_signal._place_deferred_orders(
                results, self._cfg(cash=5000.0), {})
            assert results[0]["signal"] == "hold"
            assert "资金不足" in results[0]["reason"]
            assert orders.load_pending() == []

    def test_evaluate_defer_orders_no_pending(self, monkeypatch):
        """evaluate_stock defer 模式: 不下单, 候选记录在 _deferred。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=100.0)
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0}, defer_orders=True)
            assert result["signal"] == "buy"
            assert result.get("_deferred", {}).get("kind") == "new"
            assert orders.load_pending() == []  # 未下单

    def test_evaluate_non_defer_still_places(self, monkeypatch):
        """默认模式(无限仓位等调用方): 行为不变, 立即下单。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=100.0)
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(cash=100000.0, max_pct=0.20), 2,
                {"stock": 0.0})
            assert result["signal"] == "buy"
            assert result.get("_deferred") is None
            assert len(orders.load_pending()) == 1

    def test_pending_new_consumes_slot(self, monkeypatch, tmp_path):
        """已有 pending 新建单(非持仓)占名额: 4持仓 + 创业板50单 + 新候选 → 拦截。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, pool_pos=dict(self.STOCK_POOL))
            orders.add_pending_order("000009", "已下单标的", "buy", 100,
                                     signal_price=10.0,
                                     signal_date="2026-08-11")
            results = [self._candidate("000008", "新建", "new", dev=9.9)]
            daily_signal._place_deferred_orders(results, self._cfg(), {})
            assert results[0]["signal"] == "hold"
            assert "已达最大持仓数" in results[0]["reason"]
            # pending 仍只有 1 笔(未重复下单)
            assert len(orders.load_pending()) == 1

    def test_slot_freed_by_prior_execution_before_scan(self, monkeypatch, tmp_path):
        """2026-09-01 顺序调整回归: 先执行昨日订单再扫描——昨日卖出成交
        释放名额发生在扫描前, 阶段 2 名额检查(实时读 positions)基于执行后
        持仓 → 当日候选直接下单, 不再出现"拦截+补仓"两步(大族 9-1 场景)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pool_pos = dict(self.STOCK_POOL)
            pool_pos["000005"] = {"name": "标的E", "shares": 100,
                                  "avg_cost": 10.0}   # 满仓 5 只
            _setup_common(monkeypatch, tmp, pool_pos=pool_pos)
            # 模拟执行昨日卖出成交(执行函数更新 positions → 名额释放)
            del pool_pos["000005"]
            results = [self._candidate("000006", "新候选", "new", dev=5.0)]
            daily_signal._place_deferred_orders(results, self._cfg(), {})
            # 直接下单, 不拦截
            assert results[0]["signal"] == "buy"
            assert "已达最大持仓数" not in results[0].get("reason", "")
            pend = orders.load_pending()
            assert len(pend) == 1
            assert pend[0]["symbol"] == "000006"

    def test_buy_qty_zero_marks_hold(self, monkeypatch):
        """股价过高买不起 1 手(buy_qty=0) → signal 改 hold, 不落盘 buy。"""
        with tempfile.TemporaryDirectory() as tmp:
            _setup_common(monkeypatch, tmp, signal="buy", target_pct=0.95,
                          last_close=500.0)  # 19万/500/100<1手
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(cash=100000.0, max_pct=0.20), 2,
                {"stock": 0.0})
            assert result["signal"] == "hold"
            assert result.get("_skipped") is True
            assert "资金不足" in result["reason"]
            assert orders.load_pending() == []


class TestPrefetchData:
    """并行预取(2026-08-24): 主循环前线程池下载缺失/过期数据。

    下载无副作用可并行; evaluate_stock 有下单副作用, 评估阶段保持串行。
    """

    def test_prefetch_downloads_missing(self, monkeypatch, tmp_path):
        """缓存缺失的标的全部触发下载。"""
        calls = []

        def _dl(sym, years, at="stock"):
            calls.append(sym)
            return pd.DataFrame()

        monkeypatch.setattr(daily_signal, "download_with_cache", _dl)
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(tmp_path / "nc"))
        n = daily_signal.prefetch_data(
            [{"symbol": "000001"}, {"symbol": "000002"}], 2, quiet=True)
        assert n == 2
        assert set(calls) == {"000001", "000002"}

    def test_prefetch_skips_fresh_cache(self, monkeypatch, tmp_path):
        """缓存已含今日数据 → 不下载。"""
        cache = tmp_path / "cache"
        cache.mkdir()
        df = pd.DataFrame({"date": [pd.Timestamp.now().normalize()],
                           "close": [1.0]})
        df.to_parquet(cache / "000001.parquet", index=False)
        calls = []
        monkeypatch.setattr(daily_signal, "download_with_cache",
                            lambda *a, **k: calls.append(1))
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(cache))
        n = daily_signal.prefetch_data([{"symbol": "000001"}], 2, quiet=True)
        assert n == 0 and calls == []

    def test_prefetch_stale_cache_downloads(self, monkeypatch, tmp_path):
        """缓存数据早于今日 → 触发下载(对齐 download_with_cache 判断)。"""
        cache = tmp_path / "cache"
        cache.mkdir()
        df = pd.DataFrame({"date": [pd.Timestamp.now().normalize()
                                    - pd.Timedelta(days=2)], "close": [1.0]})
        df.to_parquet(cache / "000001.parquet", index=False)
        calls = []
        monkeypatch.setattr(daily_signal, "download_with_cache",
                            lambda *a, **k: calls.append(a[0]))
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(cache))
        n = daily_signal.prefetch_data([{"symbol": "000001"}], 2, quiet=True)
        assert n == 1 and calls == ["000001"]

    def test_prefetch_failure_skipped(self, monkeypatch, tmp_path):
        """下载失败不抛错(主循环串行阶段重试/回退缓存)。"""
        def _fail(*a, **k):
            raise ValueError("模拟失败")

        monkeypatch.setattr(daily_signal, "download_with_cache", _fail)
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(tmp_path / "nc"))
        n = daily_signal.prefetch_data([{"symbol": "000001"}], 2, quiet=True)
        assert n == 1  # 尝试数(失败静默)

    def test_prefetch_is_parallel(self, monkeypatch, tmp_path):
        """并行生效: 4 只各耗时 0.3s → 总耗时显著小于串行 1.2s。"""
        import time

        def _slow(sym, years, at="stock"):
            time.sleep(0.3)
            return pd.DataFrame()

        monkeypatch.setattr(daily_signal, "download_with_cache", _slow)
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(tmp_path / "nc"))
        t0 = time.time()
        daily_signal.prefetch_data(
            [{"symbol": f"{i:06d}"} for i in range(4)], 2, quiet=True)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"并行未生效: 耗时 {elapsed:.2f}s (串行应 ~1.2s)"

    def test_no_pending_returns_zero(self, monkeypatch, tmp_path):
        """全部缓存新鲜 → 返回 0, 无下载。"""
        calls = []
        monkeypatch.setattr(daily_signal, "download_with_cache",
                            lambda *a, **k: calls.append(1))
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(tmp_path / "nc"))
        assert daily_signal.prefetch_data([], 2, quiet=True) == 0
        assert calls == []


class TestPlaceDeferredOrdersPool:
    """阶段2下单修复(2026-08-27): 股票/ETF 分池名额 + _block 原始 target。

    原 bug: ①placed_new 全局计数跨池串扰(股票下单挤占 ETF 名额);
    ②_block 存 capped_pct(0.19), _auto_fill_pool 再乘 max_pct 双重 cap
    (0.038) → 高价股补仓永远买不起 1 手、低价股补仓量减半。
    """

    def _deferred(self, sym, atype, dev, kind="new"):
        return {"_deferred": {"kind": kind, "symbol": sym, "name": sym,
                              "shares": 100, "price": 10.0,
                              "target_pct": 0.19, "raw_target": 0.95,
                              "reason": "x", "deviation": dev,
                              "asset_type": atype}}

    def _cfg(self, stock_max=2, etf_max=2):
        return {"stock": {"initial_cash": 100000, "max_positions": stock_max,
                          "single_position_pct": 0.2},
                "etf": {"initial_cash": 100000, "max_positions": etf_max,
                        "single_position_pct": 0.2}}

    def _mock_log(self, monkeypatch, tmp_path):
        """防污染(2026-08-27 教训): 隔离状态文件 + 拦截写入函数。"""
        monkeypatch.setattr(orders, "PENDING_FILE", str(tmp_path / "pending.json"))
        monkeypatch.setattr(daily_signal, "log_pending", lambda *a, **k: None)
        monkeypatch.setattr(daily_signal, "log_skipped", lambda *a, **k: None)
        monkeypatch.setattr(daily_signal, "add_pending_order",
                            lambda *a, **k: None)

    def test_pool_slots_independent(self, monkeypatch, tmp_path):
        """股票池下单不挤占 ETF 池名额: 股票满 2 + ETF 满 2 各自下单。"""
        self._mock_log(monkeypatch, tmp_path)
        monkeypatch.setattr(daily_signal, "load_positions", lambda: {})
        # 阶段 2 处理中 pending 会增长——mock 为空(不读真实 pending)
        monkeypatch.setattr(orders, "load_pending", lambda: [])
        results = [
            self._deferred("000001", "stock", 0.05),
            self._deferred("000002", "stock", 0.04),
            self._deferred("510050", "etf", 0.03),
            self._deferred("510300", "etf", 0.02),
        ]
        placed = []
        monkeypatch.setattr(daily_signal, "add_pending_order",
                            lambda *a, **k: placed.append(a[0]))
        daily_signal._place_deferred_orders(results, self._cfg(), {})
        assert set(placed) == {"000001", "000002", "510050", "510300"}
        # 全部下单(各池独立名额) — 修复前股票 2 单使 ETF 被拦

    def test_stock_exhausts_only_stock_slots(self, monkeypatch, tmp_path):
        """股票池满额只拦股票, ETF 不受影响。"""
        self._mock_log(monkeypatch, tmp_path)
        monkeypatch.setattr(daily_signal, "load_positions", lambda: {})
        monkeypatch.setattr(orders, "load_pending", lambda: [])
        results = [
            self._deferred("000001", "stock", 0.05),
            self._deferred("000002", "stock", 0.04),
            self._deferred("000003", "stock", 0.03),
            self._deferred("510050", "etf", 0.06),   # 偏离最高但 ETF 池
        ]
        placed = []
        monkeypatch.setattr(daily_signal, "add_pending_order",
                            lambda *a, **k: placed.append(a[0]))
        daily_signal._place_deferred_orders(results, self._cfg(stock_max=2, etf_max=1), {})
        # ETF 名额 1 → 510050 下单; 股票名额 2 → 前 2 只下单、000003 拦截
        assert set(placed) == {"000001", "000002", "510050"}
        blocked = [r for r in results if r.get("_skipped")]
        assert [b["_deferred"]["symbol"] for b in blocked] == ["000003"]

    def test_block_keeps_raw_target(self, monkeypatch, tmp_path):
        """_block 后 _target_pct 为原始(0.95), 非 capped(0.19)——双 cap 修复。"""
        self._mock_log(monkeypatch, tmp_path)
        monkeypatch.setattr(daily_signal, "load_positions", lambda: {})
        monkeypatch.setattr(orders, "load_pending", lambda: [])
        results = [self._deferred("000001", "stock", 0.05),
                   self._deferred("000002", "stock", 0.04)]
        daily_signal._place_deferred_orders(results, self._cfg(stock_max=1), {})
        blocked = [r for r in results if r.get("_skipped")]
        assert len(blocked) == 1
        assert blocked[0]["_target_pct"] == 0.95  # 原始 target, 供 _auto_fill_pool 乘 max_pct


class TestPendingNewSnapshot:
    """pool_pending_new 本轮前快照(2026-08-27): 本轮新下的单不重复计数。

    原实时读 pending → 本轮已下订单计入 pool_pending_new, 与 placed_new
    重复 → 名额减半(股票池 3 名额只下 2 单, 光智等被误拦)。
    """

    def test_same_round_orders_not_double_counted(self, monkeypatch, tmp_path):
        """本轮下 3 单(名额 3): 新下的单不占用 pool_pending_new。"""
        monkeypatch.setattr(daily_signal, "log_pending", lambda *a, **k: None)
        monkeypatch.setattr(daily_signal, "load_positions", lambda: {})
        monkeypatch.setattr(orders, "load_pending", lambda: [])  # 本轮前无 pending
        results = [
            {"_deferred": {"kind": "new", "symbol": s, "name": s,
                           "shares": 100, "price": 10.0, "target_pct": 0.19,
                           "raw_target": 0.95, "reason": "x",
                           "deviation": d, "asset_type": "stock"}}
            for s, d in [("000001", 0.05), ("000002", 0.04), ("000003", 0.03)]
        ]
        placed = []
        monkeypatch.setattr(daily_signal, "add_pending_order",
                            lambda *a, **k: placed.append(a[0]))
        daily_signal._place_deferred_orders(
            results, {"stock": {"initial_cash": 100000, "max_positions": 3,
                                "single_position_pct": 0.2}}, {})
        # 名额 3(无持仓无遗留 pending) → 3 单全部下单(修复前只下 2)
        assert set(placed) == {"000001", "000002", "000003"}
        assert not any(r.get("_skipped") for r in results)


class TestSellWithoutPositionDowngraded:
    """无持仓卖出信号降级 hold(2026-08-28 修复)。

    背景: 持仓在今日开盘已全部卖出, 收盘扫描又触发卖出信号——
    原逻辑 signal 保持 sell, 打印 🔴 卖出 仓位 0%, 但无持仓不生成
    订单, 展示误导(518880 黄金ETF 案例)。对齐 2026-08-12 买入侧
    "买不起 1 手降级 hold" 修复: 落盘/快照/状态检查不误显卖出。
    """

    def test_sell_no_position_downgraded(self, monkeypatch):
        """无持仓 + sell 信号 → 降级 hold, 不生成订单, 记 skipped(action=sell)。"""
        with tempfile.TemporaryDirectory() as tmp:
            calls = _setup_common(
                monkeypatch, tmp, signal="sell", target_pct=0.0,
                last_close=100.0, pos=None, pool_pos={},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert result["signal"] == "hold"
            assert result["target_pct"] == 0.0
            assert result["_skipped"] is True
            assert "卖出信号忽略" in result["reason"]
            assert calls["log_pending"] == []
            assert len(calls["log_skipped"]) == 1
            args, kwargs = calls["log_skipped"][0]
            assert kwargs.get("action") == "sell"

    def test_sell_with_position_still_queues(self, monkeypatch):
        """有持仓 + sell 信号 → 正常生成待执行订单(不回归)。"""
        with tempfile.TemporaryDirectory() as tmp:
            pos = {"shares": 100, "avg_cost": 100.0,
                   "first_buy_date": "2026-01-01", "name": "宁德时代"}
            calls = _setup_common(
                monkeypatch, tmp, signal="sell", target_pct=0.0,
                last_close=100.0, pos=pos, pool_pos={"300750": pos},
            )
            result = daily_signal.evaluate_stock(
                STOCK, _make_config(), 2, {"stock": 0.0},
            )
            assert result["signal"] == "sell"
            assert result["shares"] == 100
            assert calls["log_pending"] != []
            assert calls["log_skipped"] == []
