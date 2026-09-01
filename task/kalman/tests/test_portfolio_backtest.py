"""portfolio_backtest 组合回测与实盘语义对齐(2026-09-01 修复)的测试。

覆盖 6 项修复:
① 涨停/跌停基准 = 前交易日收盘(信号日), 非执行日收盘
② 手续费: 买入 avg_cost 含费、卖出 pnl 减费(portfolio.calc_fee)
③ 资金分池: 股票/ETF 池现金独立
④ 名额检查: pending 排除卖单与已持仓标的补仓单
⑤ 补仓模拟: hold+target_pct>0 生成 refill 订单, 成交合并持仓
⑥ T+1 信号日过滤(防御)

运行: /home/renyu/miniconda3/envs/akquant_032/bin/python -m pytest tests/test_portfolio_backtest.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

import portfolio_backtest as pb


# ---------------------------------------------------------------------------
# 合成数据与脚本化引擎
# ---------------------------------------------------------------------------
def _make_ohlcv(dates, closes, opens=None):
    """合成 OHLCV: open 可单独指定(测试涨停/正常成交)。"""
    opens = opens if opens is not None else closes
    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": 1000000,
    }).set_index("date")


class _ScriptedEngine:
    """按 update 调用序号返回脚本化信号。

    所有标的数据日期对齐时 update 同步步进(calls 序号一致):
    1..warmup_days 为预热(结果被忽略), warmup 之后为信号期。
    script: {call_no: (signal, target_pct)}, 缺失返回 ("hold", 0.0)。
    """

    def __init__(self, script, **p):
        self.script = script
        self.calls = 0

    def update(self, c, ma20_c, ma20_p, high, low, volume):
        self.calls += 1
        sig, tgt = self.script.get(self.calls, ("hold", 0.0))
        return {"signal": sig, "target_pct": tgt, "reason": "scripted",
                "kalman_price": ma20_c, "trend": "up"}

    def set_position(self, has, cost=0.0):
        pass


def _make_data(days=60, prices=None, opens=None):
    """构造 (all_data, dates, warmup_days)。prices: {sym: closes}; opens: {sym: opens}。"""
    dates = pd.bdate_range("2026-01-05", periods=days)
    all_data = {}
    for sym, closes in (prices or {"600001": [10.0] * days}).items():
        all_data[sym] = _make_ohlcv(dates, closes,
                                    opens.get(sym) if opens else None)
    dates_list = sorted(set().union(*[set(d.index) for d in all_data.values()]))
    return all_data, dates_list


def _cfg(stock_cash=200000.0, etf_cash=100000.0, max_pos=5):
    return {
        "strategy": {},
        "stock": {"initial_cash": stock_cash, "max_positions": max_pos,
                  "single_position_pct": 0.20},
        "etf": {"initial_cash": etf_cash, "max_positions": max_pos,
                "single_position_pct": 0.20},
    }



def _script(mapping):
    """信号期第 k 个信号日的脚本: _script({1: (signal, target), ...})。

    数据 60 天、warmup 50 天, 且 warmup 内 idx>=20 才调 update(20 天不调)
    → 每标的 warmup 计数 30 次, 信号期首次 update 为第 31 次调用(calls=31)。
    """
    return {30 + k: v for k, v in mapping.items()}


def _falling_prices(days=60):
    """51 天 10 元 → 之后 5 元(建仓后腰斩, 触发补仓); 执行日开盘同价。"""
    dates = pd.bdate_range("2026-01-05", periods=days)
    closes = [10.0] * 51 + [5.0] * (days - 51)
    opens = [10.0] * 52 + [5.0] * (days - 52)
    return _make_ohlcv(dates, closes, opens), dates


def _run(monkeypatch, all_data, dates, cfg, script=None):
    """monkeypatch SignalEngine 后跑 _run_backtest_core(配置直接透传)。"""
    script = script if script is not None else _script({1: ("buy", 0.95)})

    def factory(**p):
        return _ScriptedEngine(script)

    monkeypatch.setattr(pb, "SignalEngine", factory)
    scfg = pb._pool_config(cfg, "stock")
    ecfg = pb._pool_config(cfg, "etf")
    return pb._run_backtest_core(all_data, dates, cfg, scfg, ecfg)


# ===========================================================================
# 纯函数: 涨停/跌停拦截(修复 ①)
# ===========================================================================
class TestLimitBlocked:
    def test_buy_blocked_on_limit_up(self):
        """买入: open ≥ 昨收×1.1×0.999 → 拦截(主板)。"""
        assert pb._limit_blocked("buy", 11.0, 10.0, "600001") is True
        assert pb._limit_blocked("buy", 10.98, 10.0, "600001") is False  # 容差内放行

    def test_sell_blocked_on_limit_down(self):
        """卖出: open ≤ 昨收×0.9×1.001 → 拦截。"""
        assert pb._limit_blocked("sell", 9.0, 10.0, "600001") is True
        assert pb._limit_blocked("sell", 9.02, 10.0, "600001") is False

    def test_kcb_20pct_limit(self):
        """科创板/创业板 20% 涨跌停。"""
        assert pb._limit_blocked("buy", 11.99, 10.0, "688001") is True
        assert pb._limit_blocked("buy", 11.5, 10.0, "300001") is False
        assert pb._limit_blocked("buy", 11.5, 10.0, "600001") is True  # 主板 10%

    def test_prev_close_of_previous_day(self):
        """前交易日收盘(非执行日当天)。"""
        df = pd.DataFrame({"close": [9.0, 10.0, 10.5]})
        assert pb._prev_close_of(df, 2, 8.0) == 10.0
        assert pb._prev_close_of(df, 0, 8.0) == 8.0  # 首日兜底


# ===========================================================================
# 纯函数: 合并持仓与补仓数量(修复 ⑤)
# ===========================================================================
class TestMergeAndRefill:
    def test_merge_position_weighted(self):
        """加权均价合并, 保留首次买入日。"""
        old = {"shares": 100, "avg_cost": 10.0, "first_buy": "2026-01-05"}
        merged = pb._merge_position(old, 300, 12.0)
        assert merged["shares"] == 400
        assert merged["avg_cost"] == pytest.approx((1000 + 3600) / 400)
        assert merged["first_buy"] == "2026-01-05"

    def test_refill_qty_formula(self):
        """target 30 万×0.95×0.2=57000; 现持 1000 市值 → 补至整手。"""
        qty = pb._refill_qty("600001", 100, 10.0, 0.95, 300000.0, 0.20, 100000.0)
        assert qty == int((57000 - 1000) / 10 / 100) * 100

    def test_refill_zero_when_target_met(self):
        """current×1.05 ≥ target → 不补。"""
        assert pb._refill_qty("600001", 6000, 10.0, 0.95, 300000.0, 0.20,
                              100000.0) == 0

    def test_refill_zero_when_no_cash(self):
        """池现金不足 → 不补。"""
        assert pb._refill_qty("600001", 100, 10.0, 0.95, 300000.0, 0.20,
                              500.0) == 0

    def test_refill_kcb_lot_200(self):
        """科创板补仓按 200 股整手。"""
        qty = pb._refill_qty("688001", 200, 100.0, 0.95, 300000.0, 0.20,
                             100000.0)
        assert qty % 200 == 0


# ===========================================================================
# core: 手续费(修复 ②)与分池(修复 ③)
# ===========================================================================
class TestCoreFeesAndPool:
    def test_buy_fee_in_avg_cost_and_cash(self, monkeypatch):
        """买入: 现金扣 cost+fee, avg_cost 含费(对齐实盘 daily_signal:1111)。"""
        all_data, dates = _make_data()
        core = _run(monkeypatch, all_data, dates, _cfg(), _script({1: ("buy", 0.95)}))
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert len(buys) == 1
        row = buys.iloc[0]
        # 3800 股 @10(19% 目标仓位): 佣金 38000×0.0003=11.4 + 过户 0.38 = 11.78
        assert row["shares"] == 3800
        assert row["fee"] == pytest.approx(11.78)
        assert row["cost"] == pytest.approx((38000 + 11.78) / 3800)
        # 现金: 股票池 200000 - 38011.78(ETF 池未动)
        assert core["final_cash"] == pytest.approx(200000 - 38011.78 + 100000)

    def test_sell_fee_in_pnl_and_cash(self, monkeypatch):
        """卖出: pnl 减费(佣金+印花税+过户), 现金加金额-fee。"""
        # 第 51 天 buy(100 股 @10), 第 52 天 sell @11
        all_data, dates = _make_data()
        script = _script({1: ("buy", 0.95), 2: ("sell", 0.0)})
        core = _run(monkeypatch, all_data, dates, _cfg(), script)
        sells = core["trades_df"][core["trades_df"]["action"] == "sell"]
        assert len(sells) == 1
        row = sells.iloc[0]
        # 买入费 11.78 → avg_cost 10.0031; 卖出 3800 股 @10(数据全 10 元):
        # fee = 佣金 11.4 + 印花 38 + 过户 0.38 = 49.78
        fee_sell = pytest.approx(11.4 + 38 + 0.38)
        assert row["fee"] == fee_sell
        assert row["pnl"] == pytest.approx((10 - 10.0031) * 3800 - 49.78)
        # 现金: 初始 30 万 - 买入 38011.78 + 卖出 38000 - 49.78
        assert core["final_cash"] == pytest.approx(300000 - 38011.78 + 38000 - 49.78)

    def test_pool_cash_independent(self, monkeypatch):
        """分池: 股票买入只扣股票池, ETF 池现金不受影响(修复 ③)。"""
        all_data, dates = _make_data(
            prices={"600001": [10.0] * 60, "510001": [1.0] * 60})
        core = _run(monkeypatch, all_data, dates, _cfg(),
                    _script({1: ("buy", 0.95)}))
        # 股票 3800 股@10(股票池), ETF 19000 股@1(ETF 池)
        # 费用: 股票 11.78; ETF 佣金 5.7 + 过户 0.19 = 5.89
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert len(buys) == 2
        # 最终现金 = 股票池 200000-38011.78 + ETF 池 100000-19005.89
        assert core["final_cash"] == pytest.approx(200000 - 38011.78 + 100000 - 19005.89)

    def test_equity_cash_is_pool_sum(self, monkeypatch):
        """权益曲线 cash = 两池现金之和(修复 ③)。"""
        all_data, dates = _make_data(
            prices={"600001": [10.0] * 60, "510001": [1.0] * 60})
        core = _run(monkeypatch, all_data, dates, _cfg(),
                    _script({1: ("buy", 0.95)}))
        last = core["equity_df"].iloc[-1]
        assert last["cash"] == pytest.approx(core["final_cash"])


# ===========================================================================
# core: 名额口径(修复 ④)与补仓(修复 ⑤)
# ===========================================================================
class TestCoreSlotsAndRefill:
    def test_slot_limit_blocks_second(self, monkeypatch):
        """max_positions=1: 第 2 只标的被名额拦截(修复 ④ 名额生效)。"""
        all_data, dates = _make_data(
            prices={"600001": [10.0] * 60, "600002": [10.0] * 60})
        core = _run(monkeypatch, all_data, dates, _cfg(max_pos=1),
                    _script({1: ("buy", 0.95)}))
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert len(buys) == 1
        assert buys.iloc[0]["symbol"] == "600001"

    def test_refill_merges_position(self, monkeypatch):
        """补仓成交合并持仓(加权均价, 修复 ⑤)。

        10 万池: 51 天信号 buy(close=10) → 1900 股订单; 52 天跌至 5 元
        → 52 天成交 @5 建仓 1900 股, 52 天 hold+target(close=5, current=9500
        < 19000×1.05) → refill 1900 股; 53 天 refill 成交, 合并 3800 股。"""
        df, dates = _falling_prices()
        all_data = {"600001": df}
        script = _script({1: ("buy", 0.95), 2: ("hold", 0.95), 3: ("hold", 0.95), 4: ("hold", 0.0)})
        core = _run(monkeypatch, all_data, dates, _cfg(stock_cash=100000.0), script)
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert len(buys) == 2
        merged = core["final_positions"]["600001"]
        assert merged["shares"] == 1900 + 1900
        # 建仓 @10: avg1 = (19000+5.89)/1900 = 10.0031(费: 佣金 5.7+过户 0.19)
        avg1 = (1900 * 10.0 + 5.89) / 1900
        # refill @5: avg2 = (9500+5.095)/1900(费: 佣金 max(2.85,5)=5+过户 0.095)
        avg2 = (1900 * 5.0 + 5.095) / 1900
        assert merged["avg_cost"] == pytest.approx((1900 * avg1 + 1900 * avg2) / 3800)

    def test_refill_not_consume_slot(self, monkeypatch):
        """补仓不占新增名额(修复 ④⑤): max_pos=1 时 A 补仓正常, B 新仓被拦。"""
        df, dates = _falling_prices()
        all_data = {"600001": df, "600002": df.copy()}
        script = _script({1: ("buy", 0.95), 2: ("hold", 0.95)})
        core = _run(monkeypatch, all_data, dates, _cfg(max_pos=1), script)
        # A: 建仓 + 补仓成交; B: 全程被名额拦(仅 A 的 2 笔买入)
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert set(buys["symbol"]) == {"600001"}
        assert len(buys) == 2

    def test_sell_replaces_blocked_refill(self, monkeypatch):
        """refill 单被拦截期间出现 sell → 移除 refill 仅保留 sell(修复 ⑤C)。

        52 天 refill 生成; 53 天 open=11.2(涨停)拦截 refill, 同日 sell 信号
        清理 refill 单 → 仅初始 1900 股被卖出(若 refill 成交则卖出 3800)。"""
        days = 60
        dates = pd.bdate_range("2026-01-05", periods=days)
        closes = [10.0] * 51 + [5.0] * (days - 51)   # 51 天起跌至 5
        opens = [10.0] * 52 + [11.2, 5.0] + [5.0] * (days - 54)  # 53 天涨停开盘
        all_data = {"600001": _make_ohlcv(dates, closes, opens)}
        dates_list = sorted(all_data["600001"].index)
        script = _script({1: ("buy", 0.95), 2: ("hold", 0.95), 3: ("sell", 0.0)})
        core = _run(monkeypatch, all_data, dates_list, _cfg(stock_cash=100000.0), script)
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        sells = core["trades_df"][core["trades_df"]["action"] == "sell"]
        assert len(buys) == 1       # refill 被清理, 未成交
        assert len(sells) == 1
        assert sells.iloc[0]["shares"] == 1900


# ===========================================================================
# core: 涨停拦截(修复 ①)与 T+1(修复 ⑥)与恒等式
# ===========================================================================
class TestCoreLimitAndTiming:
    def test_limit_up_blocks_then_executes(self, monkeypatch):
        """高开涨停(open≥昨收×1.1)买入拦截, 次日回落成交(修复 ①)。"""
        days = 60
        dates = pd.bdate_range("2026-01-05", periods=days)
        closes = [10.0] * days
        opens = [10.0] * days
        # 第一个信号日 = 第 51 天(索引 50), 执行日 = 第 52 天(索引 51)
        opens[51] = 11.2   # 涨停开盘(昨收 10 × 1.1)
        closes[51] = 10.5
        opens[52] = 10.5   # 次日回落成交
        all_data = {"600001": _make_ohlcv(dates, closes, opens)}
        dates_list = sorted(all_data["600001"].index)
        core = _run(monkeypatch, all_data, dates_list, _cfg(),
                    _script({1: ("buy", 0.95)}))
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert len(buys) == 1
        assert buys.iloc[0]["date"].date() == dates[52].date()  # 第 53 个工作日成交
        assert buys.iloc[0]["price"] == 10.5

    def test_signal_date_today_deferred(self, monkeypatch):
        """T+1: 信号日(第 51 天)不成交, 次日开盘成交(修复 ⑥ 防御)。"""
        all_data, dates = _make_data()
        core = _run(monkeypatch, all_data, dates, _cfg(),
                    _script({1: ("buy", 0.95)}))
        buys = core["trades_df"][core["trades_df"]["action"] == "buy"]
        assert len(buys) == 1
        assert buys.iloc[0]["date"].date() == dates[51].date()  # 第 52 个工作日

    def test_equity_identity(self, monkeypatch):
        """端到端恒等式: 最终权益 = 两池现金 + 持仓市值(修复 ②③ 口径守恒)。"""
        all_data, dates = _make_data(
            prices={"600001": [10.0] * 60, "510001": [1.0] * 60})
        script = _script({1: ("buy", 0.95), 2: ("hold", 0.95), 10: ("sell", 0.0)})
        core = _run(monkeypatch, all_data, dates, _cfg(), script)
        pos_mkt = sum(
            p["shares"] * float(all_data[s]["close"].iloc[-1])
            for s, p in core["final_positions"].items())
        assert core["final_cash"] + pos_mkt == pytest.approx(core["final_equity"])
        # trades 记录含 fee 字段
        assert "fee" in core["trades_df"].columns

    def test_no_position_no_sell(self, monkeypatch):
        """恒 sell 且无持仓 → 无卖出交易(防御)。"""
        all_data, dates = _make_data()
        core = _run(monkeypatch, all_data, dates, _cfg(),
                    _script({1: ("sell", 0.0)}))
        sells = (core["trades_df"][core["trades_df"]["action"] == "sell"]
                 if "action" in core["trades_df"].columns else core["trades_df"])
        assert len(sells) == 0
