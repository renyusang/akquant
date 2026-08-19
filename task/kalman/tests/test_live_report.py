"""Tests for live_report.py 权益曲线/月度收益。

背景(2026-08-13):
1. 权益曲线历史点原用"当前价"估算持仓市值——7 月等历史月份收益
   随每日价格漂移(7月 +0.4% 隔日变 -0.2%)。
   修复: 历史点用当日收盘价(price_history), 缺失回退当前价。
"""

import json
import os
import re
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


class TestMonthlyHeatmap:
    """月度热力图(2026-08-19 修复): 表格须包横向滚动容器。

    14 列(年份+12月+全年)固有宽度 ~880px, 未包 _wrap_table 时
    手机上被 body overflow-x:clip 裁剪, 5 月后月份及全年列不可访问。
    """

    def _eq(self):
        return pd.Series(
            [100000.0, 102000.0, 98000.0],
            index=pd.to_datetime(["2026-07-20", "2026-07-31", "2026-08-29"]),
        )

    def test_wrapped_in_scroll_container(self):
        """输出以 tscroll 容器包裹, 表格在容器内部(与其他 7 张表一致)。"""
        html = live_report._monthly_heatmap(self._eq(), pd.DataFrame(), 100000.0)
        assert "class='tscroll'" in html or 'class="tscroll"' in html
        # 容器在前、表格在后(表格是容器的子元素)
        assert html.find("tscroll") < html.find("<table")

    def test_scroll_style_has_touch_scrolling(self):
        """容器带 overflow-x:auto + iOS 惯性滚动。"""
        html = live_report._monthly_heatmap(self._eq(), pd.DataFrame(), 100000.0)
        assert "overflow-x:auto" in html
        assert "-webkit-overflow-scrolling:touch" in html

    def test_content_unchanged_inside_wrapper(self):
        """包裹不改变表格内容: 年份行/月份列/全年列/热力单元格齐全。"""
        html = live_report._monthly_heatmap(self._eq(), pd.DataFrame(), 100000.0)
        assert "<th>年份</th>" in html
        assert "<th>7月</th>" in html and "<th>12月</th>" in html
        assert "<th>全年</th>" in html
        assert 'class="heat-cell"' in html
        # 7 月 +2.0% / 8 月 -3.9%: 值仍正确渲染
        assert "+2.0%" in html
        assert "-3.9%" in html
        # 全年列 = 各月之和
        assert "-1.9%" in html

    def test_empty_equity_returns_empty(self):
        """无月度数据 → 空字符串(边界; eq 实际总是 DatetimeIndex)。"""
        out = live_report._monthly_heatmap(
            pd.Series([], index=pd.to_datetime([]), dtype=float),
            pd.DataFrame(), 100000.0)
        assert out == ""

    def test_single_month_renders(self):
        """仅 1 个月数据也能渲染(实盘运行初期场景)。"""
        eq = pd.Series([100000.0, 105000.0],
                       index=pd.to_datetime(["2026-07-20", "2026-07-31"]))
        html = live_report._monthly_heatmap(eq, pd.DataFrame(), 100000.0)
        assert "+5.0%" in html
        assert "tscroll" in html


class TestMobileAdaptation:
    """移动端适配(2026-08-19): P2 图高自适应 + P3 内层滚动取消。"""

    def test_chart_box_attrs(self):
        """chart-box 携带 data-dh/data-mh, 内部图表 html 原样保留。"""
        box = live_report._chart_box("<div id='c'>chart</div>", 800, 560)
        assert "class='chart-box'" in box
        assert "data-dh='800'" in box
        assert "data-mh='560'" in box
        assert "<div id='c'>chart</div>" in box

    def test_chart_box_empty_passthrough(self):
        """空图表(无月度数据时 chart2 为空串)不产生空容器。"""
        assert live_report._chart_box("", 300, 260) == ""

    def test_adaptive_js_guards(self):
        """自适应脚本: CDN 失败守卫 + 竖/横屏双条件 + relayout + resize 防抖。"""
        js = live_report._ADAPTIVE_CHART_JS
        assert "typeof Plotly === 'undefined'" in js
        assert "Plotly.relayout" in js
        assert "(max-width:600px)" in js   # 竖屏手机
        assert "(max-height:500px)" in js  # 横屏手机(宽>600 但高≤500)
        assert "data-mh" in js and "data-dh" in js
        assert "addEventListener('resize'" in js

    def test_css_vscroll_rules(self):
        """vscroll: 桌面 600px 滚动盒; ≤600px 取消(避免三层嵌套滚动)。"""
        css = live_report._css()
        assert ".vscroll{max-height:600px;overflow-y:auto" in css
        assert "@media (max-width:600px){.vscroll{max-height:none" in css


class TestBuildReportSmoke:
    """端到端冒烟: 临时目录造数据跑完整 build_live_report, 验证生成的
    HTML 含全部移动端适配要素(不碰生产数据文件, 沪深300强制离线)。"""

    def _prepare(self, monkeypatch, tmp_path):
        (tmp_path / "stocks.yaml").write_text(
            "stock:\n  initial_cash: 200000\n"
            "etf:\n  initial_cash: 100000\n"
            "watchlist:\n  stocks:\n  - symbol: '000001'\n    name: 平安银行\n"
            "  etfs: []\n", encoding="utf-8")
        (tmp_path / "positions.json").write_text(json.dumps({
            "000001": {"name": "平安银行", "shares": 100,
                       "avg_cost": 10.0, "first_buy_date": "2026-08-01"}}),
            encoding="utf-8")
        (tmp_path / "trades.csv").write_text(
            "entry_date,exit_date,symbol,name,shares,entry_price,exit_price,"
            "pnl,pnl_pct,fee,reason\n"
            "2026-07-01,2026-07-15,000001,平安银行,100,9.0,10.0,95,10.6,5,x\n",
            encoding="utf-8")
        (tmp_path / "execution_log.csv").write_text(
            "signal_date,exec_date,symbol,name,action,target_pct,shares,"
            "signal_reason,exec_price,status,reason\n"
            "2026-07-31,2026-08-01,000001,平安银行,buy,0.19,100,价格突破,"
            "10.0,executed,\n", encoding="utf-8")
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        import data_utils
        import orders
        # orders.load_pending 读模块级 PENDING_FILE, 重定向到临时目录
        monkeypatch.setattr(orders, "PENDING_FILE",
                            str(tmp_path / "pending_orders.json"))

        def _offline(*_a, **_k):
            raise RuntimeError("offline test")

        monkeypatch.setattr(data_utils, "fetch_hs300", _offline)
        return tmp_path / "live_report.html"

    def test_report_contains_mobile_adaptations(self, monkeypatch, tmp_path):
        out = self._prepare(monkeypatch, tmp_path)
        live_report.build_live_report()
        html = out.read_text(encoding="utf-8")

        # P2: 两个图表容器(主图 800→560, 月度 300→260) + 自适应脚本注入
        assert "class='chart-box' data-dh='800' data-mh='560'" in html
        assert "class='chart-box' data-dh='300' data-mh='260'" in html
        assert "Plotly.relayout" in html

        # P3: 交易区滚动盒改为类(桌面 CSS 控制, 手机取消); 内联样式已移除
        assert html.count("class='vscroll'") == 2
        # 内联滚动样式已移除(注意: CSS 类规则中合法存在同子串, 只查内联形式)
        assert "style='max-height:600px" not in html

        # P1 回归: 热力图仍在 tscroll 滚动容器内
        m = re.search(r"<summary><h2>月度收益</h2></summary>(.*?)</details>",
                      html, re.S)
        assert m and "tscroll" in m.group(1)

        # 基本内容未丢: 指标卡片 / 持仓表 / 交易表 / 免责声明
        assert "class='mcard'" in html
        assert "pos-table" in html
        assert "trades-table" in html
        assert "免责声明" in html

    def test_empty_monthly_chart_no_box(self, monkeypatch, tmp_path):
        """无月度数据时 chart2 为空 → 不产生空 chart-box 容器(边界)。"""
        out = self._prepare(monkeypatch, tmp_path)
        # 交易日集中在同一个月内且当月无月末数据 → 权益点均在 7 月,
        # 月度收益仅 1 行(首月), chart2 仍渲染; 再造极端: 无任何事件
        (tmp_path / "execution_log.csv").write_text(
            "signal_date,exec_date,symbol,name,action,target_pct,shares,"
            "signal_reason,exec_price,status,reason\n"
            "2026-07-31,2026-07-31,000001,平安银行,buy,0.19,100,价格突破,"
            "10.0,pending,\n", encoding="utf-8")
        (tmp_path / "trades.csv").write_text(
            "entry_date,exit_date,symbol,name,shares,entry_price,exit_price,"
            "pnl,pnl_pct,fee,reason\n", encoding="utf-8")
        live_report.build_live_report()
        html = out.read_text(encoding="utf-8")
        # 无成交买入 + 无卖出 → 仅剩当前持仓的事件: 至少主图容器存在,
        # 且空串图表不会被包出只有容器的空 div
        assert "class='chart-box' data-dh='800' data-mh='560'" in html
        # chart-box 数量 == 实际渲染的 plotly 图 div 数(空图表不包;
        # 自适应脚本里也引用了 .plotly-graph-div 字符串, 用 class= 精确计数)
        n_box = html.count("class='chart-box'")
        n_plot = html.count('class="plotly-graph-div"')
        assert n_box == n_plot
