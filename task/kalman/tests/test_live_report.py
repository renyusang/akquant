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
        # 现金 91000 - 买入费5.09(佣金5+过户0.09) + 1000股×当日收盘10
        assert day20 == pytest.approx(100994.91)
        # 若误用当前价会是 105,994.91
        assert day20 != pytest.approx(105994.91)

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
        assert day20 == pytest.approx(105994.91)  # 91000 - 5.09 + 1000×15

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

    def test_pool_monthly_sum_equals_total(self, monkeypatch, tmp_path):
        """逐日重估修复(2026-08-31): 基金池最后事件 7-30、股票池 7-31 有事件,
        原曲线月末点错位(基金缺 7-31 浮动重估)导致月度"股票+基金"≠总体;
        逐日重估后各池月末点同日, 金额池和 = 总体(差值 < 1 元)。"""
        rows = [
            {"signal_date": "2026-07-29", "exec_date": "2026-07-30",
             "symbol": "510000", "name": "测试ETF", "action": "buy",
             "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
             "exec_price": 1.0, "status": "executed", "reason": ""},
            {"signal_date": "2026-07-30", "exec_date": "2026-07-31",
             "symbol": "000001", "name": "测试股", "action": "buy",
             "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
             "exec_price": 1.0, "status": "executed", "reason": ""},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "execution_log.csv", index=False)
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        trades = pd.DataFrame(columns=["exit_date", "symbol", "shares", "exit_price"])
        price_history = {
            "510000": {"2026-07-30": 1.0, "2026-07-31": 1.05},
            "000001": {"2026-07-31": 1.0},
        }
        price_map = {"510000": 1.05, "000001": 1.0}

        def _eq(pool, cash):
            df = live_report._build_equity_curve(
                trades, {}, price_map, cash, price_history, pool=pool)
            df["date"] = pd.to_datetime(df["date"])
            return df.groupby("date")["equity"].last()

        eq_all = _eq(None, 200000.0)
        eq_stock = _eq("stock", 100000.0)
        eq_etf = _eq("etf", 100000.0)
        # 修复核心: 基金池曲线含 7-31 点(逐日重估, 修复前只有事件日 7-30)
        assert pd.Timestamp("2026-07-31") in eq_etf.index
        m_all = live_report._monthly_returns(eq_all, 200000.0)
        m_stock = live_report._monthly_returns(eq_stock, 100000.0)
        m_etf = live_report._monthly_returns(eq_etf, 100000.0)
        d = pd.Timestamp("2026-07-31")
        diff = (m_stock.loc[d, "amount"] + m_etf.loc[d, "amount"]
                - m_all.loc[d, "amount"])
        assert abs(diff) < 1.0


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
        # 全年列 = 复合年收益(2026-08-25 起, 非算术和): 1.02×0.961-1 = -2.0%,
        # 并显示金额(各月 amount 之和 = 2000 + (-2000) = 0)
        assert "-2.0%" in html
        assert "¥-2,000" in html

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

    def test_pool_rows_added(self):
        """传分池权益序列 → 每年渲染总体/股票/基金三行, 各池独立基准。

        股票池 7 月: 309000/300000-1 = +3.0%; 基金池 7 月: 98000/100000-1 = -2.0%;
        总体 7 月: 102000/100000-1 = +2.0% —— 三值互异可区分。"""
        idx = pd.to_datetime(["2026-07-20", "2026-07-31", "2026-08-29"])
        eq = pd.Series([100000.0, 102000.0, 98000.0], index=idx)
        eq_stock = pd.Series([300000.0, 309000.0, 300000.0], index=idx)
        eq_etf = pd.Series([100000.0, 98000.0, 103000.0], index=idx)
        html = live_report._monthly_heatmap(
            eq, pd.DataFrame(), 100000.0,
            eq_stock, 300000.0, eq_etf, 100000.0)
        assert "总体" in html and "股票" in html and "基金" in html
        assert "+2.0%" in html   # 总体 7 月
        assert "+3.0%" in html   # 股票池 7 月
        assert "-2.0%" in html   # 基金池 7 月
        # 全年列独立复合: 股票 300000/300000-1 = 0.0%, 基金 103000/100000-1 = +3.0%
        assert "0.0%" in html
        assert "+3.0%" in html

    def test_pool_rows_backward_compatible(self):
        """不传分池序列 → 仅"总体"行(与旧行为一致)。"""
        html = live_report._monthly_heatmap(self._eq(), pd.DataFrame(), 100000.0)
        assert "总体" in html
        assert "股票" not in html and "基金" not in html


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
        assert "class='chart-box' data-dh='320' data-mh='280'" in html
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


class TestMarketEnvCards:
    """环境状态栏(2026-09-04): up 占比分级 + 变化。"""

    def _write_signals(self, monkeypatch, tmp_path, rows):
        """rows: [(date, symbol, trend)]"""
        df = pd.DataFrame(rows, columns=["date", "symbol", "trend"])
        df.to_csv(tmp_path / "signals.csv", index=False)
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))

    def test_weak_env_when_low_up_ratio(self, monkeypatch, tmp_path):
        """up 11/49(22%) → 弱势(env-weak), 显示占比与分池。"""
        rows = [("2026-09-03", f"{600000+i:06d}", "down") for i in range(38)]
        rows += [("2026-09-03", f"6000{i:02d}", "up") for i in range(11)]
        rows += [("2026-09-02", "600001", "up"), ("2026-09-02", "600002", "down")]
        self._write_signals(monkeypatch, tmp_path, rows)
        html = live_report._market_env_cards()
        assert "env-weak" in html
        assert "22% 弱势" in html
        assert "股票池up" in html

    def test_strong_env_when_high_up_ratio(self, monkeypatch, tmp_path):
        """up 占比高 → 强势(env-strong), 较前日变化显示。"""
        rows = [("2026-09-03", f"600{i:04d}", "up") for i in range(40)]
        rows += [("2026-09-03", "510000", "down")]
        rows += [("2026-09-02", "600001", "down"), ("2026-09-02", "600002", "down")]
        self._write_signals(monkeypatch, tmp_path, rows)
        html = live_report._market_env_cards()
        assert "env-strong" in html
        assert "较前日" in html

    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        """无 signals.csv → 空串(不影响报告)。"""
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        assert live_report._market_env_cards() == ""

    def test_single_day_no_delta(self, monkeypatch, tmp_path):
        """仅一天数据 → 无"较前日"显示。"""
        self._write_signals(monkeypatch, tmp_path,
                            [("2026-09-03", "600001", "up")])
        html = live_report._market_env_cards()
        assert "较前日" not in html


class TestSkippedBuys:
    """被跳过买入候选(2026-08-21 修复): watchlist/近5日过滤 + 跳过当日快照。

    原实现三处误导: 无 watchlist 过滤(旧池标的永久显示)、无日期过滤
    (数周前跳过永久显示)、日期/现价混用 signals.csv 最新快照(看似
    "昨日被跳过", 实为几周前)。manage.py cmd_show 复用同一实现。
    """

    ELOG_COLS = ["signal_date", "exec_date", "symbol", "name", "action",
                 "target_pct", "shares", "signal_reason", "exec_price",
                 "status", "reason"]
    SIG_COLS = ["date", "symbol", "name", "close", "kalman_price",
                "kalman_velocity", "ma20", "ma20_rising", "trend",
                "signal", "target_pct", "reason"]

    def _d(self, days_ago):
        from datetime import datetime, timedelta
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    def _setup(self, monkeypatch, tmp_path, skips, sig_rows, positions=None):
        """skips: [(date, symbol, reason)]; sig_rows: [(date, symbol, close, kalman, trend)]"""
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        elog = pd.DataFrame([{
            "signal_date": d, "exec_date": "", "symbol": s, "name": s,
            "action": "buy", "target_pct": 0.19, "shares": 100,
            "signal_reason": "价格突破", "exec_price": "",
            "status": "skipped", "reason": r,
        } for d, s, r in skips], columns=self.ELOG_COLS)
        elog.to_csv(tmp_path / "execution_log.csv", index=False)
        sig = pd.DataFrame([{
            "date": d, "symbol": s, "name": s, "close": c,
            "kalman_price": k, "kalman_velocity": 0.0, "ma20": k,
            "ma20_rising": True, "trend": t, "signal": "hold",
            "target_pct": 0.19, "reason": "已达最大持仓数",
        } for d, s, c, k, t in sig_rows], columns=self.SIG_COLS)
        sig.to_csv(tmp_path / "signals.csv", index=False)
        (tmp_path / "positions.json").write_text(
            json.dumps(positions or {}), encoding="utf-8")

    def test_recent_skip_shows_skip_date_and_snapshot(self, monkeypatch, tmp_path):
        """昨日跳过: 日期=跳过日, 价格=跳过当日快照。"""
        y, old = self._d(1), self._d(9)
        self._setup(monkeypatch, tmp_path,
                    skips=[(y, "000001", "已达最大持仓数(5)")],
                    sig_rows=[(old, "000001", 8.0, 8.5, "up"),
                              (y, "000001", 10.0, 9.7, "up")])
        out = live_report._get_skipped_buys([], {"000001": "测试"})
        assert len(out) == 1
        assert out[0]["signal_date"] == y   # 跳过日, 而非最新信号日
        assert out[0]["close"] == 10.0      # 跳过当日快照价
        assert out[0]["deviation"] == pytest.approx((10.0 / 9.7 - 1) * 100)

    def test_old_skip_filtered(self, monkeypatch, tmp_path):
        """10 天前的跳过已失效(新买入需新信号) → 不显示。"""
        d10 = self._d(10)
        self._setup(monkeypatch, tmp_path,
                    skips=[(d10, "000001", "已达最大持仓数(5)")],
                    sig_rows=[(d10, "000001", 10.0, 9.7, "up")])
        assert live_report._get_skipped_buys([], {"000001": "测试"}) == []

    def test_cross_weekend_skip_filtered(self, monkeypatch, tmp_path):
        """跨周末(2026-08-31 修复): 2 天前(周四)的跳过在周一报告不显示——
        与 _auto_fill_pool 仅当日候选的执行语义对齐; 候选若仍有效,
        当日重新触发 buy 会生成新的当日 skipped 记录, 不丢失(长川 8-27 案例)。"""
        d2 = self._d(2)
        self._setup(monkeypatch, tmp_path,
                    skips=[(d2, "000001", "已达最大持仓数(5)")],
                    sig_rows=[(d2, "000001", 10.0, 9.7, "up"),
                              (self._d(0), "000001", 9.0, 8.5, "down")])
        assert live_report._get_skipped_buys([], {"000001": "测试"}) == []

    def test_same_day_skip_shown(self, monkeypatch, tmp_path):
        """当日被跳过 → 显示(1 天窗口下仅当日候选有效)。"""
        today = self._d(0)
        self._setup(monkeypatch, tmp_path,
                    skips=[(today, "000001", "已达最大持仓数(5)")],
                    sig_rows=[(today, "000001", 10.0, 9.7, "up")])
        out = live_report._get_skipped_buys([], {"000001": "测试"})
        assert len(out) == 1
        assert out[0]["signal_date"] == today

    def test_removed_from_watchlist_filtered(self, monkeypatch, tmp_path):
        """不在当前 watchlist(旧池标的, 永不复扫) → 不显示, 即使跳过很近。"""
        y = self._d(1)
        self._setup(monkeypatch, tmp_path,
                    skips=[(y, "000002", "已达最大持仓数(5)")],
                    sig_rows=[(y, "000002", 10.0, 9.7, "up")])
        assert live_report._get_skipped_buys([], {"000001": "在池"}) == []

    def test_held_and_pending_excluded(self, monkeypatch, tmp_path):
        """已持仓 / 已有待执行订单的标的不列入候选。"""
        y = self._d(1)
        rows = [(y, s, 10.0, 9.7, "up") for s in ("000001", "000002", "000003")]
        self._setup(monkeypatch, tmp_path,
                    skips=[(y, s, "x") for s in ("000001", "000002", "000003")],
                    sig_rows=rows,
                    positions={"000001": {"shares": 100, "avg_cost": 9}})
        out = live_report._get_skipped_buys(
            [{"symbol": "000002", "action": "buy"}],
            {"000001": "a", "000002": "b", "000003": "c"})
        assert [s["symbol"] for s in out] == ["000003"]

    def test_negative_deviation_excluded(self, monkeypatch, tmp_path):
        """跳过当日偏离为负(实际是卖出方向) → 不列入待买入。"""
        y = self._d(1)
        self._setup(monkeypatch, tmp_path,
                    skips=[(y, "000001", "x")],
                    sig_rows=[(y, "000001", 9.0, 9.7, "up")])
        assert live_report._get_skipped_buys([], {"000001": "a"}) == []

    def test_sort_by_date_then_deviation(self, monkeypatch, tmp_path):
        """排序(2026-09-01): 最近日期在前, 同日期内偏离度降序。

        构造: 今天两个(偏离 13.4% / 1.0%) + 2 天前一个(偏离 3.1%)。
        预期: 今13.4% → 今1.0% → 2天前(日期倒序优先, 旧日期即使偏离
        更大也排后)。"""
        today, d2 = self._d(0), self._d(2)
        self._setup(monkeypatch, tmp_path,
                    skips=[(d2, "000003", "x"), (today, "000001", "x"),
                           (today, "000002", "x")],
                    sig_rows=[(d2, "000003", 10.0, 9.7, "up"),
                              (today, "000001", 10.0, 9.9, "up"),
                              (today, "000002", 11.0, 9.7, "up")])
        out = live_report._get_skipped_buys(
            [], {"000001": "a", "000002": "b", "000003": "c"}, within_days=10)
        assert [s["symbol"] for s in out] == ["000002", "000001", "000003"]
        # 同日期内按偏离降序(000002 偏离 > 000001)
        assert out[0]["deviation"] > out[1]["deviation"]

    def test_latest_skip_wins(self, monkeypatch, tmp_path):
        """同一标的多次跳过 → 取最新一次记录。"""
        d3, d1 = self._d(3), self._d(1)
        self._setup(monkeypatch, tmp_path,
                    skips=[(d3, "000001", "旧原因"), (d1, "000001", "新原因")],
                    sig_rows=[(d3, "000001", 10.0, 9.7, "up"),
                              (d1, "000001", 11.0, 9.7, "up")])
        out = live_report._get_skipped_buys([], {"000001": "a"})
        assert len(out) == 1
        assert out[0]["signal_date"] == d1
        assert out[0]["close"] == 11.0

    def test_skip_day_snapshot_not_latest(self, monkeypatch, tmp_path):
        """跳过后又有更新信号(已转跌) → 仍显示跳过当日快照(时间点一致)。"""
        d1, today = self._d(1), self._d(0)
        self._setup(monkeypatch, tmp_path,
                    skips=[(d1, "000001", "x")],
                    sig_rows=[(d1, "000001", 10.0, 9.7, "up"),
                              (today, "000001", 8.0, 8.5, "down")])
        out = live_report._get_skipped_buys([], {"000001": "a"})
        assert len(out) == 1
        assert out[0]["close"] == 10.0
        assert out[0]["trend"] == "up"

    def test_no_skip_day_signal_excluded(self, monkeypatch, tmp_path):
        """跳过当日无信号快照(无法还原当日状态) → 不显示。"""
        y, d2 = self._d(1), self._d(2)
        self._setup(monkeypatch, tmp_path,
                    skips=[(y, "000001", "x")],
                    sig_rows=[(d2, "000001", 10.0, 9.7, "up")])
        assert live_report._get_skipped_buys([], {"000001": "a"}) == []

    def test_missing_files_returns_empty(self, monkeypatch, tmp_path):
        """无 execution_log/signals 文件 → 空列表(边界)。"""
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        assert live_report._get_skipped_buys([], {"000001": "a"}) == []

    def test_manage_reuses_shared_helper(self):
        """manage.py 复用 live_report 的规范实现(消除内联复制)。"""
        import manage
        assert manage._get_skipped_buys is live_report._get_skipped_buys


class TestEquityCurvePool:
    """分池权益曲线(2026-08-21): pool 参数按代码前缀过滤事件与持仓。

    权益走势图叠加股票池/基金池单独曲线: 各自从本池初始资金起算,
    只含本池买卖事件, 最终快照只统计本池持仓。
    注意: 历史点无 price_history 时回退当前价(price_map), 与合并曲线同口径。
    """

    ROWS = [
        {"signal_date": "2026-07-01", "exec_date": "2026-07-02",
         "symbol": "000001", "name": "测试股", "action": "buy",
         "target_pct": 0.19, "shares": 100, "signal_reason": "x",
         "exec_price": 10.0, "status": "executed", "reason": ""},
        {"signal_date": "2026-07-01", "exec_date": "2026-07-02",
         "symbol": "510050", "name": "测试ETF", "action": "buy",
         "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
         "exec_price": 1.0, "status": "executed", "reason": ""},
        {"signal_date": "2026-07-03", "exec_date": "2026-07-04",
         "symbol": "000001", "name": "测试股", "action": "sell",
         "target_pct": 0.0, "shares": 100, "signal_reason": "x",
         "exec_price": 11.0, "status": "executed", "reason": ""},
    ]

    def _setup(self, monkeypatch, tmp_path):
        pd.DataFrame(self.ROWS).to_csv(tmp_path / "execution_log.csv", index=False)
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        return pd.DataFrame([{
            "entry_date": "2026-07-01", "exit_date": "2026-07-04",
            "symbol": "000001", "name": "测试股", "shares": 100,
            "entry_price": 10.0, "exit_price": 11.0, "pnl": 100.0,
            "pnl_pct": 10.0, "fee": 0.0, "reason": "x",
        }])

    POS = {"000001": {"name": "a", "shares": 100, "avg_cost": 10.0501},
           "510050": {"name": "b", "shares": 1000, "avg_cost": 1.00501}}
    PM = {"000001": 12.0, "510050": 1.2}

    def test_stock_pool_only(self, monkeypatch, tmp_path):
        """股票池: 只含股票事件, 初始=股票池现金, ETF 事件不进入。"""
        trades = self._setup(monkeypatch, tmp_path)
        raw = live_report._build_equity_curve(
            trades, self.POS, self.PM, 300000.0, pool="stock")
        assert raw["equity"].iloc[0] == pytest.approx(300000.0)
        eq = raw.groupby("date")["equity"].last()
        # 7-02: 现金298,994.99 + 000001持仓100×当前价12 = 300,194.99(ETF事件被过滤)
        assert eq["2026-07-02"] == pytest.approx(300194.99)
        # 7-04 卖出后: equity = 初始 + realized(pnl=100) + 0 持仓 = 300,100
        assert eq["2026-07-04"] == pytest.approx(300100.0)
        # 最终快照: 300,000 + 100 + 持仓市值1200 - 含费成本1005.01 = 300,294.99
        assert eq.iloc[-1] == pytest.approx(300294.99)

    def test_etf_pool_only(self, monkeypatch, tmp_path):
        """基金池: 只含 ETF 事件, 股票事件不进入; 无 ETF 卖出日无行。"""
        trades = self._setup(monkeypatch, tmp_path)
        raw = live_report._build_equity_curve(
            trades, self.POS, self.PM, 100000.0, pool="etf")
        assert raw["equity"].iloc[0] == pytest.approx(100000.0)
        eq = raw.groupby("date")["equity"].last()
        # 7-02: 初始 + 市值1200 - 含费成本1005.01 = 100,194.99
        assert eq["2026-07-02"] == pytest.approx(100194.99)
        assert "2026-07-04" not in eq.index  # 无 ETF 事件
        # 最终快照: 100,000 + 0 + 市值1200 - 含费成本1005.01 = 100,194.99
        assert eq.iloc[-1] == pytest.approx(100194.99)

    def test_merged_includes_both(self, monkeypatch, tmp_path):
        """pool=None(合并): 两池事件都计入, 初始=总初始。"""
        trades = self._setup(monkeypatch, tmp_path)
        raw = live_report._build_equity_curve(
            trades, self.POS, self.PM, 400000.0)
        assert raw["equity"].iloc[0] == pytest.approx(400000.0)
        eq = raw.groupby("date")["equity"].last()
        assert eq["2026-07-02"] == pytest.approx(400389.98)  # 双池事件(含2笔买入费)
        assert eq["2026-07-04"] == pytest.approx(400294.99)  # realized=100
        # 最终快照: 400,000 + 100 + 市值2400 - 含费成本2010.02 = 400,489.98
        assert eq.iloc[-1] == pytest.approx(400489.98)


class TestBuildReportPoolTraces:
    """分池权益独立成图(2026-08-21): 股票池/基金池各自权益+回撤图,
    不并入总权益图(col2 并排, 手机自动堆叠)。"""

    def test_pool_fig_has_equity_and_drawdown(self):
        """分池图: 2 行子图(权益+回撤), 权益起点=池初始资金, 回撤从 cummax 算。"""
        import pandas as pd
        eq = pd.Series([300000.0, 306000.0, 297000.0],
                       index=pd.to_datetime(
                           ["2026-08-19", "2026-08-20", "2026-08-21"]))
        fig = live_report._build_pool_fig(eq, "stock", "股票池")
        assert len(fig.data) == 2  # 权益 + 回撤
        # 权益线: 起点/终点正确
        assert fig.data[0].y[0] == 300000.0
        assert fig.data[0].y[-1] == 297000.0
        # 回撤线: 峰值后回撤 = (297000-306000)/306000*100
        assert fig.data[1].y[0] == 0.0
        assert fig.data[1].y[-1] == pytest.approx((297000 - 306000) / 306000 * 100)
        # 子图标题
        assert "股票池权益" in fig.layout.annotations[0].text

    def test_report_has_separate_pool_charts(self, monkeypatch, tmp_path):
        """生成的 HTML: 主图不含股票池线; 股票/基金独立图各含权益+回撤。"""
        TestBuildReportSmoke()._prepare(monkeypatch, tmp_path)
        live_report.build_live_report()
        html = (tmp_path / "live_report.html").read_text(encoding="utf-8")
        decoded = html.encode("utf-8").decode("unicode_escape", errors="ignore")
        # 独立图子图标题存在(2 张分池图 × 权益+回撤)
        assert "股票池权益" in decoded
        assert "基金池权益" in decoded
        assert "回撤(%)" in decoded
        # col2 布局容器存在(两分池图并排)
        assert "class='col2'" in html or 'class="col2"' in html
        # chart-box 共 4 个: 主图 800/560 + 股票池 480/400 + 基金池 480/400
        # + 月度图 320/280(冒烟数据含月度数据)
        assert html.count("class='chart-box'") == 4
        assert html.count("data-dh='480' data-mh='400'") == 2
        assert "data-dh='320' data-mh='280'" in html


class TestEquityCurveFees:
    """权益曲线扣手续费(2026-08-25): 与卡片总盈亏口径一致。

    原裸价重建不含费 → 曲线终点比卡片(已实现+浮动)高 ~764 元,
    月度收益/全年列与总盈亏口径不一致(7月-1.5%+8月+1.7%算术和+0.2%
    显示正, 总盈亏却是负)。
    """

    def test_buy_fee_deducted(self, monkeypatch, tmp_path):
        """买入事件扣手续费: 曲线终点 = 初始 - 买入(含费) + 市值。"""
        rows = [{
            "signal_date": "2026-07-01", "exec_date": "2026-07-02",
            "symbol": "000001", "name": "测试", "action": "buy",
            "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
            "exec_price": 10.0, "status": "executed", "reason": "",
        }]
        pd.DataFrame(rows).to_csv(tmp_path / "execution_log.csv", index=False)
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        trades = pd.DataFrame(columns=["exit_date", "symbol", "shares", "exit_price"])
        eq = live_report._build_equity_curve(
            trades, {}, {"000001": 10.0}, 100000.0)
        # 含费: 佣金5 + 过户0.1 = 5.1 → 现金 89994.9; 事件点市值 10000
        # → 7-02 事件点 = 99994.9(最终快照用 positions 参数, 空则无市值)
        day = eq[eq["date"] == "2026-07-02"]["equity"].iloc[-1]
        assert day == pytest.approx(99994.9)

    def test_sell_fee_deducted(self, monkeypatch, tmp_path):
        """卖出后 equity = 初始 + trades pnl(权威含费, 不再重算卖出费)。"""
        rows = [
            {"signal_date": "2026-07-01", "exec_date": "2026-07-02",
             "symbol": "000001", "name": "a", "action": "buy",
             "target_pct": 0.19, "shares": 1000, "signal_reason": "x",
             "exec_price": 10.0, "status": "executed", "reason": ""},
            {"signal_date": "2026-07-03", "exec_date": "2026-07-04",
             "symbol": "000001", "name": "a", "action": "sell",
             "target_pct": 0.0, "shares": 1000, "signal_reason": "x",
             "exec_price": 11.0, "status": "executed", "reason": ""},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "execution_log.csv", index=False)
        monkeypatch.setattr(live_report, "TASK_DIR", str(tmp_path))
        # pnl = (11-10)×1000 - 卖出费16.11 = 983.89(含费权威)
        trades = pd.DataFrame([{
            "entry_date": "2026-07-01", "exit_date": "2026-07-04",
            "symbol": "000001", "name": "a", "shares": 1000,
            "entry_price": 10.0, "exit_price": 11.0, "pnl": 983.89,
            "pnl_pct": 9.8, "fee": 16.11, "reason": "x",
        }])
        eq = live_report._build_equity_curve(
            trades, {}, {"000001": 11.0}, 100000.0)
        # 卖出后无持仓: equity = 100,000 + 983.89
        assert eq["equity"].iloc[-1] == pytest.approx(100983.89)

    def test_yearly_compound_not_sum(self):
        """全年列 = 复合年收益(2026-08-25 修复, 原为各月 pct 算术和)。

        7 月 -1.5%、8 月 +1.42% → 算术和 -0.08%, 复合 -0.098%——
        复合 = 区间真实收益, 与算术和不同(修复核心)。"""
        eq = pd.Series(
            [394000.0, 399608.0],
            index=pd.to_datetime(["2026-07-31", "2026-08-31"]))
        m = live_report._monthly_returns(eq, 400000.0)
        f = 1.0
        for _, r in m.iterrows():
            f *= (1 + r["pct"] / 100)
        comp = (f - 1) * 100
        arith = m["pct"].sum()
        # 复合 = 区间真实收益
        assert comp == pytest.approx((399608.0 / 400000.0 - 1) * 100)
        # 复合与算术和不同(修复点)
        assert abs(comp - arith) > 0.01


