"""Tests for 无限仓位系统: 归一化收益 + 首笔买入价 + 最小单位。"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unlimited_report as ur


class TestLotSize:
    def test_star_board_200(self):
        assert ur._lot_size("688041") == 200
        assert ur._lot_size("688256") == 200

    def test_others_100(self):
        assert ur._lot_size("002594") == 100
        assert ur._lot_size("300308") == 100
        assert ur._lot_size("512500") == 100


class TestFirstBuyPrice:
    def test_from_elog_executed(self):
        elog = pd.DataFrame([
            {"symbol": "002594", "action": "buy", "status": "pending",
             "exec_price": ""},
            {"symbol": "002594", "action": "buy", "status": "executed",
             "exec_price": 95.2},
            {"symbol": "002594", "action": "buy", "status": "executed",
             "exec_price": 100.5},
        ])
        assert ur._first_buy_price("002594", elog) == 95.2  # 首笔

    def test_no_executed_returns_zero(self):
        elog = pd.DataFrame([
            {"symbol": "002594", "action": "buy", "status": "pending",
             "exec_price": ""},
        ])
        assert ur._first_buy_price("002594", elog) == 0.0


class TestNormalizedPnl:
    """归一化收益 = 总利润 / (3 × 最小单位 × 首笔买入价)。"""

    def _elog(self):
        return pd.DataFrame([
            {"symbol": "002594", "action": "buy", "status": "executed",
             "exec_price": 100.0},
        ])

    def test_held_position_floating(self):
        """持仓中: 总利润 = 浮动盈亏。"""
        positions = {"002594": {"name": "比亚迪", "shares": 300,
                                "avg_cost": 100.0}}
        trades = pd.DataFrame(columns=["symbol", "pnl"])
        elog = self._elog()
        prices = {"002594": 110.0}
        total, norm, realized, floating = ur._normalized_pnl(
            "002594", positions, trades, elog, prices)
        # 总资金 = 3×100×100 = 30000; 浮动 = (110-100)×300 = 3000
        assert total == pytest.approx(3000.0)
        assert norm == pytest.approx(0.10)
        assert realized == pytest.approx(0.0)
        assert floating == pytest.approx(3000.0)

    def test_realized_plus_floating(self):
        """已清仓(已实现) + 无持仓。"""
        positions = {}
        trades = pd.DataFrame([
            {"symbol": "002594", "pnl": -1500.0},
            {"symbol": "002594", "pnl": 500.0},
        ])
        elog = self._elog()
        total, norm, realized, floating = ur._normalized_pnl(
            "002594", positions, trades, elog, {})
        # 总利润 = -1500+500 = -1000; 总资金 30000 → -3.33%
        assert total == pytest.approx(-1000.0)
        assert norm == pytest.approx(-1000.0 / 30000.0)
        assert realized == pytest.approx(-1000.0)

    def test_no_first_buy_price_norm_zero(self):
        """无成交记录: 总资金为 0, 归一化收益为 0。"""
        positions = {}
        trades = pd.DataFrame(columns=["symbol", "pnl"])
        elog = pd.DataFrame(columns=["symbol", "action", "status",
                                     "exec_price"])
        total, norm, realized, floating = ur._normalized_pnl(
            "002594", positions, trades, elog, {})
        assert total == 0.0
        assert norm == 0.0

    def test_star_board_capital_uses_lot200(self):
        """科创板: 总资金 = 3 × 200 × 首笔价。"""
        positions = {"688041": {"name": "海光信息", "shares": 600,
                                "avg_cost": 100.0}}
        trades = pd.DataFrame(columns=["symbol", "pnl"])
        elog = pd.DataFrame([
            {"symbol": "688041", "action": "buy", "status": "executed",
             "exec_price": 100.0},
        ])
        prices = {"688041": 110.0}
        total, norm, _, _ = ur._normalized_pnl(
            "688041", positions, trades, elog, prices)
        # 总资金 = 3×200×100 = 60000; 浮动 = 10×600 = 6000 → 10%
        assert total == pytest.approx(6000.0)
        assert norm == pytest.approx(0.10)


class TestTradesTable:
    """已完成交易配对表(2026-08-25 新增, 对齐实盘口径)。"""

    def _trades(self):
        return pd.DataFrame([{
            "entry_date": "2026-08-21", "exit_date": "2026-08-25",
            "symbol": "300502", "name": "新易盛", "shares": 100,
            "entry_price": 445.15, "exit_price": 405.0, "pnl": -4068.0,
            "pnl_pct": -9.1, "fee": 53.05, "reason": "止损(-7.4%)",
        }, {
            "entry_date": "2026-08-21", "exit_date": "2026-08-25",
            "symbol": "300373", "name": "扬杰科技", "shares": 300,
            "entry_price": 94.58, "exit_price": 95.0, "pnl": 89.0,
            "pnl_pct": 0.3, "fee": 37.33, "reason": "价格回归(偏离-2.4%)",
        }])

    def test_table_structure(self):
        """配对表: 汇总(笔数/盈利/胜率/总盈亏) + 表格含盈亏/收益率/持有天数。"""
        trades = self._trades()
        html = ur._trades_table(trades, {}, pd.DataFrame())
        assert "共 2 笔" in html
        assert "盈利 1 笔" in html
        assert "胜率 50%" in html
        assert "¥-3,979" in html  # 总盈亏 -4068+89
        assert "-9.1%" in html and "+0.3%" in html
        assert "持有天数" in html
        assert "table-wrap" in html  # 滚动容器

    def test_empty(self):
        """无交易 → 占位文案。"""
        html = ur._trades_table(
            pd.DataFrame(), {}, pd.DataFrame())
        assert "暂无已完成交易" in html

    def test_name_fallback(self):
        """trades 无 name 列时从其他源取名称。"""
        trades = self._trades().drop(columns=["name"])
        elog = pd.DataFrame([{
            "symbol": "300502", "name": "新易盛", "action": "buy",
            "exec_date": "2026-08-21", "exec_price": 445.15,
            "shares": 100, "status": "executed",
        }])
        html = ur._trades_table(trades, {}, elog)
        assert "新易盛" in html


UNLIMITED_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
KALMAN_DIR = os.path.normpath(os.path.join(UNLIMITED_DIR, "..", "..", "kalman"))


class TestPeakStateIsolation:
    """P0 峰值状态(2026-10-09 与实盘同步)在本系统的隔离与接线。

    背景: 本系统复用实盘模块(daily_signal/portfolio/exec_log), 靠模块级路径重定向
    隔离状态。新增的峰值台账(portfolio.POSITIONS_FILE)与影子日志(daily_signal.LOG_DIR)
    也必须落在本目录 —— 漏了 LOG_DIR 会把影子行写进实盘 kalman/logs(串台)。
    """

    def test_state_files_all_redirected_to_unlimited_dir(self):
        """导入 unlimited_signal 后, 5 个状态路径全部指向无限仓位目录。

        用子进程导入: 该模块在 import 时会**全局**改写 daily_signal/portfolio 的
        路径常量, 在本进程内导入会污染其它测试。
        """
        import subprocess
        code = (
            "import unlimited_signal\n"
            "import daily_signal as ds, portfolio, orders, exec_log\n"
            "print(ds.SIGNALS_CSV)\n"
            "print(portfolio.POSITIONS_FILE)\n"
            "print(orders.PENDING_FILE)\n"
            "print(exec_log.EXEC_LOG)\n"
            "print(ds.LOG_DIR)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=UNLIMITED_DIR,
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-800:]
        paths = [p for p in out.stdout.strip().split("\n") if p.strip()]
        assert len(paths) == 5, paths
        for p in paths:
            assert "unlimited" in p, f"未重定向到无限仓位目录: {p}"
        assert any(p.rstrip("/").endswith("logs") for p in paths), (
            "ds.LOG_DIR 必须重定向(否则峰值影子日志写进实盘 kalman/logs)")

    def test_persist_functions_follow_module_paths(self, tmp_path, monkeypatch):
        """_persist_peaks / _persist_peak_shadow 只认模块级路径, 不写死 kalman。"""
        import json

        sys.path.insert(0, KALMAN_DIR)
        import daily_signal as ds
        import portfolio

        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps({
            "603893": {"name": "瑞芯微", "shares": 300, "avg_cost": 190.0,
                       "first_buy_date": "2026-09-18", "peak_close": 200.0},
        }), encoding="utf-8")
        monkeypatch.setattr(portfolio, "POSITIONS_FILE", str(pos_file))
        monkeypatch.setattr(ds, "LOG_DIR", str(tmp_path))

        n = ds._persist_peaks([{"symbol": "603893", "shares": 300, "peak_price": 223.01}])
        assert n == 1
        saved = json.loads(pos_file.read_text(encoding="utf-8"))
        assert saved["603893"]["peak_close"] == 223.01       # 只增不减, 已抬高

        rows = [{"symbol": "603893", "name": "瑞芯微", "date": "2026-10-08",
                 "shares": 300,
                 "_shadow": ds._peak_shadow(peak_prev=223.01, avg_cost=190.0,
                                            close=211.0,
                                            result={"signal": "hold", "reason": "持仓中"},
                                            params={"trailing_stop_pct": 0.04,
                                                    "stop_loss_pct": 0.05})}]
        n2 = ds._persist_peak_shadow(rows, "2026-10-08")
        assert n2 == 1                                        # 回撤 5.4% ≥ 4% → 会提前卖
        shadow_csv = tmp_path / "peak_shadow.csv"
        assert shadow_csv.exists()
        assert "603893" in shadow_csv.read_text(encoding="utf-8")

    def test_unlimited_yaml_has_switches(self):
        """开关必须显式存在于配置(默认 persist=false 保证交易行为不变)。"""
        import yaml
        cfg = yaml.safe_load(open(os.path.join(UNLIMITED_DIR, "unlimited.yaml"),
                                  encoding="utf-8"))
        st = cfg["strategy"]
        assert st.get("peak_persist_enabled") is False
        assert st.get("peak_shadow_enabled") is True

    def test_main_calls_peak_persistence(self):
        """护栏: unlimited_signal.main 必须调用峰值台账与影子对照(防"函数没接线")。

        用 AST 而非文本匹配 —— 注释里也出现了函数名, 文本匹配会误判通过。
        """
        import subprocess
        code = (
            "import ast, inspect, unlimited_signal\n"
            "src = inspect.getsource(unlimited_signal.main)\n"
            "tree = ast.parse(src)\n"
            "names = [n.func.attr for n in ast.walk(tree)\n"
            "         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]\n"
            "print('_persist_peaks' in names, '_persist_peak_shadow' in names)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=UNLIMITED_DIR,
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-800:]
        assert out.stdout.strip() == "True True", (
            f"unlimited_signal.main 未同时调用两个峰值函数: {out.stdout!r}")
