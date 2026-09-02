"""manage.py split 除权调整 + validate.py 除权监控的测试(2026-09-01)。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd
import pytest

import manage
import portfolio
import validate


def _setup_positions(monkeypatch, tmp_path, positions):
    """隔离 POSITIONS_FILE 并写入初始持仓。"""
    pos_file = tmp_path / "positions.json"
    pos_file.write_text(json.dumps(positions), encoding="utf-8")
    monkeypatch.setattr(portfolio, "POSITIONS_FILE", str(pos_file))
    monkeypatch.setattr(portfolio, "TRADES_FILE",
                        str(tmp_path / "trades.csv"))
    return pos_file


class TestCmdSplit:
    """manage.py split: 送转后 shares×ratio, avg_cost÷ratio, 总成本不变。"""

    def test_split_doubles_shares(self, monkeypatch, tmp_path):
        """10送10(ratio=2): 100股@20 → 200股@10, 总成本 2000 不变。"""
        _setup_positions(monkeypatch, tmp_path,
                         {"600001": {"name": "测试", "shares": 100,
                                     "avg_cost": 20.0}})
        manage.cmd_split("600001", 2.0)
        pos = portfolio.load_positions()["600001"]
        assert pos["shares"] == 200
        assert pos["avg_cost"] == pytest.approx(10.0)
        assert pos["shares"] * pos["avg_cost"] == pytest.approx(2000.0)

    def test_split_ratio_15(self, monkeypatch, tmp_path):
        """每10股送转5股(ratio=1.5): 100股@20 → 150股@13.333。"""
        _setup_positions(monkeypatch, tmp_path,
                         {"600001": {"name": "测试", "shares": 100,
                                     "avg_cost": 20.0}})
        manage.cmd_split("600001", 1.5)
        pos = portfolio.load_positions()["600001"]
        assert pos["shares"] == 150
        assert pos["avg_cost"] == pytest.approx(20.0 / 1.5)

    def test_split_not_in_positions(self, monkeypatch, tmp_path, capsys):
        """非持仓标的 → 提示且不报错。"""
        _setup_positions(monkeypatch, tmp_path, {})
        manage.cmd_split("600001", 2.0)
        assert "不在持仓中" in capsys.readouterr().out

    def test_split_invalid_ratio(self, monkeypatch, tmp_path, capsys):
        """ratio 越界(0/负数/>10) → 拒绝。"""
        _setup_positions(monkeypatch, tmp_path,
                         {"600001": {"name": "测试", "shares": 100,
                                     "avg_cost": 20.0}})
        for bad in (0, -1, 11):
            manage.cmd_split("600001", bad)
        out = capsys.readouterr().out
        assert out.count("必须在") == 3

    def test_split_rounds_shares(self, monkeypatch, tmp_path):
        """股数取整(101股×1.5 → 151股), 总成本微量偏差可接受。"""
        _setup_positions(monkeypatch, tmp_path,
                         {"600001": {"name": "测试", "shares": 101,
                                     "avg_cost": 20.0}})
        manage.cmd_split("600001", 1.5)
        pos = portfolio.load_positions()["600001"]
        assert pos["shares"] == 151
        assert pos["avg_cost"] == pytest.approx(20.0 / 1.5)


class TestCheckExRights:
    """validate.check_ex_rights: 持仓除权除息预警。"""

    def _mock_ak(self, monkeypatch, rows):
        """rows: [{除权除息日, 方案进度, 送转股份-送转总比例}]

        check_ex_rights 函数内 `import akshare as ak` 取 sys.modules 中的
        模块对象 → 直接 patch akshare 模块属性即可拦截。"""
        import akshare
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        monkeypatch.setattr(akshare, "stock_fhps_detail_em",
                            lambda symbol: df)

    def test_future_ex_rights_warns(self, monkeypatch, tmp_path):
        """3 天后除权 → 预警(提示 split 倍数)。"""
        self._mock_ak(monkeypatch, [{
            "除权除息日": (pd.Timestamp.now() + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
            "方案进度": "实施分配", "送转股份-送转总比例": 20.0,
        }])
        pos = {"600001": {"name": "测试", "shares": 100}}
        issues = validate.check_ex_rights(pos)
        assert len(issues) == 1
        assert issues[0]["check"] == "除权除息"
        assert issues[0]["level"] == "warn"
        assert "split" in issues[0]["detail"]
        assert "3.0" in issues[0]["detail"]  # 倍数 1+20/10

    def test_past_ex_rights_urgent(self, monkeypatch, tmp_path):
        """已除权 1 天(持仓未调整)→ 紧急提示。"""
        self._mock_ak(monkeypatch, [{
            "除权除息日": (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "方案进度": "实施分配", "送转股份-送转总比例": None,
        }])
        issues = validate.check_ex_rights({"600001": {"name": "测试", "shares": 100}})
        assert len(issues) == 1
        assert "已除权" in issues[0]["detail"]

    def test_cash_dividend_no_split(self, monkeypatch, tmp_path):
        """纯现金分红(送转为空)→ 提示无需 split(2026-09-02 德福科技案例修复)。

        原实现一律提示"执行 split", 对纯分红误导。"""
        self._mock_ak(monkeypatch, [{
            "除权除息日": (pd.Timestamp.now() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            "方案进度": "实施分配", "送转股份-送转总比例": None,
            "现金分红-现金分红比例": 1.0,
            "现金分红-现金分红比例描述": "10派1.00元(含税)",
        }])
        issues = validate.check_ex_rights({"600001": {"name": "测试", "shares": 100}})
        assert len(issues) == 1
        assert "无需 split" in issues[0]["detail"]
        assert "10派1.00元" in issues[0]["detail"]
        assert "split" not in issues[0]["detail"].replace("无需 split", "")

    def test_far_future_no_warn(self, monkeypatch, tmp_path):
        """30 天后除权 → 不在窗口, 无预警。"""
        self._mock_ak(monkeypatch, [{
            "除权除息日": (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            "方案进度": "实施分配", "送转股份-送转总比例": 10.0,
        }])
        assert validate.check_ex_rights({"600001": {"name": "测试"}}) == []

    def test_api_failure_silent(self, monkeypatch, tmp_path):
        """接口异常 → 静默跳过(不阻断主流程)。"""
        import akshare
        def boom(symbol):
            raise RuntimeError("network down")
        monkeypatch.setattr(akshare, "stock_fhps_detail_em", boom)
        assert validate.check_ex_rights({"600001": {"name": "测试"}}) == []

    def test_no_positions_no_check(self, monkeypatch, tmp_path):
        """无持仓 → 无检查(不会触发网络)。"""
        import akshare
        calls = {"n": 0}
        def spy(symbol):
            calls["n"] += 1
            return pd.DataFrame()
        monkeypatch.setattr(akshare, "stock_fhps_detail_em", spy)
        assert validate.check_ex_rights({}) == []
        assert calls["n"] == 0
