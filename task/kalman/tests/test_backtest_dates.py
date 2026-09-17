"""Tests for 回测结束日期默认值(2026-09-18): 默认 = **回测当天**。

背景: CLI(`main.py`)与优化器(`optimize.py`)的 `--end` 默认值原本是
写死的 `"20241231"` —— 不显式传参会**静默截断到 2024 年底**(组合回测
因此少跑近两年数据); `run_portfolio_backtest` 的默认则是写死的未来
日期 `"20261231"`。统一改为"回测当天"(实际右端仍取 .catalog 末根 bar)。
"""

import inspect
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backtest                                                    # noqa: E402
import main                                                        # noqa: E402
import optimize                                                    # noqa: E402


def _today_or_yesterday():
    """容忍测试跨零点执行(模块常量在 import 时求值)。"""
    now = datetime.now()
    return {now.strftime("%Y%m%d"),
            (now - timedelta(days=1)).strftime("%Y%m%d")}


class TestEndDateDefaults:
    def test_resolve_none_is_today(self):
        assert backtest._resolve_end_date(None) in _today_or_yesterday()

    def test_resolve_explicit_passthrough(self):
        assert backtest._resolve_end_date("20240101") == "20240101"

    def test_cli_default_is_today_not_hardcoded(self):
        assert main.END_DEFAULT in _today_or_yesterday()
        assert optimize.END_DEFAULT in _today_or_yesterday()
        assert main.END_DEFAULT != "20241231", "CLI 默认仍是写死的旧日期"
        assert optimize.END_DEFAULT != "20241231"

    def test_portfolio_signature_default_is_none(self):
        sig = inspect.signature(backtest.run_portfolio_backtest)
        assert sig.parameters["end_date"].default is None

    def test_cli_parser_end_default(self, monkeypatch):
        """端到端: 不传 --end 时解析出的就是当天。"""
        monkeypatch.setattr(sys, "argv", ["main.py"])
        args = main.parse_args()
        assert args.end in _today_or_yesterday()

    def test_cli_parser_explicit_end_wins(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["main.py", "--end", "20240101"])
        assert main.parse_args().end == "20240101"
