"""Tests for 订单 tag 归因 (entry_tag / exit_tag).

2026-09-28 新增: 此前 `portfolio_trades.csv` 的 `entry_tag`/`exit_tag` **全为空**,
无法按"入场原因 / 出场规则"统计交易(浮盈保护研究需要它)。
本文件锁定:
1. 每笔交易都带入场/出场归因;
2. 出场 tag 来自引擎 reason(出场规则可枚举), 入场 tag 区分 建仓/加仓;
3. 加 tag 不改变交易集合与盈亏(与无 tag 版本逐笔一致)。
"""

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.disable(logging.CRITICAL)

import strategy as strategy_mod  # noqa: E402
from backtest import run_kalman_backtest  # noqa: E402

CATALOG = os.path.join(os.path.dirname(__file__), "..", ".catalog")

BASE_PARAMS = dict(
    initial_cash=100000,
    trend_filter_enabled=True,
    entry_threshold=0.02,
    exit_threshold=0.005,
    stop_loss_pct=0.05,
)

ENTRY_PREFIXES = ("入场:", "加仓:")
# 引擎 _evaluate_exit 可能产出的出场规则词表(kalman.md 信号优先级)
EXIT_RULES = (
    "价格回归", "峰值回撤止盈", "止损", "速度反转",
    "目标利润止盈", "吊灯止盈", "ADX衰减止盈", "SAR跟踪止损",
    "时间止盈", "均线止盈",
)


@pytest.fixture(scope="module")
def df_002594():
    """比亚迪 2020-2021 前 600 根(资金/仓位参数与生产池口径一致)。"""
    path = os.path.join(CATALOG, "002594", "data.parquet")
    if not os.path.exists(path):
        pytest.skip("catalog 数据不存在")
    df = pd.read_parquet(path).iloc[:600].reset_index()
    df["symbol"] = "002594"
    return df


def _bt(df, **overrides):
    params = dict(
        BASE_PARAMS,
        trend_bear_position_pct=0.3,
        downtrend_entry_threshold=0.03,
        single_position_pct=1.0,
        max_positions=5,
        **overrides,
    )
    return run_kalman_backtest(df, symbol="002594", strategy_params=params,
                               show_progress=False)


class TestOrderTags:
    """入场/出场归因覆盖全部交易。"""

    def test_all_trades_have_tags(self, df_002594):
        """100% 交易带入场与出场 tag(修复前两列全空)。"""
        r = _bt(df_002594)
        t = r.trades_df
        assert not t.empty
        assert t.entry_tag.notna().all() and (t.entry_tag.astype(str) != "").all()
        assert t.exit_tag.notna().all() and (t.exit_tag.astype(str) != "").all()

    def test_entry_tags_distinguish_entry_and_add(self, df_002594):
        """入场 tag 以 入场:/加仓: 开头(可区分建仓与趋势翻转加仓)。"""
        t = _bt(df_002594).trades_df
        assert t.entry_tag.map(lambda s: str(s).startswith(ENTRY_PREFIXES)).all()

    def test_exit_tags_are_known_rules(self, df_002594):
        """出场 tag 以 出场: 开头, 且规则名落在引擎词表内(可枚举统计)。"""
        t = _bt(df_002594).trades_df
        assert t.exit_tag.map(lambda s: str(s).startswith("出场:")).all()
        unknown = [
            s for s in t.exit_tag.astype(str)
            if not any(rule in s for rule in EXIT_RULES)
        ]
        assert not unknown, f"出现词表外的出场规则: {sorted(set(unknown))[:5]}"

    def test_tags_do_not_change_trades(self, df_002594, monkeypatch):
        """加 tag 不改变交易集合与盈亏: 与"剥离 tag"版本逐笔一致。

        做法: 把 close_position 的 tag 参数丢弃、order_target_value 的 tag kwarg 丢弃,
        其余完全不变, 两次回测的 trades 应逐笔相同。
        """
        base = _bt(df_002594)

        orig_close = strategy_mod.KalmanStrategy.close_position

        def close_no_tag(self, symbol=None, tag=None):
            return orig_close(self, symbol, tag=None)

        monkeypatch.setattr(strategy_mod.KalmanStrategy, "close_position", close_no_tag)
        stripped = _bt(df_002594)
        monkeypatch.undo()

        cols = ["entry_time", "exit_time", "entry_price", "exit_price",
                "quantity", "pnl", "net_pnl"]
        pd.testing.assert_frame_equal(
            base.trades_df[cols].reset_index(drop=True),
            stripped.trades_df[cols].reset_index(drop=True),
        )
        assert base.trades_df.exit_tag.astype(str).ne("").any()
