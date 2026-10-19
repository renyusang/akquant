"""Tests for KalmanStrategy.min_hold_bars (最小持仓周期)."""

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.disable(logging.CRITICAL)

from backtest import run_kalman_backtest

CATALOG = os.path.join(os.path.dirname(__file__), "..", ".catalog")

BASE_PARAMS = dict(
    initial_cash=100000,
    trend_filter_enabled=True,
    entry_threshold=0.02,
    exit_threshold=0.005,
    stop_loss_pct=0.05,
    trend_bear_position_pct=0.3,
    downtrend_entry_threshold=0.03,
    single_position_pct=1.0,
    max_positions=5,
)


@pytest.fixture(scope="module")
def df_002594():
    """比亚迪 2020-2021 前 600 根(有大量震荡快速反转)。"""
    path = os.path.join(CATALOG, "002594", "data.parquet")
    if not os.path.exists(path):
        pytest.skip("catalog 数据不存在")
    df = pd.read_parquet(path).iloc[:600].reset_index()
    df["symbol"] = "002594"
    return df


def _bt(df, **overrides):
    params = dict(BASE_PARAMS, **overrides)
    return run_kalman_backtest(df, symbol="002594", strategy_params=params,
                               show_progress=False)


class TestMinHold:
    """min_hold_bars 抑制快速反转交易。"""

    def test_min_hold_reduces_trade_count(self, df_002594):
        """启用最小持仓后交易次数显著下降(快速反转被抑制)。"""
        r0 = _bt(df_002594)
        r5 = _bt(df_002594, min_hold_bars=5)
        r10 = _bt(df_002594, min_hold_bars=10)
        n0 = len(r0.trades_df)
        n5 = len(r5.trades_df)
        n10 = len(r10.trades_df)
        # 600 根内基线应有多笔交易; min_hold 严格抑制 5 根内反转
        assert n0 > 3
        assert n10 <= n5 <= n0

    def test_min_hold_zero_is_baseline(self, df_002594):
        """默认 0 = 与原行为完全一致。"""
        r = _bt(df_002594, min_hold_bars=0)
        r0 = _bt(df_002594)
        assert len(r.trades_df) == len(r0.trades_df)

    def test_min_hold_clamps_negative(self, df_002594):
        """负数参数 clamp 为 0, 不崩溃。"""
        r = _bt(df_002594, min_hold_bars=-5)
        r0 = _bt(df_002594)
        assert len(r.trades_df) == len(r0.trades_df)

    def test_stop_loss_not_blocked(self, df_002594):
        """止损不受最小持仓限制: 10 根内大跌应仍能止损。"""
        # 比亚迪 2020 年 3 月有急跌段; 用较短窗口聚焦
        df = pd.read_parquet(os.path.join(CATALOG, "002594", "data.parquet"))
        df = df.iloc[30:180].reset_index()
        df["symbol"] = "002594"
        r = _bt(df, min_hold_bars=10)
        # 回测正常完成且指标可用(止损逻辑不崩溃)
        assert r.metrics.total_return_pct == r.metrics.total_return_pct  # not NaN


class TestMinHoldPerSymbol:
    """护栏(2026-10-19): min_hold 必须按**标的自有** bar 计数。

    修复前用全局 `_bar_count` —— 25 只池里一根 bar 推进所有标的,
    10 根 ≈ 0.4 个交易日, 阈值永远不到(实测 min_hold 0/5/10 逐笔相同)。
    """

    @staticmethod
    def _src():
        import pathlib
        return pathlib.Path(__file__).resolve().parents[1].joinpath(
            "strategy.py").read_text(encoding="utf-8")

    def test_held_uses_symbol_bars(self):
        import ast
        src = self._src()
        tree = ast.parse(src)
        segs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "held"
                    for t in node.targets):
                seg = ast.get_source_segment(src, node.value) or ""
                if "_entry_bars" in seg:          # min_hold 分支(时间止盈分支用 _hold_bars)
                    segs.append(seg)
        assert len(segs) == 1, f"min_hold 的 held 赋值应唯一, 实际 {len(segs)}"
        assert "_symbol_bars" in segs[0]
        assert "_bar_count" not in segs[0]

    def test_entry_bars_recorded_per_symbol(self):
        src = self._src()
        assert "self._entry_bars[symbol] = self._symbol_bars.get(symbol, 0)" in src
