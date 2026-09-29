"""Tests for ``close_position(tag=...)`` pass-through.

2026-09-28: ``close_position`` 之前没有 ``tag`` 参数, 导致"按出场规则归因"
(成交记录的 ``exit_tag``)无法覆盖平仓路径 —— 只能改写调用点为 ``sell()``.
本文件锁定新契约: tag 透传到成交记录, 且不传 tag 时行为与原来一致。
"""

from typing import Optional

import pandas as pd
from akquant import Bar, Strategy, run_backtest


class TaggedCloseStrategy(Strategy):
    """第 1 根买入, 第 3 根带 tag 平仓。"""

    def __init__(self) -> None:
        """Initialize step counter."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Buy once, then close with a tag."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100, tag="entry:breakout")
        elif self.step == 3:
            self.close_position(symbol=bar.symbol, tag="exit:peak_drawdown")


class UntaggedPositionalCloseStrategy(Strategy):
    """第 1 根买入, 第 3 根用位置参数平仓(不传 tag) —— 回归保护。"""

    def __init__(self) -> None:
        """Initialize step counter."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Buy once, then close positionally without a tag."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100, tag="entry:breakout")
        elif self.step == 3:
            self.close_position(bar.symbol)


def _build_data() -> list[Bar]:
    closes = [10.0, 10.5, 11.0, 10.8, 11.2, 11.4]
    return [
        Bar(
            timestamp=pd.Timestamp(f"2023-01-{i + 1:02d} 10:00:00").value,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=5000.0,
            symbol="TEST",
        )
        for i, close in enumerate(closes)
    ]


def _run(strategy_cls: type) -> object:
    return run_backtest(
        data=_build_data(),
        strategy=strategy_cls,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )


def _single_trade(result: object) -> pd.Series:
    trades = result.trades_df
    assert not trades.empty, "应产生一笔平仓交易"
    assert len(trades) == 1, f"应只有一笔交易, 实际 {len(trades)}"
    return trades.iloc[0]


def test_close_position_forwards_tag_to_exit_tag() -> None:
    """close_position 的 tag 必须进入成交记录的 exit_tag。"""
    trade = _single_trade(_run(TaggedCloseStrategy))
    assert trade["exit_tag"] == "exit:peak_drawdown"
    assert trade["entry_tag"] == "entry:breakout"


def test_close_position_without_tag_keeps_empty_exit_tag() -> None:
    """不传 tag(含位置参数调用)时, 行为与新增 tag 参数之前一致。"""
    trade = _single_trade(_run(UntaggedPositionalCloseStrategy))
    tag: Optional[str] = trade["exit_tag"]
    assert tag is None or tag == "" or pd.isna(tag)
    assert trade["entry_tag"] == "entry:breakout"
