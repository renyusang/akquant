"""Dual Moving Average strategy for parameter optimization demo."""

from typing import Any, Dict

import numpy as np
from akquant import Bar, Strategy


class DualMovingAverageStrategy(Strategy):
    """Dual Moving Average Strategy for parameter optimization."""

    def __init__(self, short_window: int, long_window: int):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    def on_bar(self, bar: Bar) -> None:
        hist = self.get_history(count=self.long_window + 1, field="close")
        if len(hist) < self.long_window:
            return

        closes = hist
        ma_short = np.mean(closes[-self.short_window :])
        ma_long = np.mean(closes[-self.long_window :])
        prev_ma_short = np.mean(closes[-self.short_window - 1 : -1])
        prev_ma_long = np.mean(closes[-self.long_window - 1 : -1])

        position = self.get_position(bar.symbol)

        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            if position == 0:
                self.buy(bar.symbol, 100)
                print(f"[{bar.timestamp_str}] 金叉买入 {bar.symbol} @ {bar.close:.2f}")

        elif prev_ma_short >= prev_ma_long and ma_short < ma_long:
            if position > 0:
                self.sell(bar.symbol, 100)
                print(f"[{bar.timestamp_str}] 死叉卖出 {bar.symbol} @ {bar.close:.2f}")


def warmup_calc(params: Dict[str, Any]) -> int:
    return int(params["long_window"] + 1)


def param_constraint(params: Dict[str, Any]) -> bool:
    return bool(params["short_window"] < params["long_window"])


def result_filter(metrics: Dict[str, Any]) -> bool:
    return bool(metrics.get("trade_count", 0) >= 2)
