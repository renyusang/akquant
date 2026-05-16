"""
Parameter optimization example.

Demonstrates how to use grid search for parameter optimization.
"""

import numpy as np
import pandas as pd

from strategies.dual_ma_optimization import (
    DualMovingAverageStrategy,
    param_constraint,
    result_filter,
    warmup_calc,
)


if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(1024)

    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    price = 100 + np.cumsum(np.random.randn(len(dates)))  # 随机游走价格

    df = pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 10000,
            "symbol": "DEMO",
        }
    )
    # run_walk_forward 需要 DatetimeIndex
    df.set_index("date", inplace=True)

    # 运行回测
    print("开始回测...")

    from akquant import run_grid_search

    # Define parameter grid
    param_grid = {
        "short_window": list(range(5, 30, 5)),  # [5, 10, 15, 20, 25]
        "long_window": list(range(20, 60, 10)),  # [20, 30, 40, 50]
    }

    # Run optimization
    results_df = run_grid_search(
        strategy=DualMovingAverageStrategy,
        param_grid=param_grid,
        data=df,
        initial_cash=100_000.0,
        sort_by=["sharpe_ratio", "total_return"],  # 多字段排序：先按夏普，再按总收益
        ascending=[False, False],  # 都是降序
        warmup_calc=warmup_calc,  # 动态计算预热期
        constraint=param_constraint,  # 参数约束
        result_filter=result_filter,  # 结果筛选
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # DataFrame 返回显示的行内的所有内容，不要缩略

    # Print top 5
    print(results_df)

    # ------------------------------
    # 演示 Walk-Forward Optimization
    # ------------------------------
    print("\n\n开始 Walk-Forward Optimization...")
    from akquant import run_walk_forward

    # 确保数据足够长
    # train=100, test=50 -> need 150+
    # current len is 365

    wfo_result = run_walk_forward(
        strategy=DualMovingAverageStrategy,
        param_grid=param_grid,
        data=df,
        train_period=100,
        test_period=50,
        metric=["sharpe_ratio", "total_return"],  # 多目标排序
        ascending=[False, False],
        initial_cash=100_000.0,
        warmup_calc=warmup_calc,
        constraint=param_constraint,
        result_filter=result_filter,  # 同样可以使用结果过滤
    )

    print("\nWFO Result Head:")
    print(wfo_result.head())
