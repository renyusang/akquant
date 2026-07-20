#!/usr/bin/env python
"""
卡尔曼滤波策略 —— 参数网格搜索优化。

使用 akquant.run_grid_search 对卡尔曼滤波器参数和信号阈值进行网格搜索，
自动寻找最优参数组合。

用法:
    python optimize.py                              # 默认比亚迪，快速搜索
    python optimize.py --symbol 600519 --cash 500000 # 茅台，更多资金
    python optimize.py --full                        # 完整网格搜索（组合数多，耗时长）
    python optimize.py --db results.db               # 启用断点续传

参数网格:
    卡尔曼滤波器:
        kalman_q_price:  [1e-5, 1e-4, 1e-3]    # 价格过程噪声
        kalman_q_vel:    [1e-6, 1e-5, 1e-4]    # 速度过程噪声
        kalman_r:        [1e-3, 1e-2, 1e-1]    # 观测噪声

    信号阈值:
        entry_threshold:  [0.01, 0.02, 0.03]    # 买入偏离阈值
        exit_threshold:   [0.002, 0.005, 0.01]  # 卖出偏离阈值
        stop_loss_pct:    [0.03, 0.05, 0.08]    # 止损比例
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from akquant import run_grid_search

# 确保可以从 task 目录导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import download_stock_data, get_data_summary, preprocess_data
from strategy import KalmanStrategy

# ---- 默认配置 ----
SYMBOL_DEFAULT = "002594"
START_DEFAULT = "20200101"
END_DEFAULT = "20241231"
CASH_DEFAULT = 100000.0

# ---- 预设参数网格 ----

# 快速搜索（3 × 3 × 3 × 3 = 81 种组合）
GRID_QUICK: Dict[str, List[Any]] = {
    "kalman_q_price": [1e-5, 1e-4, 1e-3],
    "kalman_q_vel": [1e-6, 1e-5, 1e-4],
    "kalman_r": [1e-3, 1e-2, 1e-1],
    "entry_threshold": [0.01, 0.02, 0.03],
    "exit_threshold": [0.005, 0.01],
    "stop_loss_pct": [0.03, 0.05, 0.08],
}

# 完整搜索（3 × 3 × 4 × 4 × 3 × 3 = 1296 种组合）
GRID_FULL: Dict[str, List[Any]] = {
    "kalman_q_price": [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "kalman_q_vel": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    "kalman_r": [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "entry_threshold": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
    "exit_threshold": [0.002, 0.005, 0.01, 0.02],
    "stop_loss_pct": [0.03, 0.05, 0.08, 0.10],
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="卡尔曼滤波策略参数网格搜索优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python optimize.py                                    # 快速搜索，比亚迪
  python optimize.py --symbol 600519 --cash 500000      # 茅台
  python optimize.py --full                             # 完整搜索
  python optimize.py --db results.db                    # 断点续传
  python optimize.py --sort-by total_return_pct         # 按总收益排序
        """,
    )

    parser.add_argument(
        "--symbol",
        default=SYMBOL_DEFAULT,
        help=f"股票代码（默认: {SYMBOL_DEFAULT} 比亚迪）",
    )
    parser.add_argument(
        "--start",
        default=START_DEFAULT,
        help=f"起始日期（默认: {START_DEFAULT}）",
    )
    parser.add_argument(
        "--end",
        default=END_DEFAULT,
        help=f"结束日期（默认: {END_DEFAULT}）",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=CASH_DEFAULT,
        help=f"初始资金（默认: {CASH_DEFAULT:,.0f}）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="使用完整网格搜索（{len(GRID_FULL)} 组合 vs 快速 {len(GRID_QUICK)} 组合）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="并行进程数（默认: CPU 核心数）",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="sharpe_ratio",
        help="排序指标（默认: sharpe_ratio）。可选: total_return_pct, max_drawdown_pct, win_rate",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite 数据库路径，用于断点续传和增量保存",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="显示前 N 个最优结果（默认: 10）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="单次回测超时时间（秒，默认: 60）。设为 0 禁用超时",
    )

    return parser.parse_args()


def print_grid_info(grid: Dict[str, List[Any]], total: int) -> None:
    """打印参数网格信息。"""
    print(f"\n参数网格（共 {total} 种组合）:")
    print("-" * 50)
    for key, values in grid.items():
        val_strs = []
        for v in values:
            if isinstance(v, float) and v < 0.01:
                val_strs.append(f"{v:.0e}")
            elif isinstance(v, float):
                val_strs.append(f"{v:.3f}")
            else:
                val_strs.append(str(v))
        print(f"  {key:20s}: [{', '.join(val_strs)}]")
    print("-" * 50)


def print_top_results(
    results: pd.DataFrame, top_n: int, sort_by: str
) -> None:
    """打印最优结果。"""
    if results.empty:
        print("\n无有效结果。")
        return

    print(f"\n{'='*80}")
    print(f"  最优参数（按 {sort_by} 降序排列，前 {min(top_n, len(results))} 组）")
    print(f"{'='*80}")

    display_cols = [
        "kalman_q_price",
        "kalman_q_vel",
        "kalman_r",
        "entry_threshold",
        "exit_threshold",
        "stop_loss_pct",
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate",
    ]
    available_cols = [c for c in display_cols if c in results.columns]

    for i in range(min(top_n, len(results))):
        row = results.iloc[i]
        print(f"\n  #{i+1}")
        for col in available_cols:
            val = row[col]
            if col in ("kalman_q_price", "kalman_q_vel", "kalman_r"):
                if isinstance(val, float) and val < 0.01:
                    print(f"    {col:20s}: {val:.1e}")
                else:
                    print(f"    {col:20s}: {val}")
            elif col in ("total_return_pct", "max_drawdown_pct", "win_rate"):
                print(f"    {col:20s}: {val:.2f}%")
            elif col == "sharpe_ratio":
                print(f"    {col:20s}: {val:.2f}")
            else:
                print(f"    {col:20s}: {val}")

    # 汇总统计
    print(f"\n{'='*80}")
    print(f"  汇总统计（{len(results)} 个有效结果）")
    print(f"{'='*80}")
    for col in ["sharpe_ratio", "total_return_pct", "max_drawdown_pct", "win_rate"]:
        if col in results.columns:
            vals = results[col].dropna()
            if len(vals) > 0:
                print(
                    f"  {col:20s}: "
                    f"均值={vals.mean():.2f}  "
                    f"中位数={vals.median():.2f}  "
                    f"最小={vals.min():.2f}  "
                    f"最大={vals.max():.2f}"
                )


def _constraint(params: Dict[str, Any]) -> bool:
    """参数约束：entry_threshold 必须大于 exit_threshold。"""
    entry = float(params.get("entry_threshold", 0.02))
    exit_ = float(params.get("exit_threshold", 0.005))
    return entry > exit_


def main() -> None:
    """主流程。"""
    args = parse_args()
    grid = GRID_FULL if args.full else GRID_QUICK

    # 计算总组合数
    total = 1
    for v in grid.values():
        total *= len(v)

    print("\n" + "=" * 60)
    print("  卡尔曼滤波策略 —— 参数网格搜索优化")
    print("=" * 60)
    print(f"  标的:       {args.symbol}")
    print(f"  时间范围:   {args.start} ~ {args.end}")
    print(f"  初始资金:   {args.cash:,.0f}")
    print(f"  搜索模式:   {'完整' if args.full else '快速'}")
    print(f"  排序指标:   {args.sort_by}")

    print_grid_info(grid, total)

    # ---- 1. 下载并预处理数据 ----
    print("\n[1/3] 准备数据...")
    df = download_stock_data(args.symbol, args.start, args.end, adjust="qfq")
    df = preprocess_data(df)
    effective_bars = len(df) - 40  # warmup_period = 40
    print(f"  数据: {len(df)} 条（有效 bar 数: {effective_bars}，扣除预热期 40）")
    print(get_data_summary(df))

    if effective_bars < 30:
        print(f"\n  ⚠ 警告: 有效 bar 数仅 {effective_bars}（< 30），回测结果可能不稳定。")
        print(f"  建议扩大日期范围，例如 --start 20240101 --end 20250715")

    # ---- 2. 运行网格搜索 ----
    # 多进程安全设置
    max_workers = args.max_workers
    _timeout = args.timeout if args.timeout > 0 else None
    _max_tasks = 1 if max_workers is None or max_workers > 1 else None

    n_workers = max_workers if max_workers else os.cpu_count() or 4
    est_seconds = total * 0.5 / n_workers  # 粗略估计每组合 ~0.5 秒
    print(f"\n[2/3] 运行网格搜索（{total} 种组合）...")
    if _timeout:
        est_min = est_seconds / 60
        suffix = f"预计耗时: ~{est_min:.0f} 分钟" if est_min >= 1 else "预计耗时: < 1 分钟"
        print(f"  并行进程: {n_workers} | 超时: {_timeout}s/任务 | {suffix}")
    else:
        print(f"  并行进程: {n_workers} | 无超时限制")
    start_time = time.time()

    results = run_grid_search(
        strategy=KalmanStrategy,
        param_grid=grid,
        data=df,
        symbols=args.symbol,
        initial_cash=args.cash,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        min_commission=5.0,
        lot_size=100,
        start_time=args.start,
        end_time=args.end,
        sort_by=args.sort_by,
        ascending=False,
        return_df=True,
        max_workers=max_workers,
        constraint=_constraint,
        timeout=_timeout,
        max_tasks_per_child=_max_tasks,
        db_path=args.db,
        show_progress=False,
    )

    elapsed = time.time() - start_time
    print(f"  耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

    # ---- 3. 输出结果 ----
    print(f"\n[3/3] 分析结果...")

    if isinstance(results, pd.DataFrame):
        # 过滤错误结果
        if "error" in results.columns:
            error_count = results["error"].notna().sum()
            if error_count > 0:
                print(f"  错误/超时: {error_count} 个组合")
            results = results[results["error"].isna()]

        print_top_results(results, args.top, args.sort_by)

        # 保存 CSV（始终保存到 task/kalman 目录）
        task_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(task_dir, f"grid_search_{args.symbol}.csv")
        results.to_csv(csv_path, index=False)
        print(f"\n完整结果已保存: {csv_path}")
    else:
        print(f"  返回 {len(results)} 个 OptimizationResult 对象")
        for i, r in enumerate(results[: args.top]):
            print(f"\n  #{i+1}")
            print(f"    params:  {r.params}")
            print(f"    metrics: {r.metrics}")


if __name__ == "__main__":
    main()
