#!/usr/bin/env python
"""
卡尔曼滤波股票交易策略 —— 主入口。

用法:
    python main.py                              # 默认参数：比亚迪(002594)
    python main.py --symbol 600519              # 切换为贵州茅台
    python main.py --symbol 000858 --start 20150101 --end 20251231
    python main.py --plot-only                  # 仅绘制 K 线图，跳过回测
    python main.py --backtest-only              # 仅运行回测，跳过画图
    python main.py --walk-forward               # 滚动窗口回测
    python main.py --use-best                   # 自动加载网格搜索最优参数
    python main.py --params-csv grid_search.csv # 从指定 CSV 加载参数

参数:
    --symbol          股票代码（默认 002594 比亚迪）
    --start           起始日期 YYYYMMDD（默认 20200101）
    --end             结束日期 YYYYMMDD（默认 20241231）
    --adjust          复权方式 qfq/hfq/""（默认 qfq）
    --cash            初始资金（默认 100000）
    --plot-only       仅画图
    --backtest-only   仅回测
    --walk-forward    滚动窗口回测
    --use-best        自动加载 optimize.py 产出的最优参数
    --params-csv      指定网格搜索结果 CSV 文件路径
    --sort-by         加载最优参数时的排序指标（默认 sharpe_ratio）
    --kalman-q-price  卡尔曼价格过程噪声（默认 1e-4）
    --kalman-q-vel    卡尔曼速度过程噪声（默认 1e-5）
    --kalman-r        卡尔曼观测噪声（默认 1e-2）
    --entry-threshold 买入偏离阈值（默认 0.02）
    --exit-threshold  卖出偏离阈值（默认 0.005）
    --stop-loss       止损比例（默认 0.05）
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd

# 确保可以从 task 目录导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import download_stock_data, get_data_summary, preprocess_data
from plot_kline import plot_kline_with_indicators
from backtest import (
    export_trades_csv,
    print_metrics,
    run_kalman_backtest,
    run_walk_forward,
)


# ---- 默认配置 ----
SYMBOL_DEFAULT = "002594"  # 比亚迪
START_DEFAULT = "20200101"
END_DEFAULT = "20241231"
ADJUST_DEFAULT = "qfq"
CASH_DEFAULT = 100000.0

# 策略参数字段名列表（对应 CSV 列名和策略构造函数参数名）
_PARAM_FIELDS = [
    "kalman_q_price",
    "kalman_q_vel",
    "kalman_r",
    "entry_threshold",
    "exit_threshold",
    "stop_loss_pct",
    "trend_filter_enabled",
    "trend_filter_confirm_bars",
    "trend_bear_position_pct",
    "downtrend_entry_threshold",
]


def load_best_params(
    csv_path: str,
    sort_by: str = "sharpe_ratio",
) -> Dict[str, Any]:
    """从网格搜索 CSV 文件中加载最优参数。

    参数:
        csv_path: optimize.py 产出的 CSV 文件路径
        sort_by:  排序指标（默认 sharpe_ratio）

    返回:
        最优参数字典，键名与 KalmanStrategy 构造函数参数一致

    用法:
        params = load_best_params("grid_search_002594.csv")
        # 然后用 params 传给 run_kalman_backtest
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"网格搜索结果文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    # 过滤错误结果
    if "error" in df.columns:
        df = df[df["error"].isna()]

    if df.empty:
        raise ValueError(f"{csv_path} 中没有有效的优化结果")

    # 按指定指标排序
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    else:
        available = [c for c in df.columns if c not in _PARAM_FIELDS + ["error", "_duration"]]
        raise ValueError(
            f"排序指标 '{sort_by}' 不在 CSV 中。可用指标: {', '.join(sorted(available))}"
        )

    best = df.iloc[0]
    params: Dict[str, Any] = {}
    for field in _PARAM_FIELDS:
        if field in df.columns:
            val = best[field]
            # 将 numpy 类型转为 Python 原生类型
            if hasattr(val, "item"):
                val = val.item()
            params[field] = float(val)

    return params


def detect_csv_path(symbol: str) -> Optional[str]:
    """根据股票代码自动检测网格搜索结果 CSV 文件。

    按以下优先级查找:
        1. task/kalman/grid_search_{symbol}.csv
        2. task/kalman/grid_search_*.csv（任意匹配）
        3. 项目根目录/grid_search_{symbol}.csv

    返回找到的文件路径，或 None。
    """
    import glob

    task_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(task_dir)  # akquant 项目根目录

    for base in [task_dir, project_dir]:
        # 精确匹配
        exact = os.path.join(base, f"grid_search_{symbol}.csv")
        if os.path.exists(exact):
            return exact
        # 模糊匹配
        candidates = glob.glob(os.path.join(base, "grid_search_*.csv"))
        if candidates:
            return sorted(candidates)[-1]

    return None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="卡尔曼滤波股票交易策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 比亚迪，默认参数
  python main.py --symbol 600519                    # 贵州茅台
  python main.py --symbol 000858 --start 20150101   # 五粮液，2015年起
  python main.py --plot-only                        # 仅画图
  python main.py --walk-forward                     # 滚动窗口回测
        """,
    )

    # 数据和回测参数
    parser.add_argument(
        "--symbol",
        default=SYMBOL_DEFAULT,
        help=f"股票代码（默认: {SYMBOL_DEFAULT} 比亚迪）",
    )
    parser.add_argument(
        "--start",
        default=START_DEFAULT,
        help=f"起始日期 YYYYMMDD（默认: {START_DEFAULT}）",
    )
    parser.add_argument(
        "--end",
        default=END_DEFAULT,
        help=f"结束日期 YYYYMMDD（默认: {END_DEFAULT}）",
    )
    parser.add_argument(
        "--adjust",
        default=ADJUST_DEFAULT,
        choices=["qfq", "hfq", ""],
        help="复权方式: qfq=前复权, hfq=后复权, ''=不复权（默认: qfq）",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=CASH_DEFAULT,
        help=f"初始资金（默认: {CASH_DEFAULT:,.0f}）",
    )

    # 运行模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--plot-only",
        action="store_true",
        help="仅绘制 K 线图，跳过回测",
    )
    mode_group.add_argument(
        "--backtest-only",
        action="store_true",
        help="仅运行回测，跳过画图",
    )
    mode_group.add_argument(
        "--walk-forward",
        action="store_true",
        help="运行滚动窗口回测（Walk-Forward Analysis）",
    )

    # 图表参数
    parser.add_argument(
        "--save-plot",
        default=None,
        help="保存 K 线图到指定路径（支持 .html / .png）",
    )
    parser.add_argument(
        "--save-report",
        default=None,
        help="保存回测报告到指定路径（.html）",
    )

    # 参数加载（从优化结果）
    opt_group = parser.add_argument_group("参数加载（网格搜索优化结果）")
    opt_group.add_argument(
        "--use-best",
        action="store_true",
        help="自动加载网格搜索最优参数（从 grid_search_{symbol}.csv）",
    )
    opt_group.add_argument(
        "--params-csv",
        default=None,
        help="从指定 CSV 文件加载参数（optimize.py 产出的结果）",
    )
    opt_group.add_argument(
        "--sort-by",
        default="sharpe_ratio",
        help="加载参数时的排序指标（默认: sharpe_ratio）",
    )

    # 卡尔曼滤波器参数
    kf_group = parser.add_argument_group("卡尔曼滤波器参数")
    kf_group.add_argument(
        "--kalman-q-price",
        type=float,
        default=1e-4,
        help="价格过程噪声（默认: 1e-4）",
    )
    kf_group.add_argument(
        "--kalman-q-vel",
        type=float,
        default=1e-5,
        help="速度过程噪声（默认: 1e-5）",
    )
    kf_group.add_argument(
        "--kalman-r",
        type=float,
        default=1e-2,
        help="观测噪声（默认: 1e-2）",
    )

    # 信号阈值
    sig_group = parser.add_argument_group("信号阈值")
    sig_group.add_argument(
        "--entry-threshold",
        type=float,
        default=0.02,
        help="买入价格偏离阈值（默认: 0.02 = 2%%）",
    )
    sig_group.add_argument(
        "--exit-threshold",
        type=float,
        default=0.005,
        help="卖出价格偏离阈值（默认: 0.005 = 0.5%%）",
    )
    sig_group.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="止损比例（默认: 0.05 = 5%%）",
    )

    # 趋势过滤
    trend_group = parser.add_argument_group("趋势过滤")
    trend_group.add_argument(
        "--trend-filter",
        action="store_true",
        default=False,
        help="启用趋势过滤（close < MA20 时禁止买入并强制平仓）",
    )
    trend_group.add_argument(
        "--trend-confirm",
        type=int,
        default=3,
        help="趋势确认所需连续天数（默认: 3，避免频繁切换）",
    )
    trend_group.add_argument(
        "--trend-bear-pct",
        type=float,
        default=0.30,
        help="下跌趋势中的仓位比例（默认: 0.30 = 30%%。0 = 硬封堵）",
    )
    trend_group.add_argument(
        "--downtrend-entry",
        type=float,
        default=0.03,
        help="下跌趋势中买入所需的价格偏离阈值（默认: 0.03 = 3%%。正常为 2%%）",
    )

    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="运行全量组合回测(股票池+ETF池,用 stocks.yaml,替代单标的回测)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="回测完成后自动部署报告到服务器",
    )
    return parser.parse_args()


def _resolve_strategy_params(args: argparse.Namespace) -> Dict[str, Any]:
    """解析策略参数，优先使用 CSV 加载的最优参数。

    优先级: --params-csv > --use-best > CLI 直接指定的参数
    """
    params: Dict[str, Any] = {
        "kalman_q_price": args.kalman_q_price,
        "kalman_q_vel": args.kalman_q_vel,
        "kalman_r": args.kalman_r,
        "entry_threshold": args.entry_threshold,
        "exit_threshold": args.exit_threshold,
        "stop_loss_pct": args.stop_loss,
        "trend_filter_enabled": args.trend_filter,
        "trend_filter_confirm_bars": args.trend_confirm,
        "trend_bear_position_pct": args.trend_bear_pct,
        "downtrend_entry_threshold": args.downtrend_entry,
    }

    # 确定 CSV 路径
    csv_path: Optional[str] = None
    if args.params_csv:
        csv_path = args.params_csv
    elif args.use_best:
        csv_path = detect_csv_path(args.symbol)

    if csv_path:
        if not os.path.exists(csv_path):
            print(f"[警告] 未找到优化结果文件: {csv_path}，使用 CLI 参数")
        else:
            try:
                best = load_best_params(csv_path, sort_by=args.sort_by)
                print(f"[参数] 从 {csv_path} 加载最优参数 "
                      f"(排序指标: {args.sort_by}):")
                for k, v in best.items():
                    if isinstance(v, float) and v < 0.01:
                        print(f"  {k}: {v:.1e}")
                    else:
                        print(f"  {k}: {v}")
                print()
                params = best
            except Exception as e:
                print(f"[警告] 加载参数失败: {e}，使用 CLI 参数")

    return params


def main() -> None:
    """主流程。"""
    args = parse_args()

    # ---- 组合回测模式(用 AKQuant 引擎,替代 portfolio_backtest.py) ----
    if args.portfolio:
        from backtest import run_portfolio_backtest

        run_portfolio_backtest(
            start_date=args.start,
            end_date=args.end,
            show_progress=True,
            save=True,
            report=True,
        )
        return

    # ---- 打印运行配置 ----
    print("\n" + "=" * 60)
    print("  卡尔曼滤波股票交易策略")
    print("=" * 60)
    print(f"  标的:       {args.symbol}")
    print(f"  时间范围:   {args.start} ~ {args.end}")
    print(f"  复权方式:   {args.adjust or '不复权'}")
    print(f"  初始资金:   {args.cash:,.0f}")
    print(f"  运行模式:   {'仅画图' if args.plot_only else '仅回测' if args.backtest_only else '滚动回测' if args.walk_forward else '完整流程'}")

    # ---- 1. 下载数据 ----
    print("\n[1/4] 下载数据...")
    df = download_stock_data(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
    )
    print(f"  获取到 {len(df)} 条记录")

    # ---- 2. 预处理数据 ----
    print("\n[2/4] 预处理数据...")
    df = preprocess_data(df)
    print(f"  预处理后: {len(df)} 条记录")
    print(get_data_summary(df))

    # ---- 策略参数 ----
    strategy_params = _resolve_strategy_params(args)

    # ---- 3. 绘制 K 线图 ----
    if not args.backtest_only:
        print("\n[3/4] 绘制 K 线图...")
        save_plot = args.save_plot or f"kline_{args.symbol}.html"
        plot_kline_with_indicators(
            df=df,
            symbol=args.symbol,
            save_path=save_plot,
            show=args.save_plot is None,
        )

    # ---- 4. 回测 ----
    if not args.plot_only:
        print("\n[4/4] 运行回测...")

        if args.walk_forward:
            # 滚动窗口回测
            run_walk_forward(
                df=df,
                symbol=args.symbol,
                initial_cash=args.cash,
                strategy_params=strategy_params,
            )
        else:
            # 单次回测
            result = run_kalman_backtest(
                df=df,
                symbol=args.symbol,
                initial_cash=args.cash,
                start_time=args.start,
                end_time=args.end,
                strategy_params=strategy_params,
            )

            # 打印指标
            print_metrics(result)

            # 导出买卖点 CSV
            csv_base = f"trades_{args.symbol}"
            export_trades_csv(result, csv_base)

            # 生成 HTML 报告
            report_file = args.save_report or f"report_{args.symbol}.html"
            try:
                from akquant.plot import plot_report
                plot_report(
                    result,
                    title=f"卡尔曼滤波策略 - {args.symbol}",
                    filename=report_file,
                    show=args.save_report is None,
                    market_data=df,
                    plot_symbol=args.symbol,
                    include_trade_kline=True,
                )
                print(f"[报告] 已保存: {report_file}")
            except Exception as e:
                print(f"[报告] 生成失败: {e}")

    # ---- 5. 部署报告到服务器 ----
    if args.deploy:
        _deploy_reports()

    print("\n完成！")


def _deploy_reports() -> None:
    """调用 deploy.sh 将回测报告部署到服务器。"""
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_script = os.path.join(script_dir, "deploy.sh")
    if not os.path.exists(deploy_script):
        print("[部署] deploy.sh 不存在，跳过")
        return
    print("\n[部署] 上传报告到服务器...")
    try:
        result = subprocess.run(
            ["bash", deploy_script],
            capture_output=True, text=True, timeout=120,
            cwd=script_dir,
        )
        if result.returncode == 0:
            print("[部署] ✅ 报告已部署")
        else:
            print(f"[部署] ⚠️ 失败: {result.stderr.strip()}")
    except Exception as e:
        print(f"[部署] ⚠️ 异常: {e}")


if __name__ == "__main__":
    main()
