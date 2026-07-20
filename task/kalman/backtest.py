"""
回测运行模块。

封装 akquant.run_backtest 的参数配置，运行卡尔曼策略回测并输出指标。
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from akquant import BacktestResult, run_backtest

from strategy import KalmanStrategy


def run_kalman_backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_cash: float = 100000.0,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    min_commission: float = 5.0,
    lot_size: int = 100,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
) -> BacktestResult:
    """运行卡尔曼策略回测。

    参数:
        df:              预处理后的 OHLCV 数据（需包含 date, open, high, low, close, volume, symbol 列）
        symbol:          股票代码
        initial_cash:    初始资金（默认 10 万）
        commission_rate: 佣金费率（默认万分之三）
        stamp_tax_rate:  印花税率（默认千分之一，仅卖出收取）
        min_commission:  最低佣金（默认 5 元）
        lot_size:        每手股数（A 股为 100）
        start_time:      回测起始时间（截取数据）
        end_time:        回测结束时间
        strategy_params: 策略参数字典，传入 KalmanStrategy 构造函数
        show_progress:   是否显示进度条

    返回:
        BacktestResult 对象，包含 .metrics, .trades_df, .orders_df, .equity_curve 等
    """
    if strategy_params is None:
        strategy_params = {}

    # 打印策略参数
    _print_strategy_params(strategy_params)

    # 创建策略实例（显式传参）
    strategy = KalmanStrategy(**strategy_params) if strategy_params else KalmanStrategy()

    # 数据准备：确保 date 列和 symbol 列存在
    backtest_df = df.copy()
    if "date" in backtest_df.columns:
        backtest_df["date"] = pd.to_datetime(backtest_df["date"])

    # 运行回测
    result: BacktestResult = run_backtest(
        data=backtest_df,
        strategy=strategy,
        symbols=symbol,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        min_commission=min_commission,
        lot_size=lot_size,
        start_time=start_time,
        end_time=end_time,
        show_progress=show_progress,
    )

    return result


def print_metrics(result: BacktestResult) -> None:
    """格式化打印回测指标。"""
    m = result.metrics

    print("\n" + "=" * 60)
    print("                     回 测 结 果")
    print("=" * 60)

    # 基本指标
    print(f"  初始资金:       {m.initial_market_value:>12,.2f}")
    print(f"  最终权益:       {m.end_market_value:>12,.2f}")
    print(f"  总收益率:       {m.total_return_pct:>11.2f}%")
    print(f"  年化收益率:     {m.annualized_return:>11.2%}")
    print(f"  夏普比率:       {m.sharpe_ratio:>11.2f}")
    print(f"  波动率:         {m.volatility:>11.2%}")
    print(f"  最大回撤:       {m.max_drawdown_pct:>11.2f}%")
    print(f"  胜率:           {m.win_rate:>11.2f}%")
    print(f"  总交易次数:     {len(result.trades_df):>11}")
    print("-" * 60)

    # 交易详情
    trades = result.trades_df
    if trades is not None and not trades.empty and len(trades) > 0:
        _print_trade_stats(trades)

    # 卡尔曼策略特定统计
    strategy = result.strategy
    if strategy is not None and hasattr(strategy, "trade_count"):
        print(f"  策略信号次数:   {strategy.trade_count:>11}")

    print("=" * 60 + "\n")

    # 买卖点列表
    print_trade_list(result)
    print()

    # 交易配对表
    print_trade_pairs(result)


def _print_trade_stats(trades: pd.DataFrame) -> None:
    """打印交易统计详情。"""
    print("-" * 60)
    print("  交易详情:")
    print(f"    总交易数:     {len(trades)}")

    # 尝试获取盈亏列
    pnl_col = None
    for col_name in ["pnl", "net_pnl", "profit", "net_profit"]:
        if col_name in trades.columns:
            pnl_col = col_name
            break

    if pnl_col:
        pnl = trades[pnl_col].dropna()
        if len(pnl) > 0:
            winning = pnl[pnl > 0]
            losing = pnl[pnl < 0]
            print(f"    盈利交易:     {len(winning)} ({100*len(winning)/len(pnl):.1f}%)")
            print(f"    亏损交易:     {len(losing)} ({100*len(losing)/len(pnl):.1f}%)")
            if len(winning) > 0:
                print(f"    平均盈利:     {winning.mean():.2f}")
                print(f"    最大盈利:     {winning.max():.2f}")
            if len(losing) > 0:
                print(f"    平均亏损:     {losing.mean():.2f}")
                print(f"    最大亏损:     {losing.min():.2f}")
            print(f"    总盈亏:       {pnl.sum():.2f}")


def print_trade_list(result: BacktestResult) -> None:
    """打印所有买卖点列表（按时间排序）。"""
    orders = result.orders_df
    if orders is None or orders.empty:
        print("  (无买卖记录)")
        return

    # 只保留已成交订单
    filled = orders[orders["status"] == "filled"].copy()
    if filled.empty:
        print("  (无成交订单)")
        return

    print("-" * 80)
    print("  买 卖 点 列 表")
    print("-" * 80)
    header = f"  {'序号':>4s}  {'时间':<20s}  {'方向':>4s}  {'价格':>10s}  {'数量':>8s}  {'金额':>12s}"
    print(header)
    print("  " + "-" * 72)

    for i, (_, row) in enumerate(filled.iterrows(), 1):
        ts = str(row.get("created_at", ""))[:19]  # 截断到秒
        # 转换时间格式
        if len(ts) >= 10:
            ts = ts[:10]  # 只显示日期
        side = "买入" if str(row.get("side", "")).lower() == "buy" else "卖出"
        price = float(row.get("avg_price", 0))
        qty = float(row.get("filled_quantity", 0))
        value = float(row.get("filled_value", 0))

        print(
            f"  {i:>4d}  {ts:<20s}  {side:>4s}  "
            f"{price:>10.2f}  {qty:>8.0f}  {value:>12.0f}"
        )

    print("  " + "-" * 72)
    print(f"  共 {len(filled)} 笔成交")


def print_trade_pairs(result: BacktestResult) -> None:
    """打印交易配对表（买入→卖出，含盈亏）。"""
    trades = result.trades_df
    if trades is None or trades.empty:
        return

    print("")
    print("-" * 100)
    print("  交 易 明 细 表")
    print("-" * 100)
    header = (
        f"  {'#':>3s}  {'入场日期':<12s}  {'入场价':>8s}  "
        f"{'出场日期':<12s}  {'出场价':>8s}  {'数量':>6s}  "
        f"{'盈亏':>10s}  {'收益率':>8s}  {'持有':>5s}"
    )
    print(header)
    print("  " + "-" * 94)

    total_pnl = 0.0
    for i, (_, t) in enumerate(trades.iterrows(), 1):
        entry_time = str(t.get("entry_time", ""))[:10]
        exit_time = str(t.get("exit_time", ""))[:10]
        entry_price = float(t.get("entry_price", 0))
        exit_price = float(t.get("exit_price", 0))
        qty = int(t.get("quantity", 0))
        pnl = float(t.get("net_pnl", t.get("pnl", 0)))
        ret_pct = float(t.get("return_pct", 0))
        bars = int(t.get("duration_bars", 0)) if pd.notna(t.get("duration_bars")) else 0

        total_pnl += pnl
        pnl_marker = "+" if pnl >= 0 else ""

        print(
            f"  {i:>3d}  {entry_time:<12s}  {entry_price:>8.2f}  "
            f"{exit_time:<12s}  {exit_price:>8.2f}  {qty:>6d}  "
            f"{pnl_marker}{pnl:>9.2f}  {ret_pct:>7.2f}%  {bars:>4d}d"
        )

    print("  " + "-" * 94)
    total_marker = "+" if total_pnl >= 0 else ""
    print(f"  {'':>51s}  总盈亏: {total_marker}{total_pnl:>9.2f}")
    print("-" * 100)


def export_trades_csv(result: BacktestResult, path: str) -> None:
    """导出买卖点和交易配对到 CSV 文件。

    生成两个 sheet/文件:
        {path}_orders.csv  — 买卖点列表
        {path}_trades.csv  — 交易配对表（含盈亏）
    """
    base = path.replace(".csv", "")

    # 买卖点
    orders = result.orders_df
    if orders is not None and not orders.empty:
        filled = orders[orders["status"] == "filled"]
        cols = ["created_at", "side", "avg_price", "filled_quantity", "filled_value"]
        cols = [c for c in cols if c in filled.columns]
        filled[cols].to_csv(f"{base}_orders.csv", index=False, encoding="utf-8-sig")
        print(f"[导出] 买卖点列表: {base}_orders.csv")

    # 交易配对
    trades = result.trades_df
    if trades is not None and not trades.empty:
        cols = [
            "entry_time", "exit_time", "entry_price", "exit_price",
            "quantity", "side", "pnl", "net_pnl", "return_pct",
            "duration_bars", "entry_tag", "exit_tag",
        ]
        cols = [c for c in cols if c in trades.columns]
        trades[cols].to_csv(f"{base}_trades.csv", index=False, encoding="utf-8-sig")
        print(f"[导出] 交易明细: {base}_trades.csv")


def _print_strategy_params(params: dict) -> None:
    """打印策略参数。"""
    print("\n" + "-" * 40)
    print("  策略参数:")
    defaults = [
        ("kalman_q_price", 1e-4),
        ("kalman_q_vel", 1e-5),
        ("kalman_r", 1e-2),
        ("entry_threshold", 0.02),
        ("exit_threshold", 0.005),
        ("stop_loss_pct", 0.05),
        ("use_price_signal", True),
        ("use_velocity_signal", True),
        ("trend_filter_enabled", False),
        ("trend_filter_confirm_bars", 3),
        ("trend_bear_position_pct", 0.30),
        ("downtrend_entry_threshold", 0.03),
    ]
    for key, default in defaults:
        value = params.get(key, default)
        marker = " (默认)" if key not in params else ""
        if isinstance(value, float) and value < 0.01:
            print(f"    {key}: {value:.0e}{marker}")
        elif isinstance(value, float):
            print(f"    {key}: {value}{marker}")
        else:
            print(f"    {key}: {value}{marker}")
    print("-" * 40)


def run_walk_forward(
    df: pd.DataFrame,
    symbol: str,
    initial_cash: float = 100000.0,
    train_years: int = 2,
    test_months: int = 6,
    strategy_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """简单的滚动窗口回测（Walk-Forward Analysis）。

    将数据按时间分割为多个训练/测试窗口，在每个窗口上运行回测。

    参数:
        df:          完整数据
        symbol:      股票代码
        train_years: 训练窗口年数
        test_months: 测试窗口月数
    """
    if strategy_params is None:
        strategy_params = {}

    if "date" not in df.columns:
        raise ValueError("数据必须包含 date 列")

    dates = pd.to_datetime(df["date"])
    start = dates.min()
    end = dates.max()

    window_results = []
    train_start = start

    print(f"\n{'='*60}")
    print(f"  滚动窗口回测 (Walk-Forward Analysis)")
    print(f"  训练窗口: {train_years} 年 | 测试窗口: {test_months} 月")
    print(f"{'='*60}")

    window_idx = 0
    while train_start < end:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end:
            test_end = end

        if train_end >= end or test_end <= train_end:
            break

        window_idx += 1
        train_df = df[(dates >= train_start) & (dates < train_end)]
        test_df = df[(dates >= train_end) & (dates <= test_end)]

        if len(test_df) < 20:
            train_start = test_end
            continue

        print(f"\n  窗口 {window_idx}: "
              f"训练 {train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')} | "
              f"测试 {train_end.strftime('%Y-%m')} ~ {test_end.strftime('%Y-%m')}")

        try:
            result = run_kalman_backtest(
                df=df[(dates >= train_start) & (dates <= test_end)],
                symbol=symbol,
                initial_cash=initial_cash,
                start_time=train_end.strftime("%Y%m%d"),
                end_time=test_end.strftime("%Y%m%d"),
                strategy_params=strategy_params,
                show_progress=False,
                **kwargs,
            )
            window_results.append({
                "window": window_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_end": test_end,
                "total_return_pct": result.metrics.total_return_pct,
                "sharpe": result.metrics.sharpe_ratio,
                "max_drawdown_pct": result.metrics.max_drawdown_pct,
                "trades": len(result.trades_df),
            })
        except Exception as e:
            print(f"    窗口 {window_idx} 回测失败: {e}")

        train_start = test_end

    # 汇总
    if window_results:
        print(f"\n{'='*60}")
        print(f"  滚动窗口汇总 ({len(window_results)} 个窗口)")
        print(f"{'='*60}")
        for wr in window_results:
            print(f"  窗口 {wr['window']:2d}: "
                  f"收益={wr['total_return_pct']:7.2f}% | "
                  f"夏普={wr['sharpe']:5.2f} | "
                  f"回撤={wr['max_drawdown_pct']:6.2f}% | "
                  f"交易={wr['trades']}")

        avg_return = np.mean([w["total_return_pct"] for w in window_results])
        avg_sharpe = np.mean([w["sharpe"] for w in window_results])
        print(f"  平均收益: {avg_return:.2f}% | 平均夏普: {avg_sharpe:.2f}")
