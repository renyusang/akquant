"""
回测运行模块。

封装 akquant.run_backtest 的参数配置，运行卡尔曼策略回测并输出指标。
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from akquant import BacktestResult, run_backtest
from akquant.backtest.engine import make_fill_policy
from akquant.plot import plot_report

from strategy import KalmanStrategy


def run_kalman_backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_cash: float = 100000.0,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    transfer_fee_rate: float = 0.00001,
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

    # 创建策略实例（显式传参,传入 initial_cash 供 order_target_value 使用）
    params = dict(strategy_params) if strategy_params else {}
    params.setdefault("initial_cash", initial_cash)
    strategy = KalmanStrategy(**params)

    # 数据准备：确保 date 列和 symbol 列存在
    backtest_df = df.copy()
    if "date" in backtest_df.columns:
        backtest_df["date"] = pd.to_datetime(backtest_df["date"])

    # NextOpen:T日 on_bar 下单 → T+1 开盘价撮合(与实盘次日开盘成交一致)
    fill_policy = make_fill_policy(
        price_basis="open", temporal="next_event", bar_offset=1
    )
    # 运行回测
    result: BacktestResult = run_backtest(
        data=backtest_df,
        strategy=strategy,
        symbols=symbol,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        transfer_fee_rate=transfer_fee_rate,
        min_commission=min_commission,
        lot_size=lot_size,
        t_plus_one=True,
        fill_policy=fill_policy,
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


# =============================================================================
# 组合回测(替代 portfolio_backtest.py,用 AKQuant 引擎)
# =============================================================================

# stocks.yaml 参数名 → KalmanStrategy 构造参数名映射
_PARAM_MAP = {
    "trend_confirm_bars": "trend_filter_confirm_bars",
    "trend_bear_pct": "trend_bear_position_pct",
    "downtrend_entry": "downtrend_entry_threshold",
}


def _map_strategy_params(yaml_params: Dict[str, Any]) -> Dict[str, Any]:
    """将 stocks.yaml 的策略参数名映射到 KalmanStrategy 构造参数名。"""
    return {_PARAM_MAP.get(k, k): v for k, v in yaml_params.items()}


def _lot_size_map(symbols: list) -> Dict[str, int]:
    """按标的生成 lot_size(科创板688→200,其余100)。"""
    return {s: (200 if str(s).startswith("688") else 100) for s in symbols}


def _run_pool(
    data_map: Dict[str, pd.DataFrame],
    symbols: list,
    initial_cash: float,
    strategy_params: Dict[str, Any],
    show_progress: bool = False,
) -> Optional[BacktestResult]:
    """单池回测:T+1 开盘价执行,真实手续费(佣金万3双边最低5元+印花税千1卖出+过户费万0.1)。

    2026-09-01 更新 docstring: 原"手续费置零(对齐旧基线)"为过时注释——
    实际费率与实盘 portfolio.calc_fee 一致(kalman.md 注意事项 12)。
    """
    if not data_map or not symbols:
        return None
    # NextOpen:T日 on_bar 下单 → T+1 开盘价撮合(与实盘次日开盘成交一致)
    fill_policy = make_fill_policy(
        price_basis="open", temporal="next_event", bar_offset=1
    )
    strategy = KalmanStrategy(**strategy_params)
    return run_backtest(
        data=data_map,
        strategy=strategy,
        symbols=symbols,
        initial_cash=initial_cash,
        t_plus_one=True,
        fill_policy=fill_policy,
        lot_size=_lot_size_map(symbols),
        commission_rate=0.0003,      # 佣金万3(双边,最低5元)
        stamp_tax_rate=0.001,        # 印花税千1(仅卖出)
        transfer_fee_rate=0.00001,   # 过户费万0.1(双边)
        min_commission=5.0,          # 最低佣金5元
        show_progress=show_progress,
    )


def _combine_equity(
    stock_result: Optional[BacktestResult],
    etf_result: Optional[BacktestResult],
    stock_cash: float,
    etf_cash: float,
) -> pd.Series:
    """合并两池权益曲线(逐日相加,起始前用各池初始资金填充)。"""
    parts = {}
    if stock_result is not None:
        parts["stock"] = stock_result.equity_curve.copy()
    if etf_result is not None:
        parts["etf"] = etf_result.equity_curve.copy()
    df = pd.DataFrame(parts)
    if "stock" in df.columns:
        df["stock"] = df["stock"].ffill().fillna(stock_cash)
    if "etf" in df.columns:
        df["etf"] = df["etf"].ffill().fillna(etf_cash)
    combined = df.sum(axis=1)
    combined.name = "equity"
    return combined


def _get_open_positions(result, data_map: dict) -> pd.DataFrame:
    """从 BacktestResult 提取最终未平仓持仓。

    使用 result.positions (日度持仓快照) + get_positions_dict (引擎计算的市值/浮盈)。
    """
    if result is None:
        return pd.DataFrame()

    pos_df = result.positions
    if pos_df is None or pos_df.empty:
        return pd.DataFrame()

    # 最后一天有持仓的标的
    last_row = pos_df.iloc[-1]
    open_syms = last_row[last_row > 0]
    if len(open_syms) == 0:
        return pd.DataFrame()

    # 从 get_positions_dict 提取引擎计算的最终数据(市值+浮盈准确)
    pos_dict = result.get_positions_dict()
    final_data = {}
    if pos_dict and "symbol" in pos_dict:
        n = len(pos_dict["symbol"])
        for i in range(n):
            sym = str(pos_dict["symbol"][i]).zfill(6)
            shares = float(pos_dict["long_shares"][i])
            if shares > 0:
                final_data[sym] = {
                    "shares": shares,
                    "entry_price": float(pos_dict["entry_price"][i]),
                    "close": float(pos_dict["close"][i]),
                    "market_value": float(pos_dict["market_value"][i]),
                    "unrealized_pnl": float(pos_dict["unrealized_pnl"][i]),
                }

    rows = []
    for sym, shares in open_syms.items():
        sym = str(sym).zfill(6)
        shares = float(shares)
        if shares <= 0:
            continue
        fd = final_data.get(sym, {})
        entry_price = fd.get("entry_price", 0)
        market_value = fd.get("market_value", 0)
        unrealized_pnl = fd.get("unrealized_pnl", 0)
        last_close = fd.get("close", 0)
        cost_value = market_value - unrealized_pnl
        pnl_pct = (unrealized_pnl / cost_value * 100) if cost_value > 0 else 0

        # 用 data_map 的最后收盘价覆盖(更精确)
        if sym in data_map:
            last_close = float(data_map[sym].iloc[-1]["close"])
            market_value = shares * last_close
            unrealized_pnl = market_value - cost_value
            pnl_pct = (unrealized_pnl / cost_value * 100) if cost_value > 0 else 0

        rows.append({
            "symbol": sym,
            "shares": shares,
            "avg_entry": round(entry_price, 3),
            "last_close": round(last_close, 3),
            "cost_value": round(cost_value, 2),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    return pd.DataFrame(rows).sort_values("unrealized_pnl", ascending=False)


def build_combined_report(
    combined_equity: pd.Series,
    combined_trades: pd.DataFrame,
    metrics: Dict[str, Any],
    stock_cash: float,
    etf_cash: float,
    filename: str,
    benchmark: Optional[pd.Series] = None,
    open_positions: Optional[pd.DataFrame] = None,
) -> None:
    """生成组合汇总 HTML 报告(股票池+ETF池 合并权益/指标/交易)。

    两池各自有 result.report(report_stock.html/report_etf.html,含K线复盘);
    本报告汇总两池合并的组合权益曲线、回撤、月度/年度收益、指标、交易明细。
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = combined_equity.copy()
    eq.index = pd.to_datetime(eq.index)
    if getattr(eq.index, "tz", None) is not None:
        eq.index = eq.index.tz_convert("Asia/Shanghai")

    # 回撤
    dd = (eq - eq.cummax()) / eq.cummax() * 100

    # 年度收益
    yearly_ret = eq.resample("YE").last().pct_change() * 100

    # 月度收益(热力图)
    monthly_ret = eq.resample("ME").last().pct_change() * 100

    # ---- 主图:权益 + 回撤 + 年度收益 ----
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("组合权益曲线(股票池+ETF池)", "回撤(%)", "年度收益(%)"),
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(x=eq.index, y=eq.values, name="权益",
                   line=dict(color="#2c7fb8", width=1.5)),
        row=1, col=1,
    )
    # 沪深300基准(归一化到初始权益,便于对比)
    if benchmark is not None:
        # 统一 index(eq 可能 tz-aware,benchmark naive)
        eq_naive = eq.index.tz_localize(None) if hasattr(eq.index, "tz") and eq.index.tz else eq.index
        bench_naive = (
            benchmark.index.tz_localize(None)
            if hasattr(benchmark.index, "tz") and benchmark.index.tz
            else benchmark.index
        )
        bench = pd.Series(benchmark.values, index=bench_naive).reindex(eq_naive).ffill()
        if len(bench.dropna()) > 0:
            bench_norm = bench / bench.iloc[0] * eq.iloc[0]
            fig.add_trace(
                go.Scatter(x=eq_naive, y=bench_norm.values, name="沪深300",
                           line=dict(color="#ff7f0e", dash="dash", width=1.2)),
                row=1, col=1,
            )
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name="回撤", fill="tozeroy",
                   line=dict(color="#d62728", width=1)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[d.year for d in yearly_ret.index],
            y=yearly_ret.values, name="年度收益",
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in yearly_ret.values],
        ),
        row=3, col=1,
    )
    fig.update_layout(
        title=(f"组合汇总报告 | 总收益 {metrics['total_return_pct']:+.1f}% | "
               f"Sharpe {metrics['sharpe']:.2f} | 最大回撤 {metrics['max_drawdown_pct']:.1f}%"),
        height=900, showlegend=False, template="plotly_white",
    )
    fig.update_yaxes(title_text="权益(¥)", row=1, col=1)
    fig.update_yaxes(title_text="回撤(%)", row=2, col=1)
    fig.update_yaxes(title_text="收益(%)", row=3, col=1)

    # ---- 月度热力图 ----
    heat_html = ""
    if len(monthly_ret) > 0:
        mdf = monthly_ret.to_frame("ret")
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot_table(index="year", columns="month", values="ret")
        heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{m}月" for m in pivot.columns],
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}", hovertemplate="%{y}年 %{x}: %{z:.1f}%<extra></extra>",
        ))
        heat.update_layout(title="月度收益热力图(%)", height=400, template="plotly_white")
        heat_html = heat.to_html(full_html=False, include_plotlyjs=False)

    # ---- 指标表 ----
    m = metrics
    metrics_html = f"""
    <table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse;font-size:14px;margin:10px 0'>
      <tr><th>指标</th><th>值</th></tr>
      <tr><td>回测区间</td><td>{m['period_start']} ~ {m['period_end']} ({m['days']}天)</td></tr>
      <tr><td>初始资金</td><td>¥{m['initial_cash']:,.0f} (股票池¥{stock_cash:,.0f} + ETF池¥{etf_cash:,.0f})</td></tr>
      <tr><td>最终权益</td><td>¥{m['final_equity']:,.0f}</td></tr>
      <tr><td>总收益率</td><td>{m['total_return_pct']:+.1f}%</td></tr>
      <tr><td>夏普比率</td><td>{m['sharpe']:.2f}</td></tr>
      <tr><td>最大回撤</td><td>{m['max_drawdown_pct']:.1f}%</td></tr>
      <tr><td>总交易数</td><td>{m['trade_count']}</td></tr>
    </table>"""

    # ---- 未平仓持仓 ----
    positions_html = ""
    if open_positions is not None and not open_positions.empty:
        pos = open_positions.copy()
        total_upnl = pos["unrealized_pnl"].sum()
        upnl_cls = "positive" if total_upnl > 0 else "negative"
        pos_html = "<h3>未平仓持仓明细</h3>"
        pos_html += f'<div class="summary-text">共 {len(pos)} 只 | 总浮盈 <span class="{upnl_cls}">¥{total_upnl:+,.0f}</span></div>'
        pos_disp = pos[["symbol", "shares", "avg_entry", "last_close", "cost_value", "market_value", "unrealized_pnl", "pnl_pct"]].copy()
        pos_disp.columns = ["代码", "股数", "均成本", "现价", "成本额", "市值", "浮盈", "收益率%"]
        pos_disp["股数"] = pos_disp["股数"].apply(lambda x: f"{int(x):,}")
        pos_disp["均成本"] = pos_disp["均成本"].apply(lambda x: f"{float(x):.2f}")
        pos_disp["现价"] = pos_disp["现价"].apply(lambda x: f"{float(x):.2f}")
        pos_disp["成本额"] = pos_disp["成本额"].apply(lambda x: f"{float(x):,.0f}")
        pos_disp["市值"] = pos_disp["市值"].apply(lambda x: f"{float(x):,.0f}")
        pos_disp["浮盈"] = pos_disp["浮盈"].apply(
            lambda x: f"<span class='{'positive' if float(x)>0 else 'negative'}'>{float(x):+,.0f}</span>")
        pos_disp["收益率%"] = pos_disp["收益率%"].apply(
            lambda x: f"<span class='{'positive' if float(x)>0 else 'negative'}'>{float(x):+.1f}%</span>")
        pos_html += pos_disp.to_html(index=False, classes="data-table", border=0, justify="center", escape=False)
        positions_html = pos_html

    # ---- 交易明细(前20笔)----
    trades_html = "<h3>交易明细(前20笔)</h3>"
    if not combined_trades.empty:
        show_cols = [c for c in [
            "entry_time", "exit_time", "symbol", "side",
            "entry_price", "exit_price", "quantity", "pnl", "pool",
        ] if c in combined_trades.columns]
        tdisp = combined_trades[show_cols].head(20).copy()
        for c in tdisp.select_dtypes(include=["float"]).columns:
            tdisp[c] = tdisp[c].round(2)
        trades_html += tdisp.to_html(index=False, border=1)

    full = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>卡尔曼组合汇总报告</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>body{{font-family:sans-serif;margin:20px;}} h1{{color:#2c3e50;}} table{{font-size:13px;}}</style>
</head><body>
<div style="background:#fff5f5;border:3px solid #e74c3c;border-radius:8px;padding:20px 24px;margin-bottom:24px;font-size:16px;color:#721c24;line-height:1.8;text-align:center">
<div style="font-size:24px;margin-bottom:8px">⚠️</div>
<strong style="font-size:18px">免责声明</strong><br>
本报告仅为<u>个人量化策略研究记录</u>，<strong>不构成任何投资建议</strong>。<br>
回测基于<strong>历史数据</strong>，过往表现<strong>不代表未来收益</strong>。<br>
股市有风险，投资需谨慎。
</div>
<h1>卡尔曼组合汇总报告</h1>
{metrics_html}
{positions_html}
{fig.to_html(full_html=False, include_plotlyjs=False)}
{heat_html}
{trades_html}
</body></html>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full)


def run_portfolio_backtest(
    config_path: Optional[str] = None,
    start_date: str = "20200101",
    end_date: str = "20261231",
    show_progress: bool = False,
    save: bool = True,
    report: bool = True,
) -> Dict[str, Any]:
    """全量投资组合回测:股票池+ETF池各跑一次 run_backtest,合并结果。

    替代旧 portfolio_backtest.py。用 AKQuant 引擎,T+1 开盘价执行,
    分池资金管理(股票20万/ETF10万),手续费置零对齐旧基线。

    返回 dict: stock_result, etf_result, combined_equity, combined_trades, metrics。
    """
    from data_utils import load_watchlist_data

    task_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = config_path or os.path.join(task_dir, "stocks.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stock_map, etf_map, _stock_items, _etf_items = load_watchlist_data(
        config, start_date, end_date, quiet=not show_progress
    )
    stock_syms = list(stock_map.keys())
    etf_syms = list(etf_map.keys())

    base_params = _map_strategy_params(config.get("strategy", {}))
    stock_pool = config.get("stock", {})
    etf_pool = config.get("etf", {})
    stock_cash = float(stock_pool.get("initial_cash", 200000))
    etf_cash = float(etf_pool.get("initial_cash", 100000))
    stock_max = int(stock_pool.get("max_positions", 5))
    etf_max = int(etf_pool.get("max_positions", 5))
    stock_pct = float(stock_pool.get("single_position_pct", 0.20))
    etf_pct = float(etf_pool.get("single_position_pct", 0.20))
    total_cash = stock_cash + etf_cash

    print(
        f"\n组合回测 {start_date}~{end_date}: "
        f"股票 {len(stock_syms)} 只(¥{stock_cash:,.0f}) + "
        f"ETF {len(etf_syms)} 只(¥{etf_cash:,.0f})"
    )

    stock_result = (
        _run_pool(
            stock_map, stock_syms, stock_cash,
            {**base_params, "single_position_pct": stock_pct, "max_positions": stock_max, "initial_cash": stock_cash},
            show_progress,
        )
        if stock_syms
        else None
    )
    etf_result = (
        _run_pool(
            etf_map, etf_syms, etf_cash,
            {**base_params, "single_position_pct": etf_pct, "max_positions": etf_max, "initial_cash": etf_cash},
            show_progress,
        )
        if etf_syms
        else None
    )

    combined = _combine_equity(stock_result, etf_result, stock_cash, etf_cash)

    # 合并 trades
    trades_parts = []
    if stock_result is not None:
        t = stock_result.trades_df.copy()
        t["pool"] = "stock"
        trades_parts.append(t)
    if etf_result is not None:
        t = etf_result.trades_df.copy()
        t["pool"] = "etf"
        trades_parts.append(t)
    combined_trades = (
        pd.concat(trades_parts, ignore_index=True) if trades_parts else pd.DataFrame()
    )

    # 组合指标(口径与旧 portfolio_backtest 一致)
    final_equity = float(combined.iloc[-1])
    total_return = (final_equity / total_cash - 1) * 100
    dr = combined.pct_change().dropna()
    sharpe = (
        float(dr.mean() / dr.std() * np.sqrt(252))
        if len(dr) > 1 and dr.std() > 0
        else 0.0
    )
    max_dd = float(((combined - combined.cummax()) / combined.cummax()).min() * 100)

    print(f"\n{'='*60}")
    print(f"  组合回测结果(AKQuant 引擎)")
    print(f"{'='*60}")
    print(f"  回测区间: {combined.index[0]} ~ {combined.index[-1]} ({len(combined)} 天)")
    print(f"  初始资金: ¥{total_cash:,.0f}")
    print(f"  最终权益: ¥{final_equity:,.0f}")
    print(f"  总收益率: {total_return:+.1f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  总交易数: {len(combined_trades)}")

    result = {
        "stock_result": stock_result,
        "etf_result": etf_result,
        "combined_equity": combined,
        "combined_trades": combined_trades,
        "metrics": {
            "total_return_pct": total_return,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "final_equity": final_equity,
            "initial_cash": total_cash,
            "trade_count": len(combined_trades),
            "period_start": str(combined.index[0]),
            "period_end": str(combined.index[-1]),
            "days": int(len(combined)),
        },
    }

    if save:
        idx = pd.to_datetime(combined.index)
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("Asia/Shanghai")
        eq_df = pd.DataFrame(
            {"date": idx.strftime("%Y-%m-%d"), "equity": combined.values}
        )
        eq_df.to_csv(
            os.path.join(task_dir, "portfolio_equity.csv"),
            index=False, encoding="utf-8-sig",
        )
        combined_trades.to_csv(
            os.path.join(task_dir, "portfolio_trades.csv"),
            index=False, encoding="utf-8-sig",
        )
        print(f"  已保存 portfolio_equity.csv / portfolio_trades.csv")

        # 导出未平仓持仓
        stock_positions = _get_open_positions(stock_result, stock_map) if stock_result else pd.DataFrame()
        etf_positions = _get_open_positions(etf_result, etf_map) if etf_result else pd.DataFrame()
        all_positions = pd.concat([stock_positions, etf_positions], ignore_index=True)
        if not all_positions.empty:
            pos_path = os.path.join(task_dir, "portfolio_positions.csv")
            all_positions.to_csv(pos_path, index=False, encoding="utf-8-sig")
            print(f"  已保存 portfolio_positions.csv ({len(all_positions)} 只未平仓)")

    if report:
        try:
            # 沪深300基准(收益序列给 result.report,价格序列给 build_combined_report)
            hs300_price = None
            hs300_returns = None
            try:
                from data_utils import fetch_hs300

                hs300_price = fetch_hs300(start_date, end_date)
                hs300_returns = hs300_price.pct_change().dropna()
                hs300_returns.name = "沪深300"
                # result.to_quantstats() returns 为 naive datetime(00:00:00),保持 naive 对齐
            except Exception as e:
                print(f"  ⚠️ 沪深300数据获取失败({e}),报告无基准对比")

            # BacktestResult.report(含K线买卖点复盘、拒单原因、完整指标、沪深300基准)
            if stock_result is not None:
                stock_plot = (
                    "002594" if "002594" in stock_map
                    else (stock_syms[0] if stock_syms else None)
                )
                plot_report(
                    stock_result,
                    title=f"卡尔曼组合回测 - 股票池({len(stock_syms)}只)",
                    filename=os.path.join(task_dir, "report_stock.html"),
                    market_data=stock_map,
                    plot_symbol=stock_plot,
                    include_trade_kline=True,
                    benchmark=hs300_returns,
                    show=False,
                )
            if etf_result is not None:
                etf_plot = (
                    "510050" if "510050" in etf_map
                    else (etf_syms[0] if etf_syms else None)
                )
                plot_report(
                    etf_result,
                    title=f"卡尔曼组合回测 - ETF池({len(etf_syms)}只)",
                    filename=os.path.join(task_dir, "report_etf.html"),
                    market_data=etf_map,
                    plot_symbol=etf_plot,
                    include_trade_kline=True,
                    benchmark=hs300_returns,
                    show=False,
                )
            # 组合汇总报告(股票池+ETF池合并权益/指标/交易,含沪深300基准)
            build_combined_report(
                combined, combined_trades, result["metrics"],
                stock_cash, etf_cash,
                os.path.join(task_dir, "report_portfolio.html"),
                benchmark=hs300_price,
                open_positions=all_positions,
            )
            print(f"  已生成 report_stock.html / report_etf.html(含K线复盘+沪深300) + report_portfolio.html(组合汇总)")
        except Exception as e:
            print(f"  ⚠️ 报告生成失败: {e}")

    return result
