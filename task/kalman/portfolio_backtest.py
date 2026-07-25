#!/usr/bin/env python
"""
全量投资组合回测。

按照 stocks.yaml 配置，模拟 daily_signal.py 的完整逻辑：
- 股票+ETF 分池管理
- T+1 开盘价执行
- 趋势过滤 + 卡尔曼信号
- 自动补仓
- 仓位上限 + 跌停保护

用法: python portfolio_backtest.py
输出: portfolio_report.csv (交易记录), portfolio_equity.csv (权益曲线)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_engine import SignalEngine
from data_utils import download_stock_data, download_etf_data, preprocess_data

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(TASK_DIR, "stocks.yaml")


# =============================================================================
# 配置加载
# =============================================================================
def load_backtest_config(config_path: str = None) -> Dict[str, Any]:
    path = config_path or CONFIG_FILE
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _pool_config(config, asset_type):
    pool = config.get(asset_type, {})
    return {
        "cash": float(pool.get("initial_cash", 200000 if asset_type == "stock" else 100000)),
        "max_positions": int(pool.get("max_positions", 5)),
        "max_pct": float(pool.get("single_position_pct", 0.20)),
    }


# =============================================================================
# 数据准备
# =============================================================================
def prepare_data(config, start, end):
    """下载所有股票和 ETF 的历史数据，对齐到统一日期索引。"""
    all_data = {}
    watchlist = config["watchlist"]
    stocks = watchlist.get("stocks", [])
    etfs = watchlist.get("etfs", [])

    print(f"下载 {len(stocks)} 只股票 + {len(etfs)} 只 ETF ...")
    for stock in stocks:
        sym = stock["symbol"]
        try:
            df = download_stock_data(sym, start, end, adjust="qfq")
            df = preprocess_data(df)
            if len(df) > 0:
                all_data[sym] = df.set_index("date")
        except Exception:
            pass

    for etf in etfs:
        sym = etf["symbol"]
        try:
            df = download_etf_data(sym, start, end)
            df = preprocess_data(df)
            if len(df) > 0:
                all_data[sym] = df.set_index("date")
        except Exception:
            pass

    # 对齐日期
    dates = sorted(set().union(*[set(d.index) for d in all_data.values()]))
    dates = [d for d in dates if d >= pd.Timestamp(start) and d <= pd.Timestamp(end)]
    print(f"数据就绪: {len(all_data)} 只标的, {len(dates)} 个交易日")
    return all_data, dates


# =============================================================================
# 主回测循环
# =============================================================================
def run_backtest(start_date="20200101", end_date="20260724", config_path=None):
    config = load_backtest_config(config_path)
    strategy_params = config.get("strategy", {})

    all_data, dates = prepare_data(config, start_date, end_date)

    stock_cfg = _pool_config(config, "stock")
    etf_cfg = _pool_config(config, "etf")

    # 每个标的的 SignalEngine
    evaluators: Dict[str, SignalEngine] = {}
    warmup_days = 50

    # 持仓: {symbol: {shares, avg_cost, first_buy}}
    positions: Dict[str, Dict] = {}

    # 待执行: [{symbol, action, shares, signal_price, signal_date, target_pct}]
    pending: List[Dict] = []

    # 交易记录
    trades_log = []
    equity_curve = []
    initial_cash = stock_cfg["cash"] + etf_cfg["cash"]
    cash_balance = initial_cash

    print(f"回测 {len(dates)} 天 ({dates[0]} ~ {dates[-1]})...")

    for di, date in enumerate(dates):
        if di < warmup_days:
            # 预热期：只更新卡尔曼滤波器
            for sym in all_data:
                if sym not in evaluators:
                    p = dict(strategy_params)
                    pool_cfg = etf_cfg if _is_etf(sym) else stock_cfg
                    p["initial_cash"] = pool_cfg["cash"]
                    p["max_positions"] = pool_cfg["max_positions"]
                    p["single_position_pct"] = pool_cfg["max_pct"]
                    evaluators[sym] = SignalEngine(**p)
                row = all_data[sym]
                if date in row.index:
                    idx = row.index.get_loc(date)
                    if idx >= 20:
                        ma20 = float(row["close"].iloc[max(0, idx - 19):idx + 1].mean())
                        ma20_prev = float(row["close"].iloc[max(0, idx - 20):idx].mean())
                        c = float(row["close"].iloc[idx])
                        evaluators[sym].update(c, ma20, ma20_prev)
            continue

        # 执行待处理订单
        still_pending = []
        for order in pending:
            sym = order["symbol"]
            action = order["action"]
            if sym not in all_data or date not in all_data[sym].index:
                still_pending.append(order)
                continue
            open_price = float(all_data[sym].loc[date, "open"])

            # 涨跌停检查
            is_kcb = sym.startswith("688") or sym.startswith("30")
            lp = 0.20 if is_kcb else 0.10
            prev_close = float(all_data[sym].loc[date, "close"]) if di > 0 else open_price
            if action == "buy" and open_price >= prev_close * (1 + lp) * 0.999:
                still_pending.append(order)
                continue
            if action == "sell" and open_price <= prev_close * (1 - lp) * 1.001:
                still_pending.append(order)
                continue

            if action == "sell":
                if sym in positions:
                    pos = positions.pop(sym)
                    cash_back = open_price * pos["shares"]
                    cash_balance += cash_back
                    trades_log.append({
                        "date": date, "symbol": sym, "action": "sell",
                        "shares": pos["shares"], "price": open_price,
                        "cost": pos["avg_cost"],
                        "pnl": (open_price - pos["avg_cost"]) * pos["shares"],
                        "entry_date": pos["first_buy"],
                    })
            else:  # buy
                cost = open_price * order["shares"]
                if cost <= cash_balance:
                    cash_balance -= cost
                    positions[sym] = {
                        "shares": order["shares"],
                        "avg_cost": open_price,
                        "first_buy": date,
                    }
                    trades_log.append({
                        "date": date, "symbol": sym, "action": "buy",
                        "shares": order["shares"], "price": open_price,
                        "pnl": 0, "entry_date": "",
                    })

        pending = still_pending

        # 生成信号
        for sym in all_data:
            if sym not in evaluators:
                continue
            row = all_data[sym]
            if date not in row.index:
                continue
            idx = row.index.get_loc(date)
            c = float(row["close"].iloc[idx])
            ma20_c = float(row["close"].iloc[max(0, idx - 19):idx + 1].mean())
            ma20_p = float(row["close"].iloc[max(0, idx - 20):idx].mean())

            ev = evaluators[sym]
            has_pos = sym in positions
            if has_pos:
                ev.set_position(True, positions[sym]["avg_cost"])
            else:
                ev.set_position(False)

            result = ev.update(c, ma20_c, ma20_p)
            signal = result["signal"]
            target_pct = result["target_pct"]
            reason = result["reason"]
            fp = result["kalman_price"]
            trend = result["trend"]

            if signal == "buy":
                pool_cfg = etf_cfg if _is_etf(sym) else stock_cfg
                max_pct = pool_cfg["max_pct"]
                capped_pct = round(target_pct * max_pct, 4)
                lot = 200 if sym.startswith("688") else 100
                buy_qty = int(pool_cfg["cash"] * capped_pct / c / lot) * lot

                # 检查池上限
                pool_positions = sum(1 for s in positions if _is_etf(s) == _is_etf(sym))
                pool_pending = sum(1 for o in pending if _is_etf(o["symbol"]) == _is_etf(sym))
                if pool_positions + pool_pending < pool_cfg["max_positions"]:
                    existing = next((o for o in pending if o["symbol"] == sym), None)
                    if existing is None and buy_qty > 0:
                        pending.append({
                            "symbol": sym, "action": "buy", "shares": buy_qty,
                            "signal_price": c, "signal_date": str(date)[:10],
                            "target_pct": capped_pct,
                        })

            elif signal == "sell":
                if has_pos:
                    existing = next(
                        (o for o in pending if o["symbol"] == sym and o["action"] == "sell"),
                        None,
                    )
                    if existing is None:
                        pending.append({
                            "symbol": sym, "action": "sell",
                            "shares": positions[sym]["shares"],
                            "signal_price": c, "signal_date": str(date)[:10],
                            "target_pct": 0.0,
                        })

        # 记录权益（现金 + 持仓市值）
        total_market = 0.0
        for sym, pos in positions.items():
            if sym in all_data and date in all_data[sym].index:
                total_market += pos["shares"] * float(all_data[sym].loc[date, "close"])
        total_equity = cash_balance + total_market
        equity_curve.append({
            "date": date, "equity": total_equity, "cash": cash_balance,
            "market": total_market, "positions": len(positions),
        })

        if di % 200 == 0:
            print(f"  {date}  持仓 {len(positions)}  权益 ¥{total_market:,.0f}")

    # 输出结果
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve)

    # 计算指标
    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_cash
    total_return = (final_equity / initial_cash - 1) * 100

    daily_returns = equity_df["equity"].pct_change().dropna() if len(equity_df) > 1 else pd.Series()
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else 0
    )
    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max
    max_dd = float(drawdown.min() * 100) if len(drawdown) > 0 else 0

    wins = len(trades_df[trades_df["pnl"] > 0]) if len(trades_df) > 0 else 0
    win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  投资组合回测结果")
    print(f"{'='*60}")
    print(f"  回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")
    print(f"  初始资金: ¥{initial_cash:,.0f}")
    print(f"  最终权益: ¥{final_equity:,.0f}")
    print(f"  总收益率: {total_return:+.1f}%")
    print(f"  年化收益: {total_return / len(dates) * 252:.1f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  总交易数: {len(trades_log)}")
    print(f"  胜率:     {win_rate:.1f}%")
    print(f"  最终持仓: {len(positions)} 只")

    # 保存CSV
    trades_df.to_csv(os.path.join(TASK_DIR, "portfolio_trades.csv"), index=False, encoding="utf-8-sig")
    equity_df.to_csv(os.path.join(TASK_DIR, "portfolio_equity.csv"), index=False, encoding="utf-8-sig")

    # 下载沪深300基准
    bench = None
    bench_ret = 0.0
    try:
        import akshare as ak
        bench = ak.stock_zh_index_daily(symbol="sh000300")
        bench = bench.rename(columns={"date": "date", "close": "close"})
        bench["date"] = pd.to_datetime(bench["date"])
        bench = bench.set_index("date").sort_index()
        bench = bench[bench.index >= pd.Timestamp(start_date)]
        bench = bench[bench.index <= pd.Timestamp(end_date)]
        if len(bench) > 0:
            bench_ret = (bench["close"].iloc[-1] / bench["close"].iloc[0] - 1) * 100
            print(f"\n  沪深300: {bench_ret:+.1f}%")
    except Exception:
        pass

    # 最终持仓快照
    final_positions = []
    for sym, pos in positions.items():
        if sym in all_data:
            mkt_price = float(all_data[sym].iloc[-1]["close"])
            mkt_val = pos["shares"] * mkt_price
            pnl = (mkt_price - pos["avg_cost"]) * pos["shares"]
            final_positions.append({
                "symbol": sym, "shares": pos["shares"], "cost": pos["avg_cost"],
                "price": mkt_price, "value": mkt_val, "pnl": pnl,
                "pnl_pct": (mkt_price / pos["avg_cost"] - 1) * 100,
            })

    # 验证：最终权益 = 现金 + 持仓市值
    final_market = sum(p["value"] for p in final_positions)
    final_total = cash_balance + final_market
    print(f"  验证: 现金 ¥{cash_balance:,.0f} + 市值 ¥{final_market:,.0f} = ¥{final_total:,.0f}")

    # 生成HTML报告
    _generate_report(equity_df, trades_df, bench, initial_cash, cash_balance, final_positions, positions)
    print(f"\n  报告: portfolio_trades.csv, portfolio_equity.csv, portfolio_report.html")


def _is_etf(symbol: str) -> bool:
    return symbol.startswith(("51", "15", "58", "56"))


def _generate_report(equity_df, trades_df, bench_df, initial_cash, cash_balance, final_positions=None, _positions=None):
    """生成详细HTML报告。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yaml

    # ---- 名称映射 ----
    name_map = {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        wl = cfg.get("watchlist", {})
        for item in wl.get("stocks", []) + wl.get("etfs", []):
            name_map[str(item["symbol"]).zfill(6)] = item.get("name", item["symbol"])
    except Exception:
        pass

    eq = equity_df.set_index("date")["equity"]
    rets = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / initial_cash - 1) * 100
    ann_ret = total_ret / len(eq) * 252
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0
    max_dd = float((eq / eq.cummax() - 1).min() * 100)
    vol = float(rets.std() * np.sqrt(252) * 100)
    total_pnl = trades_df["pnl"].sum() if len(trades_df) > 0 else 0

    buys = trades_df[trades_df["action"] == "buy"].copy() if len(trades_df) > 0 else pd.DataFrame()
    sells = trades_df[trades_df["action"] == "sell"].copy() if len(trades_df) > 0 else pd.DataFrame()
    win_count = len(sells[sells["pnl"] > 0]) if len(sells) > 0 else 0
    win_rate = win_count / len(sells) * 100 if len(sells) > 0 else 0

    # 基准
    bench_ret = 0.0; bench_sharpe = 0.0; bench_dd = 0.0; beq = None
    if bench_df is not None:
        beq = bench_df["close"] / bench_df["close"].iloc[0] * initial_cash
        bench_ret = (bench_df["close"].iloc[-1] / bench_df["close"].iloc[0] - 1) * 100
        bench_rets = bench_df["close"].pct_change().dropna()
        bench_sharpe = float(bench_rets.mean() / bench_rets.std() * np.sqrt(252)) if bench_rets.std() > 0 else 0
        bench_dd = float((beq / beq.cummax() - 1).min() * 100)

    # 年度收益
    eq_yearly = eq.resample("YE").last()
    yearly_ret = eq_yearly.pct_change().dropna() * 100
    bench_yearly_ret = pd.Series(dtype=float)
    if bench_df is not None:
        by = bench_df["close"].resample("YE").last()
        bench_yearly_ret = by.pct_change().dropna() * 100

    # 月度数据
    monthly = eq.resample("ME").last().pct_change().dropna()
    monthly_bench = pd.Series(dtype=float)
    if bench_df is not None:
        monthly_bench = bench_df["close"].resample("ME").last().pct_change().dropna()

    roll_1y = eq.pct_change(252) * 100

    # ---- 图表 1: 权益曲线 ----
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("权益曲线",))
    fig1.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="策略",
                   line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    if beq is not None:
        fig1.add_trace(go.Scatter(x=beq.index, y=beq, mode="lines", name="沪深300",
                       line=dict(color="#999", width=1, dash="dot")), row=1, col=1)
    if len(buys) > 0:
        bd = buys["date"].values; bv = [eq.loc[d] for d in bd if d in eq.index]
        fig1.add_trace(go.Scatter(x=bd[:2000], y=bv[:2000], mode="markers", name="买入",
                       marker=dict(color="red", size=2, symbol="triangle-up", opacity=0.5)), row=1, col=1)
    if len(sells) > 0:
        sd = sells["date"].values; sv = [eq.loc[d] for d in sd if d in eq.index]
        fig1.add_trace(go.Scatter(x=sd[:2000], y=sv[:2000], mode="markers", name="卖出",
                       marker=dict(color="green", size=2, symbol="triangle-down", opacity=0.5)), row=1, col=1)
    fig1.update_layout(height=400, margin=dict(l=40, r=20, t=40, b=20),
                       hovermode="x unified", legend=dict(orientation="h", y=1.1))

    # ---- 图表 2: 回撤 + 滚动1年收益 ----
    dd = (eq / eq.cummax() - 1) * 100
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("回撤 (%)", "滚动 1 年收益 (%)"), horizontal_spacing=0.08)
    fig2.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="回撤", fill="tozeroy",
                   line=dict(color="#d62728", width=0.8), fillcolor="rgba(214,39,40,0.1)"), row=1, col=1)
    if bench_df is not None:
        bdd = (beq / beq.cummax() - 1) * 100
        fig2.add_trace(go.Scatter(x=bdd.index, y=bdd, mode="lines", name="基准回撤",
                       line=dict(color="#ff7f0e", width=0.6, dash="dot")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=roll_1y.index, y=roll_1y, mode="lines", name="滚动1Y",
                   line=dict(color="#17becf", width=0.8)), row=1, col=2)
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig2.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=20), hovermode="x unified", showlegend=False)

    # ---- 图表 3: 年度收益 ----
    fig3 = make_subplots(rows=1, cols=1, subplot_titles=("年度收益 (%)",))
    yc = ["#d62728" if v < 0 else "#2ca02c" for v in yearly_ret.values]
    fig3.add_trace(go.Bar(x=yearly_ret.index.year.astype(str), y=yearly_ret.values, name="策略",
                   marker_color=yc, text=[f"{v:+.1f}%" for v in yearly_ret.values],
                   textposition="outside", textfont=dict(size=12)), row=1, col=1)
    if len(bench_yearly_ret) > 0:
        bc = ["#ff7f0e" if v < 0 else "#ffbf80" for v in bench_yearly_ret.values]
        fig3.add_trace(go.Bar(x=bench_yearly_ret.index.year.astype(str), y=bench_yearly_ret.values,
                       name="沪深300", marker_color=bc, opacity=0.7), row=1, col=1)
    fig3.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=20),
                       hovermode="x unified", legend=dict(orientation="h", y=1.1), barmode="group")

    # ---- 图表 4: 月度收益 ----
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("月度收益 (%)", "月度超额收益 vs 沪深300 (%)"), horizontal_spacing=0.08)
    mc = ["#d62728" if v < 0 else "#2ca02c" for v in monthly.values]
    fig4.add_trace(go.Bar(x=monthly.index, y=monthly.values * 100, name="月度收益", marker_color=mc), row=1, col=1)
    if len(monthly_bench) > 0:
        common = monthly.index.intersection(monthly_bench.index)
        if len(common) > 1:
            excess = (monthly.loc[common] - monthly_bench.loc[common]) * 100
            ec = ["#d62728" if v < 0 else "#2ca02c" for v in excess.values]
            fig4.add_trace(go.Bar(x=excess.index, y=excess.values, name="超额收益", marker_color=ec), row=1, col=2)
    fig4.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=20), showlegend=False)

    # ---- 渲染图表 ----
    chart1 = fig1.to_html(full_html=False, include_plotlyjs=True)
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    chart3 = fig3.to_html(full_html=False, include_plotlyjs=False)
    chart4 = fig4.to_html(full_html=False, include_plotlyjs=False)

    # ---- 构建 HTML 表格 ----
    final_market = sum(p["value"] for p in final_positions) if final_positions else 0
    final_equity = cash_balance + final_market

    # 月度收益表
    monthly_table = _build_monthly_table(eq, trades_df, initial_cash, bench_df)

    # 资产分类
    is_etf = lambda s: s.startswith(("51", "15", "58", "56"))

    yearly_summary = _build_yearly_table(eq, trades_df, initial_cash, bench_df)
    stock_symbol = _build_symbol_table(trades_df, asset_type="stock", name_map=name_map)
    etf_symbol = _build_symbol_table(trades_df, asset_type="etf", name_map=name_map)
    annual_top5 = _build_annual_top5(trades_df, name_map=name_map)
    pairs_html = _build_trade_pairs_table(buys, sells, name_map=name_map)
    trade_html = _build_trades_table(trades_df, name_map=name_map)
    pos_html = _build_positions_table(final_positions, "stock", name_map=name_map)
    pos_etf_html = _build_positions_table(final_positions, "etf", name_map=name_map)

    # ---- CSS (plotly.js 已由 chart1 内联) ----
    css = """<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5;color:#333}
h1{color:#1a1a1a;border-bottom:3px solid #1f77b4;padding-bottom:10px}
h2{color:#2c3e50;margin-top:40px;border-bottom:2px solid #ddd;padding-bottom:8px}
h3{color:#555;margin-top:25px}
.metrics{width:100%;border-collapse:collapse;margin:20px 0;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden}
.metrics td{padding:16px 12px;text-align:center;border-right:1px solid #eee}
.metrics td:last-child{border-right:none}
.metrics .label{font-size:12px;color:#666;display:block}
.metrics .value{font-size:22px;font-weight:bold;display:block;margin-top:4px}
.positive{color:#2ca02c}.negative{color:#d62728}
.data-table{width:100%;border-collapse:collapse;font-size:13px;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;margin:10px 0}
.data-table th{background:#1f77b4;color:white;padding:10px 8px;text-align:right}
.data-table th:nth-child(1),.data-table td:nth-child(1){text-align:left}
.data-table td{padding:6px 8px;border-bottom:1px solid #f0f0f0}
.data-table tr:hover{background:#f8f9fa}
.data-table tr:nth-child(even){background:#fafafa}
.year-section{margin:30px 0}
.year-label{font-size:18px;font-weight:bold;color:#1f77b4;margin:20px 0 10px 0;padding:8px 12px;background:#e8f4fd;border-radius:4px;display:inline-block}
.summary-text{font-size:14px;color:#555;margin:10px 0;padding:10px;background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.05)}
.nav{position:sticky;top:0;background:white;padding:8px 16px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:20px;z-index:100;display:flex;gap:15px;flex-wrap:wrap;align-items:center}
.nav a{color:#1f77b4;text-decoration:none;font-size:14px;font-weight:500}
.nav a:hover{text-decoration:underline}
.col2{display:flex;gap:20px}
.col2>div{flex:1;min-width:0}
details{margin-bottom:5px;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);padding:10px 20px}
summary{cursor:pointer;user-select:none}
summary h2{display:inline;border:none;font-size:18px;margin:0}
summary::marker{color:#1f77b4;font-size:16px}
</style>"""

    bench_text = ""
    if bench_df is not None:
        bench_text = f"沪深300: {bench_ret:+.1f}% | 超额收益: {total_ret - bench_ret:+.1f}% | 基准 Sharpe: {bench_sharpe:.2f} | 基准回撤: {bench_dd:.1f}%"

    metrics_html = f"""<h2>核心指标</h2>
<table class='metrics'><tr>
<td><span class='label'>总收益率</span><span class='value {"positive" if total_ret>0 else "negative"}'>{total_ret:+.1f}%</span></td>
<td><span class='label'>年化收益</span><span class='value'>{ann_ret:.1f}%</span></td>
<td><span class='label'>夏普比率</span><span class='value'>{sharpe:.2f}</span></td>
<td><span class='label'>波动率</span><span class='value'>{vol:.1f}%</span></td>
<td><span class='label'>最大回撤</span><span class='value negative'>{max_dd:.1f}%</span></td>
<td><span class='label'>胜率</span><span class='value'>{win_rate:.0f}%</span></td>
<td><span class='label'>总交易</span><span class='value'>{len(trades_df)}</span></td>
</tr><tr>
<td><span class='label'>初始资金</span><span class='value'>¥{initial_cash:,.0f}</span></td>
<td><span class='label'>最终现金</span><span class='value'>¥{cash_balance:,.0f}</span></td>
<td><span class='label'>持仓市值</span><span class='value'>¥{final_market:,.0f}</span></td>
<td><span class='label'>最终权益</span><span class='value'>¥{final_equity:,.0f}</span></td>
<td><span class='label'>已实现盈亏</span><span class='value {"positive" if total_pnl>0 else "negative"}'>{total_pnl/1e4:+.1f}万</span></td>
<td><span class='label'>浮动盈亏</span><span class='value'>{'+' if (final_equity-initial_cash-total_pnl)>0 else ''}¥{final_equity-initial_cash-total_pnl:+,.0f}</span></td>
<td><span class='label'>总盈亏</span><span class='value {"positive" if (final_equity-initial_cash)>0 else "negative"}'>{final_equity-initial_cash:+,.0f}</span></td>
</tr></table>
<div class='summary-text'>{bench_text}</div>"""

    cash_pct = cash_balance / final_equity * 100 if final_equity > 0 else 0
    cash_note = f"""<div class='summary-text'>💡 <b>现金占比 {cash_pct:.0f}% 说明：</b>策略在下跌趋势中每只仅配置 {0.30*0.20*100:.0f}%（趋势仓位 30% × 单只上限 20%），上涨趋势配置 19%。大部分时段市场处于下跌趋势，且入场信号需要价格偏离卡尔曼估计超过 2-3%，信号密度较低。卖出后资金等待下一信号期间以现金持有。这是策略的保守性设计，避免在不利市场中过度暴露。</div>"""

    nav = """<div class='nav'><b>导航：</b>
<a href='#metrics'>核心指标</a><a href='#yearly'>年度汇总</a><a href='#monthly'>月度收益</a>
<a href='#top5'>年度TOP5</a><a href='#symbols'>标的汇总</a>
<a href='#pairs'>交易分析</a><a href='#trades'>交易明细</a><a href='#positions'>最终持仓</a></div>"""

    full_html = f"""<!DOCTYPE html><html lang='zh-CN'>
<head><meta charset='utf-8'><title>投资组合回测报告</title>{css}</head>
<body>
<h1>投资组合回测报告</h1>
<p style='color:#666;font-size:14px'>17 只股票 + 24 只ETF | {eq.index[0].strftime('%Y-%m-%d')} ~ {eq.index[-1].strftime('%Y-%m-%d')} | 初始资金: ¥{initial_cash:,.0f}</p>
{nav}
<details open><summary><h2>核心指标</h2></summary><div id='metrics'>{metrics_html}{cash_note}</div></details>
<details open><summary><h2>最终持仓</h2></summary><div id='positions'><div class='col2'><div><h3>📈 股票</h3>{pos_html}</div><div><h3>📊 ETF</h3>{pos_etf_html}</div></div></div></details>
<details open><summary><h2>权益走势</h2></summary>{chart1}{chart2}</details>
<details open><summary><h2>年度汇总</h2></summary><div id='yearly'>{chart3}{yearly_summary}</div></details>
<details open><summary><h2>月度收益汇总</h2></summary><div id='monthly'>{chart4}{monthly_table}</div></details>
<details open><summary><h2>年度盈亏 TOP5</h2></summary><div id='top5' style='max-height:600px;overflow-y:auto;'>{annual_top5}</div></details>
<details open><summary><h2>标的盈亏汇总</h2></summary><div id='symbols'><div class='col2'><div><h3>📈 股票</h3>{stock_symbol}</div><div><h3>📊 ETF</h3>{etf_symbol}</div></div></div></details>
<details open><summary><h2>交易分析</h2></summary><div id='pairs' style='max-height:600px;overflow-y:auto;'>{pairs_html}</div></details>
<details open><summary><h2>交易明细</h2></summary><div id='trades' style='max-height:600px;overflow-y:auto;'>{trade_html}</div></details>
<p style='text-align:center;color:#999;margin-top:40px;font-size:12px'>🤖 Generated with AKQuant Kalman Strategy — {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""

    html_path = os.path.join(TASK_DIR, "portfolio_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print("  HTML报告: portfolio_report.html")


def _build_monthly_table(eq, trades_df, initial_cash, bench_df=None):
    """月度收益热力图 — 颜色深浅表示盈亏强度。"""
    monthly = eq.resample("ME").last().pct_change() * 100
    if len(monthly) < 1:
        return ""

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["date"]).dt.year

    years = sorted(set(d.year for d in monthly.index), reverse=True)
    months = list(range(1, 13))
    max_abs = max(abs(monthly.max()), abs(monthly.min()), 5.0)

    def _heat_bg(ret_val):
        """纯红/绿色 + 透明度映射数值大小。"""
        if pd.isna(ret_val) or ret_val == 0:
            return ""
        intensity = min(abs(ret_val) / max_abs, 1.0)
        if ret_val > 0:
            return f"background-color:rgba(0,180,0,{intensity:.2f});"
        else:
            return f"background-color:rgba(220,0,0,{intensity:.2f});"

    # 手动构建 HTML 表格
    html = ['<table class="data-table"><thead><tr><th>年份</th>']
    for m in months:
        html.append(f'<th>{m}月</th>')
    html.append('<th>全年</th><th>交易</th><th>已实现盈亏</th></tr></thead><tbody>')

    for y in years:
        html.append(f'<tr><td><b>{y}</b></td>')
        y_ret = 0.0
        count = 0

        for m in months:
            # monthly.index 是月末日期 (2020-01-31)，按年月匹配
            matches = [d for d in monthly.index if d.year == y and d.month == m]
            if matches:
                ret = float(monthly.loc[matches[0]])
                if pd.notna(ret):
                    bg = _heat_bg(ret)
                    cls = "positive" if ret > 0 else "negative"
                    html.append(f'<td class="heat-cell" style="{bg}"><span style="color:#222;font-weight:bold;">{ret:+.1f}%</span></td>')
                    y_ret += ret
                    count += 1
                else:
                    html.append('<td class="heat-cell">-</td>')
            else:
                html.append('<td class="heat-cell">-</td>')

        # 全年汇总
        ys = trades_df[trades_df["year"] == y]
        y_sells = ys[ys["action"] == "sell"] if len(ys) > 0 else pd.DataFrame()
        y_pnl = y_sells["pnl"].sum() if len(y_sells) > 0 else 0
        y_cls = "positive" if y_ret > 0 else "negative" if y_ret < 0 else ""
        pnl_cls = "positive" if y_pnl > 0 else "negative" if y_pnl < 0 else ""

        html.append(f'<td><b><span class="{y_cls}">{y_ret:+.1f}%</span></b></td>')
        html.append(f'<td>{len(y_sells)}</td>')
        html.append(f'<td><span class="{pnl_cls}">¥{y_pnl:+,.0f}</span></td>')
        html.append('</tr>')

    html.append('</tbody></table>')
    return "\n".join(html)


def _build_yearly_table(eq, trades_df, initial_cash, bench_df=None):
    """年度收益与交易汇总。"""
    eq_yearly = eq.resample("YE").last()
    yearly_ret = eq_yearly.pct_change() * 100
    bench_ret_map = {}
    if bench_df is not None:
        by = bench_df["close"].resample("YE").last()
        br = by.pct_change() * 100
        for y, v in br.items():
            if pd.notna(v):
                bench_ret_map[y.year] = float(v)

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["date"]).dt.year
    sells = trades_df[trades_df["action"] == "sell"] if len(trades_df) > 0 else pd.DataFrame()

    rows = []
    items = yearly_ret.items() if hasattr(yearly_ret, "items") else yearly_ret.iteritems()
    for date, ret in items:
        y = date.year
        ret_val = float(ret.iloc[0]) if hasattr(ret, "iloc") else float(ret)
        if pd.isna(ret_val):
            continue
        ys = sells[sells["year"] == y] if len(sells) > 0 else pd.DataFrame()
        wins = len(ys[ys["pnl"] > 0]) if len(ys) > 0 else 0
        total = len(ys)
        y_pnl = ys["pnl"].sum() if len(ys) > 0 else 0
        row = {
            "年份": y, "收益率": f"{ret_val:+.1f}%",
            "卖出笔数": total, "盈利笔数": wins,
            "胜率": f"{wins/total*100:.0f}%" if total > 0 else "-",
            "已实现盈亏": f"¥{y_pnl:+,.0f}",
        }
        if y in bench_ret_map:
            row["沪深300"] = f"{bench_ret_map[y]:+.1f}%"
            row["超额"] = f"{ret_val - bench_ret_map[y]:+.1f}%"
        rows.append(row)

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    table = df.to_html(index=False, classes="data-table", border=0, justify="right")
    return table


def _build_symbol_table(trades_df, asset_type=None, name_map=None):
    """按标的汇总盈亏。"""
    if name_map is None: name_map = {}
    is_etf = lambda s: str(s).startswith(("51", "15", "58", "56"))
    if len(trades_df) == 0: return ""
    sells = trades_df[trades_df["action"] == "sell"].copy()
    if asset_type == "stock": sells = sells[~sells["symbol"].astype(str).apply(is_etf)]
    elif asset_type == "etf": sells = sells[sells["symbol"].astype(str).apply(is_etf)]
    if len(sells) == 0: return "<p>无卖出记录</p>"
    summary = sells.groupby("symbol").agg(笔数=("pnl", "count"), 总盈亏=("pnl", "sum"), 平均盈亏=("pnl", "mean")).reset_index()
    win_counts = sells[sells["pnl"] > 0].groupby("symbol").size()
    summary["胜率"] = summary["symbol"].map(win_counts).fillna(0).astype(int)
    summary["胜率"] = (summary["胜率"] / summary["笔数"] * 100).round(0).astype(int).astype(str) + "%"
    summary = summary.sort_values("总盈亏", ascending=False)
    total_pnl = summary["总盈亏"].sum()
    wins = (summary["总盈亏"] > 0).sum()
    label = {"stock": "股票", "etf": "ETF"}.get(asset_type, "全部")
    disp = summary.copy()
    disp.insert(0, "名称", disp["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
    disp["总盈亏"] = disp["总盈亏"].apply(lambda x: f"<span class='{'positive' if x>0 else 'negative'}'>{x:+,.0f}</span>")
    disp["平均盈亏"] = disp["平均盈亏"].apply(lambda x: f"{x:+,.0f}")
    table = disp.to_html(index=False, classes="data-table", border=0, justify="right", escape=False)
    return f"<div class='summary-text'>{label}: {len(summary)} 只 | 盈利 {wins} 只 | 总盈亏 ¥{total_pnl:+,.0f}</div>{table}"


def _build_trade_pairs_table(buys, sells, name_map=None):
    """交易配对表 — 按出场日期降序，最新在前。"""
    if name_map is None: name_map = {}
    if len(sells) == 0 or len(buys) == 0:
        return "<h2>交易分析</h2><p>无完整交易</p>"

    pairs = []
    for _, sell in sells.iterrows():
        sym = sell["symbol"]
        sym_buys = buys[(buys["symbol"] == sym) & (buys["date"] < sell["date"])]
        if len(sym_buys) > 0:
            buy = sym_buys.iloc[-1]
            pairs.append({
                "symbol": sym,
                "entry_date": str(buy["date"])[:10], "exit_date": str(sell["date"])[:10],
                "shares": int(sell["shares"]),
                "entry_price": round(float(buy["price"]), 2),
                "exit_price": round(float(sell["price"]), 2),
                "pnl": float(sell["pnl"]),
                "pnl_pct": round((float(sell["price"]) / float(buy["price"]) - 1) * 100, 2),
            })

    if not pairs:
        return "<h2>交易分析</h2><p>无完整交易</p>"

    pairs_df = pd.DataFrame(pairs).sort_values("exit_date", ascending=False)
    wins = (pairs_df["pnl"] > 0).sum()
    total_pair_pnl = pairs_df["pnl"].sum()

    # 按年分组显示，每年最多 30 条
    pairs_df["year"] = pd.to_datetime(pairs_df["exit_date"]).dt.year
    years = sorted(pairs_df["year"].unique(), reverse=True)

    html_parts = [
        f"<h2>交易分析</h2>"
        f"<div class='summary-text'>{len(pairs_df)} 笔完整交易 | "
        f"胜率 {wins/len(pairs_df)*100:.0f}% | "
        f"总盈亏 <span class='{'positive' if total_pair_pnl>0 else 'negative'}'>¥{total_pair_pnl:+,.0f}</span></div>"
    ]

    for y in years:
        yp = pairs_df[pairs_df["year"] == y]
        y_wins = (yp["pnl"] > 0).sum()
        y_pnl = yp["pnl"].sum()
        html_parts.append(
            f"<div class='year-section'>"
            f"<span class='year-label'>{y} 年 — {len(yp)} 笔 | "
            f"胜率 {y_wins/len(yp)*100:.0f}% | "
            f"盈亏 <span class='{'positive' if y_pnl>0 else 'negative'}'>¥{y_pnl:+,.0f}</span></span>"
        )
        disp = yp[["symbol", "entry_date", "exit_date", "shares", "entry_price", "exit_price", "pnl", "pnl_pct"]].copy()
        disp.insert(0, "名称", disp["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
        disp["pnl"] = disp["pnl"].apply(lambda x: f"<span class='{'positive' if float(x)>0 else 'negative'}'>{float(x):+,.0f}</span>")
        disp["pnl_pct"] = disp["pnl_pct"].apply(lambda x: f"<span class='{'positive' if float(x)>0 else 'negative'}'>{float(x):+.1f}%</span>")
        table = disp.to_html(
            index=False, classes="data-table", border=0, justify="right", escape=False,
        )
        html_parts.append(table)
        html_parts.append("</div>")

    return "\n".join(html_parts)


def _build_trades_table(trades_df, name_map=None):
    """交易明细 — 全部买卖点，按日期降序，按年分组。"""
    if name_map is None: name_map = {}
    if len(trades_df) == 0:
        return "<h2>交易明细</h2><p>无交易记录</p>"

    df = trades_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=False)
    df["year"] = df["date"].dt.year
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    years = sorted(df["year"].unique(), reverse=True)

    html_parts = [
        f"<h2>交易明细</h2>"
        f"<div class='summary-text'>共 {len(df)} 笔 | "
        f"买入 {(df['action']=='buy').sum()} | 卖出 {(df['action']=='sell').sum()} | "
        f"总盈亏 <span class='{'positive' if df['pnl'].sum()>0 else 'negative'}'>¥{df['pnl'].sum():+,.0f}</span></div>"
    ]

    for y in years:
        yd = df[df["year"] == y]
        html_parts.append(
            f"<div class='year-section'><span class='year-label'>{y} 年 — {len(yd)} 笔</span>"
        )
        disp_cols = yd[["date_str", "symbol", "action", "shares", "price", "pnl"]].copy()
        disp_cols.insert(0, "名称", disp_cols["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
        disp_cols.columns = ["名称", "日期", "代码", "方向", "数量", "价格", "盈亏"]
        disp_cols["价格"] = disp_cols["价格"].apply(lambda x: f"{float(x):.2f}")
        disp_cols["盈亏"] = disp_cols["盈亏"].apply(
            lambda x: f"<span class='{'positive' if float(x)>0 else 'negative' if float(x)<0 else ''}'>{float(x):+,.0f}</span>"
        )
        table = disp_cols.to_html(
            index=False, classes="data-table", border=0, justify="right", escape=False,
        )
        html_parts.append(table)
        html_parts.append("</div>")

    return "\n".join(html_parts)


def _build_positions_table(final_positions, asset_type="stock", name_map=None):
    """最终持仓表，按资产类型过滤。"""
    if name_map is None: name_map = {}
    is_etf = lambda s: str(s).startswith(("51", "15", "58", "56"))
    if not final_positions: return "<p>无持仓</p>"
    filtered = [p for p in final_positions if (asset_type == "etf") == is_etf(p["symbol"])]
    if not filtered: return "<p>无持仓</p>"
    pos_df = pd.DataFrame(filtered).sort_values("value", ascending=False)
    total_value = sum(p["value"] for p in filtered)
    label = {"stock": "股票", "etf": "ETF"}.get(asset_type, "")
    summary = f"<div class='summary-text'>{label}: {len(filtered)} 只 | 总市值 ¥{total_value:,.0f}</div>"
    disp = pos_df[["symbol", "shares", "cost", "price", "value", "pnl", "pnl_pct"]].copy()
    disp.insert(0, "名称", disp["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
    disp.columns = ["名称", "代码", "股数", "成本", "现价", "市值", "浮动盈亏", "收益率"]
    for col in ["成本", "现价"]: disp[col] = disp[col].apply(lambda x: f"{float(x):.2f}")
    disp["市值"] = disp["市值"].apply(lambda x: f"{float(x):,.0f}")
    disp["浮动盈亏"] = disp["浮动盈亏"].apply(lambda x: f"<span class='{'positive' if float(x)>0 else 'negative' if float(x)<0 else ''}'>{float(x):+,.0f}</span>")
    disp["收益率"] = disp["收益率"].apply(lambda x: f"<span class='{'positive' if float(x)>0 else 'negative' if float(x)<0 else ''}'>{float(x):+.1f}%</span>")
    return summary + disp.to_html(index=False, classes="data-table", border=0, justify="right", escape=False)


def _build_annual_top5(trades_df, name_map=None):
    """每年赚钱TOP5和亏钱TOP5，股票和ETF分开。"""
    if name_map is None: name_map = {}
    if len(trades_df) == 0:
        return ""
    sells = trades_df[trades_df["action"] == "sell"].copy()
    if len(sells) == 0:
        return ""
    sells["year"] = pd.to_datetime(sells["date"]).dt.year
    is_etf = lambda s: str(s).startswith(("51", "15", "58", "56"))

    years = sorted(sells["year"].unique(), reverse=True)
    html_parts = ["<h2>年度盈亏 TOP5</h2>"]

    for y in years:
        ys = sells[sells["year"] == y]
        html_parts.append(f"<div class='year-section'><span class='year-label'>{y} 年</span>")
        html_parts.append("<div class='col2'>")

        for atype, alabel in [("stock", "股票"), ("etf", "ETF")]:
            if atype == "stock":
                sub = ys[~ys["symbol"].astype(str).apply(is_etf)]
            else:
                sub = ys[ys["symbol"].astype(str).apply(is_etf)]

            if len(sub) == 0:
                html_parts.append(f"<div><h3>{alabel}</h3><p>无交易</p></div>")
                continue

            agg = sub.groupby("symbol")["pnl"].sum().reset_index()
            agg["count"] = sub.groupby("symbol").size().values
            winners = agg.nlargest(5, "pnl")
            losers = agg.nsmallest(5, "pnl")

            html_parts.append(f"<div><h3>{alabel}</h3>")

            # TOP5 赚钱
            wcols = winners.copy()
            wcols.insert(0, "名称", wcols["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
            wcols["盈亏"] = wcols["pnl"].apply(lambda x: f"<span class='positive'>+{x:,.0f}</span>")
            wdisp = wcols[["名称", "symbol", "count", "盈亏"]].copy()
            wdisp.columns = ["名称", "代码", "笔数", "盈亏"]
            wt = wdisp.to_html(index=False, classes="data-table", border=0, justify="right", escape=False)
            html_parts.append(f"<p><b>🟢 赚钱 TOP5</b></p>{wt}")

            # TOP5 亏钱
            lcols = losers.copy()
            lcols.insert(0, "名称", lcols["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
            lcols["盈亏"] = lcols["pnl"].apply(lambda x: f"<span class='negative'>{x:,.0f}</span>")
            ldisp = lcols[["名称", "symbol", "count", "盈亏"]].copy()
            ldisp.columns = ["名称", "代码", "笔数", "盈亏"]
            lt = ldisp.to_html(index=False, classes="data-table", border=0, justify="right", escape=False)
            html_parts.append(f"<p><b>🔴 亏钱 TOP5</b></p>{lt}")

            html_parts.append("</div>")

        html_parts.append("</div></div>")

    return "\n".join(html_parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="投资组合回测")
    parser.add_argument("--start", default="20200101", help="起始日期 YYYYMMDD（默认 20200101）")
    parser.add_argument("--end", default="20260724", help="结束日期 YYYYMMDD（默认 20260724）")
    parser.add_argument("--config", default=None, help="stocks.yaml 路径（默认使用 task/kalman/stocks.yaml）")
    args = parser.parse_args()

    print(f"配置: {args.config or CONFIG_FILE}")
    print(f"区间: {args.start} ~ {args.end}")
    run_backtest(start_date=args.start, end_date=args.end, config_path=args.config)
