"""
实盘交易报告生成器。

从 positions.json / trades.csv / signals.csv / .cache/ 读取实际交易数据，
生成与 portfolio_backtest.py 相同风格的 HTML 报告。

用法:
    from live_report import build_live_report
    build_live_report()
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def build_live_report():
    """生成实盘交易报告。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ---- 加载数据 ----
    config = _load_config()
    positions = _load_positions()
    trades = _load_trades()
    name_map = _build_name_map(config)

    stock_cash = float(config.get("stock", {}).get("initial_cash", 200000))
    etf_cash = float(config.get("etf", {}).get("initial_cash", 100000))
    initial_cash = stock_cash + etf_cash

    # ---- 获取最新价格 ----
    price_map = _get_latest_prices()

    # ---- 计算持仓市值 ----
    pos_list = []
    pos_value = 0.0
    pos_pnl = 0.0
    for sym, p in positions.items():
        price = price_map.get(sym, p["avg_cost"])
        value = p["shares"] * price
        pnl = (price - p["avg_cost"]) * p["shares"]
        pos_value += value
        pos_pnl += pnl
        pos_list.append({
            "symbol": sym, "name": p["name"], "shares": p["shares"],
            "cost": p["avg_cost"], "price": price, "value": value,
            "pnl": pnl, "pnl_pct": (price / p["avg_cost"] - 1) * 100,
            "first_buy": p.get("first_buy_date", ""),
        })

    # ---- 交易统计（含 execution_log 中的买入成本） ----
    elog_path = os.path.join(TASK_DIR, "execution_log.csv")
    total_buy_cost = 0.0
    if os.path.exists(elog_path):
        elog = pd.read_csv(elog_path, dtype={"symbol": str, "exec_date": str})
        elog["symbol"] = elog["symbol"].str.zfill(6)
        exec_buys = elog[(elog["action"] == "buy") & (elog["status"] == "executed")]
        total_buy_cost = (exec_buys["shares"].astype(float) * exec_buys["exec_price"].astype(float)).sum()

    total_sell_proceeds = 0.0
    if len(trades) > 0:
        total_sell_proceeds = (trades["shares"].astype(float) * trades["exit_price"].astype(float)).sum()

    realized_pnl = trades["pnl"].sum() if len(trades) > 0 else 0.0
    total_pnl = realized_pnl + pos_pnl
    cash_balance = initial_cash - total_buy_cost + total_sell_proceeds
    final_equity = cash_balance + pos_value
    total_ret = total_pnl / initial_cash * 100

    sells = trades  # trades.csv 全部是已完成卖出
    win_count = len(sells[sells["pnl"] > 0]) if len(sells) > 0 else 0
    win_rate = win_count / len(sells) * 100 if len(sells) > 0 else 0

    # ---- 权益曲线（累积已实现盈亏 + 估算持仓市值） ----
    equity_curve = _build_equity_curve(trades, positions, price_map, initial_cash)
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    # 去重：同一天取最后一条
    eq = equity_curve.groupby("date")["equity"].last()
    # 沪深300基准(归一化到初始资金,与权益曲线对比)
    hs300_norm = None
    try:
        from data_utils import fetch_hs300
        hs300 = fetch_hs300(str(eq.index[0].date()).replace("-", ""),
                            str(eq.index[-1].date()).replace("-", ""))
        hs300 = hs300.reindex(eq.index).ffill()
        if len(hs300.dropna()) > 0:
            hs300_norm = hs300 / hs300.iloc[0] * eq.iloc[0]
    except Exception:
        pass

    # 回撤 + 年度收益(对齐回测报告 build_combined_report 格式)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    yearly_ret = eq.resample("YE").last().pct_change() * 100

    fig1 = make_subplots(rows=3, cols=1,
                         subplot_titles=("权益曲线(含沪深300对比)", "回撤(%)", "年度收益(%)"),
                         row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.12)
    fig1.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="权益",
                   line=dict(color="#1f77b4", width=1.5), fill="tozeroy",
                   fillcolor="rgba(31,119,180,0.1)"), row=1, col=1)
    if hs300_norm is not None:
        fig1.add_trace(go.Scatter(x=hs300_norm.index, y=hs300_norm.values, name="沪深300",
                       line=dict(color="#ff7f0e", dash="dash", width=1.2)), row=1, col=1)
    # 买入标记
    elog_path2 = os.path.join(TASK_DIR, "execution_log.csv")
    buys_log = pd.DataFrame()
    if os.path.exists(elog_path2):
        elog2 = pd.read_csv(elog_path2, dtype={"symbol": str, "exec_date": str})
        elog2["symbol"] = elog2["symbol"].str.zfill(6)
        buys_log = elog2[(elog2["action"] == "buy") & (elog2["status"] == "executed")].copy()
        if len(buys_log) > 0:
            bd = pd.to_datetime(buys_log["exec_date"].values)
            bv = [eq.loc[d] for d in bd if d in eq.index]
            bd_f = [d for d in bd if d in eq.index]
            # 去重坐标，显示名称
            buy_text = [f"{name_map.get(str(s).zfill(6), str(s))}" for s in buys_log["symbol"]]
            fig1.add_trace(go.Scatter(x=bd_f, y=bv, mode="markers", name="买入",
                           marker=dict(color="red", size=10, symbol="triangle-up",
                                       line=dict(width=1, color="darkred")),
                           text=buy_text, hoverinfo="text+x+y"), row=1, col=1)

    # 卖出标记
    if len(trades) > 0:
        sd = pd.to_datetime(trades["exit_date"].values)
        sv = [eq.loc[d] for d in sd if d in eq.index]
        sd_f = [d for d in sd if d in eq.index]
        sell_text = [f"{name_map.get(str(s).zfill(6), str(s))}" for s in trades["symbol"]]
        fig1.add_trace(go.Scatter(x=sd_f, y=sv, mode="markers", name="卖出",
                       marker=dict(color="green", size=10, symbol="triangle-down",
                                   line=dict(width=1, color="darkgreen")),
                       text=sell_text, hoverinfo="text+x+y"), row=1, col=1)
    # 回撤
    fig1.add_trace(go.Scatter(x=dd.index, y=dd.values, name="回撤", fill="tozeroy",
                   line=dict(color="#d62728", width=1)), row=2, col=1)
    # 年度收益
    fig1.add_trace(go.Bar(x=[d.year for d in yearly_ret.index], y=yearly_ret.values, name="年度收益",
                   marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in yearly_ret.values]), row=3, col=1)
    fig1.update_layout(height=800, margin=dict(l=40, r=20, t=40, b=20),
                       hovermode="x unified", legend=dict(orientation="h", y=1.06))

    # 图表 2: 月度收益
    if len(eq) > 20:
        monthly = eq.resample("ME").last().pct_change().dropna() * 100
        fig2 = make_subplots(rows=1, cols=1, subplot_titles=("月度收益 (%)",))
        mc = ["#d62728" if v < 0 else "#2ca02c" for v in monthly.values]
        fig2.add_trace(go.Bar(x=monthly.index, y=monthly.values, name="月度收益",
                       marker_color=mc, text=[f"{v:+.1f}%" for v in monthly.values],
                       textposition="outside", textfont=dict(size=11)), row=1, col=1)
        fig2.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=20), showlegend=False)
        chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    else:
        chart2 = ""

    chart1 = fig1.to_html(full_html=False, include_plotlyjs=False)

    # ---- 构建表格 ----
    def _nm(sym):
        s = str(sym).zfill(6)
        return name_map.get(s, s)

    is_etf = lambda s: str(s).startswith(("51", "15", "58", "56"))
    stock_pos = [p for p in pos_list if not is_etf(p["symbol"])]
    etf_pos = [p for p in pos_list if is_etf(p["symbol"])]

    pos_stock_html = _positions_table(stock_pos, name_map, "股票", stock_cash)
    pos_etf_html = _positions_table(etf_pos, name_map, "ETF", etf_cash)
    trades_html = _trades_table(trades, name_map)
    details_html = _trade_details(buys_log, trades, name_map)
    monthly_table = _monthly_heatmap(eq, trades)

    # ---- 指标卡片 ----
    stock_used = sum(p["value"] for p in stock_pos)
    etf_used = sum(p["value"] for p in etf_pos)
    ret_cls = "positive" if total_ret > 0 else "negative"
    rpnl_cls = "positive" if realized_pnl > 0 else "negative"
    upnl_cls = "positive" if pos_pnl > 0 else "negative"

    metrics = f"""<h2>核心指标</h2>
<table class='metrics'><tr>
<td><span class='label'>总收益率</span><span class='value {ret_cls}'>{total_ret:+.1f}%</span></td>
<td><span class='label'>已实现盈亏</span><span class='value {rpnl_cls}'>¥{realized_pnl:+,.0f}</span></td>
<td><span class='label'>浮动盈亏</span><span class='value {upnl_cls}'>¥{pos_pnl:+,.0f}</span></td>
<td><span class='label'>总盈亏</span><span class='value {ret_cls}'>¥{total_pnl:+,.0f}</span></td>
<td><span class='label'>初始资金</span><span class='value'>¥{initial_cash:,.0f}</span></td>
<td><span class='label'>最终现金</span><span class='value'>¥{cash_balance:,.0f}</span></td>
<td><span class='label'>持仓市值</span><span class='value'>¥{pos_value:,.0f}</span></td>
<td><span class='label'>最终权益</span><span class='value'>¥{final_equity:,.0f}</span></td>
</tr><tr>
<td><span class='label'>总交易</span><span class='value'>{len(trades)}</span></td>
<td><span class='label'>盈利笔数</span><span class='value'>{win_count}</span></td>
<td><span class='label'>胜率</span><span class='value'>{win_rate:.0f}%</span></td>
<td><span class='label'>当前持仓</span><span class='value'>{len(positions)} 只</span></td>
<td><span class='label'>股票池已用</span><span class='value'>¥{stock_used:,.0f}</span></td>
<td><span class='label'>ETF池已用</span><span class='value'>¥{etf_used:,.0f}</span></td>
<td></td><td></td>
</tr></table>"""

    # ---- 完整 HTML ----
    css = _css()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    start_date = eq.index[0].strftime("%Y-%m-%d") if len(eq) > 0 else "N/A"
    end_date = eq.index[-1].strftime("%Y-%m-%d") if len(eq) > 0 else "N/A"

    html = f"""<!DOCTYPE html><html lang='zh-CN'>
<head><meta charset='utf-8'><title>实盘交易报告</title>{css}
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>
<body>
<h1>实盘交易报告</h1>
<p style='color:#666;font-size:14px'>交易区间: {start_date} ~ {end_date} | 初始资金: ¥{initial_cash:,.0f} | 生成时间: {now}</p>
<details open><summary><h2>核心指标</h2></summary>{metrics}</details>
<details open><summary><h2>权益走势</h2></summary>{chart1}</details>
<details open><summary><h2>月度收益</h2></summary>{chart2}{monthly_table}</details>
<details open><summary><h2>当前持仓</h2></summary>
<div class='col2'><div><h3>股票</h3>{pos_stock_html}</div><div><h3>ETF</h3>{pos_etf_html}</div></div></details>
<details open><summary><h2>已完成交易</h2></summary>
<div style='max-height:600px;overflow-y:auto;'>{trades_html}</div></details>
<details open><summary><h2>交易明细</h2></summary>
<div style='max-height:600px;overflow-y:auto;'>{details_html}</div></details>
<p style='text-align:center;color:#999;margin-top:40px;font-size:12px'>🤖 Generated by AKQuant Live Report — {now}</p>
</body></html>"""

    report_path = os.path.join(TASK_DIR, "live_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  实盘报告: live_report.html")


# =============================================================================
# 数据加载 helpers
# =============================================================================
def _load_config():
    with open(os.path.join(TASK_DIR, "stocks.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_positions():
    path = os.path.join(TASK_DIR, "positions.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_trades():
    path = os.path.join(TASK_DIR, "trades.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["entry_date", "exit_date", "symbol", "name",
                                      "shares", "entry_price", "exit_price", "pnl", "pnl_pct"])
    df = pd.read_csv(path, dtype={"symbol": str})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    # 所有 trades.csv 中的行都是已完成交易（卖出）
    df["action"] = "sell"
    return df


def _build_name_map(config):
    name_map = {}
    wl = config.get("watchlist", {})
    for item in wl.get("stocks", []) + wl.get("etfs", []):
        name_map[str(item["symbol"]).zfill(6)] = item.get("name", item["symbol"])
    return name_map


def _get_latest_prices():
    """从缓存读取每只股票的最新收盘价。"""
    price_map = {}
    cache_dir = os.path.join(TASK_DIR, ".cache")
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".parquet"):
                try:
                    df = pd.read_parquet(os.path.join(cache_dir, fname))
                    if len(df) > 0:
                        sym = fname.replace(".parquet", "").zfill(6)
                        price_map[sym] = float(df["close"].iloc[-1])
                except Exception:
                    pass
    return price_map


def _build_equity_curve(trades, positions, price_map, initial_cash):
    """从 execution_log.csv + trades.csv + positions.json 重建权益曲线。

    数据来源:
    - execution_log.csv: 所有已执行的买入事件
    - trades.csv: 所有已完成的卖出事件
    - positions.json: 当前持仓（未平仓）
    """
    elog_path = os.path.join(TASK_DIR, "execution_log.csv")

    # 收集所有买卖事件
    events = []

    # 买入事件：从 execution_log.csv
    if os.path.exists(elog_path):
        elog = pd.read_csv(elog_path, dtype={"symbol": str, "exec_date": str})
        elog["symbol"] = elog["symbol"].str.zfill(6)
        executed_buys = elog[(elog["action"] == "buy") & (elog["status"] == "executed")]
        for _, b in executed_buys.iterrows():
            events.append({
                "date": str(b["exec_date"])[:10],
                "type": "buy",
                "symbol": str(b["symbol"]).zfill(6),
                "shares": int(b["shares"]),
                "price": float(b["exec_price"]),
            })

    # 卖出事件：从 trades.csv
    for _, t in trades.iterrows():
        exit_date = str(t.get("exit_date", ""))[:10]
        if exit_date and exit_date != "nan":
            events.append({
                "date": exit_date,
                "type": "sell",
                "symbol": str(t["symbol"]).zfill(6),
                "shares": int(t["shares"]),
                "price": float(t["exit_price"]),
            })

    events.sort(key=lambda e: e["date"])

    # 模拟持仓变化
    cash = initial_cash
    holdings = {}
    equity_rows = [{"date": events[0]["date"] if events else datetime.now().strftime("%Y-%m-%d"),
                    "equity": initial_cash}]

    for ev in events:
        if ev["type"] == "buy":
            cost = ev["shares"] * ev["price"]
            cash -= cost
            if ev["symbol"] in holdings:
                h = holdings[ev["symbol"]]
                total_shares = h["shares"] + ev["shares"]
                h["price"] = (h["shares"] * h["price"] + ev["shares"] * ev["price"]) / total_shares
                h["shares"] = total_shares
            else:
                holdings[ev["symbol"]] = {"shares": ev["shares"], "price": ev["price"]}
        elif ev["type"] == "sell":
            if ev["symbol"] in holdings:
                cash += ev["shares"] * ev["price"]
                del holdings[ev["symbol"]]

        market_value = sum(h["shares"] * price_map.get(s, h["price"])
                           for s, h in holdings.items())
        equity_rows.append({"date": ev["date"], "equity": cash + market_value})

    # 最终快照
    current_market = sum(
        p["shares"] * price_map.get(sym, p["avg_cost"])
        for sym, p in positions.items()
    )
    equity_rows.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "equity": cash + current_market,
    })

    return pd.DataFrame(equity_rows)


# =============================================================================
# HTML table builders
# =============================================================================
def _positions_table(pos_list, name_map, label, pool_cash):
    if not pos_list:
        return f"<div class='summary-text'>{label}: 无持仓 (可用 ¥{pool_cash:,.0f})</div>"
    df = pd.DataFrame(pos_list).sort_values("value", ascending=False)
    total = sum(p["value"] for p in pos_list)
    total_pnl = sum(p["pnl"] for p in pos_list)
    pnl_cls = "positive" if total_pnl > 0 else "negative" if total_pnl < 0 else ""
    summary = f"<div class='summary-text'>{label}: {len(pos_list)} 只 | 市值 ¥{total:,.0f} | 浮动 <span class='{pnl_cls}'>¥{total_pnl:+,.0f}</span></div>"
    disp = df[["symbol", "name", "shares", "cost", "price", "value", "pnl", "pnl_pct"]].copy()
    disp.columns = ["代码", "名称", "股数", "成本", "现价", "市值", "浮动盈亏", "收益率"]
    disp["成本"] = disp["成本"].apply(lambda x: f"{float(x):.2f}")
    disp["现价"] = disp["现价"].apply(lambda x: f"{float(x):.2f}")
    disp["市值"] = disp["市值"].apply(lambda x: f"{float(x):,.0f}")
    disp["浮动盈亏"] = disp["浮动盈亏"].apply(
        lambda x: f"<span class='{'positive' if float(x)>0 else 'negative' if float(x)<0 else ''}'>{float(x):+,.0f}</span>")
    disp["收益率"] = disp["收益率"].apply(
        lambda x: f"<span class='{'positive' if float(x)>0 else 'negative' if float(x)<0 else ''}'>{float(x):+.1f}%</span>")
    table = disp.to_html(index=False, classes="data-table", border=0, justify="center", escape=False)
    return summary + table


def _trades_table(trades, name_map):
    if len(trades) == 0:
        return "<p>暂无已完成交易</p>"
    sells = trades.sort_values("exit_date", ascending=False).copy()
    total_pnl = sells["pnl"].sum() if len(sells) > 0 else 0
    wins = (sells["pnl"] > 0).sum() if len(sells) > 0 else 0

    disp = sells[["symbol", "entry_date", "exit_date", "shares", "entry_price", "exit_price", "pnl", "pnl_pct"]].copy()
    disp.insert(0, "名称", disp["symbol"].apply(lambda s: name_map.get(str(s).zfill(6), "")))
    disp.columns = ["名称", "代码", "入场日期", "出场日期", "股数", "入场价", "出场价", "盈亏", "收益率"]
    disp["盈亏"] = disp["盈亏"].apply(
        lambda x: f"<span class='{'positive' if float(x)>0 else 'negative'}'>{float(x):+,.0f}</span>")
    disp["收益率"] = disp["收益率"].apply(
        lambda x: f"<span class='{'positive' if float(x)>0 else 'negative'}'>{float(x):+.1f}%</span>")

    summary = f"<div class='summary-text'>共 {len(sells)} 笔 | 盈利 {wins} 笔 | 胜率 {wins/len(sells)*100:.0f}% | 总盈亏 <span class='{'positive' if total_pnl>0 else 'negative'}'>¥{total_pnl:+,.0f}</span></div>"
    return summary + disp.to_html(index=False, classes="data-table", border=0, justify="center", escape=False)


def _trade_details(buys_log, trades, name_map):
    """交易明细：全部买入 + 卖出事件，按时间排序。"""
    rows = []
    for _, b in buys_log.iterrows():
        rows.append({
            "日期": str(b["exec_date"])[:10],
            "代码": str(b["symbol"]).zfill(6),
            "名称": name_map.get(str(b["symbol"]).zfill(6), ""),
            "方向": "买入",
            "数量": int(b["shares"]),
            "价格": f'{float(b["exec_price"]):.2f}',
            "金额": f'{int(b["shares"]) * float(b["exec_price"]):,.0f}',
            "盈亏": "",
        })
    for _, t in trades.iterrows():
        rows.append({
            "日期": str(t["exit_date"])[:10],
            "代码": str(t["symbol"]).zfill(6),
            "名称": name_map.get(str(t["symbol"]).zfill(6), ""),
            "方向": "卖出",
            "数量": int(t["shares"]),
            "价格": f'{float(t["exit_price"]):.2f}',
            "金额": f'{int(t["shares"]) * float(t["exit_price"]):,.0f}',
            "盈亏": f"<span class='{'positive' if float(t['pnl'])>0 else 'negative'}'>{float(t['pnl']):+,.0f}</span>",
        })
    if not rows:
        return "<p>暂无交易记录</p>"
    df = pd.DataFrame(rows).sort_values("日期", ascending=False)
    return df.to_html(index=False, classes="data-table", border=0, justify="right", escape=False)


def _monthly_heatmap(eq, trades):
    """月度收益热力图。"""
    if len(eq) < 20:
        return ""
    monthly = eq.resample("ME").last().pct_change().dropna() * 100
    if len(monthly) < 1:
        return ""
    years = sorted(set(d.year for d in monthly.index), reverse=True)
    months = list(range(1, 13))
    max_abs = max(abs(monthly.max()), abs(monthly.min()), 5.0)

    html = ['<table class="data-table"><thead><tr><th>年份</th>']
    for m in months:
        html.append(f'<th>{m}月</th>')
    html.append('<th>全年</th></tr></thead><tbody>')

    for y in years:
        html.append(f'<tr><td><b>{y}</b></td>')
        y_ret = 0.0
        for m in months:
            matches = [d for d in monthly.index if d.year == y and d.month == m]
            if matches:
                ret = float(monthly.loc[matches[0]])
                if pd.notna(ret):
                    intensity = min(abs(ret) / max_abs, 1.0)
                    if ret > 0:
                        bg = f"background-color:rgba(0,180,0,{intensity:.2f});"
                    else:
                        bg = f"background-color:rgba(220,0,0,{intensity:.2f});"
                    html.append(f'<td class="heat-cell" style="{bg}"><span style="color:#222;font-weight:bold;">{ret:+.1f}%</span></td>')
                    y_ret += ret
                else:
                    html.append('<td class="heat-cell">-</td>')
            else:
                html.append('<td class="heat-cell">-</td>')
        cls = "positive" if y_ret > 0 else "negative" if y_ret < 0 else ""
        html.append(f'<td><b><span class="{cls}">{y_ret:+.1f}%</span></b></td></tr>')

    html.append('</tbody></table>')
    return "\n".join(html)


def _css():
    return """<style>
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
.data-table{width:100%;border-collapse:collapse;font-size:13px;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;margin:12px 0}
.data-table th{background:#1f77b4;color:white;padding:11px 10px;text-align:center;font-weight:600;white-space:nowrap;letter-spacing:0.5px}
.data-table th:first-child,.data-table td:first-child{text-align:left}
.data-table td{padding:9px 10px;border-bottom:1px solid #f0f0f0;text-align:right;white-space:nowrap}
.data-table td:nth-child(2){text-align:left}
.data-table tr:hover{background:#e8f4fd;transition:background 0.2s}
.data-table tr:nth-child(even){background:#f8fbff}
.summary-text{font-size:14px;color:#555;margin:10px 0;padding:10px;background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.05)}
.nav{position:sticky;top:0;background:white;padding:8px 16px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:20px;z-index:100}
.col2{display:flex;gap:20px}
.col2>div{flex:1;min-width:0}
details{margin-bottom:5px;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);padding:10px 20px}
summary{cursor:pointer;user-select:none}
summary h2{display:inline;border:none;font-size:18px;margin:0}
summary::marker{color:#1f77b4;font-size:16px}
.heat-cell{text-align:center;font-weight:bold;font-size:12px;padding:6px 4px;min-width:55px}
</style>"""
