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
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

TASK_DIR = os.path.dirname(os.path.abspath(__file__))

# 被跳过买入候选的显示窗口(天): 过期候选已失效(新买入需新信号),
# 与 daily_signal _auto_fill_pool 只用当日候选的设计一致
_SKIPPED_WINDOW_DAYS = 5


# 图表高度自适应脚本(2026-08-19, P2): 窄屏(≤600px 宽或 ≤500px 高, 覆盖
# 竖屏/横屏手机)时 Plotly.relayout 压缩图高。plotly 高度是布局参数(内部
# svg 按像素绘制), 纯 CSS 缩放 div 只会等比缩小全部元素(文字不可读),
# 必须 relayout 重算; 子图用分数域(domain)布局, 高度变化时比例自动保持。
_ADAPTIVE_CHART_JS = """
<script>
(function(){
  if (typeof Plotly === 'undefined') return;  // CDN 加载失败时静默跳过
  var boxes = document.querySelectorAll('.chart-box[data-mh]');
  if (!boxes.length) return;
  function apply(){
    var narrow = window.matchMedia('(max-width:600px)').matches
              || window.matchMedia('(max-height:500px)').matches;
    Array.prototype.forEach.call(boxes, function(box){
      var gd = box.querySelector('.plotly-graph-div');
      if (!gd || !gd.data) return;  // 图表未初始化(空容器)跳过
      var h = parseInt(narrow ? box.getAttribute('data-mh')
                              : box.getAttribute('data-dh'), 10);
      if (h > 0 && gd.layout.height !== h) {
        Plotly.relayout(gd, {height: h});
      }
    });
  }
  var timer = null;
  window.addEventListener('resize', function(){
    clearTimeout(timer);
    timer = setTimeout(apply, 150);  // 防抖: 旋转屏幕/拖拽窗口
  });
  apply();                // 脚本在 body 末尾, 解析到此图表已渲染
  setTimeout(apply, 400); // 兜底: 慢速设备上再校一次
})();
</script>
"""


def _chart_box(chart_html: str, desktop_h: int, mobile_h: int) -> str:
    """图表自适应高度容器: 桌面/手机高度经 data-dh/data-mh 传给自适应脚本。

    空图表(如无月度数据时 chart2 为空串)原样返回, 不产生空容器。
    """
    if not chart_html:
        return chart_html
    return (f"<div class='chart-box' data-dh='{desktop_h}' "
            f"data-mh='{mobile_h}'>{chart_html}</div>")


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

    # 基金(ETF)判断: 51/15/58/56 开头(代码统一 6 位)
    is_etf = lambda s: str(s).zfill(6).startswith(("51", "15", "58", "56"))

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
    # 分池已实现盈亏(股票池/基金池)
    if len(trades) > 0 and "symbol" in trades.columns:
        trade_sym = trades["symbol"].astype(str).str.zfill(6)
        realized_stock = trades.loc[~trade_sym.apply(is_etf), "pnl"].sum()
        realized_etf = trades.loc[trade_sym.apply(is_etf), "pnl"].sum()
    else:
        realized_stock = realized_etf = 0.0
    total_pnl = realized_pnl + pos_pnl
    # 累计手续费(卖出费来自 trades.csv fee; 买入费含在 avg_cost 中不可分)
    total_fees = float(trades["fee"].sum()) if len(trades) > 0 else 0.0
    # 权益口径修正(2026-08-05): 旧口径用裸价现金流(不含手续费/手动买入),
    # 导致 最终权益 - 初始资金 ≠ 总盈亏。改为含费口径:
    #   cash = 初始 + 已实现(含卖出费) + 浮动(含买入费) - 持仓市值
    #   → 最终权益 = cash + 市值 = 初始 + 总盈亏 (恒等)
    cash_balance = initial_cash + realized_pnl + pos_pnl - pos_value
    final_equity = cash_balance + pos_value
    total_ret = total_pnl / initial_cash * 100

    sells = trades  # trades.csv 全部是已完成卖出
    win_count = len(sells[sells["pnl"] > 0]) if len(sells) > 0 else 0
    win_rate = win_count / len(sells) * 100 if len(sells) > 0 else 0

    # ---- 权益曲线（累积已实现盈亏 + 估算持仓市值） ----
    price_history = _get_price_history()
    equity_curve = _build_equity_curve(trades, positions, price_map,
                                       initial_cash, price_history)
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    # 去重：同一天取最后一条
    eq = equity_curve.groupby("date")["equity"].last()
    # 分池权益曲线(2026-08-21): 股票池/基金池各自单独曲线, 供走势图叠加
    eq_stock = _build_equity_curve(trades, positions, price_map, stock_cash,
                                   price_history, pool="stock")
    eq_stock["date"] = pd.to_datetime(eq_stock["date"])
    eq_stock = eq_stock.groupby("date")["equity"].last()
    eq_etf = _build_equity_curve(trades, positions, price_map, etf_cash,
                                 price_history, pool="etf")
    eq_etf["date"] = pd.to_datetime(eq_etf["date"])
    eq_etf = eq_etf.groupby("date")["equity"].last()
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
                         subplot_titles=("总权益(含沪深300对比)", "回撤(%)", "年度收益(%)"),
                         row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.12)
    fig1.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="权益",
                   line=dict(color="#1f77b4", width=1.5), fill="tozeroy",
                   fillcolor="rgba(31,119,180,0.1)"), row=1, col=1)
    if hs300_norm is not None:
        fig1.add_trace(go.Scatter(x=hs300_norm.index, y=hs300_norm.values, name="沪深300",
                       line=dict(color="#ff7f0e", dash="dash", width=1.2)), row=1, col=1)
    # 分池独立图(2026-08-21): 股票池/基金池各自权益+回撤, 不并入总图
    fig_stock = _build_pool_fig(eq_stock, "stock", "股票池")
    fig_etf = _build_pool_fig(eq_etf, "etf", "基金池")
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
    # 权益曲线纵轴自适应(2026-08-06): 紧贴数据范围±5% padding,
    # 避免 plotly 取整范围导致波动显示不明显。
    # 修复: 范围须同时覆盖沪深300对比线, 否则其曲线被截断不可见
    if len(eq) > 0:
        eq_min = float(eq.min())
        eq_max = float(eq.max())
        if hs300_norm is not None and len(hs300_norm.dropna()) > 0:
            eq_min = min(eq_min, float(hs300_norm.min()))
            eq_max = max(eq_max, float(hs300_norm.max()))
        eq_pad = (eq_max - eq_min) * 0.05 or 1.0
        fig1.update_yaxes(range=[eq_min - eq_pad, eq_max + eq_pad],
                          row=1, col=1)

    # 图表 2: 月度收益
    # 修复(2026-08-12): 原门槛 len(eq) > 20 —— 实盘运行初期权益曲线不足
    # 20 个点(7-20 首批成交至今仅 16 点)时月度收益整块空白。
    # 改为以"存在月度数据"为门槛(monthly 内部 dropna 已保护空数据)。
    monthly = _monthly_returns(eq, initial_cash)
    if len(monthly) > 0:
        pct_v = monthly["pct"].values
        amt_v = monthly["amount"].values
        fig2 = make_subplots(rows=1, cols=1, subplot_titles=("月度收益 (%)",))
        mc = ["#d62728" if v < 0 else "#2ca02c" for v in pct_v]
        fig2.add_trace(go.Bar(x=monthly.index, y=pct_v, name="月度收益",
                       marker_color=mc,
                       text=[f"{v:+.1f}%<br>¥{a:+,.0f}"
                             for v, a in zip(pct_v, amt_v)],
                       textposition="outside", textfont=dict(size=11)),
                       row=1, col=1)
        # y 轴按数据自动留白: 顶部 ≥1.5× 最高柱, 标签(两行)不被裁剪
        pmax = float(max(pct_v)) if len(pct_v) else 0.0
        pmin = float(min(pct_v)) if len(pct_v) else 0.0
        y_hi = max(pmax * 1.5, 0.5)
        y_lo = min(pmin * 1.5, -0.5)
        fig2.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=20), showlegend=False)
        fig2.update_yaxes(range=[y_lo, y_hi])
        chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    else:
        chart2 = ""

    chart1 = fig1.to_html(full_html=False, include_plotlyjs=False)
    # 分池独立图(2026-08-21): 股票池/基金池各自权益+回撤, col2 并排
    chart_stock = _chart_box(
        fig_stock.to_html(full_html=False, include_plotlyjs=False), 480, 400)
    chart_etf = _chart_box(
        fig_etf.to_html(full_html=False, include_plotlyjs=False), 480, 400)

    # ---- 构建表格 ----
    def _nm(sym):
        s = str(sym).zfill(6)
        return name_map.get(s, s)

    stock_pos = [p for p in pos_list if not is_etf(p["symbol"])]
    etf_pos = [p for p in pos_list if is_etf(p["symbol"])]

    pos_stock_html = _positions_table(stock_pos, name_map, "股票", stock_cash)
    pos_etf_html = _positions_table(etf_pos, name_map, "ETF", etf_cash)
    trades_html = _trades_table(trades, name_map)
    details_html = _trade_details(buys_log, trades, name_map)
    monthly_table = _monthly_heatmap(eq, trades, initial_cash)
    pending_html = _pending_orders_html(name_map, stock_cash, etf_cash)

    # ---- 指标卡片 ----
    stock_used = sum(p["value"] for p in stock_pos)
    etf_used = sum(p["value"] for p in etf_pos)
    # 分池总盈亏 = 池内已实现 + 池内浮动
    stock_pnl = sum(p["pnl"] for p in stock_pos)
    etf_pnl = sum(p["pnl"] for p in etf_pos)
    stock_total_pnl = realized_stock + stock_pnl
    etf_total_pnl = realized_etf + etf_pnl
    ret_cls = "positive" if total_ret > 0 else "negative"
    rpnl_cls = "positive" if realized_pnl > 0 else "negative"
    upnl_cls = "positive" if pos_pnl > 0 else "negative"
    s_cls = "positive" if stock_total_pnl > 0 else "negative"
    e_cls = "positive" if etf_total_pnl > 0 else "negative"

    metrics = f"""<h2>核心指标</h2>
<div class='metrics'>
<div class='mcard'><span class='label'>总收益率</span><span class='value {ret_cls}'>{total_ret:+.1f}%</span></div>
<div class='mcard'><span class='label'>已实现盈亏</span><span class='value {rpnl_cls}'>¥{realized_pnl:+,.0f}</span></div>
<div class='mcard'><span class='label'>浮动盈亏</span><span class='value {upnl_cls}'>¥{pos_pnl:+,.0f}</span></div>
<div class='mcard'><span class='label'>总盈亏</span><span class='value {ret_cls}'>¥{total_pnl:+,.0f}</span></div>
<div class='mcard'><span class='label'>股票池总盈亏</span><span class='value {s_cls}'>¥{stock_total_pnl:+,.0f}</span></div>
<div class='mcard'><span class='label'>基金池总盈亏</span><span class='value {e_cls}'>¥{etf_total_pnl:+,.0f}</span></div>
<div class='mcard'><span class='label'>累计卖出手续费</span><span class='value'>¥{total_fees:,.2f}</span></div>
<div class='mcard'><span class='label'>初始资金</span><span class='value'>¥{initial_cash:,.0f}</span></div>
<div class='mcard'><span class='label'>最终现金</span><span class='value'>¥{cash_balance:,.0f}</span></div>
<div class='mcard'><span class='label'>持仓市值</span><span class='value'>¥{pos_value:,.0f}</span></div>
<div class='mcard'><span class='label'>最终权益</span><span class='value'>¥{final_equity:,.0f}</span></div>
<div class='mcard'><span class='label'>总交易</span><span class='value'>{len(trades)}</span></div>
<div class='mcard'><span class='label'>盈利笔数</span><span class='value'>{win_count}</span></div>
<div class='mcard'><span class='label'>胜率</span><span class='value'>{win_rate:.0f}%</span></div>
<div class='mcard'><span class='label'>当前持仓</span><span class='value'>{len(positions)} 只</span></div>
<div class='mcard'><span class='label'>股票池已用</span><span class='value'>¥{stock_used:,.0f}</span></div>
<div class='mcard'><span class='label'>ETF池已用</span><span class='value'>¥{etf_used:,.0f}</span></div>
</div>"""

    # ---- 完整 HTML ----
    css = _css()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    start_date = eq.index[0].strftime("%Y-%m-%d") if len(eq) > 0 else "N/A"
    end_date = eq.index[-1].strftime("%Y-%m-%d") if len(eq) > 0 else "N/A"

    html = f"""<!DOCTYPE html><html lang='zh-CN'>
<head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>实盘交易报告</title>{css}
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>
<body>
<div style="background:#fff5f5;border:3px solid #e74c3c;border-radius:8px;padding:20px 24px;margin-bottom:24px;font-size:16px;color:#721c24;line-height:1.8;text-align:center">
<div style="font-size:24px;margin-bottom:8px">⚠️</div>
<strong style="font-size:18px">免责声明</strong><br>
本报告仅为<u>个人量化策略研究记录</u>，<strong>不构成任何投资建议</strong>。<br>
信号和回测基于<strong>历史数据</strong>，过往表现<strong>不代表未来收益</strong>。<br>
股市有风险，投资需谨慎。使用者应<strong>独立判断并承担全部投资风险</strong>，<br>
作者不对因使用本报告信息产生的任何直接或间接损失承担责任。
</div>
<h1>实盘交易报告</h1>
<p style='color:#666;font-size:14px'>交易区间: {start_date} ~ {end_date} | 初始资金: ¥{initial_cash:,.0f} | 生成时间: {now}</p>
<details open><summary><h2>核心指标</h2></summary>{metrics}</details>
<details open><summary><h2>权益走势</h2></summary>{_chart_box(chart1, 800, 560)}
<div class='col2'><div><h4>股票池</h4>{chart_stock}</div><div><h4>基金池</h4>{chart_etf}</div></div></details>
<details open><summary><h2>月度收益</h2></summary>{_chart_box(chart2, 300, 260)}{monthly_table}</details>
<details open><summary><h2>当前持仓</h2></summary>
<div class='col2'><div><h3>股票</h3>{pos_stock_html}</div><div><h3>ETF</h3>{pos_etf_html}</div></div></details>
<details open><summary><h2>待执行订单</h2></summary>{pending_html}</details>
<details open><summary><h2>已完成交易</h2></summary>
<div class='vscroll'>{trades_html}</div></details>
<details open><summary><h2>交易明细</h2></summary>
<div class='vscroll'>{details_html}</div></details>
<p style='text-align:center;color:#999;margin-top:40px;font-size:12px'>🤖 Generated by AKQuant Live Report — {now}</p>
{_ADAPTIVE_CHART_JS}
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


def _get_price_history():
    """从缓存读取全部历史收盘价 → {symbol: {date_str: close}}。

    修复(2026-08-13): 权益曲线历史点原用"当前价"估算持仓市值,
    导致 7 月等历史月份收益随每日价格漂移(7月 +0.4% 隔日变 -0.2%)。
    历史点改用当日收盘价: 历史月份收益固定, 只有当月随行情变化。
    """
    hist = {}
    cache_dir = os.path.join(TASK_DIR, ".cache")
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".parquet"):
                try:
                    df = pd.read_parquet(os.path.join(cache_dir, fname),
                                         columns=["date", "close"])
                    sym = fname.replace(".parquet", "").zfill(6)
                    dates = pd.to_datetime(df["date"]).astype(str).str[:10]
                    hist[sym] = dict(zip(dates, df["close"].astype(float)))
                except Exception:
                    pass
    return hist


def _build_equity_curve(trades, positions, price_map, initial_cash,
                        price_history=None, pool=None):
    """从 execution_log.csv + trades.csv + positions.json 重建权益曲线。

    数据来源:
    - execution_log.csv: 所有已执行的买入事件
    - trades.csv: 所有已完成的卖出事件
    - positions.json: 当前持仓（未平仓）

    历史点持仓市值用当日收盘价(price_history), 缺失回退当前价;
    最终快照(今日)用当前价。

    pool: None=合并曲线(股票+ETF); "stock"/"etf"=仅该池——按代码前缀
    过滤事件与持仓, 初始资金用该池的 initial_cash(2026-08-21 新增,
    供权益走势图叠加股票池/基金池单独曲线)。
    """
    is_etf = lambda s: str(s).zfill(6).startswith(("51", "15", "58", "56"))

    def _in_pool(sym: str) -> bool:
        if pool is None:
            return True
        return is_etf(sym) == (pool == "etf")

    elog_path = os.path.join(TASK_DIR, "execution_log.csv")

    # 收集所有买卖事件
    events = []

    # 买入事件：从 execution_log.csv
    if os.path.exists(elog_path):
        elog = pd.read_csv(elog_path, dtype={"symbol": str, "exec_date": str})
        elog["symbol"] = elog["symbol"].str.zfill(6)
        executed_buys = elog[(elog["action"] == "buy") & (elog["status"] == "executed")]
        for _, b in executed_buys.iterrows():
            if not _in_pool(str(b["symbol"]).zfill(6)):
                continue
            events.append({
                "date": str(b["exec_date"])[:10],
                "type": "buy",
                "symbol": str(b["symbol"]).zfill(6),
                "shares": int(b["shares"]),
                "price": float(b["exec_price"]),
            })

    # 卖出事件：从 trades.csv
    for _, t in trades.iterrows():
        if not _in_pool(str(t["symbol"]).zfill(6)):
            continue
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

        # 历史点市值用当日收盘价(修复 2026-08-13: 原用当前价导致
        # 历史月份收益随每日价格漂移), 缺失回退当前价
        market_value = 0.0
        for s, h in holdings.items():
            px = None
            if price_history:
                px = price_history.get(s, {}).get(ev["date"])
            if px is None or pd.isna(px):
                px = price_map.get(s, h["price"])
            market_value += h["shares"] * float(px)
        equity_rows.append({"date": ev["date"], "equity": cash + market_value})

    # 最终快照(分池时只统计该池持仓)
    current_market = sum(
        p["shares"] * price_map.get(sym, p["avg_cost"])
        for sym, p in positions.items()
        if _in_pool(sym)
    )
    equity_rows.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "equity": cash + current_market,
    })

    return pd.DataFrame(equity_rows)


# =============================================================================
# HTML table builders
# =============================================================================
def _build_pool_fig(eq_pool, pool: str, title: str):
    """构建分池独立权益图(2026-08-21): 权益 + 回撤 两个子图。

    股票池/基金池各自的权益曲线与回撤(从本池初始资金起算的 cummax),
    不并入总权益图——两池规模不同(30万 vs 10万), 合画会互相压缩波动。
    eq_pool: 已按日期去重的权益 Series。
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    color = "#2ca02c" if pool == "stock" else "#9467bd"
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"{title}权益", "回撤(%)"),
                        row_heights=[0.65, 0.35], vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=eq_pool.index, y=eq_pool, mode="lines",
                   line=dict(color=color, width=1.5), fill="tozeroy",
                   fillcolor="rgba(0,0,0,0.04)"), row=1, col=1)
    dd = (eq_pool - eq_pool.cummax()) / eq_pool.cummax() * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                   line=dict(color="#d62728", width=1)), row=2, col=1)
    fig.update_layout(height=480, margin=dict(l=40, r=20, t=40, b=20),
                      showlegend=False)
    # 权益纵轴自适应(与 fig1 同款: 紧贴数据±5%)
    if len(eq_pool) > 0:
        lo = float(eq_pool.min())
        hi = float(eq_pool.max())
        pad = (hi - lo) * 0.05 or 1.0
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    return fig


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
    table = disp.to_html(index=False, classes="data-table pos-table", border=0, justify="center", escape=False)
    return summary + _wrap_table(table)


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
    return summary + _wrap_table(disp.to_html(index=False, classes="data-table trades-table", border=0, justify="center", escape=False))


def _trade_details(buys_log, trades, name_map):
    """交易明细：全部买入 + 卖出事件，按时间排序。"""
    rows = []
    for _, b in buys_log.iterrows():
        rows.append({
            "代码": str(b["symbol"]).zfill(6),
            "名称": name_map.get(str(b["symbol"]).zfill(6), ""),
            "日期": str(b["exec_date"])[:10],
            "方向": "买入",
            "数量": int(b["shares"]),
            "价格": f'{float(b["exec_price"]):.2f}',
            "金额": f'{int(b["shares"]) * float(b["exec_price"]):,.0f}',
            "盈亏": "",
        })
    for _, t in trades.iterrows():
        rows.append({
            "代码": str(t["symbol"]).zfill(6),
            "名称": name_map.get(str(t["symbol"]).zfill(6), ""),
            "日期": str(t["exit_date"])[:10],
            "方向": "卖出",
            "数量": int(t["shares"]),
            "价格": f'{float(t["exit_price"]):.2f}',
            "金额": f'{int(t["shares"]) * float(t["exit_price"]):,.0f}',
            "盈亏": f"<span class='{'positive' if float(t['pnl'])>0 else 'negative'}'>{float(t['pnl']):+,.0f}</span>",
        })
    if not rows:
        return "<p>暂无交易记录</p>"
    df = pd.DataFrame(rows).sort_values("日期", ascending=False)
    return _wrap_table(df.to_html(index=False, classes="data-table details-table", border=0, justify="right", escape=False))


def _pending_orders_html(name_map, stock_cash, etf_cash):
    """生成待执行订单表格（对齐 manage.py show 输出）。

    包含: 待买入/待卖出 + 被跳过的买入信号（仓位满/资金不足）。
    """
    import pandas as pd
    from orders import load_pending

    pending = load_pending()
    is_etf = lambda s: str(s).startswith(("51", "15", "58", "56"))

    # 待卖出
    sells = [o for o in pending if o.get("action") == "sell"]

    # 待买入（pending 中的 buy 订单）
    buys = [o for o in pending if o.get("action") != "sell"]

    # 被跳过的买入信号（从 execution_log + signals 提取，同 manage.py show）
    skipped = _get_skipped_buys(pending, name_map)

    html_parts = ['<div class="col2">']

    # ---- 待卖出 ----
    html_parts.append('<div><h3>🔴 待卖出</h3>')
    if not sells:
        html_parts.append('<p style="color:#888">暂无</p>')
    else:
        _build_pending_table(html_parts, sells, name_map, "sell")
    html_parts.append('</div>')

    # ---- 待买入（合并 pending buys + skipped buys, 分股票/基金） ----
    def _split_by_pool(items):
        st, et = [], []
        for o in items:
            (et if is_etf(o.get("symbol", "")) else st).append(o)
        return st, et

    buys_stock, buys_etf = _split_by_pool(buys)
    skip_stock, skip_etf = _split_by_pool(skipped)

    def _render_buy_block(title, buys_x, skip_x):
        html_parts.append(f'<div style="margin-bottom:8px"><h4>{title}</h4>')
        if not buys_x and not skip_x:
            html_parts.append('<p style="color:#888">暂无</p>')
        else:
            if buys_x:
                _build_pending_table(html_parts, buys_x, name_map, "buy")
            if skip_x:
                _build_skipped_table(html_parts, skip_x, name_map)
        html_parts.append('</div>')

    html_parts.append('<div><h3>🟢 待买入</h3>')
    if not buys and not skipped:
        html_parts.append('<p style="color:#888">暂无待买入信号</p>')
    else:
        _render_buy_block('📈 股票', buys_stock, skip_stock)
        _render_buy_block('📊 基金(ETF)', buys_etf, skip_etf)
    html_parts.append('</div>')

    html_parts.append('</div>')  # close col2

    return "\n".join(html_parts)


def _get_skipped_buys(pending, name_map, within_days: int = _SKIPPED_WINDOW_DAYS):
    """从 execution_log.csv + signals.csv 提取近期被跳过的买入候选。

    修复(2026-08-21)——原实现三处误导:
    1. 无 watchlist 过滤: 已移出监控列表的旧池标的(永不复扫)仍显示为候选
    2. 无日期过滤: 数周前的历史跳过永久显示(池清空后仍显示旧候选)
    3. 日期列误用 signals.csv 最新信号日期: 跳过发生在几周前却显示为
       "昨日被跳过"; 现价/偏离同样混用最新快照, 与跳过事件时间点不一致

    现行为:
    - 仅显示当前 watchlist 内标的(name_map), 排除已持仓/已有待执行订单
    - 每标的取最新一次跳过记录, 信号价/偏离/趋势取**跳过当日**的
      signals.csv 快照(行内时间点一致, 不混用最新价)
    - 仅显示近 within_days 天内的跳过——过期候选已失效(新买入需新信号,
      与 daily_signal _auto_fill_pool 只用当日候选的设计一致)
    - 负偏离(应卖出)不列入

    manage.py cmd_show 复用本函数。
    """
    import pandas as pd

    elog_path = os.path.join(TASK_DIR, "execution_log.csv")
    sig_path = os.path.join(TASK_DIR, "signals.csv")
    pos_path = os.path.join(TASK_DIR, "positions.json")

    if not os.path.exists(elog_path) or not os.path.exists(sig_path):
        return []

    elog = pd.read_csv(elog_path, dtype={"symbol": str, "signal_date": str})
    elog["symbol"] = elog["symbol"].str.zfill(6)
    sig_df = pd.read_csv(sig_path, dtype={"symbol": str})
    sig_df["symbol"] = sig_df["symbol"].str.zfill(6)

    # 每标的最新一次 skip 记录
    sk = elog[(elog["status"] == "skipped") & (elog["action"] == "buy")].copy()
    if sk.empty:
        return []
    sk = sk[sk["signal_date"].notna()]
    sk = sk.sort_values("signal_date").groupby("symbol").last().reset_index()

    # 跳过当日的信号快照 {(symbol, date): row}
    sig_idx = {(str(r["symbol"]).zfill(6), str(r["date"])[:10]): r
               for _, r in sig_df.iterrows()}

    # 当前持仓
    pos_set = set()
    if os.path.exists(pos_path):
        with open(pos_path) as f:
            pos_set = set(json.load(f).keys())

    pend_set = {o["symbol"] for o in pending}
    cutoff = (datetime.now() - timedelta(days=within_days)).strftime("%Y-%m-%d")

    skipped = []
    for _, r in sk.iterrows():
        sym = r["symbol"]
        if sym in pos_set or sym in pend_set:
            continue
        if sym not in name_map:  # 已移出 watchlist 的旧池标的
            continue
        skip_date = str(r["signal_date"])[:10]
        if not skip_date or skip_date < cutoff:  # 过期跳过, 候选已失效
            continue
        row = sig_idx.get((sym, skip_date))
        if row is None:  # 跳过当日无信号快照, 无法还原当日状态
            continue
        c = float(row.get("close", 0))
        k = float(row.get("kalman_price", c))
        dev = (c / k - 1) * 100 if k > 0 else 0
        if dev <= 0:
            continue  # 负偏离=卖出信号, 不列入待买入
        skipped.append({
            "symbol": sym,
            "name": row.get("name", sym),
            "close": c,
            "deviation": dev,
            "trend": row.get("trend", "?"),
            "target_pct": float(row.get("target_pct", 0.95)),
            "signal_date": skip_date,
        })

    # 按偏离度降序
    skipped.sort(key=lambda x: x["deviation"], reverse=True)
    return skipped


def _build_skipped_table(html_parts, skipped, name_map):
    """构建被跳过买入信号表格(跳过当日快照, 日期/价格时间点一致)。"""
    import pandas as pd

    rows = []
    for s in skipped:
        sym = str(s["symbol"]).zfill(6)
        name = name_map.get(sym, s["name"])
        rows.append({
            "代码": sym,
            "名称": name,
            "信号价": f'{s["close"]:.2f}',
            "偏离": f'{s["deviation"]:+.1f}%',
            "趋势": s["trend"],
            "跳过日": s["signal_date"],
        })

    df = pd.DataFrame(rows)
    html_parts.append(
        f'<p style="margin-top:16px;color:#e67e22;font-weight:bold">'
        f'⏸️ 被跳过 ({len(skipped)} 笔，跳过时仓位满/资金不足，'
        f'近{_SKIPPED_WINDOW_DAYS}日内)</p>'
    )
    html_parts.append(
        _wrap_table(df.to_html(index=False, classes="data-table skipped-table", border=0,
                               justify="center", escape=False))
    )


def _wrap_table(html: str) -> str:
    """表格横向滚动容器(手机窄屏时保持列宽可读, 不挤压重叠)。

    -webkit-overflow-scrolling:touch: iOS Safari 惯性滚动(否则 flex 容器内
    可能无法滚动); width:100% 保证容器不超父宽。
    """
    return (f"<div class='tscroll' style='overflow-x:auto;"
            f"-webkit-overflow-scrolling:touch;width:100%'>{html}</div>")


def _build_pending_table(html_parts, orders, name_map, action,
                         stock_cash=0, etf_cash=0):
    """构建单个待执行订单表格。"""
    import pandas as pd

    rows = []
    for o in sorted(orders, key=lambda x: x.get("signal_date", ""), reverse=True):
        sym = str(o["symbol"]).zfill(6)
        name = name_map.get(sym, o.get("name", sym))
        pct = float(o.get("target_pct", 0)) * 100
        rows.append({
            "代码": sym,
            "名称": name,
            "股数": int(o["shares"]),
            "信号价": f'{float(o["signal_price"]):.2f}',
            "仓位": f"{pct:.0f}%" if action == "buy" else "—",
            "信号日": o.get("signal_date", ""),
        })

    if not rows:
        html_parts.append(f'<p style="color:#888">暂无待{action=="buy" and "买入" or "卖出"}订单</p>')
        return

    df = pd.DataFrame(rows)
    table = _wrap_table(df.to_html(index=False, classes="data-table pending-table", border=0, justify="center", escape=False))

    # 汇总
    total_val = sum(
        int(o["shares"]) * float(o["signal_price"]) for o in orders
    )
    html_parts.append(
        f'<div class="summary-text">{action=="buy" and "待买入" or "待卖出"}: '
        f'{len(orders)} 笔 | 预估金额 ¥{total_val:,.0f}</div>'
    )
    html_parts.append(table)


def _monthly_returns(eq, initial_cash):
    """月度收益: 首月以初始资金为基准, 之后各月环比。

    修复(2026-08-12): 原 resample+pct_change 方案中首月无环比基准
    被 dropna 掉 —— 实盘 7-20 开始交易后 7 月月度收益永远缺失。
    首月基准改为初始资金: (首月末权益 / 初始资金 - 1) × 100。
    返回 DataFrame[pct(%), amount(¥)]: amount = 当月权益变化额。
    """
    monthly_equity = eq.resample("ME").last()
    if len(monthly_equity) == 0:
        return pd.DataFrame({"pct": pd.Series(dtype=float),
                             "amount": pd.Series(dtype=float)})
    prev = monthly_equity.shift(1)
    prev.iloc[0] = float(initial_cash)
    return pd.DataFrame({
        "pct": (monthly_equity / prev - 1) * 100,
        "amount": monthly_equity - prev,
    })


def _monthly_heatmap(eq, trades, initial_cash):
    """月度收益热力图。"""
    # 修复(2026-08-12): 原门槛 len(eq) < 20 导致运行初期月度收益空白,
    # 统一以"存在月度数据"为门槛。
    monthly = _monthly_returns(eq, initial_cash)
    if len(monthly) < 1:
        return ""
    years = sorted(set(d.year for d in monthly.index), reverse=True)
    months = list(range(1, 13))
    max_abs = max(abs(monthly["pct"].max()), abs(monthly["pct"].min()), 5.0)

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
                ret = float(monthly.loc[matches[0], "pct"])
                if pd.notna(ret):
                    amt = float(monthly.loc[matches[0], "amount"])
                    intensity = min(abs(ret) / max_abs, 1.0)
                    if ret > 0:
                        bg = f"background-color:rgba(0,180,0,{intensity:.2f});"
                    else:
                        bg = f"background-color:rgba(220,0,0,{intensity:.2f});"
                    html.append(
                        f'<td class="heat-cell" style="{bg}">'
                        f'<span style="color:#222;font-weight:bold;">{ret:+.1f}%</span>'
                        f'<br><span style="color:#444;font-size:11px;">¥{amt:+,.0f}</span></td>')
                    y_ret += ret
                else:
                    html.append('<td class="heat-cell">-</td>')
            else:
                html.append('<td class="heat-cell">-</td>')
        cls = "positive" if y_ret > 0 else "negative" if y_ret < 0 else ""
        html.append(f'<td><b><span class="{cls}">{y_ret:+.1f}%</span></b></td></tr>')

    html.append('</tbody></table>')
    # 修复(2026-08-19): 14 列热力表(年份+12月+全年)固有宽度 ~880px,
    # 未包滚动容器时手机上被 body overflow-x:clip 裁剪, 5 月后月份
    # 及"全年"列不可见。与其他 7 张表一致包 _wrap_table 横向滚动。
    return _wrap_table("\n".join(html))


def _css():
    return """<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5;color:#333;overflow-x:clip}
h1{color:#1a1a1a;border-bottom:3px solid #1f77b4;padding-bottom:10px}
h2{color:#2c3e50;margin-top:40px;border-bottom:2px solid #ddd;padding-bottom:8px}
h3{color:#555;margin-top:25px}
h4{color:#777;margin:14px 0 4px;font-size:15px;border-left:3px solid #1f77b4;padding-left:8px}
/* 核心指标: CSS Grid 自动换行(auto-fit), 任意宽度自适应多行 */
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:20px 0;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden}
.mcard{padding:16px 12px;text-align:center;border-right:1px solid #eee;border-bottom:1px solid #eee}
.mcard:nth-child(odd){background:#fafafa}
.mcard .label{font-size:12px;color:#666;display:block}
.mcard .value{font-size:20px;font-weight:bold;display:block;margin-top:4px}
.positive{color:#2ca02c}.negative{color:#d62728}
.data-table{width:max-content;min-width:max(640px,100%);border-collapse:separate;border-spacing:0;font-size:13px;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin:12px 0}
/* 注: 不能有 overflow:hidden——它是 sticky 首列的祖先, 会使其失效 */
/* 恢复 collapse 的细分隔线视觉(separate 下) */
.data-table th,.data-table td{border-bottom:1px solid #eee}
.data-table th{background:#1f77b4;color:white;padding:11px 10px;text-align:center;font-weight:600;white-space:nowrap;letter-spacing:0.5px}
.data-table th:first-child,.data-table td:first-child{text-align:left}
.data-table td{padding:9px 10px;border-bottom:1px solid #f0f0f0;text-align:right;white-space:nowrap}
.data-table td:nth-child(2){text-align:left}
.data-table tr:hover{background:#e8f4fd;transition:background 0.2s}
.data-table tr:nth-child(even){background:#f8fbff}
.summary-text{font-size:14px;color:#555;margin:10px 0;padding:10px;background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.05)}
.nav{position:sticky;top:0;background:white;padding:8px 16px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:20px;z-index:100}
/* 持仓双列: flex-wrap + 最小宽度, 宽度不足时平滑换行(无固定断点) */
.col2{display:flex;flex-wrap:wrap;gap:20px}
.col2>div{flex:1 1 340px;min-width:0}
/* min-width:0 关键——flex 子项可压缩到容器宽, 表格溢出由内部 overflow 容器
   接管滚动; 否则子项被 max-content 表格撑宽导致页面级横向滚动 */
/* ≤900px: col2 用 block 替代 flex(绕开 iOS flex+overflow 兼容问题),
   子项固定 100% 宽, 表格溢出由内部容器滚动 */
@media (max-width: 900px){
  .col2{display:block}
  .col2>div{width:100%}
}
/* 手机窄屏: 缩小卡片字号/间距, 布局由 grid 自动重排 */
@media (max-width: 480px){
  .mcard{padding:10px 8px}
  .mcard .value{font-size:16px}
  .mcard .label{font-size:11px}
  h1{font-size:22px}
  .data-table{font-size:11px}
  /* plotly 图例(6项横排)在窄屏重叠 → 隐藏, 悬停仍显示名称 */
  .js-plotly-plot .legend{display:none}
  /* 当前持仓/待执行订单: 手机强制上下单列排布 */
  .col2{flex-direction:column}
  .col2>div{min-width:0}
}
/* 表格横向滑动时冻结首列(名称/代码) — 2026-08-06
   蓝色系与网页主色 #1f77b4(标题边框/权益曲线)同源匹配 */
.data-table th:first-child,.data-table td:first-child{
  position:sticky;left:0;background:#eef4fa;
  box-shadow:1px 0 0 #d0e3f2;z-index:2;font-weight:bold;white-space:nowrap
}
/* 表头首列: 置顶且继承 th 深蓝底白字(不得覆盖其背景) */
.data-table thead th:first-child{z-index:3}
.col2>div{flex:1;min-width:0}
details{margin-bottom:5px;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);padding:10px 20px}
/* 长表滚动盒(桌面, 已完成交易/交易明细): ≤600px 手机取消内层竖向滚动,
   避免三层嵌套滚动(页面竖滚→容器竖滚→表格横滚), 表格随页面自然展开 */
.vscroll{max-height:600px;overflow-y:auto;-webkit-overflow-scrolling:touch}
@media (max-width:600px){.vscroll{max-height:none;overflow-y:visible}}
summary{cursor:pointer;user-select:none}
summary h2{display:inline;border:none;font-size:18px;margin:0}
summary::marker{color:#1f77b4;font-size:16px}
.heat-cell{text-align:center;font-weight:bold;font-size:12px;padding:6px 4px;min-width:55px}
</style>"""
