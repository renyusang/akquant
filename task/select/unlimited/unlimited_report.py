"""无限仓位信号报告生成器。

独立报告(unlimited_report.html):
- 搜索框: 输入股票代码/名称, 过滤所有表格
- 归一化收益: 每只股票 总利润 / (3×最小单位×首笔成交价), 排序 + 柱状图
- 当前持仓 / 最近信号 / 交易明细

数据源: 本目录状态文件(positions/trades/execution_log/signals)
价格: 复用实盘 .cache(数据共用)
"""

import json
import os
from datetime import datetime

import yaml

import numpy as np
import pandas as pd

UNLIMITED_DIR = os.path.dirname(os.path.abspath(__file__))
KALMAN_DIR = os.path.normpath(os.path.join(UNLIMITED_DIR, "..", "..", "kalman"))
CACHE_DIR = os.path.join(KALMAN_DIR, ".cache")


def _load_json(name, default=None):
    path = os.path.join(UNLIMITED_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _load_csv(name, **kw):
    path = os.path.join(UNLIMITED_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


def _latest_prices():
    prices = {}
    if not os.path.isdir(CACHE_DIR):
        return prices
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".parquet"):
            sym = f.replace(".parquet", "")
            try:
                df = pd.read_parquet(os.path.join(CACHE_DIR, f))
                prices[sym] = float(df["close"].iloc[-1])
            except Exception:
                pass
    return prices


def _lot_size(symbol: str) -> int:
    return 200 if str(symbol).startswith("688") else 100


def _is_etf(symbol: str) -> bool:
    return str(symbol).zfill(6).startswith(("51", "15", "58", "56"))


def _first_buy_price(symbol: str, elog: pd.DataFrame) -> float:
    """首笔买入成交价(归一化收益的总资金基准)。"""
    b = elog[(elog["symbol"] == symbol) & (elog["action"] == "buy")
             & (elog["status"] == "executed")]
    if len(b):
        return float(b["exec_price"].iloc[0])
    return 0.0


def _normalized_pnl(symbol, positions, trades, elog, prices):
    """单只股票: (总利润, 归一化收益)。"""
    lot = _lot_size(symbol)
    first_px = _first_buy_price(symbol, elog)
    total_capital = 3 * lot * first_px if first_px > 0 else 0.0

    # 已实现: trades.csv 该股 pnl 累计
    realized = 0.0
    if len(trades) and "symbol" in trades.columns:
        t = trades[trades["symbol"].astype(str).str.zfill(6) == symbol]
        realized = float(t["pnl"].sum()) if len(t) else 0.0

    # 浮动: 当前持仓
    floating = 0.0
    if symbol in positions:
        p = positions[symbol]
        px = prices.get(symbol, float(p["avg_cost"]))
        floating = (px - float(p["avg_cost"])) * int(p["shares"])

    total = realized + floating
    norm = total / total_capital if total_capital > 0 else 0.0
    return total, norm, realized, floating


def _watchlist():
    """读取监控列表(unlimited.yaml watchlist)。"""
    path = os.path.join(UNLIMITED_DIR, "unlimited.yaml")
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        cfg = yaml.safe_load(f)
    wl = cfg.get("watchlist", {})
    if isinstance(wl, list):
        stocks, etfs = wl, []
    else:
        stocks = wl.get("stocks", [])
        etfs = wl.get("etfs", [])
    return stocks, etfs


def build_unlimited_report() -> None:
    import plotly.graph_objects as go

    positions = _load_json("positions.json", {}) or {}
    trades = _load_csv("trades.csv", dtype={"symbol": str})
    elog = _load_csv("execution_log.csv", dtype={"symbol": str})
    signals = _load_csv("signals.csv", dtype={"symbol": str})
    prices = _latest_prices()

    if len(elog):
        elog["symbol"] = elog["symbol"].str.zfill(6)
    if len(trades) and "symbol" in trades.columns:
        trades["symbol"] = trades["symbol"].str.zfill(6)

    # 全部出现过信号的标的(positions + elog + trades 并集)
    symbols = set(positions.keys())
    if len(elog):
        symbols |= set(elog["symbol"])
    if len(trades) and "symbol" in trades.columns:
        symbols |= set(trades["symbol"])
    symbols = sorted(symbols)

    # ---- 归一化收益(每只股票) ----
    rows = []
    for sym in symbols:
        total, norm, realized, floating = _normalized_pnl(
            sym, positions, trades, elog, prices)
        rows.append({
            "代码": sym, "名称": _name_of(sym, positions, elog, trades),
            "最小单位": f"{_lot_size(sym)}股",
            "首笔买入价": _first_buy_price(sym, elog),
            "总资金": 3 * _lot_size(sym) * _first_buy_price(sym, elog),
            "已实现": realized, "浮动": floating, "总利润": total,
            "归一化收益": norm,
        })
    norm_df = pd.DataFrame(rows)
    if len(norm_df):
        norm_df = norm_df.sort_values("归一化收益", ascending=False)

    # ---- 监控列表(股票/基金分开) ----
    wl_stocks, wl_etfs = _watchlist()
    wl_stock_df = pd.DataFrame([
        {"代码": s.get("symbol"), "名称": s.get("name", s.get("symbol"))}
        for s in wl_stocks]) if wl_stocks else pd.DataFrame()
    wl_etf_df = pd.DataFrame([
        {"代码": e.get("symbol"), "名称": e.get("name", e.get("symbol"))}
        for e in wl_etfs]) if wl_etfs else pd.DataFrame()
    wl_stock_table = (wl_stock_df.to_html(index=False, classes="data-table wl-table",
                                          border=0, justify="center",
                                          escape=False) if len(wl_stock_df)
                      else "<p>暂无</p>")
    wl_etf_table = (wl_etf_df.to_html(index=False, classes="data-table wl-table",
                                      border=0, justify="center",
                                      escape=False) if len(wl_etf_df)
                    else "<p>暂无</p>")

    # ---- 核心指标 ----
    n_symbols = len(symbols)  # 本轮产生过信号的标的数
    n_watch = len(wl_stocks) + len(wl_etfs)  # 监控列表总数
    n_pos = len(positions)
    n_buy_signals = len(elog[(elog["action"] == "buy")]) if len(elog) else 0
    realized_all = float(trades["pnl"].sum()) if len(trades) and "pnl" in trades.columns else 0.0
    floating_all = sum(r["浮动"] for r in rows)
    avg_norm = float(norm_df["归一化收益"].mean()) if len(norm_df) else 0.0

    # 分池: 股票/基金/整体
    def _split(subset, key):
        st = [r for r in subset if not _is_etf(r["代码"])]
        et = [r for r in subset if _is_etf(r["代码"])]
        return sum(r[key] for r in st), sum(r[key] for r in et)

    realized_stock, realized_etf = _split(rows, "已实现")
    floating_stock, floating_etf = _split(rows, "浮动")
    if len(norm_df):
        nd = norm_df.to_dict("records")
        ns = [r["归一化收益"] for r in nd if not _is_etf(r["代码"])]
        ne = [r["归一化收益"] for r in nd if _is_etf(r["代码"])]
        avg_norm_stock = float(np.mean(ns)) if ns else 0.0
        avg_norm_etf = float(np.mean(ne)) if ne else 0.0
    else:
        avg_norm_stock = avg_norm_etf = 0.0

    # 总盈亏 = 已实现 + 浮动 (对齐实盘 live_report 口径, 2026-08-12)
    total_pnl = realized_all + floating_all
    total_stock = realized_stock + floating_stock
    total_etf = realized_etf + floating_etf

    def _cls(v):
        return "positive" if v > 0 else "negative"

    pos_pnl_cls = _cls(floating_all)
    rpnl_cls = _cls(realized_all)
    norm_cls = _cls(avg_norm)
    total_cls = _cls(total_pnl)
    total_stock_cls = _cls(total_stock)
    total_etf_cls = _cls(total_etf)

    # ---- 归一化收益柱状图(股票/基金分开) ----
    def _norm_chart(sub, title):
        if not len(sub):
            return ""
        fig = go.Figure(go.Bar(
            x=sub["代码"], y=sub["归一化收益"],
            marker_color=["#2ca02c" if v >= 0 else "#d62728"
                          for v in sub["归一化收益"]],
            text=[f"{v:+.1%}" for v in sub["归一化收益"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=title, height=360, margin=dict(l=40, r=20, t=50, b=20),
            yaxis_tickformat=".0%", hovermode="x",
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    chart_html = ""
    chart_etf_html = ""
    if len(norm_df):
        chart_html = _norm_chart(norm_df.head(30), "📈 股票归一化收益 TOP30")
        chart_etf_html = _norm_chart(
            norm_df[norm_df["代码"].apply(_is_etf)].head(30),
            "📊 基金归一化收益 TOP30")

    # ---- 表格 ----
    def _fmt_pnl(x):
        cls = "positive" if x > 0 else "negative" if x < 0 else ""
        return f"<span class='{cls}'>{x:+,.0f}</span>"

    norm_disp = norm_df.copy()
    if len(norm_disp):
        norm_disp["已实现"] = norm_disp["已实现"].apply(_fmt_pnl)
        norm_disp["浮动"] = norm_disp["浮动"].apply(_fmt_pnl)
        norm_disp["总利润"] = norm_disp["总利润"].apply(_fmt_pnl)
        norm_disp["归一化收益"] = norm_disp["归一化收益"].apply(
            lambda x: f"<span class='{'positive' if x>0 else 'negative' if x<0 else ''}'>{x:+.1%}</span>")
        norm_disp["首笔买入价"] = norm_disp["首笔买入价"].apply(lambda x: f"{x:.2f}")
        norm_disp["总资金"] = norm_disp["总资金"].apply(lambda x: f"{x:,.0f}")
    def _norm_table(sub):
        if not len(sub):
            return "<p>暂无数据</p>"
        return sub.to_html(index=False, classes="data-table",
                           border=0, justify="center", escape=False)

    if len(norm_disp):
        norm_table = _norm_table(norm_disp[~norm_disp["代码"].apply(_is_etf)])
        norm_etf_table = _norm_table(norm_disp[norm_disp["代码"].apply(_is_etf)])
    else:
        norm_table = norm_etf_table = "<p>暂无数据</p>" 

    # 当前持仓
    pos_rows = []
    for sym, p in positions.items():
        px = prices.get(sym, float(p["avg_cost"]))
        val = int(p["shares"]) * px
        pnl = (px - float(p["avg_cost"])) * int(p["shares"])
        pos_rows.append({
            "代码": sym, "名称": p.get("name", sym),
            "股数": int(p["shares"]), "成本": float(p["avg_cost"]),
            "现价": px, "市值": val, "浮动盈亏": pnl,
        })
    pos_df = pd.DataFrame(pos_rows)
    if len(pos_df):
        pos_df["浮动盈亏"] = pos_df["浮动盈亏"].apply(_fmt_pnl)
        pos_df["现价"] = pos_df["现价"].apply(lambda x: f"{x:.2f}")
        pos_df["市值"] = pos_df["市值"].apply(lambda x: f"{x:,.0f}")
    pos_stock = pos_df[~pos_df["代码"].apply(_is_etf)] if len(pos_df) else pos_df
    pos_etf = pos_df[pos_df["代码"].apply(_is_etf)] if len(pos_df) else pos_df
    pos_stock_table = (pos_stock.to_html(index=False, classes="data-table",
                                         border=0, justify="center",
                                         escape=False) if len(pos_stock)
                       else "<p>暂无持仓</p>")
    pos_etf_table = (pos_etf.to_html(index=False, classes="data-table",
                                     border=0, justify="center",
                                     escape=False) if len(pos_etf)
                     else "<p>暂无持仓</p>")

    # 信号状态: 每只标的的最新信号 + 未成交待执行订单
    # (2026-08-06: 新扫描的 hold 会掩盖历史 buy 信号, 需体现 pending 中的
    #  34 笔未成交买入——待买订单覆盖显示为"待成交")
    pending_orders = _load_json("pending_orders.json", []) or []
    pending_buy = {str(o["symbol"]).zfill(6): o for o in pending_orders
                   if o.get("action") == "buy"}
    pending_sell = {str(o["symbol"]).zfill(6): o for o in pending_orders
                    if o.get("action") == "sell"}

    # pending 订单的原始信号原因(execution_log.signal_reason)
    pending_reason = {}
    if len(elog):
        pe = elog[elog["status"] == "pending"]
        for _, r in pe.iterrows():
            sym = str(r["symbol"]).zfill(6)
            sr = str(r.get("signal_reason", "")).strip()
            if sr and sr != "nan":
                pending_reason[sym] = sr

    sig_disp = pd.DataFrame()
    if len(signals):
        s = signals.sort_values("date").groupby("symbol").last().reset_index()
        s["symbol"] = s["symbol"].str.zfill(6)
        # 优先用 signals.csv 自带的 name 列
        s["name"] = s.apply(
            lambda r: str(r.get("name", "")).strip() or _name_of(
                str(r["symbol"]).zfill(6), positions, elog, trades),
            axis=1)
        sig_disp = s[["date", "symbol", "name", "close", "trend",
                      "signal", "target_pct", "reason"]].copy()
        # 待执行订单覆盖显示(未成交的买入/卖出必须可见)
        def _status(r):
            sym = str(r["symbol"]).zfill(6)
            if sym in pending_buy:
                o = pending_buy[sym]
                return f"🟢待成交{int(o['shares'])}股"
            if sym in pending_sell:
                return "🔴待卖出"
            return {"buy": "🟢买入", "sell": "🔴卖出", "hold": "⚪持有"}.get(
                r["signal"], r["signal"])
        sig_disp["signal"] = sig_disp.apply(_status, axis=1)
        # 待成交标的: reason 用原始信号原因(而非"已有待执行订单"拦截文案)
        sig_disp["reason"] = sig_disp.apply(
            lambda r: pending_reason.get(
                str(r["symbol"]).zfill(6), r["reason"]),
            axis=1)
        sig_disp["target_pct"] = sig_disp["target_pct"].apply(
            lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) else "")
        # 待成交优先展示
        sig_disp = sig_disp.sort_values(
            "signal", key=lambda c: c.str.contains("待", na=False),
            ascending=False)
    sig_table = (sig_disp.to_html(index=False, classes="data-table",
                                  border=0, justify="center", escape=False)
                 if len(sig_disp) else "<p>暂无信号</p>")

    # 交易明细
    det_rows = []
    if len(elog):
        ex = elog[elog["status"] == "executed"]
        for _, r in ex.iterrows():
            det_rows.append({
                "日期": str(r.get("exec_date", ""))[:10],
                "代码": r["symbol"],
                "名称": _name_of(r["symbol"], positions, elog, trades),
                "方向": "买入" if r["action"] == "buy" else "卖出",
                "股数": int(r.get("shares", 0)),
                "价格": float(r.get("exec_price", 0)),
            })
    det_df = pd.DataFrame(det_rows).sort_values("日期", ascending=False) if det_rows else pd.DataFrame()
    det_table = (det_df.to_html(index=False, classes="data-table",
                                border=0, justify="center", escape=False)
                 if len(det_df) else "<p>暂无交易</p>")

    # 已完成交易配对表(2026-08-25 新增, 对齐实盘)
    trades_table = _trades_table(trades, positions, elog)

    # ---- 组装 HTML ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html><html lang='zh-CN'>
<head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>无限仓位信号报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5;color:#333;overflow-x:clip}}
h1{{color:#1a1a1a;border-bottom:3px solid #1f77b4;padding-bottom:10px}}
h2{{color:#2c3e50;margin-top:40px;border-bottom:2px solid #ddd;padding-bottom:8px}}
.positive{{color:#2ca02c}}.negative{{color:#d62728}}
.summary-text{{font-size:14px;color:#555;margin:10px 0;padding:10px;background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.05)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
margin:20px 0;background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
.mcard{{padding:16px 12px;text-align:center;border-right:1px solid #eee;border-bottom:1px solid #eee}}
.mcard .label{{font-size:12px;color:#666;display:block}}
.mcard .value{{font-size:20px;font-weight:bold;display:block;margin-top:4px}}
.search-box{{margin:16px 0}}
.search-box input{{width:100%;max-width:400px;padding:10px 14px;font-size:15px;
border:2px solid #1f77b4;border-radius:8px}}
.data-table{{width:max-content;min-width:max(640px,100%);border-collapse:separate;
border-spacing:0;font-size:13px;background:white;border-radius:8px;
box-shadow:0 2px 8px rgba(0,0,0,0.1);margin:12px 0}}
.data-table th{{background:#1f77b4;color:white;padding:11px 10px;text-align:center;
font-weight:600;white-space:nowrap}}
.data-table th:first-child,.data-table td:first-child{{
position:sticky;left:0;background:#eef4fa;box-shadow:1px 0 0 #d0e3f2;
z-index:2;font-weight:bold;white-space:nowrap}}
.data-table thead th:first-child{{z-index:3}}
.data-table th,.data-table td{{border-bottom:1px solid #eee;padding:8px 10px}}
.table-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch;width:100%}}
.col2{{display:flex;flex-wrap:wrap;gap:20px}}
.wl-table{{min-width:280px}}
.col2>div{{flex:1 1 340px;min-width:0}}
h3{{color:#555;margin-top:25px;border-left:3px solid #1f77b4;padding-left:8px}}
@media (max-width:900px){{.col2{{display:block}}.col2>div{{width:100%}}}}
@media (max-width:480px){{
.mcard .value{{font-size:16px}}.mcard .label{{font-size:11px}}
.data-table{{font-size:11px}}
.js-plotly-plot .legend{{display:none}}
}}
</style>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
</head><body>
<h1>无限仓位信号报告</h1>
<p style='color:#666;font-size:14px'>生成时间: {now} | 信号引擎与实盘一致, 仅仓位规则不同(1-3手)</p>

<div class='search-box'>
<input id='stock-search' placeholder='🔍 输入股票代码或名称, 过滤所有表格...'
oninput='filterTables(this.value)'>
</div>

<div class='metrics'>
<div class='mcard'><span class='label'>监控标的</span><span class='value'>{n_watch}</span></div>
<div class='mcard'><span class='label'>本轮信号标的</span><span class='value'>{n_symbols}</span></div>
<div class='mcard'><span class='label'>当前持仓</span><span class='value'>{n_pos} 只</span></div>
<div class='mcard'><span class='label'>累计买入信号</span><span class='value'>{n_buy_signals}</span></div>
<div class='mcard'><span class='label'>已实现·股票</span><span class='value {_cls(realized_stock)}'>¥{realized_stock:+,.0f}</span></div>
<div class='mcard'><span class='label'>已实现·基金</span><span class='value {_cls(realized_etf)}'>¥{realized_etf:+,.0f}</span></div>
<div class='mcard'><span class='label'>已实现·整体</span><span class='value {rpnl_cls}'>¥{realized_all:+,.0f}</span></div>
<div class='mcard'><span class='label'>浮动·股票</span><span class='value {_cls(floating_stock)}'>¥{floating_stock:+,.0f}</span></div>
<div class='mcard'><span class='label'>浮动·基金</span><span class='value {_cls(floating_etf)}'>¥{floating_etf:+,.0f}</span></div>
<div class='mcard'><span class='label'>浮动·整体</span><span class='value {pos_pnl_cls}'>¥{floating_all:+,.0f}</span></div>
<div class='mcard'><span class='label'>总盈亏</span><span class='value {total_cls}'>¥{total_pnl:+,.0f}</span></div>
<div class='mcard'><span class='label'>股票池总盈亏</span><span class='value {total_stock_cls}'>¥{total_stock:+,.0f}</span></div>
<div class='mcard'><span class='label'>基金池总盈亏</span><span class='value {total_etf_cls}'>¥{total_etf:+,.0f}</span></div>
<div class='mcard'><span class='label'>归一化·股票</span><span class='value {_cls(avg_norm_stock)}'>{avg_norm_stock:+.1%}</span></div>
<div class='mcard'><span class='label'>归一化·基金</span><span class='value {_cls(avg_norm_etf)}'>{avg_norm_etf:+.1%}</span></div>
<div class='mcard'><span class='label'>归一化·整体</span><span class='value {norm_cls}'>{avg_norm:+.1%}</span></div>
</div>

<h2>当前监控列表</h2>
<div class='col2'>
<div><h3>📈 股票 ({len(wl_stocks)})</h3><div class='table-wrap'>{wl_stock_table}</div></div>
<div><h3>📊 基金({len(wl_etfs)})</h3><div class='table-wrap'>{wl_etf_table}</div></div>
</div>

<h2>归一化收益排序</h2>
<p style='color:#666'>每只股票以「3手 × 首笔买入价」为总资金, 归一化收益 = 总利润 / 总资金(股票/基金分开)</p>
<h3>📈 股票</h3>
{chart_html}
<div class='table-wrap'>{norm_table}</div>
<h3>📊 基金(ETF)</h3>
{chart_etf_html}
<div class='table-wrap'>{norm_etf_table}</div>

<h2>当前持仓</h2>
<div class='col2'>
<div><h3>📈 股票</h3><div class='table-wrap'>{pos_stock_table}</div></div>
<div><h3>📊 基金(ETF)</h3><div class='table-wrap'>{pos_etf_table}</div></div>
</div>

<h2>信号状态（每只标的最近信号 + 未成交订单）</h2>
<div class='table-wrap'>{sig_table}</div>

<h2>已完成交易</h2>
{trades_table}

<h2>交易明细</h2>
<div class='table-wrap'>{det_table}</div>

<script>
function filterTables(q){{
  q=(q||'').trim().toUpperCase();
  document.querySelectorAll('.data-table').forEach(function(t){{
    t.querySelectorAll('tbody tr').forEach(function(tr){{
      tr.style.display = (!q || tr.textContent.toUpperCase().includes(q)) ? '' : 'none';
    }});
  }});
}}
</script>
</body></html>"""
    with open(os.path.join(UNLIMITED_DIR, "unlimited_report.html"),
              "w", encoding="utf-8") as f:
        f.write(html)


def _name_of(sym, positions, elog, trades):
    if sym in positions:
        return positions[sym].get("name", sym)
    if len(elog):
        m = elog[elog["symbol"] == sym]
        if len(m):
            return str(m.iloc[-1].get("name", sym))
    if len(trades) and "symbol" in trades.columns:
        m = trades[trades["symbol"].astype(str).str.zfill(6) == sym]
        if len(m):
            return str(m.iloc[-1].get("name", sym))
    return sym


def _trades_table(trades, positions, elog):
    """已完成交易配对表(2026-08-25 新增, 对齐实盘 live_report 口径)。

    trades.csv 为已平仓卖出记录: 入场/出场日期、入场/出场价、股数、
    盈亏(含费)、收益率、持有天数。
    """
    if len(trades) == 0:
        return "<p>暂无已完成交易</p>"
    sells = trades.sort_values("exit_date", ascending=False).copy()
    total_pnl = float(sells["pnl"].sum()) if "pnl" in sells.columns else 0.0
    wins = int((sells["pnl"] > 0).sum()) if "pnl" in sells.columns else 0
    sells["持有天数"] = (
        pd.to_datetime(sells["exit_date"]) - pd.to_datetime(sells["entry_date"])
    ).dt.days
    sells["名称"] = sells["symbol"].apply(
        lambda s: _name_of(str(s).zfill(6), positions, elog, sells))
    disp = sells[["名称", "symbol", "entry_date", "exit_date", "shares",
                  "entry_price", "exit_price", "pnl", "pnl_pct", "持有天数"]].copy()
    disp.columns = ["名称", "代码", "入场日期", "出场日期", "股数",
                    "入场价", "出场价", "盈亏", "收益率", "持有天数"]
    disp["盈亏"] = disp["盈亏"].apply(
        lambda x: f"<span class='{'positive' if float(x) > 0 else 'negative'}'>{float(x):+,.0f}</span>")
    disp["收益率"] = disp["收益率"].apply(
        lambda x: f"<span class='{'positive' if float(x) > 0 else 'negative'}'>{float(x):+.1f}%</span>")
    summary = (f"<div class='summary-text'>共 {len(sells)} 笔 | 盈利 {wins} 笔 | "
               f"胜率 {wins / len(sells) * 100:.0f}% | 总盈亏 "
               f"<span class='{'positive' if total_pnl > 0 else 'negative'}'>"
               f"¥{total_pnl:+,.0f}</span></div>")
    table = disp.to_html(index=False, classes="data-table trades-table",
                         border=0, justify="center", escape=False)
    return summary + f"<div class='table-wrap'>{table}</div>"


if __name__ == "__main__":
    build_unlimited_report()
    print("报告已生成: unlimited_report.html")
