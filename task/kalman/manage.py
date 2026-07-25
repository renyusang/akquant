#!/usr/bin/env python
"""
持仓手动管理工具。

用法:
    python manage.py show                      # 查看当前持仓
    python manage.py add 002594 比亚迪 800 95.20 2026-07-17   # 新增持仓
    python manage.py remove 002594             # 删除持仓
    python manage.py adjust 002594 500 90.00   # 调整股数和成本
    python manage.py restore 002594            # 从 trades.csv 恢复最近删除的持仓
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml


def _dwidth(s: str) -> int:
    """估算字符串在终端中的显示宽度（CJK≈2，ASCII≈1）。"""
    w = 0
    for c in str(s):
        w += 2 if '一' <= c <= '鿿' or '　' <= c <= '〿' or '＀' <= c <= '￯' else 1
    return w


def _pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度填充字符串。"""
    need = width - _dwidth(s)
    if need <= 0:
        return s
    if align == "right":
        return " " * need + s
    return s + " " * need

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio import (
    POSITIONS_FILE,
    TRADES_FILE,
    add_position,
    get_position,
    load_positions,
    remove_position,
    save_positions,
)
import pandas as pd


def cmd_show() -> None:
    """显示持仓总览，分股票和 ETF 两个池。"""
    positions = load_positions()
    from portfolio import get_trade_summary

    max_pct = 0.20; cash = 100000; etf_cash = 100000; etf_max_pct = 0.30; etf_max_pos = 3
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.yaml")) as f:
            cfg = yaml.safe_load(f)
        stock_cfg = cfg.get("stock", cfg.get("defaults", {}))
        etf_cfg = cfg.get("etf", {})
        cash = float(stock_cfg.get("initial_cash", 200000))
        max_pct = float(stock_cfg.get("single_position_pct", 0.20))
        max_pos = int(stock_cfg.get("max_positions", 5))
        etf_cash = float(etf_cfg.get("initial_cash", 100000))
        etf_max_pct = float(etf_cfg.get("single_position_pct", 0.20))
        etf_max_pos = int(etf_cfg.get("max_positions", 5))
    except Exception:
        max_pos = 5; etf_max_pos = 5
    max_value = cash * max_pct
    etf_max_value = etf_cash * etf_max_pct

    price_map = {}
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    for fname in os.listdir(cache_dir):
        if fname.endswith(".parquet"):
            try:
                df = pd.read_parquet(os.path.join(cache_dir, fname))
                if len(df) > 0:
                    sym = fname.replace(".parquet", "").zfill(6)
                    price_map[sym] = float(df["close"].iloc[-1])
            except Exception:
                pass

    etf_prefixes = ("51", "15", "58", "56")
    stock_pos = {s: p for s, p in positions.items() if not s.startswith(etf_prefixes)}
    etf_pos = {s: p for s, p in positions.items() if s.startswith(etf_prefixes)}

    def _print_table(label, pool_positions, pool_cash, pool_max_pct, pool_max_value):
        if not pool_positions:
            print(f"\n  [{label}] 当前无持仓  (可用: ¥{pool_cash:,.0f})")
            return
        CW = [6, 8, 5, 9, 9, 9, 5, 10, 7]
        headers = ["代码", "名称", "股数", "成本", "现价", "市值", "占比", "浮动盈亏", "收益率"]
        aligns = ["left", "left", "right", "right", "right", "right", "right", "right", "right"]
        hdr = "  " + "  ".join(_pad(h, CW[i], aligns[i]) for i, h in enumerate(headers))
        sep = "  " + "  ".join("-" * w for w in CW)
        total_cost = 0.0; total_market = 0.0
        rows = []
        for sym, pos in sorted(pool_positions.items()):
            price = price_map.get(sym, pos["avg_cost"])
            cost_v = pos["shares"] * pos["avg_cost"]
            market_v = pos["shares"] * price
            pnl = market_v - cost_v
            pnl_pct = (price / pos["avg_cost"] - 1) * 100
            pct = market_v / pool_cash * 100 if pool_cash > 0 else 0
            total_cost += cost_v; total_market += market_v
            flag = " !" if market_v > pool_max_value * 1.01 else ""
            rows.append([sym, pos['name'], f"{pos['shares']:,d}",
                         f"¥{pos['avg_cost']:.2f}", f"¥{price:.2f}",
                         f"¥{market_v:,.0f}", f"{pct:.1f}%",
                         f"¥{pnl:+,.0f}{flag}", f"{pnl_pct:+.1f}%"])
        total_pnl = total_market - total_cost
        total_pnl_pct = (total_market / total_cost - 1) * 100 if total_cost > 0 else 0
        total_pct = total_market / pool_cash * 100 if pool_cash > 0 else 0
        print(f"\n  [{label}] ({len(pool_positions)} 只, 上限{pool_max_pct*100:.0f}%/只 ¥{pool_max_value:,.0f})")
        print(hdr); print(sep)
        for r in rows:
            print("  " + "  ".join(_pad(v, CW[i], aligns[i]) for i, v in enumerate(r)))
        print(sep)
        print(f"  持仓市值: ¥{total_market:,.0f} / ¥{pool_cash:,.0f} = {total_pct:.1f}%  浮动盈亏: ¥{total_pnl:+,.0f} ({total_pnl_pct:+.1f}%)")
        return total_cost, total_market

    if not positions:
        print("当前无持仓")
    else:
        stock_cost, stock_market = _print_table("股票", stock_pos, cash, max_pct, max_value)
        etf_cost, etf_market = _print_table("ETF", etf_pos, etf_cash, etf_max_pct, etf_max_value)

    total_cost = stock_cost + etf_cost
    if total_cost > 0:
        stock_pnl = stock_market - stock_cost
        etf_pnl = etf_market - etf_cost
        total_pnl = stock_pnl + etf_pnl
        print(f"\n  [股票] 成本 ¥{stock_cost:,.0f}  市值 ¥{stock_market:,.0f}  浮动 ¥{stock_pnl:+,.0f}")
        print(f"  [ETF]  成本 ¥{etf_cost:,.0f}  市值 ¥{etf_market:,.0f}  浮动 ¥{etf_pnl:+,.0f}")
        print(f"  [合计] 成本 ¥{stock_cost+etf_cost:,.0f}  市值 ¥{stock_market+etf_market:,.0f}  浮动 ¥{total_pnl:+,.0f}  ({len(positions)} 只)")

    trade_summary = get_trade_summary()
    if trade_summary["count"] > 0:
        print(f"  已实现盈亏: ¥{trade_summary['total_pnl']:+,.0f}")
        print(f"  总盈亏(浮+实): ¥{total_pnl + trade_summary['total_pnl']:+,.0f}")

    from orders import load_pending
    pending = load_pending()
    sells = [o for o in pending if o.get("action") == "sell"]
    buys = [o for o in pending if o.get("action") != "sell"]
    if sells:
        print(f"\n🔴 待卖出 ({len(sells)} 笔):")
        for o in sells:
            print(f"   {o['symbol']} {o['name']} {o['shares']}股  信号价¥{o['signal_price']:.2f}  信号日{o['signal_date']}")
    if buys:
        print(f"\n🟢 待买入 ({len(buys)} 笔):")
        for o in buys:
            pct = float(o.get("target_pct", 0)) * 100
            print(f"   {o['symbol']} {o['name']} {o['shares']}股 ({pct:.0f}%)  信号价¥{o['signal_price']:.2f}  信号日{o['signal_date']}")

    # 最近买入（从 execution_log）
    elog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "execution_log.csv")
    if os.path.exists(elog_path):
        elog = pd.read_csv(elog_path, dtype={"symbol": str})
        elog["symbol"] = elog["symbol"].str.zfill(6)
        recent_buys = elog[(elog["status"] == "executed") & (elog["action"] == "buy")]
        if len(recent_buys) > 0:
            recent_buys = recent_buys.sort_values("exec_date").tail(3)
            print(f"\n🟢 最近买入 ({len(recent_buys)} 笔):")
            for _, t in recent_buys.iterrows():
                print(f"   {t['symbol']} {t['name']} {int(t['shares'])}股  ¥{float(t['exec_price']):.2f}  {t['exec_date']}")

    # 最近卖出（从 trades.csv）
    tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.csv")
    if os.path.exists(tp):
        trades_df = pd.read_csv(tp, dtype={"symbol": str})
        if len(trades_df) > 0:
            trades_df["symbol"] = trades_df["symbol"].str.zfill(6)
            recent = trades_df.tail(3)
            print(f"\n🔴 最近卖出 ({len(recent)} 笔):")
            for _, t in recent.iterrows():
                pnl = float(t["pnl"])
                ps = f"+¥{pnl:,.0f}" if pnl >= 0 else f"¥{pnl:,.0f}"
                print(f"   {t['symbol']} {t['name']} {int(t['shares'])}股  ¥{float(t['entry_price']):.2f}→¥{float(t['exit_price']):.2f}  {ps}  {t['entry_date']}→{t['exit_date']}")

    skipped = []
    elog_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "execution_log.csv")
    sig_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.csv")
    if os.path.exists(elog_csv) and os.path.exists(sig_csv):
        elog = pd.read_csv(elog_csv, dtype={"symbol": str})
        elog["symbol"] = elog["symbol"].str.zfill(6)
        sig_df = pd.read_csv(sig_csv, dtype={"symbol": str})
        sig_df["symbol"] = sig_df["symbol"].str.zfill(6)
        latest = sig_df.sort_values("date").groupby("symbol").last()
        pos_set = set(positions.keys())
        pend_set = {o["symbol"] for o in pending}
        seen = set()
        for _, r in elog.iterrows():
            sym = r["symbol"]
            if r.get("status") != "skipped" or r.get("action") != "buy":
                continue
            if sym in pos_set or sym in pend_set or sym in seen:
                continue
            seen.add(sym)
            if sym in latest.index:
                row = latest.loc[sym]
                c = float(row.get("close", 0))
                k = float(row.get("kalman_price", c))
                dev = abs((c / k - 1) * 100) if k > 0 else 0
                target_pct = float(row.get("target_pct", 0.95))
                skipped.append({"symbol": sym, "name": row.get("name", sym), "close": c, "deviation": dev, "trend": row.get("trend", "?"), "target_pct": target_pct})
        skipped.sort(key=lambda x: x["deviation"], reverse=True)

    stock_skipped = [s for s in skipped if not s["symbol"].startswith(etf_prefixes)]
    etf_skipped = [s for s in skipped if s["symbol"].startswith(etf_prefixes)]

    def _show_group(label, items, pool_slots, pool_cash, pool_max_pct):
        if not items:
            return
        print(f"\n⏸️ [{label}] 待买入 ({len(items)} 只，空位{pool_slots}个):")
        picked = 0
        for s in items:
            sym = s["symbol"]
            blocked = False
            cp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", f"{sym}.parquet")
            if os.path.exists(cp):
                try:
                    cd = pd.read_parquet(cp)
                    if len(cd) >= 1:
                        pc = float(cd["close"].iloc[-1])
                        lp = 0.20 if sym.startswith("688") or sym.startswith("300") or sym.startswith("301") else 0.10
                        if s["close"] >= pc * (1 + lp) * 0.999:
                            blocked = True
                except Exception:
                    pass
            lot = 200 if sym.startswith("688") else 100
            # 从趋势推断原始仓位：上涨 95%，下跌 30%
            trend = s.get("trend", "up")
            target = 0.30 if trend == "down" else 0.95
            est_qty = int(pool_cash * pool_max_pct * target / s["close"] / lot) * lot
            if blocked:
                print(f"   {s['symbol']} {s['name']:<6s} ¥{s['close']:>8.2f}  偏离{s['deviation']:+.1f}%  趋势={s['trend']}  ⚠️涨停跳过")
            elif picked < pool_slots:
                print(f"   {s['symbol']} {s['name']:<6s} ¥{s['close']:>8.2f}  {est_qty}股  偏离{s['deviation']:+.1f}%  趋势={s['trend']}  ← 买入 #{picked+1}")
                picked += 1
            else:
                print(f"   {s['symbol']} {s['name']:<6s} ¥{s['close']:>8.2f}  {est_qty}股  偏离{s['deviation']:+.1f}%  趋势={s['trend']}")

    pending_sells = sum(1 for o in pending if o.get("action") == "sell")
    pending_buys = len(pending) - pending_sells
    stock_slots = max_pos - len(stock_pos) - pending_buys + pending_sells
    etf_slots = etf_max_pos - len(etf_pos)
    _show_group("股票", stock_skipped, stock_slots, cash, max_pct)
    _show_group("ETF", etf_skipped, etf_slots, etf_cash, etf_max_pct)

    print()
def cmd_add(symbol: str, name: str, shares: int, cost: float, date: str) -> None:
    """手动新增持仓。"""
    # 读取配置中的资金限制
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        stock_cfg = cfg.get("stock", cfg.get("defaults", {}))
        cash = float(stock_cfg.get("initial_cash", 200000))
        max_pct = float(stock_cfg.get("single_position_pct", 0.20))
        max_value = cash * max_pct
        add_value = shares * cost
        if add_value > max_value * 1.01:  # 1% 容差
            max_shares = int(max_value / cost / 100) * 100
            print(f"⚠️ 警告: {shares}股 × ¥{cost:.2f} = ¥{add_value:,.0f} 超过单只上限 "
                  f"¥{max_value:,.0f} ({max_pct*100:.0f}%)，建议 ≤ {max_shares}股")
            return
    except Exception:
        pass
    add_position(symbol, name, shares, cost, date)
    pos = get_position(symbol)
    print(f"✅ 已添加 {symbol} {name}: {pos['shares']}股 @ ¥{pos['avg_cost']:.2f}")


def cmd_fill(symbol: str, action: str, shares: int, price: float) -> None:
    """记录人工实际成交价(供 daily_signal 执行时覆盖 open 假设)。

    用法:人工按 pending_orders 实际下单后,用此命令记录实际成交价。
    daily_signal 下次执行时优先用此价,使 avg_cost/pnl 反映真实成交。
    """
    from orders import add_actual_fill, load_actual_fills

    action = action.lower()
    if action not in ("buy", "sell"):
        print(f"⚠️ action 必须是 buy 或 sell,收到 {action}")
        return
    add_actual_fill(symbol, action, shares, price)
    fills = load_actual_fills()
    print(f"✅ 已记录实际成交 {symbol} {action} {shares}股 @ ¥{price:.2f}")
    print(f"   daily_signal 下次执行时将用此价(而非开盘价假设)")
    if fills:
        print(f"   当前待用实际成交记录: {len(fills)} 笔")


def cmd_remove(symbol: str) -> None:
    """删除持仓。"""
    removed = remove_position(symbol)
    if removed:
        print(f"✅ 已删除 {symbol} {removed['name']}: {removed['shares']}股 @ ¥{removed['avg_cost']:.2f}")
    else:
        print(f"⚠️ {symbol} 不在持仓中")


def cmd_adjust(symbol: str, shares: int, cost: float) -> None:
    """调整持仓股数和成本。"""
    positions = load_positions()
    key = str(symbol).zfill(6)
    if key not in positions:
        print(f"⚠️ {symbol} 不在持仓中，请先用 add 添加")
        return

    # 校验仓位上限
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        stock_cfg = cfg.get("stock", cfg.get("defaults", {}))
        cash = float(stock_cfg.get("initial_cash", 200000))
        max_pct = float(stock_cfg.get("single_position_pct", 0.20))
        max_value = cash * max_pct
        new_value = shares * cost
        if new_value > max_value * 1.01:
            max_shares = int(max_value / cost / 100) * 100
            print(f"⚠️ {shares}股 × ¥{cost:.2f} = ¥{new_value:,.0f} 超过上限 "
                  f"¥{max_value:,.0f} ({max_pct*100:.0f}%)，建议 ≤ {max_shares}股")
            return
    except Exception:
        pass

    old = positions[key]
    positions[key]["shares"] = shares
    positions[key]["avg_cost"] = cost
    save_positions(positions)
    print(f"✅ 已调整 {symbol} {old['name']}: {old['shares']}→{shares}股, "
          f"¥{old['avg_cost']:.2f}→¥{cost:.2f}")


def cmd_restore(symbol: str) -> None:
    """从 trades.csv 恢复最近一笔已删除的持仓。"""
    if not os.path.exists(TRADES_FILE):
        print("⚠️ trades.csv 不存在")
        return

    trades = pd.read_csv(TRADES_FILE)
    key = str(symbol).zfill(6)
    symbol_trades = trades[trades["symbol"].astype(str).str.zfill(6) == key]
    if symbol_trades.empty:
        print(f"⚠️ {symbol} 无历史交易记录")
        return

    last = symbol_trades.iloc[-1]
    add_position(
        symbol=symbol,
        name=last.get("name", symbol),
        shares=int(last["shares"]),
        avg_cost=float(last["entry_price"]),
        buy_date=str(last["entry_date"]),
    )
    print(f"✅ 已恢复 {symbol} {last['name']}: {int(last['shares'])}股 @ ¥{last['entry_price']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="持仓手动管理工具")
    sub = parser.add_subparsers(dest="command", help="操作")

    p_show = sub.add_parser("show", help="查看持仓")

    p_add = sub.add_parser("add", help="新增持仓")
    p_add.add_argument("symbol", help="股票代码")
    p_add.add_argument("name", help="股票名称")
    p_add.add_argument("shares", type=int, help="股数")
    p_add.add_argument("cost", type=float, help="成本价")
    p_add.add_argument("date", help="首次买入日期 (YYYY-MM-DD)")

    p_remove = sub.add_parser("remove", help="删除持仓")
    p_remove.add_argument("symbol", help="股票代码")

    p_adjust = sub.add_parser("adjust", help="调整持仓")
    p_adjust.add_argument("symbol", help="股票代码")
    p_adjust.add_argument("shares", type=int, help="新股数")
    p_adjust.add_argument("cost", type=float, help="新成本价")

    p_restore = sub.add_parser("restore", help="从交易记录恢复持仓")
    p_restore.add_argument("symbol", help="股票代码")

    p_fill = sub.add_parser("fill", help="记录人工实际成交价(覆盖开盘价假设)")
    p_fill.add_argument("symbol", help="股票代码")
    p_fill.add_argument("action", help="buy 或 sell")
    p_fill.add_argument("shares", type=int, help="股数")
    p_fill.add_argument("price", type=float, help="实际成交价")

    p_backups = sub.add_parser("backups", help="列出所有备份")
    p_rollback = sub.add_parser("rollback", help="回滚到指定备份")
    p_rollback.add_argument("timestamp", nargs="?", default=None, help="备份时间戳 (默认最新)")

    args = parser.parse_args()

    # 写操作前自动备份(影响持仓的操作;fill 只写 actual_fills,不备份)
    if args.command in ("add", "remove", "adjust", "restore"):
        try:
            from backup import create_backup

            tag = create_backup()
            print(f"📦 操作前备份: {tag}")
        except Exception as e:
            print(f"⚠️ 操作前备份失败({e}),继续操作")

    if args.command == "show":
        cmd_show()
    elif args.command == "add":
        cmd_add(args.symbol, args.name, args.shares, args.cost, args.date)
    elif args.command == "remove":
        cmd_remove(args.symbol)
    elif args.command == "adjust":
        cmd_adjust(args.symbol, args.shares, args.cost)
    elif args.command == "restore":
        cmd_restore(args.symbol)
    elif args.command == "fill":
        cmd_fill(args.symbol, args.action, args.shares, args.price)
    elif args.command == "backups":
        from backup import list_backups
        backups = list_backups()
        if backups:
            print(f"\n备份快照 ({len(backups)} 个):")
            for b in backups:
                print(f"  {b}")
        else:
            print("无备份")
    elif args.command == "rollback":
        from backup import restore_backup
        restore_backup(args.timestamp)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
