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
    """显示持仓总览（含浮动盈亏和已实现盈亏）。"""
    positions = load_positions()
    from portfolio import get_trade_summary

    # 读取仓位上限配置
    max_pct = 0.20; cash = 100000
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.yaml")) as f:
            cfg = yaml.safe_load(f)
        defaults = cfg.get("defaults", {})
        cash = float(defaults.get("initial_cash", 100000))
        max_pct = float(defaults.get("single_position_pct", 0.20))
    except Exception:
        pass
    max_value = cash * max_pct

    if not positions:
        print("当前无持仓")
    else:
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

        print(f"\n{'代码':<10s} {'名称':<8s} {'股数':>6s} {'成本':>8s} {'现价':>8s} {'市值':>8s} {'占比':>5s} {'浮动盈亏':>8s} {'收益率':>8s}")
        print("-" * 78)
        total_cost_val = 0.0; total_market_val = 0.0
        for sym, pos in sorted(positions.items()):
            price = price_map.get(sym, pos["avg_cost"])
            cost_val = pos["shares"] * pos["avg_cost"]
            market_val = pos["shares"] * price
            pnl = market_val - cost_val
            pnl_pct = (price / pos["avg_cost"] - 1) * 100
            pct = market_val / cash * 100 if cash > 0 else 0
            total_cost_val += cost_val; total_market_val += market_val
            flag = " ⚠️" if market_val > max_value * 1.01 else ""
            print(f"{sym:<10s} {pos['name']:<8s} {pos['shares']:>6d}  "
                  f"¥{pos['avg_cost']:>7.2f} ¥{price:>7.2f} ¥{market_val:>7.0f} "
                  f"{pct:>4.1f}% ¥{pnl:>+7.0f} {pnl_pct:>+7.1f}%{flag}")
        print("-" * 78)
        total_pnl = total_market_val - total_cost_val
        total_pnl_pct = (total_market_val / total_cost_val - 1) * 100 if total_cost_val > 0 else 0
        total_pct = total_market_val / cash * 100 if cash > 0 else 0
        print(f"持仓市值: ¥{total_market_val:,.0f} / ¥{cash:,.0f} = {total_pct:.1f}%  "
              f"浮动盈亏: ¥{total_pnl:+,.0f} ({total_pnl_pct:+.1f}%)  ({len(positions)} 只)")
        if any(pos["shares"] * price_map.get(sym, pos["avg_cost"]) > max_value * 1.01 for sym, pos in positions.items()):
            print(f"⚠️ 有仓位超过单只上限 ¥{max_value:,.0f} ({max_pct*100:.0f}%)")

    # 已完成交易
    trade_summary = get_trade_summary()
    if trade_summary["count"] > 0:
        print(f"\n已完成交易: {trade_summary['count']} 笔  "
              f"盈利 {trade_summary['wins']} 亏损 {trade_summary['losses']}  "
              f"累计盈亏: ¥{trade_summary['total_pnl']:+,.0f}")
        # 最近 5 笔
        trades = pd.read_csv(TRADES_FILE)
        if len(trades) > 0:
            print(f"\n{'入场':<12s} {'出场':<12s} {'代码':<10s} {'名称':<8s} {'股数':>6s} {'入场价':>8s} {'出场价':>8s} {'盈亏':>10s}")
            print("-" * 80)
            for _, t in trades.tail(5).iterrows():
                print(f"{str(t['entry_date']):<12s} {str(t['exit_date']):<12s} "
                      f"{str(t['symbol']):<10s} {str(t['name']):<8s} {int(t['shares']):>6d}  "
                      f"¥{float(t['entry_price']):>7.2f}  ¥{float(t['exit_price']):>7.2f}  "
                      f"¥{float(t['pnl']):>+9.0f}")

    # 待执行订单
    from orders import load_pending
    pending = load_pending()
    if pending:
        print(f"\n⏳ 待执行 ({len(pending)} 笔):")
        for o in pending:
            print(f"   {o['symbol']} {o['name']} {o['shares']}股  信号价¥{o['signal_price']:.2f}  信号日{o['signal_date']}")
    print()


def cmd_add(symbol: str, name: str, shares: int, cost: float, date: str) -> None:
    """手动新增持仓。"""
    # 读取配置中的资金限制
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        defaults = cfg.get("defaults", {})
        cash = float(defaults.get("initial_cash", 100000))
        max_pct = float(defaults.get("single_position_pct", 0.20))
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
        defaults = cfg.get("defaults", {})
        cash = float(defaults.get("initial_cash", 100000))
        max_pct = float(defaults.get("single_position_pct", 0.20))
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

    args = parser.parse_args()

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
