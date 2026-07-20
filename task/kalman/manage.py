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
    """显示当前持仓总览。"""
    positions = load_positions()
    if not positions:
        print("当前无持仓")
        return

    print(f"\n{'代码':<10s} {'名称':<10s} {'股数':>8s} {'成本':>10s} {'首次买入':<12s}")
    print("-" * 55)
    total_value = 0
    for sym, pos in sorted(positions.items()):
        value = pos["shares"] * pos["avg_cost"]
        total_value += value
        print(
            f"{sym:<10s} {pos['name']:<10s} {pos['shares']:>8d}  "
            f"¥{pos['avg_cost']:>8.2f}  {pos['first_buy_date']:<12s}"
        )
    print("-" * 55)
    print(f"持仓市值: ¥{total_value:,.0f}  ({len(positions)} 只)")
    print(f"文件: {POSITIONS_FILE}\n")


def cmd_add(symbol: str, name: str, shares: int, cost: float, date: str) -> None:
    """手动新增持仓。"""
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
