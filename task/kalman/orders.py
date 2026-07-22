"""
待执行订单管理模块。

买入信号在当日收盘后产生，实际执行在下一个交易日的开盘价。
本模块管理 pending_orders.json，确保：
1. 涨停时订单不执行
2. 记录实际成交价格（次日开盘价）
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PENDING_FILE = os.path.join(TASK_DIR, "pending_orders.json")


def load_pending() -> List[Dict[str, Any]]:
    """加载待执行订单列表。"""
    if not os.path.exists(PENDING_FILE):
        return []
    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_pending(orders: List[Dict[str, Any]]) -> None:
    """保存待执行订单。"""
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)


def add_pending_order(
    symbol: str,
    name: str,
    action: str,
    shares: int,
    signal_price: float,
    signal_date: str,
    target_pct: float = 0.0,
) -> None:
    """新增待执行订单（买入或卖出）。

    同一股票只保留最近一条（覆盖旧订单）。
    """
    orders = load_pending()
    key = str(symbol).zfill(6)
    orders = [o for o in orders if o["symbol"] != key]
    orders.append({
        "symbol": key,
        "name": name,
        "action": action,
        "shares": shares,
        "signal_price": signal_price,
        "signal_date": signal_date,
        "target_pct": target_pct,
    })
    save_pending(orders)


def remove_pending(symbol: str) -> Dict[str, Any]:
    """移除并返回待执行订单。"""
    orders = load_pending()
    key = str(symbol).zfill(6)
    removed = None
    new_orders = []
    for o in orders:
        if o["symbol"] == key:
            removed = o
        else:
            new_orders.append(o)
    save_pending(new_orders)
    return removed or {}


def execute_pending_orders(
    price_map: Dict[str, float],
    previous_close_map: Dict[str, float],
    today_str: str = "",
) -> List[Dict[str, Any]]:
    """执行待处理订单。

    仅执行信号日期 < 今日的订单（当天生成的信号等次日执行）。
    涨停时跳过，保留到下次。
    """
    if not today_str:
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    orders = load_pending()
    executed = []
    failed = []
    deferred = []

    for order in orders:
        # 当天信号不执行，等次日
        if order["signal_date"] >= today_str:
            order["reason"] = f"等待 {order['signal_date']} 次日开盘"
            deferred.append(order)
            continue
        sym = order["symbol"]
        open_price = price_map.get(sym)
        prev_close = previous_close_map.get(sym)

        if open_price is None:
            failed.append({**order, "reason": "无开盘价数据"})
            continue

        # 涨跌停检查
        action = order.get("action", "buy")
        if prev_close and prev_close > 0:
            is_kcb = sym.startswith("688") or sym.startswith("300") or sym.startswith("301")
            pct = 0.20 if is_kcb else 0.10
            limit_up = prev_close * (1 + pct)
            limit_down = prev_close * (1 - pct)

            if action == "buy" and open_price >= limit_up * 0.999:
                failed.append({**order, "reason": f"涨停(¥{open_price:.2f}≥¥{limit_up:.2f})"})
                continue
            if action == "sell" and open_price <= limit_down * 1.001:
                failed.append({**order, "reason": f"跌停(¥{open_price:.2f}≤¥{limit_down:.2f})"})
                continue

        executed.append({
            **order,
            "exec_price": open_price,
            "exec_date": today_str or pd.Timestamp.now().strftime("%Y-%m-%d"),
        })

    # 保留：当天信号（等次日）+ 涨停未成交（等下次）
    save_pending(failed + deferred)
    return executed
