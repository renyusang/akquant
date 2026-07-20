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
    shares: int,
    signal_price: float,
    signal_date: str,
    target_pct: float,
) -> None:
    """新增待执行买单。

    同一股票只保留最近一条（覆盖旧订单）。
    """
    orders = load_pending()
    key = str(symbol).zfill(6)
    # 移除同股票旧订单
    orders = [o for o in orders if o["symbol"] != key]
    orders.append({
        "symbol": key,
        "name": name,
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
    price_map: Dict[str, float],      # symbol → 今日开盘价
    previous_close_map: Dict[str, float],  # symbol → 昨日收盘价（用于计算涨停价）
) -> List[Dict[str, Any]]:
    """执行所有待处理订单。

    对每个待执行买单：
    1. 取今日开盘价作为成交价
    2. 检查是否涨停（开盘价 ≥ 昨日收盘价 × 1.10，A股涨跌停 ±10%）
    3. 涨停 → 跳过，标记为"涨停未成交"
    4. 正常 → 返回成交记录

    返回已成交的订单列表（用于写入持仓）。
    """
    orders = load_pending()
    executed = []
    failed = []

    for order in orders:
        sym = order["symbol"]
        open_price = price_map.get(sym)
        prev_close = previous_close_map.get(sym)

        if open_price is None:
            failed.append({**order, "reason": "无开盘价数据"})
            continue

        # 检查涨停（A股 ±10%，科创板/创业板 ±20%）
        if prev_close and prev_close > 0:
            limit_up = prev_close * 1.10
            # 科创板/创业板 20% 涨跌停
            if sym.startswith("688") or sym.startswith("300") or sym.startswith("301"):
                limit_up = prev_close * 1.20
            if open_price >= limit_up * 0.999:  # 容差
                failed.append({**order, "reason": f"涨停(开盘{open_price}≥涨停价{limit_up:.2f})"})
                continue

        # 执行日期 = 当日数据的最新日期（实际交易日）
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        executed.append({
            **order,
            "exec_price": open_price,
            "exec_date": today_str,
        })

    # 保存失败订单（下次继续尝试）或直接丢弃
    save_pending(failed)
    return executed
