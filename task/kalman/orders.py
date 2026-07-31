"""
待执行订单管理模块。

买入信号在当日收盘后产生，实际执行在下一个交易日的开盘价。
本模块管理 pending_orders.json，确保：
1. 涨停时订单不执行
2. 记录实际成交价格（次日开盘价）
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def _atomic_write_json(path: str, data: Any) -> None:
    """原子写入 JSON(临时文件 + os.replace,防中断损坏)。"""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
PENDING_FILE = os.path.join(TASK_DIR, "pending_orders.json")


def load_pending() -> List[Dict[str, Any]]:
    """加载待执行订单列表。损坏时告警并提示从备份恢复。"""
    if not os.path.exists(PENDING_FILE):
        return []
    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"⚠️ {PENDING_FILE} 结构异常(非列表),已忽略。建议从备份恢复: python manage.py rollback")
            return []
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠️ {PENDING_FILE} 读取失败({e})。建议从备份恢复: python manage.py rollback")
        return []


def save_pending(orders: List[Dict[str, Any]]) -> None:
    """保存待执行订单(原子写入)。"""
    _atomic_write_json(PENDING_FILE, orders)


# 人工实际成交价记录(覆盖 open 假设,提升实盘准确性)
ACTUAL_FILLS_FILE = os.path.join(TASK_DIR, "actual_fills.json")


def load_actual_fills() -> Dict[str, Any]:
    """加载人工实际成交价记录 {symbol: {action, shares, exec_price, date}}。"""
    if not os.path.exists(ACTUAL_FILLS_FILE):
        return {}
    try:
        with open(ACTUAL_FILLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_actual_fills(fills: Dict[str, Any]) -> None:
    """保存人工实际成交价记录(原子写入)。"""
    _atomic_write_json(ACTUAL_FILLS_FILE, fills)


def add_actual_fill(
    symbol: str, action: str, shares: int, exec_price: float, date: str = ""
) -> None:
    """记录一笔人工实际成交(供 manage.py fill 命令使用)。"""
    if not date:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    fills = load_actual_fills()
    fills[str(symbol).zfill(6)] = {
        "action": action,
        "shares": int(shares),
        "exec_price": float(exec_price),
        "date": date,
    }
    save_actual_fills(fills)


def add_pending_order(
    symbol: str,
    name: str,
    action: str,
    shares: int,
    signal_price: float,
    signal_date: str,
    target_pct: float = 0.0,
) -> None:
    """新增待执行订单。

    同向: 买入保留最早的信号日期(不推迟执行); 卖出更新价格和日期(反映最新信号)。
    不同向: 替换（卖出替换买入，反之亦然）。
    """
    orders = load_pending()
    key = str(symbol).zfill(6)
    existing = next((o for o in orders if o["symbol"] == key), None)
    if existing and existing.get("action") == action:
        if action == "buy":
            # 买入保留最早信号日期,不推迟T+1执行
            return
        # 卖出: 更新 price/shares/name, 保留原 signal_date(不推迟执行)
        existing["signal_price"] = signal_price
        existing["shares"] = shares
        existing["name"] = name
        save_pending(orders)
        return
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
    actual_prices: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """执行待处理订单。

    仅执行信号日期 < 今日的订单（当天生成的信号等次日执行）。
    涨停时跳过，保留到下次。

    :param actual_prices: 人工实际成交价 {symbol: price}，优先于 open 假设。
           提供实际价表示人工已成交，跳过涨跌停检查；未提供则用开盘价假设并检查涨跌停。
    """
    if not today_str:
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    actual_prices = actual_prices or {}
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
        actual_price = actual_prices.get(sym)
        # 优先用人工实际成交价（已成交）；否则用开盘价（假设）
        exec_price = actual_price if actual_price is not None else open_price

        if exec_price is None:
            failed.append({**order, "reason": "无开盘价数据"})
            continue

        # 涨跌停检查：仅对未提供实际价的订单（人工实际成交跳过）
        action = order.get("action", "buy")
        prev_close = previous_close_map.get(sym)
        if actual_price is None and prev_close and prev_close > 0:
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
            "exec_price": exec_price,
            "exec_date": today_str or pd.Timestamp.now().strftime("%Y-%m-%d"),
        })

    # 保留：当天信号（等次日）+ 涨停未成交（等下次）
    save_pending(failed + deferred)
    return executed
