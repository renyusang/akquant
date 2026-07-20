"""
持仓与交易记录持久化模块。

提供持仓（positions.json）和已完成交易（trades.csv）的读写接口。
daily_signal.py 通过此模块管理真实持仓状态，替代从 signals.csv 推断。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- 路径 ----
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(TASK_DIR, "positions.json")
TRADES_FILE = os.path.join(TASK_DIR, "trades.csv")

TRADES_COLUMNS = [
    "entry_date", "exit_date", "symbol", "name",
    "shares", "entry_price", "exit_price",
    "pnl", "pnl_pct", "reason",
]


# =============================================================================
# 持仓读写
# =============================================================================
def load_positions() -> Dict[str, Dict[str, Any]]:
    """加载当前持仓。

    返回 {symbol: {name, shares, avg_cost, first_buy_date}} 字典。
    """
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_positions(positions: Dict[str, Dict[str, Any]]) -> None:
    """保存持仓到 JSON 文件。"""
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def add_position(
    symbol: str,
    name: str,
    shares: int,
    avg_cost: float,
    buy_date: Optional[str] = None,
) -> None:
    """新增或合并持仓（同一股票多次买入时更新均价和股数）。"""
    positions = load_positions()
    key = str(symbol).zfill(6)
    if buy_date is None:
        buy_date = datetime.now().strftime("%Y-%m-%d")

    if key in positions:
        old = positions[key]
        total_cost = old["shares"] * old["avg_cost"] + shares * avg_cost
        total_shares = old["shares"] + shares
        positions[key] = {
            "name": name,
            "shares": total_shares,
            "avg_cost": round(total_cost / total_shares, 3),
            "first_buy_date": old["first_buy_date"],
        }
    else:
        positions[key] = {
            "name": name,
            "shares": shares,
            "avg_cost": avg_cost,
            "first_buy_date": buy_date,
        }
    save_positions(positions)


def remove_position(symbol: str) -> Dict[str, Any]:
    """移除持仓并返回被删除的记录（用于记录交易盈亏）。"""
    positions = load_positions()
    key = str(symbol).zfill(6)
    removed = positions.pop(key, None)
    save_positions(positions)
    return removed or {}


def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    """获取单只股票的持仓信息。"""
    positions = load_positions()
    key = str(symbol).zfill(6)
    return positions.get(key)


def has_position(symbol: str) -> bool:
    """检查是否持有该股票。"""
    return get_position(symbol) is not None


# =============================================================================
# 交易记录读写
# =============================================================================
def load_trades() -> pd.DataFrame:
    """加载已完成交易记录。"""
    if not os.path.exists(TRADES_FILE):
        return pd.DataFrame(columns=TRADES_COLUMNS)
    try:
        return pd.read_csv(TRADES_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=TRADES_COLUMNS)


def save_trades(trades: pd.DataFrame) -> None:
    """保存交易记录。"""
    trades.to_csv(TRADES_FILE, index=False, encoding="utf-8")


def record_trade(
    symbol: str,
    name: str,
    shares: int,
    entry_price: float,
    exit_price: float,
    entry_date: str,
    exit_date: Optional[str] = None,
    reason: str = "",
) -> Dict[str, Any]:
    """记录一笔已完成交易。

    返回交易记录字典。
    """
    if exit_date is None:
        exit_date = datetime.now().strftime("%Y-%m-%d")

    pnl = (exit_price - entry_price) * shares
    pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0

    trade = {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "symbol": str(symbol).zfill(6),
        "name": name,
        "shares": shares,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "reason": reason,
    }

    trades = load_trades()
    trades = pd.concat([trades, pd.DataFrame([trade])], ignore_index=True)
    save_trades(trades)
    return trade


def get_trade_summary() -> Dict[str, float]:
    """获取交易汇总：总盈亏、盈利次数、亏损次数。"""
    trades = load_trades()
    if trades.empty:
        return {"total_pnl": 0.0, "wins": 0, "losses": 0, "count": 0}
    return {
        "total_pnl": float(trades["pnl"].sum()),
        "wins": int((trades["pnl"] > 0).sum()),
        "losses": int((trades["pnl"] < 0).sum()),
        "count": len(trades),
    }
