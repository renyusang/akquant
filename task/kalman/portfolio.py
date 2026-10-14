"""
持仓与交易记录持久化模块。

提供持仓（positions.json）和已完成交易（trades.csv）的读写接口。
daily_signal.py 通过此模块管理真实持仓状态，替代从 signals.csv 推断。
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


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


# ---- 路径 ----
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(TASK_DIR, "positions.json")
TRADES_FILE = os.path.join(TASK_DIR, "trades.csv")

TRADES_COLUMNS = [
    "entry_date", "exit_date", "symbol", "name",
    "shares", "entry_price", "exit_price",
    "pnl", "pnl_pct", "fee", "reason",
]


def calc_fee(
    shares: int,
    price: float,
    action: str,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    transfer_fee_rate: float = 0.00001,
    min_commission: float = 5.0,
) -> float:
    """计算A股交易手续费(佣金 + 印花税 + 过户费)。

    佣金:max(成交额×费率, 最低佣金),双边收取。
    印花税:成交额×费率,仅卖出收取。
    过户费:成交额×费率,双边收取。
    """
    amount = abs(shares) * price
    commission = max(amount * commission_rate, min_commission)
    stamp_tax = amount * stamp_tax_rate if action == "sell" else 0.0
    transfer_fee = amount * transfer_fee_rate
    return commission + stamp_tax + transfer_fee


# =============================================================================
# 持仓读写
# =============================================================================
def load_positions() -> Dict[str, Dict[str, Any]]:
    """加载当前持仓。

    返回 {symbol: {name, shares, avg_cost, first_buy_date, peak_close}} 字典。
    `peak_close`(2026-09-28 新增, 2026-10-14 明确口径): **持仓期内观测到的最高收盘价**
    —— 只记录真实出现过的高点, 只增不减, **不掺入成本锚点**; 加仓也不重算。
    成本锚点在**使用时**按当时加权成本取: 引擎恢复值 = `max(avg_cost, peak_close)`,
    因此加仓后止盈基准自动随加权成本调整, 而真实涨出来的高点不会被抹掉
    (见 plan_profit_protection.md §3.4-1)。
    损坏时告警并提示从备份恢复(不静默丢失)。
    """
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️ {POSITIONS_FILE} 结构异常(非字典),已忽略。建议从备份恢复: python manage.py rollback")
            return {}
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠️ {POSITIONS_FILE} 读取失败({e})。建议从备份恢复: python manage.py rollback")
        return {}


def save_positions(positions: Dict[str, Dict[str, Any]]) -> None:
    """保存持仓到 JSON 文件(原子写入,防中断损坏)。"""
    _atomic_write_json(POSITIONS_FILE, positions)


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
        # 峰值只记"观测到的最高收盘"(2026-10-14): 加仓**既不重置也不重算** ——
        # 成本锚点在引擎恢复时按 `max(avg_cost, peak_close)` 施加, 故加仓后基准
        # 自动随加权成本调整, 而真实高点不被抹掉。
        positions[key] = {
            "name": name,
            "shares": total_shares,
            "avg_cost": round(total_cost / total_shares, 3),
            "first_buy_date": old["first_buy_date"],
            "peak_close": round(float(old.get("peak_close") or 0.0), 4),
        }
    else:
        positions[key] = {
            "name": name,
            "shares": shares,
            "avg_cost": avg_cost,
            "first_buy_date": buy_date,
            # 新开仓: 尚无观测高点(0.0) —— 当日收盘由 daily_signal 的峰值台账回写,
            # 而"从未涨过成本"时由恢复端的 max(avg_cost, ...) 兜底 = -4% 成本止损
            "peak_close": 0.0,
        }
    save_positions(positions)


def effective_peak(avg_cost: float, peak_close: float) -> float:
    """止盈基准 = max(当日加权成本, 观测到的最高收盘)。

    2026-10-14 明确: `peak_close` 只存**观测高点**(不掺成本锚点), 成本锚点在使用时
    施加 —— 于是加仓后基准自动随加权成本调整, 而真实涨出来的高点不会被抹掉。
    "从未涨过成本"时结果为均价 → 规则等价 -4% 成本止损(与修复前实盘一致)。
    """
    return max(float(avg_cost or 0.0), float(peak_close or 0.0))


def update_peak(symbol: str, peak: float) -> None:
    """更新单只持仓的跨日峰值(只增不减, 2026-09-28 新增)。

    由 `daily_signal` 在每日评估后回写 `engine.peak_price`; 无持仓/值非法时静默跳过。
    """
    if not peak or peak <= 0:
        return
    positions = load_positions()
    key = str(symbol).zfill(6)
    if key not in positions:
        return
    cur = float(positions[key].get("peak_close") or 0.0)
    if peak <= cur:
        return
    positions[key]["peak_close"] = round(float(peak), 4)
    save_positions(positions)


def update_peaks(peaks: Dict[str, float]) -> None:
    """批量更新跨日峰值(一次读写, 2026-09-28 新增)。"""
    if not peaks:
        return
    positions = load_positions()
    changed = False
    for sym, peak in peaks.items():
        key = str(sym).zfill(6)
        if key not in positions or not peak or peak <= 0:
            continue
        cur = float(positions[key].get("peak_close") or 0.0)
        if float(peak) > cur:
            positions[key]["peak_close"] = round(float(peak), 4)
            changed = True
    if changed:
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
    """保存交易记录(原子写入)。"""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(TRADES_FILE) or ".", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            trades.to_csv(f, index=False, encoding="utf-8")
        os.replace(tmp_path, TRADES_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def record_trade(
    symbol: str,
    name: str,
    shares: int,
    entry_price: float,
    exit_price: float,
    entry_date: str,
    exit_date: Optional[str] = None,
    reason: str = "",
    fee: float = 0.0,
) -> Dict[str, Any]:
    """记录一笔已完成交易。

    :param fee: 出场手续费(佣金+印花税+过户费),从 pnl 扣除。
    返回交易记录字典。
    """
    if exit_date is None:
        exit_date = datetime.now().strftime("%Y-%m-%d")

    pnl = (exit_price - entry_price) * shares - fee
    pnl_pct = ((exit_price - fee / shares) / entry_price - 1) * 100 if entry_price > 0 else 0

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
        "fee": round(fee, 2),
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
