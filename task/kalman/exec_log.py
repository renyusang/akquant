"""
执行日志模块。

记录每笔信号的完整生命周期：信号生成 → 待执行 → 已成交/已取消/未成交+原因。

execution_log.csv 格式:
    signal_date, exec_date, symbol, name, action, target_pct, shares,
    signal_price, exec_price, status, reason
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
EXEC_LOG = os.path.join(TASK_DIR, "execution_log.csv")

COLUMNS = [
    "signal_date", "exec_date", "symbol", "name", "action",
    "target_pct", "shares", "signal_reason",
    "exec_price", "status", "reason",
]


def _load() -> pd.DataFrame:
    if not os.path.exists(EXEC_LOG):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(EXEC_LOG, dtype={"symbol": str, "signal_date": str, "exec_date": str})
        df["symbol"] = df["symbol"].str.zfill(6)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame) -> None:
    # 去重：同 symbol+date+action+status 只保留最后一条
    df = df.drop_duplicates(subset=["signal_date", "symbol", "action", "status"], keep="last")
    df = df.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    df.to_csv(EXEC_LOG, index=False, encoding="utf-8-sig")


def log_pending(
    symbol: str, name: str, action: str, target_pct: float,
    shares: int, signal_reason: str, signal_date: str, reason: str = "",
) -> None:
    """记录待执行订单。同 symbol+date+action 去重（保留最新）。"""
    df = _load()
    sym = str(symbol).zfill(6)
    # 删除同 symbol+date+action 的旧记录
    mask = (df["symbol"] == sym) & (df["signal_date"] == signal_date) & (df["action"] == action)
    df = df[~mask]
    new_row = pd.DataFrame([{
        "signal_date": signal_date, "exec_date": "", "symbol": sym,
        "name": name, "action": action, "target_pct": target_pct,
        "shares": shares, "signal_reason": signal_reason, "exec_price": "",
        "status": "pending", "reason": reason,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save(df)


def log_executed(symbol: str, signal_date: str, exec_price: float, exec_date: str) -> None:
    """标记为已成交。"""
    df = _load()
    sym = str(symbol).zfill(6)
    mask = (df["symbol"] == sym) & (df["signal_date"] == signal_date) & (df["status"] == "pending")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "exec_price"] = exec_price
        df.at[idx, "exec_date"] = exec_date
        df.at[idx, "status"] = "executed"
        _save(df)


def log_failed(symbol: str, signal_date: str, reason: str) -> None:
    """标记为未成交（涨停、资金不足等）。"""
    df = _load()
    sym = str(symbol).zfill(6)
    mask = (df["symbol"] == sym) & (df["signal_date"] == signal_date) & (df["status"] == "pending")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "status"] = "failed"
        df.at[idx, "reason"] = reason
        _save(df)


def log_skipped(
    symbol: str, name: str, signal_date: str, reason: str,
    signal_reason: str = "",
) -> None:
    """记录被跳过的信号。signal_reason = 原始买入信号的触发原因。"""
    df = _load()
    sym = str(symbol).zfill(6)
    new_row = pd.DataFrame([{
        "signal_date": signal_date, "exec_date": "", "symbol": sym,
        "name": name, "action": "buy", "target_pct": 0, "shares": 0,
        "signal_reason": signal_reason, "exec_price": "", "status": "skipped", "reason": reason,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save(df)


def get_pending_count() -> int:
    df = _load()
    return int((df["status"] == "pending").sum())


def get_summary() -> str:
    df = _load()
    if df.empty:
        return "无执行记录"
    executed = (df["status"] == "executed").sum()
    failed = (df["status"] == "failed").sum()
    pending = (df["status"] == "pending").sum()
    skipped = (df["status"] == "skipped").sum()
    return f"已成交 {executed} | 未成交 {failed} | 待执行 {pending} | 已跳过 {skipped}"
