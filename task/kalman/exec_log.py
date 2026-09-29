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

# get_summary 中"未成交"的统计窗口(天): 历史失败行不应长期计为当前未成交
FAILED_WINDOW_DAYS = 30

# 需要按文本(而非被推断的 float64)处理的列
_TEXT_COLS = ("exec_date", "name", "signal_reason", "exec_price", "status", "reason")


def _coerce_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    """把文本列统一为 object dtype。

    修复(2026-09-29): 整列为空时 read_csv 推断成 float64, 之后
    `df.at[idx, "reason"] = "无开盘价数据"` 触发 pandas FutureWarning
    (未来版本直接报错) —— 见 log_failed 路径。
    """
    for col in _TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def _load() -> pd.DataFrame:
    if not os.path.exists(EXEC_LOG):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(EXEC_LOG, dtype={"symbol": str, "signal_date": str, "exec_date": str})
        df["symbol"] = df["symbol"].str.zfill(6)
        return _coerce_text_cols(df)
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
    # 空表直接用新行(2026-09-29): 空 DataFrame 的列全是 NA, concat 会触发
    # pandas 弃用告警(all-NA 列的 dtype 推断将变)
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    _save(df)


def log_executed(symbol: str, signal_date: str, exec_price: float, exec_date: str) -> None:
    """标记为已成交。支持覆盖 pending/failed 状态（此前可能被误标）。

    修复(2026-08-06): 无匹配 pending/failed 记录时(如日志文件被回滚丢失),
    从该标的最近记录复制信息追加 executed——避免成交不入日志/交易明细
    (长电科技 8-6 案例: 订单手动补回但日志缺失导致交易明细无记录)。
    """
    df = _load()
    sym = str(symbol).zfill(6)
    mask = (df["symbol"] == sym) & (df["signal_date"] == signal_date) & (df["status"].isin(["pending", "failed"]))
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "exec_price"] = exec_price
        df.at[idx, "exec_date"] = exec_date
        df.at[idx, "status"] = "executed"
    else:
        ref = df[df["symbol"] == sym]
        if len(ref):
            r = ref.iloc[-1]
            new_row = pd.DataFrame([{
                "signal_date": signal_date, "exec_date": exec_date,
                "symbol": sym, "name": r.get("name", sym),
                "action": r.get("action", "buy"),
                "target_pct": r.get("target_pct", 0.0),
                "shares": r.get("shares", 0),
                "signal_reason": r.get("signal_reason", ""),
                "exec_price": exec_price, "status": "executed",
                "reason": r.get("reason", ""),
            }])
            df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
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
    signal_reason: str = "", action: str = "buy",
) -> None:
    """记录被跳过的信号。signal_reason = 原始信号的触发原因。"""
    df = _load()
    sym = str(symbol).zfill(6)
    new_row = pd.DataFrame([{
        "signal_date": signal_date, "exec_date": "", "symbol": sym,
        "name": name, "action": action, "target_pct": 0, "shares": 0,
        "signal_reason": signal_reason, "exec_price": "", "status": "skipped", "reason": reason,
    }])
    # 空表直接用新行(2026-09-29): 空 DataFrame 的列全是 NA, concat 会触发
    # pandas 弃用告警(all-NA 列的 dtype 推断将变)
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    _save(df)


def cleanup_stale_pending() -> int:
    """清理执行日志中的陈旧 pending 行, 返回清理条数。

    修复(2026-08-28): execution_log 非权威, 实际待执行以 pending_orders.json
    为准。历史遗留 pending 行(已成交但状态未更新/测试污染/拦截误标)虚高
    "待执行" 计数——持仓总览曾显示 27 待执行而实际队列为空。

    规则: pending 行仅当 (symbol, signal_date, action) 与当前待执行订单
    一致时保留, 其余删除(含空日期测试污染行)。
    """
    from orders import load_pending
    df = _load()
    stale_mask = df["status"] == "pending"
    if df.empty or not stale_mask.any():
        return 0
    cur_keys = {
        (str(o["symbol"]).zfill(6), str(o.get("signal_date", "")), o.get("action"))
        for o in load_pending()
    }
    drop = df[stale_mask & ~df.apply(
        lambda r: (str(r["symbol"]).zfill(6), str(r["signal_date"]), r["action"])
        in cur_keys, axis=1)]
    n = int(len(drop))
    if n:
        _save(df.drop(drop.index))
    return n


def get_pending_count() -> int:
    df = _load()
    return int((df["status"] == "pending").sum())


def get_summary(window_days: int = FAILED_WINDOW_DAYS) -> str:
    """执行日志摘要。

    修复(2026-09-29): `未成交` 只统计**近 window_days 天**的 failed 行 ——
    历史失败行(如 2026-07-27 长江电力)会让摘要长期显示"未成交 1", 失真;
    `待执行` 由 `cleanup_stale_pending()` 保证与 pending_orders.json 一致。
    """
    df = _load()
    if df.empty:
        return "无执行记录"
    executed = (df["status"] == "executed").sum()
    pending = (df["status"] == "pending").sum()
    skipped = (df["status"] == "skipped").sum()
    failed_mask = df["status"] == "failed"
    recent_failed = 0
    if failed_mask.any():
        d = pd.to_datetime(df.loc[failed_mask, "signal_date"], errors="coerce")
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
        recent_failed = int((d >= cutoff).sum())
    return (f"已成交 {executed} | 未成交(近{window_days}天) {recent_failed} | "
            f"待执行 {pending} | 已跳过 {skipped}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="清理 execution_log 陈旧 pending 行(以 pending_orders.json 为准)")
    parser.add_argument("--task-dir", default=TASK_DIR,
                        help="状态目录(默认 kalman; unlimited 传 unlimited 目录)")
    args = parser.parse_args()
    if os.path.abspath(args.task_dir) != os.path.abspath(TASK_DIR):
        EXEC_LOG = os.path.join(args.task_dir, "execution_log.csv")
        import orders
        orders.PENDING_FILE = os.path.join(args.task_dir, "pending_orders.json")
    n = cleanup_stale_pending()
    print(f"清理陈旧 pending {n} 条, 剩余待执行 {get_pending_count()} 条")
