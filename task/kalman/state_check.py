"""
状态快照与一致性检查。

每次运行 daily_signal.py 后保存快照，下次运行时对比，
发现矛盾时发出警告。

快照内容:
    - positions: {symbol: {shares, avg_cost}}
    - pending: [{symbol, shares, signal_date}]
    - signals: {symbol: signal}  (最后一条信号)
    - timestamp
"""

import json
import os
from typing import Any, Dict, List

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_FILE = os.path.join(TASK_DIR, "state_snapshot.json")


def save_snapshot() -> None:
    """保存当前状态快照。"""
    from portfolio import load_positions
    from orders import load_pending
    import pandas as pd

    positions = load_positions()
    pending = load_pending()

    signals = {}
    signals_csv = os.path.join(TASK_DIR, "signals.csv")
    if os.path.exists(signals_csv):
        df = pd.read_csv(signals_csv, dtype={"symbol": str})
        df["symbol"] = df["symbol"].str.zfill(6)
        if not df.empty:
            latest = df.sort_values("date").groupby("symbol").last()
            for sym, row in latest.iterrows():
                signals[sym] = row["signal"]

    from datetime import datetime
    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "positions": {s: {"shares": p["shares"], "avg_cost": p["avg_cost"]} for s, p in positions.items()},
        "pending": [{"symbol": o["symbol"], "shares": o["shares"], "signal_date": o["signal_date"]} for o in pending],
        "signals": signals,
    }
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def check_consistency() -> List[str]:
    """对比当前状态与上次快照，返回警告列表。"""
    if not os.path.exists(SNAPSHOT_FILE):
        return []

    with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
        prev = json.load(f)

    from portfolio import load_positions
    from orders import load_pending
    import pandas as pd

    curr_positions = load_positions()
    curr_pending = load_pending()

    warnings: List[str] = []

    # 1. 上次有持仓，这次消失了（且没有卖出信号）
    signals_csv = os.path.join(TASK_DIR, "signals.csv")
    curr_signals: Dict[str, str] = {}
    if os.path.exists(signals_csv):
        df = pd.read_csv(signals_csv, dtype={"symbol": str})
        df["symbol"] = df["symbol"].str.zfill(6)
        if not df.empty:
            latest = df.sort_values("date").groupby("symbol").last()
            for sym, row in latest.iterrows():
                curr_signals[sym] = row["signal"]

    for sym, pos in prev.get("positions", {}).items():
        if sym not in curr_positions:
            if curr_signals.get(sym) != "sell":
                warnings.append(
                    f"⚠️ {sym} 持仓消失但无卖出信号 (之前 {pos['shares']}股 @ ¥{pos['avg_cost']:.2f})"
                )

    # 2. 持仓股数变化（无买卖信号）
    for sym, pos in curr_positions.items():
        if sym in prev.get("positions", {}):
            prev_pos = prev["positions"][sym]
            if pos["shares"] != prev_pos["shares"]:
                warnings.append(
                    f"⚠️ {sym} 股数变化: {prev_pos['shares']}→{pos['shares']} (无对应信号)"
                )

    # 3. 待执行订单未按预期执行
    prev_pending = prev.get("pending", [])
    for po in prev_pending:
        sym = po["symbol"]
        # 检查是否已成交（进入了持仓）或仍然 pending
        in_positions = sym in curr_positions
        in_pending = any(o["symbol"] == sym for o in curr_pending)
        if not in_positions and not in_pending:
            warnings.append(
                f"⚠️ {sym} 待执行订单消失 (信号日 {po['signal_date']}, {po['shares']}股)"
            )

    # 4. 同一天出现矛盾的信号（昨天买入今天卖出）
    for sym, sig in curr_signals.items():
        prev_sig = prev.get("signals", {}).get(sym)
        if prev_sig == "buy" and sig == "sell":
            pass  # 正常：买入后卖出
        elif prev_sig == "sell" and sig == "buy" and sym in curr_positions:
            pass  # 正常：卖出后重新买入
        elif prev_sig == "buy" and sig == "buy":
            prev_pos = prev.get("positions", {}).get(sym)
            if prev_pos and sym in curr_positions:
                pass  # 已在持仓中，重复买入信号被拦截
            elif not prev_pos:
                # 昨天买入但今天还在发买入信号 → 可能没执行
                warnings.append(
                    f"⚠️ {sym} 昨日买入信号未执行，今日再次买入 (检查 execution_log)"
                )

    return warnings


def print_check_result() -> None:
    """打印一致性检查结果。"""
    warnings = check_consistency()
    if warnings:
        print(f"\n⚠️ 状态一致性检查 ({len(warnings)} 项):")
        for w in warnings:
            print(f"  {w}")
    else:
        print("  状态一致性检查: ✓ 无异常")
