"""
数据异常校验模块。

在 daily_signal.py 执行前对每只股票的数据进行检查，
发现异常时记录到 validation_log.csv 并在终端显示。

检查项:
    1. 单日涨跌幅 > 15%（可能数据错误）
    2. 成交量为 0（可能停牌）
    3. 价格连续 3 天不变（数据未更新）
    4. 待执行订单超过 3 天未成交
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(TASK_DIR, ".cache")
VALIDATION_LOG = os.path.join(TASK_DIR, "validation_log.csv")

COLUMNS = ["date", "symbol", "name", "check", "level", "detail"]


def validate_data(symbol: str, name: str) -> List[Dict[str, Any]]:
    """对单只股票的缓存数据执行异常检查。

    返回异常列表，每项包含 {symbol, name, check, level, detail}。
    level: "error"(严重) / "warn"(警告)
    """
    issues = []
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if not os.path.exists(cache_path):
        return [{"symbol": symbol, "name": name, "check": "数据缺失",
                 "level": "error", "detail": "缓存文件不存在"}]

    df = pd.read_parquet(cache_path)
    if len(df) < 2:
        return issues

    close = df["close"].values
    volume = df["volume"].values
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. 单日涨跌幅异常（根据板块使用不同阈值）
    if len(close) >= 2:
        daily_ret = abs(close[-1] / close[-2] - 1)
        # 科创板/创业板 ±20%，北交所 ±30%，主板 ±10%
        if symbol.startswith("4") or symbol.startswith("8"):
            limit = 0.30  # 北交所
        elif symbol.startswith("688") or symbol.startswith("30"):
            limit = 0.20  # 科创板/创业板
        else:
            limit = 0.10  # 主板
        # 涨停/跌停放行: 涨跌停价 = 前收×(1±limit) 四舍五入到分,
        # 实际涨幅可能因取整略超 limit(如 +10.004% 恰为涨停)。
        # 修复(2026-08-06): 按方向判断——上涨时收盘 ≤ 涨停价(+1分容差)
        # 视为涨停; 下跌时收盘 ≥ 跌停价(-1分容差) 视为跌停。
        # (注意: 不能用 or 同时判断, 否则上涨收盘必 ≥ 跌停价恒放行)
        limit_up = round(close[-2] * (1 + limit), 2)
        limit_down = round(close[-2] * (1 - limit), 2)
        rising = close[-1] >= close[-2]
        at_limit = (
            (rising and close[-1] <= limit_up + 0.01)
            or (not rising and close[-1] >= limit_down - 0.01)
        )
        if daily_ret > limit and not at_limit:
            issues.append({
                "symbol": symbol, "name": name,
                "check": "涨跌幅异常",
                "level": "error",
                "detail": "日涨跌幅 {:.1f}%（{:.2f}→{:.2f}），超过{}%限制".format(
                    daily_ret*100, close[-2], close[-1], int(limit*100)),
            })

    # 2. 成交量异常（为 0 → 可能停牌）
    if volume[-1] == 0:
        issues.append({
            "symbol": symbol, "name": name,
            "check": "成交量为零",
            "level": "warn",
            "detail": "最新日成交量为 0，可能停牌",
        })

    # 3. 价格连续 3 天不变（数据未更新）
    if len(close) >= 3:
        last_3 = close[-3:]
        if last_3[0] == last_3[1] == last_3[2]:
            last_date = str(df["date"].iloc[-1])[:10]
            issues.append({
                "symbol": symbol, "name": name,
                "check": "数据未更新",
                "level": "warn",
                "detail": "价格连续 3 天不变（最新 " + last_date + "）",
            })

    # 4. 数据滞后（最新日期 < 昨天，且不是周末）
    last_date = pd.to_datetime(df["date"].iloc[-1])
    yesterday = datetime.now() - timedelta(days=1)
    if yesterday.weekday() < 5:  # 工作日
        if last_date.date() < yesterday.date():
            issues.append({
                "symbol": symbol, "name": name,
                "check": "数据滞后",
                "level": "warn",
                "detail": "最新数据 " + str(last_date.date()) + " < " + str(yesterday.date()),
            })

    return issues


def validate_pending_orders() -> List[Dict[str, Any]]:
    """检查待执行订单是否超过 3 天未成交。"""
    from orders import load_pending

    issues = []
    pending = load_pending()
    today = datetime.now().date()

    for order in pending:
        # 防御: 缺 signal_date 的脏订单不崩溃(2026-08-05 曾因缺键 KeyError)
        if not order.get("signal_date"):
            issues.append({
                "symbol": order.get("symbol", "?"),
                "name": order.get("name", ""),
                "check": "订单缺信号日期",
                "level": "warn",
                "detail": "pending_orders.json 脏数据: {}".format(
                    {k: order.get(k) for k in ("symbol", "action", "shares")}
                ),
            })
            continue
        signal_date = datetime.strptime(order["signal_date"], "%Y-%m-%d").date()
        days_pending = (today - signal_date).days
        if days_pending >= 3:
            issues.append({
                "symbol": order["symbol"],
                "name": order.get("name", ""),
                "check": "订单停滞",
                "level": "warn",
                "detail": "待执行 {} 天（{} {}股，信号日 {}）".format(days_pending, order['action'], order['shares'], order['signal_date']),
            })

    return issues


def run_all_checks(
    watchlist: list, quiet: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """对所有监控股票执行数据校验 + 订单校验。

    返回 (data_issues, order_issues)。
    """
    all_issues = []
    for stock in watchlist:
        sym = str(stock["symbol"]).zfill(6)
        name = stock.get("name", sym)
        issues = validate_data(sym, name)
        all_issues.extend(issues)
        if issues and not quiet:
            for issue in issues:
                icon = "❌" if issue["level"] == "error" else "⚠️"
                print(f"  {icon} {sym} {name}: [{issue['check']}] {issue['detail']}")

    order_issues = validate_pending_orders()
    if order_issues and not quiet:
        for issue in order_issues:
            print(f"  ⚠️ {issue['symbol']} {issue['name']}: [{issue['check']}] {issue['detail']}")

    return all_issues, order_issues


def save_validation_log(issues: List[Dict[str, Any]]) -> None:
    """将校验异常追加到日志文件。"""
    if not issues:
        return
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_rows = []
    for i in issues:
        new_rows.append({**i, "date": today_str})

    if os.path.exists(VALIDATION_LOG):
        existing = pd.read_csv(VALIDATION_LOG, dtype={"symbol": str})
        all_rows = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        all_rows = all_rows.drop_duplicates(subset=["date", "symbol", "check"], keep="last")
    else:
        all_rows = pd.DataFrame(new_rows)

    all_rows.to_csv(VALIDATION_LOG, index=False, encoding="utf-8-sig")
