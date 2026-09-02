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


def check_ex_rights(positions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """持仓标的除权除息监控(2026-09-01 新增, 应对持仓无自动除权处理)。

    数据源: akshare stock_fhps_detail_em(东方财富分红送配详情)。
    检测: 除权除息日 ∈ [今天-3, 今天+7] 且方案进度=实施分配 → warn:
      - 送转(送股/转增)> 0 → 提示 split 调整(含倍数)
      - 纯现金分红(送转为空/0) → 提示"股数不变无需 split"(分红现金未自动入账,
        修复 2026-09-02: 德福科技 10派1 案例——原提示"执行 split"对纯分红误导)
    送转比例字段"送转股份-送转总比例"= 每 10 股送转股数(如 20 → 倍数 3)。
    网络失败/接口变更静默跳过(尽力而为, 不阻断主流程)。
    """
    issues = []
    try:
        import akshare as ak
    except ImportError:
        return issues
    today = datetime.now().date()
    for sym, pos in (positions or {}).items():
        try:
            df = ak.stock_fhps_detail_em(symbol=sym)
        except Exception:
            continue
        if df is None or df.empty or "除权除息日" not in df.columns:
            continue
        ex = pd.to_datetime(df["除权除息日"], errors="coerce").dropna()
        for d in ex:
            ex_date = d.date()
            delta = (ex_date - today).days
            if -3 <= delta <= 7:
                row = df[pd.to_datetime(df["除权除息日"], errors="coerce") == d]
                prog = str(row["方案进度"].iloc[0]) if len(row) else "?"
                ratio_field = (row["送转股份-送转总比例"].iloc[0]
                               if len(row) and "送转股份-送转总比例" in row.columns
                               else None)
                cash_field = (row["现金分红-现金分红比例"].iloc[0]
                              if len(row) and "现金分红-现金分红比例" in row.columns
                              else None)
                has_split = pd.notna(ratio_field) and float(ratio_field) > 0
                has_cash = pd.notna(cash_field) and float(cash_field) > 0
                if has_split:
                    ratio_hint = (f", 送转比例 每10股送转{float(ratio_field):.0f}股"
                                  f" → split 倍数 {1 + float(ratio_field) / 10:.1f}")
                    if delta < 0:
                        detail = (f"已除权 {abs(delta)} 天(方案:{prog}{ratio_hint}), "
                                  f"持仓 {pos.get('shares', '?')} 股未调整 → 请执行 "
                                  f"python manage.py split {sym} <倍数> 调整持仓")
                    else:
                        detail = (f"{delta} 天后除权(方案:{prog}{ratio_hint}), "
                                  f"实施后请执行 python manage.py split {sym} <倍数> "
                                  f"调整持仓")
                elif has_cash:
                    cash_desc = (str(row["现金分红-现金分红比例描述"].iloc[0])
                                 if len(row) and "现金分红-现金分红比例描述" in row.columns
                                 else "现金分红")
                    if delta < 0:
                        detail = (f"已除息 {abs(delta)} 天(方案:{prog}, {cash_desc})——"
                                  f"纯现金分红股数不变, 无需 split; 分红现金未自动入账"
                                  f"(持仓 {pos.get('shares', '?')} 股, 影响小)")
                    else:
                        detail = (f"{delta} 天后除息(方案:{prog}, {cash_desc})——"
                                  f"纯现金分红股数不变, 无需 split; 分红现金未自动入账"
                                  f"(持仓 {pos.get('shares', '?')} 股, 影响小)")
                else:
                    if delta < 0:
                        detail = (f"已除权 {abs(delta)} 天(方案:{prog}), 请人工确认"
                                  f"送转/分红方案后处理")
                    else:
                        detail = (f"{delta} 天后除权(方案:{prog}), 请人工确认"
                                  f"送转/分红方案后处理")
                issues.append({"symbol": sym, "name": pos.get("name", sym),
                               "check": "除权除息", "level": "warn",
                               "detail": detail})
    return issues


def run_all_checks(
    watchlist: list, quiet: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """对所有监控股票执行数据校验 + 订单校验 + 持仓除权监控。

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

    # 持仓除权监控(2026-09-01): 仅查持仓标的(网络调用, 失败静默)
    try:
        from portfolio import load_positions
        ex_issues = check_ex_rights(load_positions())
        all_issues.extend(ex_issues)
        if ex_issues and not quiet:
            for issue in ex_issues:
                print(f"  ⚠️ {issue['symbol']} {issue['name']}: "
                      f"[{issue['check']}] {issue['detail']}")
    except Exception:
        pass

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
