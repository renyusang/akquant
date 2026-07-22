#!/usr/bin/env python
"""
每日批量信号扫描器。

读取 stocks.yaml 配置，对监控列表中的全部股票执行信号评估，
输出买卖建议并记录到 signals.csv。

用法:
    python daily_signal.py                 # 扫描全部股票
    python daily_signal.py --quiet         # 仅输出 CSV，不打印详情
    python daily_signal.py --no-save       # 不保存到 signals.csv

输出格式:
    🟢 买入  002594 比亚迪    ¥115.50  仓位 95%  理由: 速度反转+趋势上涨
    🔴 卖出  002460 赣锋锂业  ¥67.80   仓位  0%  理由: 止损(-5.2%)
    ⚪ 持有  600519 贵州茅台  ¥1480.00 仓位 95%  理由: 上涨趋势持仓中
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kalman_filter import KalmanFilter2D
from exec_log import log_executed, log_failed, log_pending, log_skipped
from orders import add_pending_order, execute_pending_orders
from portfolio import add_position, get_position, has_position, load_positions, record_trade, remove_position
from state_check import print_check_result, save_snapshot

# ---- 常量 ----
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(TASK_DIR, "stocks.yaml")
SIGNALS_CSV = os.path.join(TASK_DIR, "signals.csv")
CACHE_DIR = os.path.join(TASK_DIR, ".cache")
WARMUP_BARS = 40

# signals.csv 列名
CSV_COLUMNS = [
    "date", "symbol", "name",
    "close", "kalman_price", "kalman_velocity",
    "ma20", "ma20_rising", "trend",
    "signal", "target_pct", "reason",
]


# =============================================================================
# 配置加载
# =============================================================================
def load_config(path: str = CONFIG_FILE) -> Dict[str, Any]:
    """加载 stocks.yaml 配置文件。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}\n"
                                f"请参考 stocks.yaml 创建配置文件。")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config or "watchlist" not in config:
        raise ValueError("配置文件格式错误: 缺少 'watchlist' 字段")
    return config


def get_stock_params(
    stock: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """合并全局默认参数和个股覆盖参数。"""
    params = dict(defaults)
    for key in defaults:
        if key in stock:
            params[key] = stock[key]
    return params


# =============================================================================
# 数据获取（带缓存）
# =============================================================================
def ensure_cache_dir() -> None:
    """确保缓存目录存在。"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def download_with_cache(symbol: str, data_years: int = 2, asset_type: str = "stock") -> pd.DataFrame:
    """下载日线数据（股票或 ETF），使用本地缓存避免重复请求。"""
    from data_utils import download_data, preprocess_data

    ensure_cache_dir()
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")

    today = datetime.now()
    start_date = (today - timedelta(days=int(data_years * 365))).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    # 检查缓存是否有效（今天的数据已有）
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if len(cached) > 0:
            last_date = pd.to_datetime(cached["date"].max())
            if last_date.date() >= today.date():
                return cached

    # 下载新数据
    try:
        df = download_data(symbol, asset_type=asset_type, start_date=start_date, end_date=end_date, adjust="qfq")
        df = preprocess_data(df)
        if len(df) > 0:
            df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        # 下载失败时尝试使用缓存
        if os.path.exists(cache_path):
            print(f"  [警告] {symbol} 下载失败({e})，使用缓存数据")
            return pd.read_parquet(cache_path)
        raise


# =============================================================================
# 当前持仓推断
# =============================================================================
# =============================================================================
# 信号计算（纯函数，不依赖 akquant 回测引擎）
# =============================================================================
class SignalEvaluator:
    """单股信号评估器。

    直接使用 kalman_filter.py 中的 KalmanFilter2D，
    不依赖 akquant 回测引擎。适合每日批量运行。
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.kf = KalmanFilter2D(
            Q_price=float(params["kalman_q_price"]),
            Q_vel=float(params["kalman_q_vel"]),
            R=float(params["kalman_r"]),
        )
        self._prev_velocity: float = 0.0
        self._trend_state: str = "up"
        self._trend_counter: int = 0
        self._bars_processed: int = 0

    def process_history(self, df: pd.DataFrame) -> None:
        """用历史数据预热卡尔曼滤波器和趋势状态。"""
        if df.empty:
            return
        for i in range(len(df)):
            close = float(df["close"].iloc[i])
            self.kf.update(close)
            self._bars_processed += 1

            # 预热趋势
            if len(df) >= 21 and i >= 20:
                ma20_cur = float(df["close"].iloc[i - 19:i + 1].mean())
                ma20_prev = float(df["close"].iloc[i - 20:i].mean())
                self._update_trend_state(close, ma20_cur, ma20_prev)

            if i >= 1:
                self._prev_velocity = self.kf.get_velocity()

    def evaluate(self, bar: Dict[str, float]) -> Dict[str, Any]:
        """评估当前 bar 的信号。

        bar 需包含: close, ma20_cur, ma20_prev
        返回信号字典。
        """
        close = float(bar["close"])
        ma20_cur = float(bar["ma20_cur"])
        ma20_prev = float(bar["ma20_prev"])

        # 更新卡尔曼
        filtered, velocity = self.kf.update(close)
        prev_velocity = self._prev_velocity

        # 更新趋势
        self._update_trend_state(close, ma20_cur, ma20_prev)

        # 判断信号
        signal, target_pct, reason = self._evaluate(
            close, filtered, velocity, prev_velocity,
        )

        self._prev_velocity = velocity
        self._bars_processed += 1

        return {
            "close": close,
            "kalman_price": filtered,
            "kalman_velocity": velocity,
            "ma20": ma20_cur,
            "ma20_rising": ma20_cur > ma20_prev,
            "trend": self._trend_state,
            "signal": signal,
            "target_pct": target_pct,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    def _update_trend_state(
        self, close: float, ma20_cur: float, ma20_prev: float
    ) -> None:
        """更新趋势状态。"""
        if not self.params.get("trend_filter_enabled", False):
            self._trend_state = "up"
            return

        is_bearish = close < ma20_cur
        ma20_rising = ma20_cur > ma20_prev
        confirm = int(self.params.get("trend_confirm_bars", 1))

        if is_bearish:
            self._trend_counter += 1
            if self._trend_counter >= confirm and self._trend_state == "up":
                self._trend_state = "down"
        elif ma20_rising:
            self._trend_counter = 0
            if self._trend_state == "down":
                self._trend_state = "up"
        else:
            self._trend_counter = max(self._trend_counter, confirm)

    # ------------------------------------------------------------------
    def _evaluate(
        self,
        close: float,
        filtered: float,
        velocity: float,
        prev_velocity: float,
    ) -> Tuple[str, float, str]:
        """评估买卖信号。

        返回 (signal, target_pct, reason)
            signal: "buy" | "sell" | "hold"
        """
        params = self.params
        has_position = self._has_position

        # 趋势决定仓位
        if params.get("trend_filter_enabled", False):
            if self._trend_state == "down":
                target_pct = float(params.get("trend_bear_pct", 0.0))
            else:
                target_pct = 0.95
        else:
            target_pct = 0.95

        # 卖出逻辑（有持仓时检查）
        if has_position:
            # 止损
            stop_loss = float(params.get("stop_loss_pct", 0.05))
            if self._entry_price > 0:
                pnl = close / self._entry_price - 1.0
                if pnl < -stop_loss:
                    return "sell", 0.0, f"止损({pnl * 100:.1f}%)"

            # 价格回归
            exit_threshold = float(params.get("exit_threshold", 0.005))
            if close < filtered * (1.0 - exit_threshold):
                return "sell", 0.0, f"价格回归(偏离{(close / filtered - 1) * 100:.1f}%)"

            # 速度反转
            if params.get("use_velocity_signal", True):
                if velocity < 0.0 and prev_velocity >= 0.0:
                    return "sell", 0.0, "速度反转(转空)"

            # 持有
            trend_info = f"趋势={self._trend_state}"
            return "hold", target_pct, f"持仓中 | {trend_info}"

        # 买入逻辑（空仓时检查）
        else:
            if params.get("trend_filter_enabled") and self._trend_state == "down":
                # 下跌趋势中需更强信号
                entry_threshold = float(params.get("downtrend_entry", 0.03))
                extra = f"下跌趋势(+{entry_threshold * 100:.0f}%阈值)"
            else:
                entry_threshold = float(params.get("entry_threshold", 0.02))
                extra = ""

            reasons = []

            # 价格突破
            if params.get("use_price_signal", True):
                if close > filtered * (1.0 + entry_threshold):
                    reasons.append(
                        f"价格突破(偏离{(close / filtered - 1) * 100:.1f}%)"
                    )

            # 速度反转
            if params.get("use_velocity_signal", True):
                if velocity > 0.0 and prev_velocity <= 0.0:
                    reasons.append("速度反转(转多)")

            if reasons:
                reason_str = " + ".join(reasons)
                if extra:
                    reason_str += f" | {extra}"
                return "buy", target_pct, reason_str

            # 无信号
            trend_info = f"趋势={self._trend_state}"
            if extra:
                trend_info += f" | {extra}"
            return "hold", 0.0, f"等待信号 | {trend_info}"

    # 持仓状态（由外部设置）
    _has_position: bool = False
    _entry_price: float = 0.0

    def set_position(self, has_position: bool, entry_price: float = 0.0) -> None:
        """设置当前持仓状态。"""
        self._has_position = has_position
        self._entry_price = entry_price


# =============================================================================
# 主流程
# =============================================================================
def evaluate_stock(
    stock: Dict[str, Any],
    defaults: Dict[str, Any],
    data_years: int,
) -> Dict[str, Any]:
    """对单只股票执行信号评估。"""
    symbol = stock["symbol"]
    name = stock.get("name", symbol)
    params = get_stock_params(stock, defaults)

    # 1. 获取数据
    asset_type = stock.get("type", "stock")
    df = download_with_cache(symbol, data_years, asset_type=asset_type)
    if df.empty:
        return {"symbol": symbol, "name": name, "error": "无数据"}

    # 2. 获取持仓（从 positions.json）
    pos_info = get_position(symbol)
    has_pos = pos_info is not None
    shares = pos_info["shares"] if pos_info else 0
    avg_price = pos_info["avg_cost"] if pos_info else 0.0

    # 3. 预热
    evaluator = SignalEvaluator(params)
    evaluator.set_position(has_pos, avg_price)

    if len(df) > WARMUP_BARS:
        evaluator.process_history(df.iloc[:-1])  # 除最后一天外的全部历史

    # 4. 评估最后一天
    last = df.iloc[-1]
    close = float(last["close"])
    # 计算 MA20
    ma20_cur = float(df["close"].iloc[-20:].mean())
    ma20_prev = float(df["close"].iloc[-21:-1].mean()) if len(df) >= 21 else ma20_cur

    result = evaluator.evaluate({
        "close": close,
        "ma20_cur": ma20_cur,
        "ma20_prev": ma20_prev,
    })

    result["symbol"] = symbol
    result["name"] = name
    result["date"] = str(last.get("date", ""))[:10]
    result["shares"] = shares
    result["kalman_price"] = evaluator.kf.get_filtered_price()
    result["error"] = None

    # 4. 过滤：已有待执行订单 / 超仓 / 资金不足 → 不生成买入信号
    entry_date = str(last.get("date", ""))[:10]
    if result["signal"] in ("buy", "sell"):
        from orders import load_pending as _load_pending
        existing = _load_pending()
        sym = str(symbol).zfill(6)
        if any(o["symbol"] == sym for o in existing):
            result["signal"] = "hold"
            result["reason"] = "已有待执行订单，等待成交"
            if result["signal"] == "sell":
                return result  # 不重复生成卖出信号
        else:
            max_pos = int(params.get("max_positions", 999))
            current_count = len(load_positions()) + len(existing)
            if max_pos > 0 and current_count >= max_pos:
                buy_reason = result.get("reason", "")  # 保存原始买入原因
                result["signal"] = "hold"
                result["target_pct"] = 0.0
                reason = f"已达最大持仓数({max_pos})"
                result["reason"] = reason
                log_skipped(symbol, name, entry_date, reason, buy_reason)
                result["_skipped"] = True

    # 5. 持久化：记录买卖操作
    if result["signal"] == "buy" and result["target_pct"] > 0:
        cash = float(stock.get("cash", params.get("initial_cash", 100000)))
        max_pct = float(params.get("single_position_pct", 0.95))
        capped_pct = min(result["target_pct"], max_pct)
        buy_qty = int(cash * capped_pct / result["close"] / 100) * 100
        if buy_qty > 0:
            used = sum(p["shares"] * p["avg_cost"] for p in load_positions().values())
            remaining_cash = cash - used
            if buy_qty * result["close"] > remaining_cash * 1.05:
                reason = f"资金不足(需¥{buy_qty * result['close']:,.0f}>可用¥{remaining_cash:,.0f})"
                log_skipped(symbol, name, entry_date, reason, result.get("reason", ""))
            else:
                add_pending_order(symbol, name, "buy", buy_qty, result["close"], entry_date, capped_pct)
                log_pending(symbol, name, "buy", capped_pct, buy_qty, result["reason"], entry_date)
                result["shares"] = buy_qty
        else:
            buy_reason = result.get("reason", "")
            log_skipped(symbol, name, entry_date, f"资金不足(单只上限¥{cash*max_pct:,.0f}, 股价¥{result['close']:.2f})", buy_reason)
    elif result["signal"] == "sell":
        # 卖出 → 待执行订单（次日开盘价成交）
        if has_pos and pos_info:
            sell_shares = pos_info["shares"]
            add_pending_order(symbol, name, "sell", sell_shares, result["close"], entry_date, 0.0)
            from exec_log import log_pending as _lp
            _lp(symbol, name, "sell", 0, sell_shares, result["reason"], entry_date, result["reason"])
            result["shares"] = sell_shares

    return result


def format_signal(result: Dict[str, Any]) -> str:
    """格式化信号为单行输出。"""
    symbol = result["symbol"]
    name = result.get("name", symbol)
    signal = result.get("signal", "?")
    close = result.get("close", 0)
    target_pct = result.get("target_pct", 0)
    reason = result.get("reason", "")
    trend = result.get("trend", "?")

    if signal == "buy":
        icon = "🟢"
    elif signal == "sell":
        icon = "🔴"
    else:
        icon = "⚪"

    return (
        f"  {icon} {signal:4s}  {symbol} {name:<6s}  "
        f"¥{close:>8.2f}  仓位{target_pct * 100:>3.0f}%  "
        f"趋势={trend:<4s}  {reason}"
    )


def append_signals_csv(results: List[Dict[str, Any]]) -> None:
    """将本次信号追加到 signals.csv，保持日期降序。"""
    new_rows = []
    for r in results:
        new_rows.append({
            "date": r.get("date", ""),
            "symbol": str(r["symbol"]).zfill(6),
            "name": r.get("name", ""),
            "close": r.get("close", 0),
            "kalman_price": r.get("kalman_price", 0),
            "kalman_velocity": r.get("kalman_velocity", 0),
            "ma20": r.get("ma20", 0),
            "ma20_rising": r.get("ma20_rising", False),
            "trend": r.get("trend", ""),
            "signal": r.get("signal", ""),
            "target_pct": r.get("target_pct", 0),
            "reason": r.get("reason", ""),
        })

    if os.path.exists(SIGNALS_CSV):
        old = pd.read_csv(SIGNALS_CSV)
        old["symbol"] = old["symbol"].astype(str).str.zfill(6)
        all_rows = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        all_rows = pd.DataFrame(new_rows)

    all_rows = all_rows.drop_duplicates(subset=["date", "symbol"], keep="last")
    all_rows = all_rows.sort_values(["date", "symbol"], ascending=[False, True])
    all_rows.to_csv(SIGNALS_CSV, index=False, encoding="utf-8")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """打印汇总报告。"""
    buys = [r for r in results if r.get("signal") == "buy"]
    sells = [r for r in results if r.get("signal") == "sell"]
    holds = [r for r in results if r.get("signal") == "hold"]
    errors = [r for r in results if r.get("error")]

    print(f"\n{'=' * 60}")
    print(f"  汇总: {len(buys)} 买入 | {len(sells)} 卖出 | "
          f"{len(holds)} 持有 | {len(errors)} 异常")
    print(f"{'=' * 60}")

    if buys:
        print("\n  [买入信号]")
        for r in buys:
            print(format_signal(r))
    if sells:
        print("\n  [卖出信号]")
        for r in sells:
            print(format_signal(r))
    if errors:
        print("\n  [异常]")
        for r in errors:
            print(f"    {r['symbol']} {r.get('name', '')}: {r['error']}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="每日批量信号扫描器"
    )
    parser.add_argument(
        "--config",
        default=CONFIG_FILE,
        help=f"配置文件路径（默认: stocks.yaml）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，仅输出 CSV",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存到 signals.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    watchlist = config.get("watchlist", [])
    data_years = int(config.get("data_years", 2))

    if not watchlist:
        print("错误: watchlist 为空")
        sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - __import__('datetime').timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n{'=' * 60}")
    print(f"  每日信号扫描  {today}")
    print(f"  监控股票: {len(watchlist)} 只")
    print(f"{'=' * 60}")

    # ---- 0. 数据新鲜度检查 ----
    data_fresh = _check_data_freshness(watchlist, yesterday)

    # ---- 1. 执行待处理订单 ----
    executed_today: List[Dict[str, Any]] = []
    if data_fresh:
        executed_today = _execute_today_pending(defaults, today)
    else:
        print(f"  ⚠️ 数据未更新（最新日期 < {yesterday}），跳过待执行订单")
        from orders import load_pending
        pending_count = len(load_pending())
        if pending_count > 0:
            print(f"  ⏳ {pending_count} 笔待执行订单等待数据更新后成交")

    positions_before = len(load_positions())
    results = []
    skipped_buys: List[Dict[str, Any]] = []  # 被跳过的买入信号（等仓位空出）
    for i, stock in enumerate(watchlist, 1):
        symbol = stock["symbol"]
        name = stock.get("name", symbol)
        if not args.quiet:
            print(f"\n[{i}/{len(watchlist)}] {symbol} {name} ...")

        try:
            result = evaluate_stock(stock, defaults, data_years)
            results.append(result)
            if result.get("_skipped"):
                skipped_buys.append(result)
            if not args.quiet:
                print(format_signal(result))
        except Exception as e:
            results.append({
                "symbol": symbol, "name": name,
                "error": str(e), "signal": "error",
            })
            if not args.quiet:
                print(f"  ❌ 错误: {e}")

    # 数据延迟时追加警告标记
    if not data_fresh and not args.quiet:
        print(f"\n  ⚠️ 警告: 数据未更新到 {yesterday}，信号基于旧数据，仅供参考")

    if not args.quiet:
        print_summary(results)

    if not args.no_save:
        valid = [r for r in results if not r.get("error")]
        if valid:
            append_signals_csv(valid)
            print(f"\n信号已保存: {SIGNALS_CSV} ({len(valid)} 条)")

    # 持仓状态表 + 一致性检查 + 快照
    # 显示被跳过的买入信号
    if skipped_buys and not args.quiet:
        print(f"\n  ⏸️ 被跳过 ({len(skipped_buys)} 笔，等仓位空出):")
        skipped_buys.sort(key=lambda x: abs(float(x.get("kalman_price", 0)) - float(x.get("close", 0))) / float(x.get("close", 1)), reverse=True)
        for s in skipped_buys[:5]:
            deviation = (float(s.get("close", 0)) / float(s.get("kalman_price", 1)) - 1) * 100
            print(f"     {s['symbol']} {s.get('name',''):<6s} ¥{s.get('close',0):>8.2f}  "
                  f"偏离{deviation:+.1f}%  趋势={s.get('trend','?')}")

    # 仓位空出时自动买入最强信号
    positions_now = len(load_positions())
    freed = positions_before - positions_now
    if freed > 0 and skipped_buys:
        remaining_slots = int(defaults.get("max_positions", 5)) - positions_now
        if remaining_slots > 0:
            skipped_buys.sort(
                key=lambda x: abs(float(x.get("close", 0)) / float(x.get("kalman_price", 1)) - 1),
                reverse=True,
            )
            filled = 0
            for s in skipped_buys:
                if filled >= remaining_slots:
                    break
                sym = str(s["symbol"]).zfill(6)
                # 检查涨停
                cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")
                blocked = False
                if os.path.exists(cache_path):
                    try:
                        cached = pd.read_parquet(cache_path)
                        if len(cached) >= 1:
                            last_close = float(cached["close"].iloc[-1])
                            limit_up = last_close * 1.20 if sym.startswith("688") or sym.startswith("300") or sym.startswith("301") else last_close * 1.10
                            if s["close"] >= limit_up * 0.999:
                                print(f"  ⚠️ {sym} {s['name']} 涨停(收盘¥{s['close']:.2f}≥涨停¥{limit_up:.2f})，跳过")
                                blocked = True
                    except Exception:
                        pass
                if blocked:
                    continue
                cash = float(defaults.get("initial_cash", 100000))
                max_pct = float(defaults.get("single_position_pct", 0.95))
                capped_pct = min(s.get("target_pct", 0.95), max_pct)
                buy_qty = int(cash * capped_pct / s["close"] / 100) * 100
                if buy_qty > 0:
                    add_pending_order(s["symbol"], s["name"], "buy", buy_qty, s["close"], s["date"], capped_pct)
                    log_pending(s["symbol"], s["name"], "buy", capped_pct, buy_qty, s.get("reason", ""), s["date"])
                    print(f"  🟢 自动补仓 {s['symbol']} {s['name']}: {buy_qty}股 "
                          f"(仓位{positions_before}→{positions_now}, 空出{freed}个, "
                          f"候选排名 #{filled+1})")
                    filled += 1

    # 持仓状态表
    if not args.quiet:
        print_position_summary()
        print_check_result()
    save_snapshot()


def _check_data_freshness(watchlist: list, yesterday: str) -> bool:
    """检查缓存数据是否覆盖到昨天（至少）。

    采样前 3 只股票检查缓存最新日期。任意一只 >= yesterday 即认为数据新鲜。
    """
    from data_utils import download_stock_data
    sample = watchlist[:3]
    for stock in sample:
        sym = str(stock["symbol"]).zfill(6)
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                last_date = str(pd.to_datetime(cached["date"].max()).date())
                if last_date >= yesterday:
                    return True
            except Exception:
                pass
    return False


def _execute_today_pending(defaults: Dict[str, Any], today: str = "") -> List[Dict[str, Any]]:
    """执行昨日的待处理订单（以今日开盘价成交）。

    返回已成交订单列表。
    """
    if not today:
        today = datetime.now().strftime("%Y-%m-%d")
    executed = []

    # 收集今日开盘价和昨日收盘价
    today_open = {}
    prev_close = {}
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".parquet"):
            try:
                df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
                if len(df) >= 2:
                    sym = fname.replace(".parquet", "")
                    today_open[sym] = float(df["open"].iloc[-1])   # 今日开盘
                    prev_close[sym] = float(df["close"].iloc[-2])  # 昨日收盘
            except Exception:
                pass

    pending = execute_pending_orders(today_open, prev_close, today_str=today)
    for order in pending:
        sym = order["symbol"]
        name = order.get("name", sym)
        exec_price = order["exec_price"]
        action = order.get("action", "buy")
        if action == "sell":
            # 执行卖出：移除持仓，记录交易
            removed = remove_position(sym)
            if removed:
                record_trade(
                    symbol=sym, name=name,
                    shares=order["shares"],
                    entry_price=removed.get("avg_cost", 0),
                    exit_price=exec_price,
                    entry_date=removed.get("first_buy_date", order["signal_date"]),
                    exit_date=today,
                    reason=order.get("signal_reason", ""),
                )
            log_executed(sym, order["signal_date"], exec_price, today)
            print(f"  ✅ 卖出 {sym} {name}: {order['shares']}股 @ ¥{exec_price:.2f} "
                  f"(信号日 {order['signal_date']})")
        else:
            add_position(sym, name, order["shares"], exec_price, order["signal_date"])
            log_executed(sym, order["signal_date"], exec_price, today)
            print(f"  ✅ 买入 {sym} {name}: {order['shares']}股 @ ¥{exec_price:.2f} "
                  f"(信号日 {order['signal_date']} 信号价 ¥{order['signal_price']:.2f})")
        executed.append(order)

    from orders import load_pending
    remaining = load_pending()
    for order in remaining:
        log_failed(order["symbol"], order["signal_date"], order.get("reason", "等待数据"))
    if remaining:
        print(f"  ⚠️ 未成交 {len(remaining)} 笔:")
        for order in remaining:
            print(f"     {order['symbol']} {order.get('name','')}: {order.get('reason','')}")

    return executed


def print_position_summary() -> None:
    """打印当前持仓总览和已实现盈亏，标记超标仓位。"""
    positions = load_positions()
    from portfolio import get_trade_summary

    if not positions:
        print("\n  (当前无持仓)")
    else:
        # 读取配置获取仓位上限
        import yaml
        max_pct = 0.20
        cash = 100000
        try:
            with open(os.path.join(TASK_DIR, "stocks.yaml")) as f:
                cfg = yaml.safe_load(f)
            defaults = cfg.get("defaults", {})
            cash = float(defaults.get("initial_cash", 100000))
            max_pct = float(defaults.get("single_position_pct", 0.20))
        except Exception:
            pass
        max_value = cash * max_pct

        print(f"\n{'=' * 80}")
        print(f"  {'持仓总览 (上限 {max_pct*100:.0f}% = ¥{max_value:,.0f})':^72s}")
        print(f"{'=' * 80}")
        print(f"  {'代码':<8s} {'名称':<8s} {'股数':>6s} {'成本':>8s} {'市值':>10s} {'占比':>6s} {'浮动盈亏':>10s} {'收益率':>8s}")
        print(f"  {'-' * 72}")

        total_value = 0.0
        total_pnl = 0.0
        try:
            if os.path.exists(SIGNALS_CSV):
                latest = pd.read_csv(SIGNALS_CSV)
                latest["symbol"] = latest["symbol"].astype(str).str.zfill(6)
                latest = latest.sort_values("date").groupby("symbol").last()
                price_map = latest["close"].to_dict()
            else:
                price_map = {}
        except Exception:
            price_map = {}

        for sym, pos in sorted(positions.items()):
            price = price_map.get(sym, pos["avg_cost"])
            value = pos["shares"] * price
            pnl = (price - pos["avg_cost"]) * pos["shares"]
            pnl_pct = (price / pos["avg_cost"] - 1) * 100
            pct = value / cash * 100 if cash > 0 else 0
            total_value += value
            total_pnl += pnl
            flag = " ⚠️超标" if value > max_value * 1.01 else ""
            print(f"  {sym:<8s} {pos['name']:<8s} {pos['shares']:>6d}  "
                  f"{pos['avg_cost']:>8.2f} {value:>10.0f} {pct:>5.1f}% {pnl:>+10.0f} {pnl_pct:>+7.1f}%{flag}")

        print(f"  {'-' * 72}")
        total_pct = total_value / cash * 100 if cash > 0 else 0
        print(f"  持仓市值: ¥{total_value:,.0f} / ¥{cash:,.0f} = {total_pct:.1f}%    浮动盈亏: ¥{total_pnl:+,.0f}")

    # 执行日志摘要
    from exec_log import get_summary as _exec_summary, get_pending_count
    print(f"\n  执行日志: {_exec_summary()}")

    # 已完成交易汇总
    trade_summary = get_trade_summary()
    if trade_summary["count"] > 0:
        print(f"  已完成交易: {trade_summary['count']} 笔 | "
              f"盈利 {trade_summary['wins']} | 亏损 {trade_summary['losses']} | "
              f"累计盈亏: ¥{trade_summary['total_pnl']:+,.0f}")

    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    main()
