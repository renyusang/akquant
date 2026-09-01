#!/usr/bin/env python
"""
每日批量信号扫描器。

读取 stocks.yaml 配置，对监控列表中的全部股票执行信号评估，
输出买卖建议并记录到 signals.csv。

用法:
    python daily_signal.py                 # 扫描全部股票
    python daily_signal.py --quiet         # 仅输出 CSV，不打印详情
    python daily_signal.py --no-save       # 不保存到 signals.csv
    python daily_signal.py --deploy        # 扫描后自动部署报告到服务器

输出格式:
    🟢 买入  002594 比亚迪    ¥115.50  仓位 95%  理由: 速度反转+趋势上涨
    🔴 卖出  002460 赣锋锂业  ¥67.80   仓位  0%  理由: 止损(-5.2%)
    ⚪ 持有  600519 贵州茅台  ¥1480.00 仓位 95%  理由: 上涨趋势持仓中
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import yaml

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_engine import SignalEngine
from exec_log import log_executed, log_failed, log_pending, log_skipped
from orders import add_pending_order, execute_pending_orders
from portfolio import (
    add_position,
    calc_fee,
    get_position,
    has_position,
    load_positions,
    record_trade,
    remove_position,
)
from state_check import print_check_result, save_snapshot
from backup import create_backup
from validate import run_all_checks, save_validation_log

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
    # 兼容旧格式（扁平 watchlist）
    if isinstance(config["watchlist"], list):
        config["watchlist"] = {"stocks": config["watchlist"], "etfs": []}
    return config


def _pool_config(config: Dict[str, Any], asset_type: str) -> Dict[str, Any]:
    """获取指定资产类型的资金池配置。"""
    pool = config.get(asset_type, {})
    return {
        "initial_cash": float(pool.get("initial_cash", 200000 if asset_type == "stock" else 100000)),
        "max_positions": int(pool.get("max_positions", 5 if asset_type == "stock" else 3)),
        "single_position_pct": float(pool.get("single_position_pct", 0.20 if asset_type == "stock" else 0.30)),
    }


def get_stock_params(
    stock: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """合并策略参数 + 资金池参数。"""
    strategy = config.get("strategy", {})
    asset_type = stock.get("type", "stock")
    pool = _pool_config(config, asset_type)
    params = dict(strategy)
    params.update(pool)
    for key in strategy:
        if key in stock:
            params[key] = stock[key]
    params["asset_type"] = asset_type
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
# 主流程
# =============================================================================
def prefetch_data(watchlist: list, data_years: int = 2, workers: int = 8,
                  quiet: bool = False) -> int:
    """并行预取缺失/过期行情数据(2026-08-24)。

    主循环 evaluate_stock 前用线程池并行下载 watchlist 中缓存缺失或
    未更新到今日的标的, 评估阶段全部缓存命中。

    设计依据:
    - 下载无副作用(各写各的 .cache 文件) → 可安全并行;
      evaluate_stock 含下单/资金计数副作用 → 评估阶段必须保持串行
    - 实测 8 路并行 8 只 5.7s vs 串行 22.3s(3.9x), 16 路无限流;
      8 路为性价比拐点(16 只 4.6s vs 8 路 5.7s)
    - 失败标的静默跳过: 主循环串行阶段自然重试或回退旧缓存
    - 配合 data_utils socket 30s 超时兜底, 单只挂起不再阻塞全部

    返回实际下载(尝试)数。
    """
    from concurrent.futures import ThreadPoolExecutor

    def _needs(symbol: str) -> bool:
        cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
        if not os.path.exists(cache_path):
            return True
        try:
            cached = pd.read_parquet(cache_path)
            if len(cached) == 0:
                return True
            last_date = pd.to_datetime(cached["date"].max())
            return last_date.date() < datetime.now().date()
        except Exception:
            return True

    pending = [s for s in watchlist if _needs(s["symbol"])]
    if not pending:
        return 0
    if not quiet:
        print(f"  ⚡ 并行预取数据 ({len(pending)} 只, {workers} 路线程)...")

    def _fetch(stock: dict) -> None:
        try:
            download_with_cache(stock["symbol"], data_years,
                                stock.get("type", "stock"))
        except Exception:
            pass  # 失败留给主循环串行阶段重试/回退缓存

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_fetch, pending):
            done += 1
            if not quiet and done % 10 == 0:
                print(f"    预取 {done}/{len(pending)}")
    if not quiet:
        print(f"  ⚡ 预取完成 ({done}/{len(pending)})")
    return len(pending)


def evaluate_stock(
    stock: Dict[str, Any],
    config: Dict[str, Any],
    data_years: int,
    allocated_cash: Dict[str, float] | None = None,
    defer_orders: bool = False,
) -> Dict[str, Any]:
    """对单只股票执行信号评估。

    allocated_cash: 本轮已承诺但未成交的资金 {pool: amount},用于防止同次扫描超额下单。
    defer_orders: 两阶段下单模式(2026-08-12)——只评估不下单,
        买入/补仓候选记录到 result["_deferred"], 由 _place_deferred_orders
        统一按"补仓优先 + 偏离度降序"排序后下单。
        卖出信号不受影响(卖出优先级最高, 评估时立即下单)。
    """
    symbol = stock["symbol"]
    name = stock.get("name", symbol)
    params = get_stock_params(stock, config)

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

    # 3. 创建引擎并预热
    engine = SignalEngine(**params)
    engine.set_position(has_pos, avg_price)

    if len(df) > WARMUP_BARS:
        engine.process_history(df.iloc[:-1])  # 除最后一天外的全部历史

    # 4. 评估最后一天
    last = df.iloc[-1]
    close = float(last["close"])
    # 计算 MA20
    ma20_cur = float(df["close"].iloc[-20:].mean())
    ma20_prev = float(df["close"].iloc[-21:-1].mean()) if len(df) >= 21 else ma20_cur

    result = engine.update(
        close, ma20_cur, ma20_prev, high=float(last["high"]),
        low=float(last["low"]),
        volume=float(last["volume"]) if "volume" in last.index else None,
    )

    result["symbol"] = symbol
    result["name"] = name
    result["date"] = str(last.get("date", ""))[:10]
    result["shares"] = shares
    result["kalman_price"] = engine.filtered_price
    result["error"] = None

    # 5. 过滤：已有待执行订单 / 超仓 / 资金不足 → 不生成买入信号
    entry_date = str(last.get("date", ""))[:10]
    if result["signal"] in ("buy", "sell"):
        from orders import load_pending as _load_pending
        existing = _load_pending()
        sym = str(symbol).zfill(6)
        existing_order = next((o for o in existing if o["symbol"] == sym), None)
        if existing_order:
            if result["signal"] == "sell":
                # 卖出信号优先级最高：取消旧订单，允许卖出
                pass
            else:
                result["signal"] = "hold"
                result["reason"] = "已有待执行订单，等待成交"
        else:
            if result["signal"] == "buy":
                max_pos = int(params.get("max_positions", 999))
                asset_type = params.get("asset_type", "stock")
                all_positions = load_positions()
                pool_positions = sum(
                    1 for s in all_positions.keys()
                    if (s.startswith("5") or s.startswith("1")) == (asset_type == "etf")
                )
                pool_pending = sum(
                    1 for o in existing
                    if (o["symbol"].startswith("5") or o["symbol"].startswith("1")) == (asset_type == "etf")
                    and o.get("action") != "sell"  # 待卖出释放仓位,不占用上限
                    and o["symbol"] not in all_positions  # 补仓单(已持仓)不占新增名额
                )
                current_count = pool_positions + pool_pending
                if max_pos > 0 and current_count >= max_pos and not defer_orders:
                    buy_reason = result.get("reason", "")
                    result["signal"] = "hold"
                    # 保留原始目标仓位供补仓使用(2026-08-05修复:
                    # 原清零导致 _auto_fill_pool 按 0 计算股数, 候选永远无法补入)
                    result["_target_pct"] = result.get("target_pct", 0.95)
                    result["target_pct"] = 0.0
                    reason = f"已达最大持仓数({max_pos})"
                    result["reason"] = reason
                    log_skipped(symbol, name, entry_date, reason, buy_reason)
                    result["_skipped"] = True

    # 6. 持久化：记录买卖操作
    if result["signal"] == "buy" and result["target_pct"] > 0:
        cash = float(stock.get("cash", params.get("initial_cash", 100000)))
        max_pct = float(params.get("single_position_pct", 0.95))
        capped_pct = round(result["target_pct"] * max_pct, 4)
        lot = 200 if str(symbol).startswith("688") else 100
        if params.get("lot_based_position"):
            # 无限仓位模式(2026-08-06): 数量=target_pct映射到[1,3]手, 不受资金限制
            qty_lots = max(1, round(result["target_pct"] * 3))
            buy_qty = qty_lots * lot
        else:
            buy_qty = int(cash * capped_pct / result["close"] / lot) * lot
        if buy_qty > 0:
            if defer_orders:
                # 两阶段模式: 只记录候选, 名额/资金检查与下单延迟到
                # _place_deferred_orders(补仓优先 + 偏离度降序)
                kp = float(result.get("kalman_price", 0) or 0)
                dev = (result["close"] / kp - 1) * 100 if kp > 0 else 0.0
                result["_deferred"] = {
                    "kind": "new", "symbol": symbol, "name": name,
                    "shares": buy_qty, "price": result["close"],
                    "target_pct": capped_pct, "raw_target": result.get("target_pct", 0.95),
                    "reason": result.get("reason", ""),
                    "deviation": dev, "asset_type": asset_type,
                }
            else:
                etf_pfx = ("51", "15", "58", "56")
                is_etf = (asset_type == "etf")
                pool_pos = {
                    s: p
                    for s, p in load_positions().items()
                    if (s.startswith(etf_pfx)) == is_etf
                }
                used = sum(p["shares"] * p["avg_cost"] for p in pool_pos.values())
                committed = allocated_cash.get(asset_type, 0.0) if allocated_cash else 0.0
                remaining_cash = cash - used - committed
                if buy_qty * result["close"] > remaining_cash * 1.05:
                    reason = f"资金不足(需¥{buy_qty * result['close']:,.0f}>可用¥{remaining_cash:,.0f})"
                    log_skipped(symbol, name, entry_date, reason, result.get("reason", ""))
                else:
                    add_pending_order(symbol, name, "buy", buy_qty, result["close"], entry_date, capped_pct)
                    log_pending(symbol, name, "buy", capped_pct, buy_qty, result["reason"], entry_date)
                    result["shares"] = buy_qty
                    # 记录已承诺资金，防止后续标的超额下单
                    if allocated_cash is not None:
                        allocated_cash[asset_type] = allocated_cash.get(asset_type, 0.0) + buy_qty * result["close"]
        else:
            # 买不起 1 手(整手归零) → 标记跳过, signal 改 hold
            # (修复 2026-08-12: 原逻辑 signal 保持 buy 落盘 signals.csv,
            #  报告/快照误显 buy, 且 state_check "昨日买入未执行" 误报)
            buy_reason = result.get("reason", "")
            result["signal"] = "hold"
            result["_target_pct"] = result.get("target_pct", 0.95)
            result["target_pct"] = 0.0
            result["reason"] = (
                f"资金不足(单只上限¥{cash * max_pct:,.0f}, "
                f"股价¥{result['close']:.2f})")
            result["_skipped"] = True
            log_skipped(symbol, name, entry_date, result["reason"],
                        buy_reason)
    elif result["signal"] == "sell":
        # 卖出 → 待执行订单（次日开盘价成交）
        if has_pos and pos_info:
            sell_shares = pos_info["shares"]
            add_pending_order(symbol, name, "sell", sell_shares, result["close"], entry_date, 0.0)
            # 用模块级 log_pending(与买入分支一致)——原本地 import 绕过
            # 测试 spy 且可向真实 execution_log 写入污染行(2026-08-28 修复)
            log_pending(symbol, name, "sell", 0, sell_shares, result["reason"], entry_date, result["reason"])
            result["shares"] = sell_shares
        else:
            # 无持仓的卖出信号 → 降级 hold, 不生成订单
            # (修复 2026-08-28: 原逻辑 signal 保持 sell 落盘 signals.csv,
            #  扫描输出显示 🔴 卖出 仓位 0%, 实际无订单可执行, 展示误导;
            #  对齐 2026-08-12 买入侧"买不起 1 手降级 hold"修复:
            #  报告/快照/状态检查不会误显卖出信号)
            sell_reason = result.get("reason", "")
            result["signal"] = "hold"
            result["_target_pct"] = result.get("target_pct", 0.0)
            result["target_pct"] = 0.0
            result["reason"] = f"无持仓, 卖出信号忽略 | {sell_reason}"
            result["_skipped"] = True
            log_skipped(symbol, name, entry_date, result["reason"],
                        sell_reason, action="sell")

    # 7. 趋势翻转补仓: 持仓标的从下跌趋势→上涨,目标仓位从30%→95%
    if has_pos and result["signal"] == "hold" and result["target_pct"] > 0:
        from orders import load_pending as _lp7
        if any(str(o["symbol"]).zfill(6) == str(symbol).zfill(6)
               for o in _lp7()):
            pass  # 已有待执行订单(含补仓单), 不重复生成补仓候选
        else:
            cash = float(stock.get("cash", params.get("initial_cash", 100000)))
            max_pct = float(params.get("single_position_pct", 0.95))
            target_value = cash * result["target_pct"] * max_pct
            # 当前市值按市价计算(对齐回测 strategy.py),避免浮盈被低估导致补仓过量
            current_value = pos_info["shares"] * close if pos_info else 0
            if target_value > current_value * 1.05:
                add_value = target_value - current_value
                capped_pct = round(result["target_pct"] * max_pct, 4)
                lot = 200 if str(symbol).startswith("688") else 100
                if params.get("lot_based_position"):
                    # 无限仓位模式: 补足到 target_pct 对应的目标手数
                    cur_lots = max(1, int(pos_info["shares"]) // lot)
                    tgt_lots = max(1, round(result["target_pct"] * 3))
                    add_qty = max(0, tgt_lots - cur_lots) * lot
                else:
                    add_qty = int(add_value / result["close"] / lot) * lot
                if add_qty > 0:
                    if defer_orders:
                        # 两阶段模式: 补仓候选延迟到 _place_deferred_orders
                        kp = float(result.get("kalman_price", 0) or 0)
                        dev = (result["close"] / kp - 1) * 100 if kp > 0 else 0.0
                        result["_deferred"] = {
                            "kind": "refill", "symbol": symbol, "name": name,
                            "shares": add_qty, "price": result["close"],
                            "target_pct": capped_pct, "raw_target": result.get("target_pct", 0.95),
                            "reason": f"趋势翻转补仓: {result.get('reason', '')}",
                            "deviation": dev, "asset_type": asset_type,
                        }
                    else:
                        etf_pfx = ("51", "15", "58", "56")
                        is_etf_pool = (asset_type == "etf")
                        pool_pos = {
                            s: p for s, p in load_positions().items()
                            if (s.startswith(etf_pfx)) == is_etf_pool
                        }
                        used = sum(p["shares"] * p["avg_cost"] for p in pool_pos.values())
                        committed = allocated_cash.get(asset_type, 0.0) if allocated_cash else 0.0
                        remaining_cash = cash - used - committed
                        if add_qty * result["close"] <= remaining_cash * 1.05:
                            add_pending_order(symbol, name, "buy", add_qty,
                                              result["close"], entry_date, capped_pct)
                            log_pending(symbol, name, "buy", capped_pct, add_qty,
                                        f"趋势翻转补仓: {result['reason']}", entry_date)
                            result["signal"] = "buy"
                            result["shares"] = add_qty
                            result["reason"] = f"趋势翻转补仓: {result['reason']}"
                            if allocated_cash is not None:
                                allocated_cash[asset_type] = (
                                    allocated_cash.get(asset_type, 0.0) + add_qty * result["close"])

    return result


def _place_deferred_orders(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    allocated_cash: Dict[str, float] | None = None,
) -> None:
    """两阶段下单(2026-08-12): 补仓优先, 新建仓按偏离度降序。

    阶段 1(evaluate_stock defer_orders=True)只收集候选(_deferred),
    阶段 2 统一排序后按 名额+资金 检查下单:
    - 补仓(已持仓)优先于新建仓: 已有持仓、趋势已确认, 先于新标的消耗资金
    - 同类型按偏离度降序: 信号最强(偏离最大)的先拿名额
      (原逻辑按 watchlist 扫描顺序先到先得, 配置顺序无业务语义却决定结果)
    - 新建仓占用 max_positions 名额, 补仓不占(成交后不新增持仓标的)
    """
    deferred = [r for r in results if r.get("_deferred") is not None]
    if not deferred:
        return

    refills = [r for r in deferred if r["_deferred"]["kind"] == "refill"]
    news = [r for r in deferred if r["_deferred"]["kind"] == "new"]

    # 修复(2026-08-27): 股票/ETF 分池排序——原混合排序 + placed_new 全局
    # 计数导致跨池名额串扰(股票池下单挤占 ETF 池名额, 如 8-27 股票 2 单
    # 使 ETF 阶段 2 仅下 1 单, 证券/半导体ETF 被挤到 _auto_fill_pool 补入)。
    # 各池独立排序(补仓优先 + 池内偏离降序), placed_new 分池计数。
    def _sort_pool(items):
        st = [r for r in items
              if r["_deferred"].get("asset_type") == "stock"]
        et = [r for r in items
              if r["_deferred"].get("asset_type") != "stock"]
        for g in (st, et):
            g.sort(key=lambda r: float(r["_deferred"].get("deviation", 0)),
                   reverse=True)
        return st + et

    candidates = (_sort_pool(refills) + _sort_pool(news))

    # 修复(2026-08-27): pool_pending_new 统计**本轮开始前**的 pending(昨日
    # 遗留防超买)——原实现实时读 pending, 本轮新下的单也计入 → 与 placed_new
    # 重复计数使名额减半(实盘股票池 3 名额实际只下 2 单, 光智等被误拦)
    from orders import load_pending as _load_pending
    pre_pending = _load_pending()

    placed_new = {"stock": 0, "etf": 0}  # 各池已下单新建数(分池占用名额)
    for r in candidates:
        d = r["_deferred"]
        symbol, name = d["symbol"], d["name"]
        asset_type = d["asset_type"]
        entry_date = str(r.get("date", ""))[:10]
        etf_pfx = ("51", "15", "58", "56")
        is_etf = (asset_type == "etf")

        def _pool_positions() -> int:
            return sum(1 for s in load_positions()
                       if (s.startswith(etf_pfx)) == is_etf)

        # 名额检查(仅新建仓; 补仓单成交后不新增持仓标的)
        if d["kind"] == "new":
            pool_cfg = config.get(asset_type, {})
            max_pos = int(pool_cfg.get(
                "max_positions", 5 if asset_type == "stock" else 3))
            # 已有 pending 中的新建单也占名额(防超买: 昨日下单未成交
            # 今日再下单 → 成交后持仓超限); 补仓单(已持仓)不占
            pool_pending_new = sum(
                1 for o in pre_pending
                if (o["symbol"].startswith(etf_pfx)) == is_etf
                and o.get("action") != "sell"
                and o["symbol"] not in load_positions())
            if (max_pos > 0
                    and _pool_positions() + placed_new[asset_type]
                    + pool_pending_new >= max_pos):
                _block(r, d, symbol, name, entry_date,
                       f"已达最大持仓数({max_pos})")
                continue

        # 资金检查
        cash = float(config.get(asset_type, {}).get(
            "initial_cash", 200000 if asset_type == "stock" else 100000))
        pool_pos = {s: p for s, p in load_positions().items()
                    if (s.startswith(etf_pfx)) == is_etf}
        used = sum(p["shares"] * p["avg_cost"] for p in pool_pos.values())
        committed = allocated_cash.get(asset_type, 0.0) if allocated_cash else 0.0
        remaining_cash = cash - used - committed
        if d["shares"] * d["price"] > remaining_cash * 1.05:
            _block(r, d, symbol, name, entry_date,
                   f"资金不足(需¥{d['shares'] * d['price']:,.0f}"
                   f">可用¥{remaining_cash:,.0f})")
            continue

        # 下单
        add_pending_order(symbol, name, "buy", d["shares"], d["price"],
                          entry_date, d["target_pct"])
        log_pending(symbol, name, "buy", d["target_pct"], d["shares"],
                    d["reason"], entry_date)
        r["signal"] = "buy"
        r["shares"] = d["shares"]
        r["reason"] = d["reason"]
        if allocated_cash is not None:
            allocated_cash[asset_type] = (
                allocated_cash.get(asset_type, 0.0) + d["shares"] * d["price"])
        if d["kind"] == "new":
            placed_new[asset_type] += 1


def _block(r, d, symbol, name, entry_date, reason: str) -> None:
    """两阶段下单被拦截: 标记 _skipped 并写执行日志(对齐原第5/6步拦截)。"""
    r["signal"] = "hold"
    # 修复(2026-08-27): _target_pct 存**原始** target_pct(未 cap)——原存
    # _deferred 的 capped_pct(0.19), _auto_fill_pool 再乘 max_pct 形成
    # 双重 cap(0.038) → 高价股(光智¥242)买不起 1 手被静默跳过、低价股
    # (巨石)补仓量减半(200 股应为 500 股)
    r["_target_pct"] = d.get("raw_target", d.get("target_pct", 0.95))
    r["target_pct"] = 0.0
    r["reason"] = reason
    r["_skipped"] = True
    log_skipped(symbol, name, entry_date, reason, d["reason"])


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
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="扫描完成后自动部署报告到服务器 (需先配置 SSH 免密登录)",
    )
    return parser.parse_args()


def _auto_deploy(quiet: bool = False) -> None:
    """调用 deploy.sh --live-only 将实盘报告部署到服务器。"""
    deploy_script = os.path.join(TASK_DIR, "deploy.sh")
    if not os.path.exists(deploy_script):
        if not quiet:
            print("  ⚠️ deploy.sh 不存在，跳过自动部署")
        return

    import subprocess
    try:
        result = subprocess.run(
            ["bash", deploy_script, "--live-only"],
            capture_output=True, text=True, timeout=60,
            cwd=TASK_DIR,
        )
        if result.returncode == 0:
            if not quiet:
                print("  ✅ 报告已部署到服务器")
        else:
            if not quiet:
                # 不显示完整错误，避免干扰主流程
                err_line = result.stderr.strip().split("\n")[-1] if result.stderr else ""
                print(f"  ⚠️ 部署失败: {err_line or result.stdout.strip()}")
    except FileNotFoundError:
        if not quiet:
            print("  ⚠️ rsync/ssh 不可用，跳过部署")
    except Exception as e:
        if not quiet:
            print(f"  ⚠️ 部署异常: {e}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_years = int(config.get("data_years", 2))
    watchlist_raw = config.get("watchlist", {})
    if isinstance(watchlist_raw, list):
        stocks_list = watchlist_raw
        etfs_list = []
    else:
        stocks_list = watchlist_raw.get("stocks", [])
        etfs_list = watchlist_raw.get("etfs", [])
    all_watchlist = stocks_list + etfs_list

    if not all_watchlist:
        print("错误: watchlist 为空")
        sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")

    # 运行前备份
    backup_tag = create_backup()
    if not args.quiet:
        print(f"  📦 备份: {backup_tag}")

    print(f"\n{'=' * 60}")
    print(f"  每日信号扫描  {today}")
    print(f"  股票: {len(stocks_list)} 只  ETF: {len(etfs_list)} 只  (共 {len(all_watchlist)} 只)")
    print(f"{'=' * 60}")

    # ---- 0. 数据校验 ----
    data_issues, order_issues = run_all_checks(all_watchlist, quiet=args.quiet)
    all_validation_issues = data_issues + order_issues

    # ---- 0.5 并行预取数据(2026-08-24): 评估循环前并行下载缺失/过期数据,
    # 主循环串行评估时全部缓存命中(下载无副作用可并行, 评估有副作用须串行)
    prefetch_data(all_watchlist, data_years, quiet=args.quiet)

    # ---- 0.6 先执行昨日待执行订单, 再扫描(2026-09-01) ----
    # 原顺序"扫描→执行→补仓": 昨日卖出成交释放的名额在扫描后才生效,
    # 当日候选先被"已达最大持仓数"拦截, 再由 _auto_fill_pool 补入——
    # 结果正确但流程绕两步, 且报告先显"被跳过"再显"补仓"。
    # 先执行后扫描: 名额检查(_pool_positions 实时读 positions.json)基于
    # 执行后的真实持仓, 昨日卖出释放的名额让当日候选直接下单。
    # 资金时序正确: 昨日卖出今日开盘成交 → 资金今日到账 → 今日收盘生成
    # 买入 → 明日开盘成交(T+2, 与 2026-08-26 名额预释放否决不冲突——
    # 该否决针对"今日卖出信号未成交即释放名额", 此处是已成交的真实释放)。
    # 今日新卖出信号仍不释放名额(名额延迟, 对齐资金正确性)。
    positions_pre_exec = len(load_positions())
    data_fresh = _check_data_freshness(all_watchlist, today)
    if not data_fresh:
        all_validation_issues.append({
            "symbol": "*", "name": "全局", "check": "数据未更新",
            "level": "warn", "detail": f"最新数据 < {today}，跳过待执行订单",
        })
        print(f"  ⚠️ 当日数据未就绪，跳过待执行订单（需 {today} 数据）")
        from orders import load_pending
        pending_count = len(load_pending())
        if pending_count > 0:
            print(f"  ⏳ {pending_count} 笔待执行订单等待数据更新后成交")
    else:
        _execute_today_pending(config, today)

    positions_before = len(load_positions())
    results = []
    allocated_cash: Dict[str, float] = {}  # 本轮已承诺资金 {pool: amount}

    # 阶段 1: 评估全部标的(只收集候选, 不下单)
    for i, stock in enumerate(all_watchlist, 1):
        symbol = stock["symbol"]
        name = stock.get("name", symbol)
        if not args.quiet:
            print(f"\n[{i}/{len(all_watchlist)}] {symbol} {name} ...")

        try:
            result = evaluate_stock(stock, config, data_years, allocated_cash,
                                    defer_orders=True)
            results.append(result)
            if not args.quiet:
                print(format_signal(result))
        except Exception as e:
            results.append({
                "symbol": symbol, "name": name,
                "error": str(e), "signal": "error",
            })
            if not args.quiet:
                print(f"  ❌ 错误: {e}")

    # 阶段 2: 统一排序下单(2026-08-12)——补仓优先, 新建按偏离度降序
    _place_deferred_orders(results, config, allocated_cash)
    skipped_buys = [r for r in results if r.get("_skipped")]
    if not args.quiet:
        for r in results:
            d = r.get("_deferred")
            if d is not None:
                if r["signal"] == "buy":
                    print(f"  📌 下单 {d['symbol']} {d['name']}: "
                          f"{r['shares']}股 @{r['close']:.2f} "
                          f"(偏离{d['deviation']:+.1f}%)")
                else:
                    print(f"  ⏸️ 拦截 {d['symbol']} {d['name']}: {r['reason']}")

    if not args.quiet:
        print_summary(results)

    if not args.no_save:
        valid = [r for r in results if not r.get("error")]
        if valid:
            append_signals_csv(valid)
            print(f"\n信号已保存: {SIGNALS_CSV} ({len(valid)} 条)")

    # 持仓状态表 + 一致性检查 + 快照
    # 显示被跳过的买入信号
    # (2026-09-01: 待执行订单执行已移至扫描前, 见"0.6 先执行昨日待执行订单")
    if skipped_buys and not args.quiet:
        print(f"\n  ⏸️ 被跳过 ({len(skipped_buys)} 笔，等仓位空出):")
        skipped_buys.sort(
            key=lambda x: abs(
                float(x.get("kalman_price", 0)) - float(x.get("close", 0))
            ) / float(x.get("close", 1)),
            reverse=True,
        )
        for s in skipped_buys[:5]:
            deviation = (
                float(s.get("close", 0)) / float(s.get("kalman_price", 1)) - 1
            ) * 100
            print(
                f"     {s['symbol']} {s.get('name',''):<6s} ¥{s.get('close',0):>8.2f}  "
                f"偏离{deviation:+.1f}%  趋势={s.get('trend','?')}"
            )

    # 仓位空出时自动买入最强信号（分池独立处理）
    # 改进(2026-08-05): 不再依赖 freed>0 总持仓变化判断——分池空位由
    # _auto_fill_pool 内部按 max_pos - 池持仓 - 池待买 计算,
    # 无条件调用保证"卖出成交后立即补仓"可靠触发(8-4 曾因触发条件未满足漏补)。
    positions_now_all = load_positions()
    positions_now = len(positions_now_all)
    # 2026-09-01: 执行已移至扫描前——"释放"指执行昨日订单的净持仓变化
    # (执行前 → 执行后), 补仓仅在仍有候选被拦截(池满无释放)时兜底触发
    exec_delta = positions_before - positions_pre_exec
    if not args.quiet and (exec_delta != 0 or skipped_buys):
        print(f"\n  🔄 补仓检查: 执行净变化 {exec_delta:+d} "
              f"({positions_pre_exec}→{positions_before}), "
              f"被跳过候选 {len(skipped_buys)} 笔")
    if skipped_buys:
        etf_pfx = ("51", "15", "58", "56")
        for pool_name, pool_cfg_key in [("股票", "stock"), ("ETF", "etf")]:
            _auto_fill_pool(
                skipped_buys, positions_pre_exec, positions_now_all,
                config, pool_cfg_key, pool_name, etf_pfx, quiet=args.quiet,
            )

    # 校验结果
    if all_validation_issues:
        errors = [i for i in all_validation_issues if i["level"] == "error"]
        warns = [i for i in all_validation_issues if i["level"] == "warn"]
        print(f"\n⚠️ 数据校验: {len(errors)} 错误 {len(warns)} 警告")
        if errors:
            for i in errors:
                print(f"  ❌ {i['symbol']} {i.get('name','')}: {i['detail']}")
        if warns:
            for i in warns:
                print(f"  ⚠️ {i['symbol']} {i.get('name','')}: {i['detail']}")
    if not args.no_save:
        save_validation_log(all_validation_issues)

    # 持仓状态表 + 实盘报告
    if not args.quiet:
        print_position_summary()
        print_check_result()
    save_snapshot()

    # 生成实盘交易报告
    try:
        from live_report import build_live_report
        build_live_report()
    except Exception as e:
        if not args.quiet:
            print(f"  ⚠️ 实盘报告生成失败: {e}")

    # 自动部署到服务器
    if hasattr(args, "deploy") and args.deploy:
        _auto_deploy(args.quiet)


def _auto_fill_pool(
    skipped_buys: List[Dict[str, Any]],
    positions_before: int,
    positions_now_all: Dict[str, Any],
    config: Dict[str, Any],
    pool_key: str,
    pool_name: str,
    etf_pfx: tuple,
    quiet: bool = False,
) -> int:
    """单一资金池的自动补仓逻辑。

    参数:
        skipped_buys: 本轮被跳过的买入信号列表
        positions_before: 执行订单前的总持仓数
        positions_now_all: 当前全部持仓 dict
        pool_key: 配置键("stock"|"etf")
        pool_name: 显示名称("股票"|"ETF")
        etf_pfx: ETF 代码前缀元组
        quiet: 是否静默

    返回补仓数量。
    """
    pool = _pool_config(config, pool_key)
    is_etf = (pool_key == "etf")
    max_pos = int(pool["max_positions"])
    cash = float(pool["initial_cash"])
    max_pct = float(pool["single_position_pct"])

    # 分池统计: 当前池内持仓数 + 待买入订单数
    from orders import load_pending as _lp
    pending = _lp()
    pool_pos_count = sum(
        1 for s in positions_now_all
        if (s.startswith(etf_pfx)) == is_etf
    )
    pool_pending_count = sum(
        1 for o in pending
        if (o["symbol"].startswith(etf_pfx)) == is_etf
        and o.get("action") != "sell"
    )
    remaining_slots = max_pos - pool_pos_count - pool_pending_count
    if remaining_slots <= 0:
        return 0

    # 计算池内可用资金
    pool_used = sum(
        p["shares"] * p["avg_cost"]
        for s, p in positions_now_all.items()
        if (s.startswith(etf_pfx)) == is_etf
    )
    pool_committed = sum(
        o["shares"] * o["signal_price"]
        for o in pending
        if (o["symbol"].startswith(etf_pfx)) == is_etf
        and o.get("action") != "sell"
    )
    remaining_cash = cash - pool_used - pool_committed

    # 筛选该池标的，按偏离度排序
    pool_skipped = [s for s in skipped_buys
                    if (str(s["symbol"]).startswith(etf_pfx)) == is_etf]
    pool_skipped.sort(
        key=lambda x: abs(
            float(x.get("close", 0)) / float(x.get("kalman_price", 1)) - 1
        ),
        reverse=True,
    )

    filled = 0
    for s in pool_skipped:
        if filled >= remaining_slots:
            break

        sym = str(s["symbol"]).zfill(6)
        # 涨停检查
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        blocked = False
        if os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                if len(cached) >= 1:
                    last_close = float(cached["close"].iloc[-1])
                    pct_limit = 0.20 if sym.startswith(("688", "300", "301")) else 0.10
                    limit_up = last_close * (1 + pct_limit)
                    if s["close"] >= limit_up * 0.999:
                        if not quiet:
                            print(f"  ⚠️ {sym} {s['name']} 涨停,跳过")
                        blocked = True
            except Exception:
                pass
        if blocked:
            continue

        # 目标仓位: 优先用被跳过时保留的原始仓位(_target_pct),
        # 回退 target_pct(0=已跳过清零, 0.95=默认满仓)
        tgt = float(s.get("_target_pct", s.get("target_pct", 0.95)))
        if tgt <= 0:
            tgt = 0.95
        capped_pct = round(tgt * max_pct, 4)
        lot = 200 if sym.startswith("688") else 100
        buy_qty = int(cash * capped_pct / s["close"] / lot) * lot
        if buy_qty <= 0:
            continue
        buy_amount = buy_qty * s["close"]

        if buy_amount > remaining_cash * 1.05:
            if not quiet:
                print(f"  ⚠️ {sym} {s['name']} 资金不足"
                      f"(需¥{buy_amount:,.0f}>可用¥{remaining_cash:,.0f})，跳过")
            continue

        add_pending_order(sym, s["name"], "buy", buy_qty,
                          s["close"], s["date"], capped_pct)
        log_pending(sym, s["name"], "buy", capped_pct,
                    buy_qty, s.get("reason", ""), s["date"])
        remaining_cash -= buy_amount
        filled += 1
        if not quiet:
            print(f"  🟢 自动补仓[{pool_name}] {sym} {s['name']}: {buy_qty}股 "
                  f"(空位{remaining_slots}→{remaining_slots - filled}, "
                  f"候选排名 #{filled})")

    return filled


def _check_data_freshness(watchlist: list, today: str) -> bool:
    """检查缓存数据是否覆盖到今天。

    只有当日数据就绪时才认为数据新鲜，才能用当日开盘价执行待处理订单。
    """
    sample = watchlist[:3]
    for stock in sample:
        sym = str(stock["symbol"]).zfill(6)
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                last_date = str(pd.to_datetime(cached["date"].max()).date())
                if last_date >= today:
                    return True
            except Exception:
                pass
    return False


def _execute_today_pending(config: Dict[str, Any], today: str = "") -> List[Dict[str, Any]]:
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

    # 读取人工实际成交价(优先于 open 假设,提升准确性)
    from orders import load_actual_fills, save_actual_fills
    actual_fills = load_actual_fills()
    actual_prices = {
        sym: float(fill["exec_price"])
        for sym, fill in actual_fills.items()
        if isinstance(fill, dict) and "exec_price" in fill
    }

    pending = execute_pending_orders(
        today_open, prev_close, today_str=today, actual_prices=actual_prices or None
    )
    # 清理已使用的实际成交价记录
    if actual_fills:
        executed_syms = {order["symbol"] for order in pending}
        remaining_fills = {
            k: v for k, v in actual_fills.items() if k not in executed_syms
        }
        if remaining_fills != actual_fills:
            save_actual_fills(remaining_fills)
    for order in pending:
        sym = order["symbol"]
        name = order.get("name", sym)
        exec_price = order["exec_price"]
        action = order.get("action", "buy")
        if action == "sell":
            # 执行卖出：移除持仓，记录交易(扣手续费)
            removed = remove_position(sym)
            if removed:
                fee_sell = calc_fee(order["shares"], exec_price, "sell")
                record_trade(
                    symbol=sym, name=name,
                    shares=order["shares"],
                    entry_price=removed.get("avg_cost", 0),
                    exit_price=exec_price,
                    entry_date=removed.get("first_buy_date", order["signal_date"]),
                    exit_date=today,
                    reason=order.get("signal_reason", ""),
                    fee=fee_sell,
                )
            log_executed(sym, order["signal_date"], exec_price, today)
            print(f"  ✅ 卖出 {sym} {name}: {order['shares']}股 @ ¥{exec_price:.2f} "
                  f"(手续费¥{fee_sell:.2f},信号日 {order['signal_date']})")
        else:
            # 执行买入:avg_cost 含手续费(佣金+过户费)
            fee_buy = calc_fee(order["shares"], exec_price, "buy")
            avg_cost = (order["shares"] * exec_price + fee_buy) / order["shares"]
            add_position(sym, name, order["shares"], avg_cost, order["signal_date"])
            log_executed(sym, order["signal_date"], exec_price, today)
            print(f"  ✅ 买入 {sym} {name}: {order['shares']}股 @ ¥{exec_price:.2f} "
                  f"(成本¥{avg_cost:.3f}含费¥{fee_buy:.2f},信号日 {order['signal_date']})")
        executed.append(order)

    from orders import load_pending
    remaining = load_pending()
    for order in remaining:
        # 仅对因涨停/缺价等原因真正失败的订单标 failed; T+1等待的不标
        if order.get("signal_date", "") < today:
            log_failed(order["symbol"], order["signal_date"],
                       order.get("reason", "未能成交"))
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
            stock_cfg = cfg.get("stock", {})
            cash = float(stock_cfg.get("initial_cash", 200000))
            max_pct = float(stock_cfg.get("single_position_pct", 0.20))
        except Exception:
            pass
        max_value = cash * max_pct

        print(f"\n{'=' * 80}")
        print(f"  {'持仓总览 (上限 ' + str(int(max_pct*100)) + '% = ¥' + f'{max_value:,.0f})':^72s}")
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
            print(
                f"  {sym:<8s} {pos['name']:<8s} {pos['shares']:>6d}  "
                f"{pos['avg_cost']:>8.2f} {value:>10.0f} {pct:>5.1f}% {pnl:>+10.0f} {pnl_pct:>+7.1f}%{flag}"
            )

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
