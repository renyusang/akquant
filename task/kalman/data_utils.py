"""
数据下载与预处理模块。

功能:
1. 使用 AKShare 下载 A 股历史日线数据（OHLCV）
2. 数据清洗：缺失值、异常值、有效性检查
3. 添加技术指标列：MA5、MA20、布林带(20,2)
"""

import os
import socket
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from akquant.data import ParquetDataCatalog
from akquant.utils import fetch_akshare_symbol

# 网络超时兜底(2026-08-24): akshare(含 akquant.utils.fetch_akshare_symbol)
# 底层 requests 无显式超时——网络挂起时进程无限等待(8-24 曾卡 10 分钟)。
# socket 级默认超时覆盖全部下载路径(股票/ETF/sina/em), 挂起 30s 抛
# socket.timeout → 调用方(download_with_cache)捕获后走缓存/报错。
# 正常下载实测 ~4s, 30s 足够宽裕; 无重试需求(偶发抖动重跑即可)。
_NETWORK_TIMEOUT = 30.0
socket.setdefaulttimeout(_NETWORK_TIMEOUT)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(TASK_DIR, ".cache")


def download_stock_data(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """下载 A 股历史日线数据。

    参数:
        symbol:     股票代码，如 "002594"（比亚迪）、"600519"（贵州茅台）
        start_date: 起始日期，格式 YYYYMMDD
        end_date:   结束日期，格式 YYYYMMDD
        adjust:     复权方式，"qfq"=前复权, "hfq"=后复权, ""=不复权

    返回:
        DataFrame，包含列: date, open, high, low, close, volume, symbol
    """
    df = fetch_akshare_symbol(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if df.empty:
        raise ValueError(f"未获取到 {symbol} 的数据 ({start_date} ~ {end_date})")

    # 统一列名
    df = df.copy()
    if "日期" in df.columns:
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )

    # 确保 date 列为 datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        # date 可能在索引上
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})

    # 确保有 symbol 列
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    # 只保留需要的列
    required_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
    df = df[[c for c in required_cols if c in df.columns]]

    # 去重、排序
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def _etf_sina_symbol(symbol: str) -> str:
    """将 ETF 代码转为 Sina 格式（sh510050 / sz159915）。

    ETF 代码规则: 5 开头 → 上海，1 开头 → 深圳。
    """
    if symbol.startswith("5"):
        return f"sh{symbol}"
    else:
        return f"sz{symbol}"


def download_etf_data(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """下载场内 ETF 历史日线数据。

    优先使用 Sina 接口（fund_etf_hist_sina），失败时回退到东方财富接口。
    Sina 接口无需日期范围参数，返回全部历史数据后再截取。
    ETF 代码为 5 位数字（如 510050=上证50ETF, 159915=创业板ETF）。

    参数:
        symbol:     ETF 代码，如 "510050"
        start_date: 起始日期，格式 YYYYMMDD
        end_date:   结束日期，格式 YYYYMMDD
        adjust:     复权方式，Sina 接口不支持复权，忽略此参数
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare 未安装")

    sina_sym = _etf_sina_symbol(symbol)
    df = None

    # 优先尝试 Sina 接口
    try:
        df = ak.fund_etf_hist_sina(symbol=sina_sym)
    except Exception:
        pass

    # 回退：东方财富接口
    if df is None or df.empty:
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception:
            pass

    if df is None or df.empty:
        raise RuntimeError(f"ETF {symbol} 数据下载失败（Sina + 东方财富均不可用）")

    # 统一列名（Sina 接口已返回英文列名，东方财富返回中文）
    rename_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 确保 date 列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})

    # 按日期范围截取
    if "date" in df.columns:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    if df.empty:
        raise ValueError(f"未获取到 ETF {symbol} 的数据 ({start_date} ~ {end_date})")

    # 确保有 symbol 列
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    # 只保留需要的列
    required_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
    df = df[[c for c in required_cols if c in df.columns]]

    # 去重、排序
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def download_data(
    symbol: str,
    asset_type: str = "stock",
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """统一下载接口，根据 asset_type 自动选择数据源。

    参数:
        symbol:     代码
        asset_type: "stock"（股票）或 "etf"（场内基金）
    """
    if asset_type == "etf":
        return download_etf_data(symbol, start_date, end_date, adjust)
    else:
        return download_stock_data(symbol, start_date, end_date, adjust)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理。

    执行步骤:
    1. 缺失值检测与填充
    2. 异常值检测与截尾处理
    3. 基本有效性校验（high>=low, close>0 等）
    4. 添加技术指标列：MA5, MA20, 布林带(20,2)

    返回处理后的 DataFrame。
    """
    df = df.copy()

    # ---- 1. 缺失值处理 ----
    ohlc_cols = ["open", "high", "low", "close"]
    vol_col = "volume"

    # 检查关键列中的 NaN
    nan_counts = df[ohlc_cols + [vol_col]].isna().sum()
    if nan_counts.sum() > 0:
        print(f"[预处理] 缺失值统计:\n{nan_counts[nan_counts > 0]}")
        # 前向填充 → 后向填充（尽量避免引入偏差）
        df[ohlc_cols] = df[ohlc_cols].ffill().bfill()
        df[vol_col] = df[vol_col].fillna(0.0)

    # ---- 2. 基本有效性校验 ----
    # close > 0
    invalid_close = df["close"] <= 0
    if invalid_close.any():
        print(f"[预处理] 发现 {invalid_close.sum()} 条 close<=0 的记录，已移除")
        df = df[~invalid_close]

    # high >= low
    invalid_hl = df["high"] < df["low"]
    if invalid_hl.any():
        print(f"[预处理] 发现 {invalid_hl.sum()} 条 high<low 的记录，交换修复")
        df.loc[invalid_hl, ["high", "low"]] = df.loc[
            invalid_hl, ["low", "high"]
        ].values

    # high >= open, high >= close
    df["high"] = df[["high", "open", "close"]].max(axis=1)

    # low <= open, low <= close
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    # volume >= 0
    df.loc[df["volume"] < 0, "volume"] = 0.0

    # ---- 3. 异常值检测与截尾 ----
    # 使用滚动窗口的 3 倍标准差进行检测
    for col in ohlc_cols:
        rolling_mean = df[col].rolling(window=20, min_periods=5).mean()
        rolling_std = df[col].rolling(window=20, min_periods=5).std()
        upper_bound = rolling_mean + 3.0 * rolling_std
        lower_bound = (rolling_mean - 3.0 * rolling_std).clip(
            lower=0.0
        )  # 价格不能为负

        outlier_mask = (df[col] > upper_bound) | (df[col] < lower_bound)
        if outlier_mask.any():
            n_outliers = outlier_mask.sum()
            pct = 100.0 * n_outliers / len(df)
            print(
                f"[预处理] {col}: {n_outliers} 个异常值 ({pct:.2f}%)，已截尾处理"
            )
            df.loc[df[col] > upper_bound, col] = upper_bound
            df.loc[df[col] < lower_bound, col] = lower_bound

    # 日收益率异常检测
    if len(df) > 1:
        daily_ret = df["close"].pct_change()
        ret_mean = daily_ret.mean()
        ret_std = daily_ret.std()
        ret_upper = ret_mean + 5.0 * ret_std
        ret_lower = ret_mean - 5.0 * ret_std
        ret_outliers = (daily_ret > ret_upper) | (daily_ret < ret_lower)
        if ret_outliers.any():
            n_ret = ret_outliers.sum()
            print(f"[预处理] 日收益率异常值: {n_ret} 个，已用前值填充")
            df.loc[ret_outliers, "close"] = df["close"].shift(1)
            # 同步修正 high/low/open
            for c in ["high", "low", "open"]:
                df.loc[ret_outliers, c] = df.loc[ret_outliers, "close"]

    # ---- 4. 排序、重置索引 ----
    df = df.sort_values("date").reset_index(drop=True)

    # ---- 5. 添加技术指标 ----
    df = _add_indicators(df)

    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标列：MA5, MA20, 布林带(20,2)。"""
    close = df["close"]

    # 均线
    df["ma5"] = close.rolling(window=5, min_periods=1).mean()
    df["ma20"] = close.rolling(window=20, min_periods=1).mean()

    # 布林带 (20, 2)
    df["bb_middle"] = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    df["bb_upper"] = df["bb_middle"] + 2.0 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2.0 * bb_std

    return df


def get_data_summary(df: pd.DataFrame) -> str:
    """返回数据摘要信息。"""
    lines = [
        f"标的:     {df['symbol'].iloc[0]}",
        f"时间范围: {df['date'].min().strftime('%Y-%m-%d')} ~ "
        f"{df['date'].max().strftime('%Y-%m-%d')}",
        f"数据条数: {len(df)}",
        f"开盘价:   {df['open'].min():.2f} ~ {df['open'].max():.2f}",
        f"收盘价:   {df['close'].min():.2f} ~ {df['close'].max():.2f}",
        f"成交量:   {df['volume'].min():.0f} ~ {df['volume'].max():.0f}",
    ]
    return "\n".join(lines)


# =============================================================================
# 回测数据加载(共享 .cache,与实盘 daily_signal 数据同源,保证对账一致)
# 注:存储沿用 .cache/{symbol}.parquet 扁平格式;ParquetDataCatalog 统一留待
# 实盘迁移阶段(阶段8),此处优先数据一致性。
# =============================================================================

# ETF 代码前缀(上海 51/58/56,深圳 15)
_ETF_PREFIXES = ("51", "15", "58", "56")


def is_etf(symbol: str) -> bool:
    """根据代码前缀判断是否为场内 ETF(51/15/58/56 开头)。"""
    sym = str(symbol).zfill(6)
    return sym.startswith(_ETF_PREFIXES)


def get_catalog(root: Optional[str] = None) -> ParquetDataCatalog:
    """获取回测专用 ParquetDataCatalog(默认 task/kalman/.catalog,与实盘 .cache 隔离)。"""
    return ParquetDataCatalog(root or os.path.join(TASK_DIR, ".catalog"))


def load_cached_data(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "20261231",
    asset_type: str = "stock",
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """加载单标的数据,优先读回测 catalog(.catalog),缺失则下载全量并缓存。

    返回 preprocess 后的 DataFrame,含 date/open/high/low/close/volume/symbol
    及 ma5/ma20/bb_* 指标列,date 列为 datetime。

    注:回测用独立 .catalog 目录存全量历史(如 2020 起),与实盘 daily_signal 的
    .cache(近2年)隔离,互不影响;数据源同为 akshare/sina,与基线同源,对账公平。
    """
    catalog = get_catalog(cache_dir)
    sym = str(symbol).zfill(6)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    df = catalog.read(sym, start_time=start_dt, end_time=end_dt)
    if df is None or df.empty:
        # catalog 缺失 → 下载全量并写入
        full_df = download_data(sym, asset_type=asset_type, start_date=start_date, end_date=end_date)
        full_df = preprocess_data(full_df)
        catalog.write(sym, full_df)
        df = full_df.copy()
    else:
        # catalog.read 返回 DatetimeIndex,还原为 date 列格式(与 .cache 一致)
        df = df.reset_index()
        if "date" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        if "symbol" not in df.columns:
            df["symbol"] = sym

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
    return df


def load_data_map(
    symbols: Union[str, List[str]],
    start_date: str = "20200101",
    end_date: str = "20261231",
    asset_types: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
    quiet: bool = True,
) -> Dict[str, pd.DataFrame]:
    """批量加载多标的数据,返回 {symbol: DataFrame}。

    供 run_backtest 多标的组合回测使用。数据源与实盘 daily_signal 共享 .cache,
    保证回测对账与实盘数据一致。

    参数:
        symbols:      标的代码或代码列表
        start_date:   起始日期 YYYYMMDD
        end_date:     结束日期 YYYYMMDD
        asset_types:  {symbol: "stock"|"etf"} 映射;None 则按代码前缀自动判断
        cache_dir:    缓存目录(默认 task/kalman/.cache)
        quiet:        是否静默(不打印进度)
    返回:
        Dict[str, pd.DataFrame],每个 df 含 date/open/high/low/close/volume/symbol 列
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    if asset_types is None:
        asset_types = {}

    def _load(sym: str):
        sym = str(sym).zfill(6)
        atype = asset_types.get(sym) or ("etf" if is_etf(sym) else "stock")
        try:
            df = load_cached_data(sym, start_date, end_date,
                                  asset_type=atype, cache_dir=cache_dir)
            return sym, atype, df
        except Exception as e:
            return sym, atype, e

    # 并行预取(2026-08-24): catalog 缺失的标的由 load_cached_data 内部下载,
    # 无副作用(各写各的 catalog 文件)可安全并行——与实盘 prefetch_data 同模式。
    # 实测 akshare 8 路并行 3.9x 加速, 组合回测首次全量(25+24 只)从 3-6 分钟
    # 降至 ~1 分钟。ex.map 保序, 失败标的与串行版一致跳过。
    from concurrent.futures import ThreadPoolExecutor

    data_map: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_load, symbols))
    for i, (sym, atype, df) in enumerate(results, 1):
        if isinstance(df, Exception):
            if not quiet:
                print(f"  [{i}/{len(symbols)}] {sym} ({atype}) 加载失败: {df}")
            continue
        if not df.empty:
            data_map[sym] = df
            if not quiet:
                print(f"  [{i}/{len(symbols)}] {sym} ({atype}): {len(df)} bars")
    return data_map


def load_watchlist_data(
    config: dict,
    start_date: str = "20200101",
    end_date: str = "20261231",
    cache_dir: Optional[str] = None,
    quiet: bool = True,
) -> tuple:
    """从 stocks.yaml 配置加载 watchlist 数据,分股票/ETF 两池返回。

    返回 (stock_map, etf_map, stock_items, etf_items):
        stock_map/etf_map: {symbol: DataFrame}
        stock_items/etf_items: watchlist 条目列表 [{symbol, name, ...}]
    """
    watchlist = config.get("watchlist", {})
    if isinstance(watchlist, list):  # 兼容旧扁平格式
        watchlist = {"stocks": watchlist, "etfs": []}
    stock_items = watchlist.get("stocks", []) or []
    etf_items = watchlist.get("etfs", []) or []

    stock_syms = [str(s["symbol"]).zfill(6) for s in stock_items]
    etf_syms = [str(s["symbol"]).zfill(6) for s in etf_items]

    stock_map = load_data_map(
        stock_syms, start_date, end_date,
        asset_types={s: "stock" for s in stock_syms},
        cache_dir=cache_dir, quiet=quiet,
    ) if stock_syms else {}
    etf_map = load_data_map(
        etf_syms, start_date, end_date,
        asset_types={s: "etf" for s in etf_syms},
        cache_dir=cache_dir, quiet=quiet,
    ) if etf_syms else {}

    return stock_map, etf_map, stock_items, etf_items


def fetch_hs300(start_date: str = "20200101", end_date: str = "20261231"):
    """获取沪深300指数日线收盘价(用于回测/实盘基准对比)。

    返回 pd.Series,index=date,dtype=float。
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare 未安装,无法获取沪深300数据")

    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
    return df.set_index("date")["close"].astype(float)
