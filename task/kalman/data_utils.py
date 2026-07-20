"""
数据下载与预处理模块。

功能:
1. 使用 AKShare 下载 A 股历史日线数据（OHLCV）
2. 数据清洗：缺失值、异常值、有效性检查
3. 添加技术指标列：MA5、MA20、布林带(20,2)
"""

from typing import Optional

import numpy as np
import pandas as pd
from akquant.utils import fetch_akshare_symbol


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
