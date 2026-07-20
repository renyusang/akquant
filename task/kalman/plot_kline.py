"""
K 线图绘制模块。

使用 mplfinance 绘制专业的 K 线图，包含:
- 主图: K 线蜡烛图 + MA5 + MA20 + 布林带（上轨/中轨/下轨）
- 副图: 成交量柱状图（涨红跌绿）

支持保存为 HTML（plotly 交互式）或 PNG 文件。
"""

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

# ---- 中文字体检测与配置 ----
def _detect_cjk_font() -> Optional[str]:
    """检测系统中可用的 CJK 字体，返回字体名称或 None。"""
    import subprocess

    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "-f", "%{family}\n"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        families = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        # 优先选择常见中文字体
        preferred = [
            "WenQuanYi",
            "Noto Sans CJK",
            "Noto Sans SC",
            "Source Han Sans",
            "SimHei",
            "Microsoft YaHei",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "AR PL",
            "CJK",
        ]
        for pref in preferred:
            for family in families:
                if pref.lower() in family.lower():
                    return family
        return families[0] if families else None
    except Exception:
        return None


_CJK_FONT = _detect_cjk_font()

if _CJK_FONT:
    matplotlib.rcParams["font.family"] = _CJK_FONT
    # 同时设置 sans-serif 回退
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    print(f"[字体] 使用中文字体: {_CJK_FONT}")
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    print("[字体] 未检测到中文字体，使用英文标签。可通过以下命令安装中文字体:")
    print("        sudo apt-get install fonts-wqy-microhei")

matplotlib.rcParams["axes.unicode_minus"] = False

# 根据是否有中文字体选择标签
_HAS_CJK = _CJK_FONT is not None
_LABELS = {
    "title_kline": "K-Line with Indicators - {}" if not _HAS_CJK else "K线图（含技术指标）- {}",
    "ylabel_price": "Price" if not _HAS_CJK else "价格",
    "ylabel_volume": "Volume" if not _HAS_CJK else "成交量",
    "ma5": "MA5",
    "ma20": "MA20",
    "bb_upper": "BB Upper" if not _HAS_CJK else "布林上轨",
    "bb_middle": "BB Middle" if not _HAS_CJK else "布林中轨",
    "bb_lower": "BB Lower" if not _HAS_CJK else "布林下轨",
}


def plot_kline_with_indicators(
    df: pd.DataFrame,
    symbol: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    ma_windows: tuple = (5, 20),
    bb_window: int = 20,
    bb_std: float = 2.0,
    plot_kalman: bool = False,
    kalman_col: Optional[str] = None,
    figsize: tuple = (16, 10),
) -> None:
    """绘制 K 线图，叠加均线、布林带和成交量。

    参数:
        df:          包含 date/open/high/low/close/volume 列的 DataFrame
        symbol:      股票代码（用于标题）
        title:       图表标题（为 None 时自动生成）
        save_path:   保存路径。支持 .html / .png / .jpg / .pdf
        show:        是否显示图表
        ma_windows:  均线窗口，默认 (5, 20)
        bb_window:   布林带窗口，默认 20
        bb_std:      布林带标准差倍数，默认 2.0
        plot_kalman: 是否叠加卡尔曼滤波估计线
        kalman_col:  卡尔曼滤波列名（当 plot_kalman=True 时需要）
        figsize:     图表大小
    """
    # ---- 准备数据 ----
    plot_df = df.copy()
    if "date" in plot_df.columns:
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        plot_df = plot_df.set_index("date")

    # 确保必要列存在
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in plot_df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    plot_df = plot_df[required].sort_index()

    # ---- 构建均线 addplot ----
    apds = []

    # 均线
    colors = ["blue", "orange", "green", "red"]
    for i, w in enumerate(ma_windows):
        ma_label = f"MA{w}"
        ma_col = f"ma{w}"
        if ma_col not in plot_df.columns:
            plot_df[ma_col] = plot_df["close"].rolling(window=w, min_periods=1).mean()
        apds.append(
            mpf.make_addplot(
                plot_df[ma_col],
                color=colors[i % len(colors)],
                width=1.0,
                label=ma_label,
            )
        )

    # 布林带
    bb_mid_col = f"bb_mid_{bb_window}"
    bb_up_col = f"bb_up_{bb_window}"
    bb_lo_col = f"bb_lo_{bb_window}"

    if bb_mid_col not in plot_df.columns:
        mid = plot_df["close"].rolling(window=bb_window, min_periods=1).mean()
        std = plot_df["close"].rolling(window=bb_window, min_periods=1).std()
        plot_df[bb_mid_col] = mid
        plot_df[bb_up_col] = mid + bb_std * std
        plot_df[bb_lo_col] = mid - bb_std * std

    # 布林带用半透明色带
    apds.append(
        mpf.make_addplot(
            plot_df[bb_up_col],
            color="gray",
            width=0.6,
            linestyle="--",
            alpha=0.6,
            label=_LABELS["bb_upper"],
        )
    )
    apds.append(
        mpf.make_addplot(
            plot_df[bb_mid_col],
            color="gray",
            width=0.8,
            linestyle="-",
            label=_LABELS["bb_middle"],
        )
    )
    apds.append(
        mpf.make_addplot(
            plot_df[bb_lo_col],
            color="gray",
            width=0.6,
            linestyle="--",
            alpha=0.6,
            label=_LABELS["bb_lower"],
        )
    )

    # 卡尔曼滤波估计线
    if plot_kalman and kalman_col and kalman_col in plot_df.columns:
        apds.append(
            mpf.make_addplot(
                plot_df[kalman_col],
                color="purple",
                width=1.2,
                label="Kalman",
            )
        )

    # ---- 样式配置 ----
    mc = mpf.make_marketcolors(
        up="red",
        down="green",
        edge="inherit",
        volume={"up": "red", "down": "green"},
        wick="inherit",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        y_on_right=False,
        figcolor="white",
    )

    # ---- 标题 ----
    if title is None:
        title = _LABELS["title_kline"].format(symbol)

    # ---- 保存格式判断 ----
    is_html = save_path and save_path.endswith(".html")

    if is_html:
        # 使用 mplfinance 的 savefig 功能 + plotly 转换
        # mplfinance 原生不支持 HTML，但可以存为 PNG 后用 plotly 重新绘制
        _save_as_html(plot_df, symbol, save_path, ma_windows, bb_window, bb_std)
    else:
        # PNG / JPG / PDF 等静态格式
        fig, axes = mpf.plot(
            plot_df,
            type="candle",
            style=style,
            title=title,
            ylabel=_LABELS["ylabel_price"],
            ylabel_lower=_LABELS["ylabel_volume"],
            volume=True,
            addplot=apds,
            figsize=figsize,
            returnfig=True,
            warn_too_much_data=len(plot_df) + 1,
        )

        # 添加图例
        ax_main = axes[0]
        lines = ax_main.get_lines()
        labels = [l.get_label() for l in lines if l.get_label()]
        if labels:
            ax_main.legend(loc="upper left", fontsize=8)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[图表] 已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def _save_as_html(
    df: pd.DataFrame,
    symbol: str,
    save_path: str,
    ma_windows: tuple = (5, 20),
    bb_window: int = 20,
    bb_std: float = 2.0,
) -> None:
    """使用 plotly 生成交互式 HTML K 线图。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_df = df.copy()
    if "date" in plot_df.columns:
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        plot_df = plot_df.set_index("date")

    # 计算指标
    close = plot_df["close"]
    for w in ma_windows:
        plot_df[f"ma{w}"] = close.rolling(window=w, min_periods=1).mean()
    plot_df["bb_mid"] = close.rolling(window=bb_window, min_periods=1).mean()
    bb_std_series = close.rolling(window=bb_window, min_periods=1).std()
    plot_df["bb_up"] = plot_df["bb_mid"] + bb_std * bb_std_series
    plot_df["bb_lo"] = plot_df["bb_mid"] - bb_std * bb_std_series

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{_LABELS['title_kline'].format(symbol)}", _LABELS["ylabel_volume"]),
    )

    # K 线蜡烛图
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="K线",
            increasing_line_color="red",
            decreasing_line_color="green",
        ),
        row=1,
        col=1,
    )

    # 均线
    colors = ["blue", "orange", "green", "red"]
    for i, w in enumerate(ma_windows):
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[f"ma{w}"],
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1),
                name=f"MA{w}",
            ),
            row=1,
            col=1,
        )

    # 布林带
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["bb_up"],
            mode="lines",
            line=dict(color="gray", width=0.5, dash="dash"),
            name=_LABELS["bb_upper"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["bb_mid"],
            mode="lines",
            line=dict(color="gray", width=0.8),
            name=_LABELS["bb_middle"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["bb_lo"],
            mode="lines",
            line=dict(color="gray", width=0.5, dash="dash"),
            name=_LABELS["bb_lower"],
        ),
        row=1,
        col=1,
    )

    # 成交量
    vol_colors = [
        "red" if plot_df["close"].iloc[i] >= plot_df["open"].iloc[i] else "green"
        for i in range(len(plot_df))
    ]
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["volume"],
            name=_LABELS["ylabel_volume"],
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 布局
    fig.update_layout(
        title=_LABELS["title_kline"].format(symbol),
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date" if not _HAS_CJK else "日期", row=2, col=1)
    fig.update_yaxes(title_text=_LABELS["ylabel_price"], row=1, col=1)
    fig.update_yaxes(title_text=_LABELS["ylabel_volume"], row=2, col=1)

    fig.write_html(save_path)
    print(f"[图表] 已保存交互式 HTML: {save_path}")
