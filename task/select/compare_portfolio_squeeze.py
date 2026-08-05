"""HS300 实盘池组合回测: squeeze 开/关对比。

用 .catalog 本地数据替代网络下载(monkeypatch prepare_data),
对比 stocks_hs300.yaml 配置(500k/10只/10%)在 bbands_squeeze_enabled
开/关下的组合表现。

用法: python compare_portfolio_squeeze.py
"""

import contextlib
import io
import os
import sys

import pandas as pd

KALMAN_DIR = os.path.join(os.path.dirname(__file__), "..", "kalman")
CATALOG_DIR = os.path.join(KALMAN_DIR, ".catalog")
sys.path.insert(0, KALMAN_DIR)

import portfolio_backtest as pb  # noqa: E402


def _local_prepare_data(config, start, end):
    """用 .catalog 本地 parquet 替代网络下载。"""
    all_data = {}
    watchlist = config["watchlist"]
    for stock in watchlist.get("stocks", []):
        sym = stock["symbol"]
        path = os.path.join(CATALOG_DIR, sym, "data.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)  # .catalog: date 为索引
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if len(df) > 0:
            all_data[sym] = df
    dates = sorted(set().union(*[set(d.index) for d in all_data.values()]))
    print(f"数据就绪: {len(all_data)} 只标的, {len(dates)} 个交易日(本地.catalog)")
    return all_data, dates


def run(config_path, tag):
    pb.prepare_data = _local_prepare_data
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pb.run_backtest(start_date="20200101", end_date="20260724",
                        config_path=config_path)
    out = buf.getvalue()
    print(f"\n{'='*50}\n[{tag}]\n{'='*50}")
    for line in out.splitlines():
        if any(k in line for k in (
            "总收益率", "年化收益", "夏普", "最大回撤", "总交易数",
            "胜率", "最终持仓", "沪深300", "验证", "报告",
        )):
            print(f"  {line.strip()}")


def main():
    import yaml

    base_path = os.path.join(KALMAN_DIR, "stocks_hs300.yaml")
    atr_path = os.path.join(os.path.dirname(__file__), "stocks_hs300_atr.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["strategy"]["atr_adaptive_exit_enabled"] = True
    cfg["strategy"]["exit_atr_factor"] = 1.0
    cfg["strategy"]["mfi_filter_enabled"] = True
    cfg["strategy"]["mfi_overbought"] = 70.0
    with open(atr_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print("=== 基线(stocks_hs300.yaml, 含atr自适应退出) ===")
    run(base_path, "基线(实盘默认)")
    print("\n=== +MFI超买过滤 ===")
    run(atr_path, "atr+MFI70")


if __name__ == "__main__":
    main()
