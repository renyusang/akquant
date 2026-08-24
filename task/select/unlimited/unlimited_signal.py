"""无限仓位信号发生器。

不限制资金和持股数量, 买入数量 = target_pct 映射到 [1,3] 手(交易所最小单位)。
与 daily_signal.py 相比仅解除资金/持仓数限制, 其余功能一致:
T+1 次日开盘执行、涨跌停保护、备份/校验/快照、报告部署。

独立运行: 状态文件(positions/pending/trades/execution_log/signals)在本目录,
数据缓存共用实盘 .cache(只读, 不重复下载)。

用法:
    python unlimited_signal.py                 # 扫描 + 信号落盘
    python unlimited_signal.py --quiet         # 静默
    python unlimited_signal.py --deploy        # 报告部署到服务器 /unlimited/
"""

import argparse
import os
import sys
from datetime import datetime

UNLIMITED_DIR = os.path.dirname(os.path.abspath(__file__))
KALMAN_DIR = os.path.normpath(os.path.join(UNLIMITED_DIR, "..", "..", "kalman"))
sys.path.insert(0, KALMAN_DIR)

# 复用实盘模块, 重定向状态文件到独立目录(数据缓存保持实盘 .cache)
import daily_signal as ds  # noqa: E402
import backup  # noqa: E402
import exec_log  # noqa: E402
import orders  # noqa: E402
import portfolio  # noqa: E402
import state_check  # noqa: E402

ds.TASK_DIR = UNLIMITED_DIR
ds.CONFIG_FILE = os.path.join(UNLIMITED_DIR, "unlimited.yaml")
ds.SIGNALS_CSV = os.path.join(UNLIMITED_DIR, "signals.csv")
portfolio.POSITIONS_FILE = os.path.join(UNLIMITED_DIR, "positions.json")
portfolio.TRADES_FILE = os.path.join(UNLIMITED_DIR, "trades.csv")
orders.PENDING_FILE = os.path.join(UNLIMITED_DIR, "pending_orders.json")
orders.ACTUAL_FILLS_FILE = os.path.join(UNLIMITED_DIR, "actual_fills.json")
exec_log.EXEC_LOG = os.path.join(UNLIMITED_DIR, "execution_log.csv")
backup.TASK_DIR = UNLIMITED_DIR
backup.BACKUP_DIR = os.path.join(UNLIMITED_DIR, "backups")
state_check.TASK_DIR = UNLIMITED_DIR
state_check.SNAPSHOT_FILE = os.path.join(UNLIMITED_DIR, "state_snapshot.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无限仓位信号发生器")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存 signals.csv(仅查看)")
    parser.add_argument("--deploy", action="store_true",
                        help="报告部署到服务器 /unlimited/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 显式传路径: load_config 的默认参数在 import 时绑定实盘 stocks.yaml,
    # 依赖默认值会读到实盘配置(2026-08-06 教训: 无限仓位配置未生效)
    config = ds.load_config(ds.CONFIG_FILE)
    data_years = int(config.get("data_years", 2))
    watchlist_raw = config.get("watchlist", {})
    if isinstance(watchlist_raw, list):
        stocks_list, etfs_list = watchlist_raw, []
    else:
        stocks_list = watchlist_raw.get("stocks", [])
        etfs_list = watchlist_raw.get("etfs", [])
    all_watchlist = stocks_list + etfs_list
    if not all_watchlist:
        print("错误: watchlist 为空")
        sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")
    backup_tag = backup.create_backup()
    if not args.quiet:
        print(f"  📦 备份: {backup_tag}")
        print(f"\n{'=' * 60}")
        print(f"  无限仓位信号扫描  {today}")
        print(f"  股票: {len(stocks_list)} 只  ETF: {len(etfs_list)} 只  "
              f"(共 {len(all_watchlist)} 只)")
        print(f"{'=' * 60}")

    # 数据校验
    data_issues, order_issues = ds.run_all_checks(all_watchlist, quiet=args.quiet)
    all_validation_issues = data_issues + order_issues

    # 并行预取数据(2026-08-24, 复用实盘实现): 扫描前并行下载缺失/过期数据
    ds.prefetch_data(all_watchlist, data_years, quiet=args.quiet)

    # 执行待处理订单(T-1 信号, 今日开盘价成交, 涨跌停保护)
    ds._execute_today_pending(config, today)

    # 扫描全部标的(无限仓位: 无池满/资金跳过, 信号直接生成订单)
    results = []
    for i, stock in enumerate(all_watchlist, 1):
        symbol = stock["symbol"]
        name = stock.get("name", symbol)
        if not args.quiet:
            print(f"\n[{i}/{len(all_watchlist)}] {symbol} {name} ...")
        try:
            result = ds.evaluate_stock(stock, config, data_years)
            results.append(result)
            if not args.quiet:
                print(ds.format_signal(result))
        except Exception as e:
            results.append({"symbol": symbol, "name": name,
                            "error": str(e), "signal": "error"})
            if not args.quiet:
                print(f"  ❌ 错误: {e}")

    if not args.quiet:
        ds.print_summary(results)

    if not args.no_save:
        valid = [r for r in results if not r.get("error")]
        if valid:
            ds.append_signals_csv(valid)
            if not args.quiet:
                print(f"\n信号已保存: {ds.SIGNALS_CSV} ({len(valid)} 条)")

    # 校验输出
    if all_validation_issues:
        errors = [i for i in all_validation_issues if i["level"] == "error"]
        warns = [i for i in all_validation_issues if i["level"] == "warn"]
        print(f"\n⚠️ 数据校验: {len(errors)} 错误 {len(warns)} 警告")
        for i in errors[:5]:
            print(f"  ❌ {i['symbol']} {i.get('name', '')}: {i['detail']}")
        for i in warns[:5]:
            print(f"  ⚠️ {i['symbol']} {i.get('name', '')}: {i['detail']}")

    # 持仓总览 + 状态一致性 + 快照
    if not args.quiet:
        ds.print_position_summary()
        state_check.print_check_result()
    state_check.save_snapshot()

    # 报告(阶段3: 搜索框 + 归一化收益)
    try:
        from unlimited_report import build_unlimited_report
        build_unlimited_report()
        if not args.quiet:
            print("  报告: unlimited_report.html")
    except Exception as e:
        if not args.quiet:
            print(f"  ⚠️ 报告生成失败: {e}")

    # 部署
    if args.deploy:
        try:
            import subprocess
            subprocess.run(["bash", os.path.join(UNLIMITED_DIR, "deploy.sh"),
                            "--unlimited"], check=False)
        except Exception as e:
            print(f"  ⚠️ 部署失败: {e}")


if __name__ == "__main__":
    main()
