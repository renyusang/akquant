# 每日信号系统

基于卡尔曼滤波 + 趋势过滤的 A 股 / ETF 批量信号生成系统。每日收盘后自动扫描监控列表，生成买卖信号，T+1 开盘价执行。

## 快速开始

```bash
conda activate akquant_test
cd /home/renyu/project/opensrc/akquant/task/kalman

# 扫描全部股票
python daily_signal.py

# 查看持仓
python manage.py show

# 定时执行
crontab -e  # 添加: 0 20 * * 1-5 /path/to/run_daily.sh
```

## 文件结构

```
task/kalman/
├── daily_signal.py          # 主程序：每日信号扫描
├── manage.py                # 管理工具：持仓/回滚/备份
├── stocks.yaml              # 配置文件：监控列表+策略参数+资金池
├── run_daily.sh             # 定时执行脚本
│
├── strategy.py              # 回测策略（历史验证用）
├── kalman_filter.py         # 2 状态卡尔曼滤波器（纯 numpy）
├── data_utils.py            # 数据下载（股票+ETF）+ 预处理
│
├── portfolio.py             # 持仓/交易持久化
├── orders.py                # 待执行订单管理（T+1 + 涨跌停保护）
├── exec_log.py              # 信号全生命周期追踪
├── backup.py                # 数据备份与回滚
├── validate.py              # 数据异常校验
├── state_check.py           # 状态一致性检查
│
├── backtest.py              # 回测封装
├── main.py                  # 回测 CLI
├── optimize.py              # 参数网格搜索
├── plot_kline.py            # K 线图
│
├── positions.json           # 当前持仓（自动生成）
├── pending_orders.json      # 待执行订单（自动生成）
├── trades.csv               # 已完成交易（自动生成）
├── execution_log.csv        # 信号执行日志（自动生成）
├── signals.csv              # 历史信号（自动生成）
├── state_snapshot.json      # 状态快照（自动生成）
├── validation_log.csv       # 异常校验日志（自动生成）
│
├── .cache/                  # 数据缓存
├── backups/                 # 运行前备份（保留 30 个）
└── logs/                    # 定时任务日志
```

## 策略设计

### 双层信号模型

```
              ┌──────────────────┐
              │  趋势判断（战略）  │
              │ close<MA20+MA20↑ │
              └────────┬─────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     上涨趋势                    下跌趋势
 entry_threshold=2%        downtrend_entry_threshold=3%
 position=95%              position=30%
          │                         │
          └────────────┬────────────┘
                       ▼
              ┌──────────────────┐
              │ 卡尔曼信号（战术） │
              │ 价格偏离+速度反转  │
              └──────────────────┘
```

- 趋势判断决定能不能买、买多少
- 卡尔曼滤波决定什么时候买
- 下跌趋势中要求更强的信号（3% vs 2%）才允许以 30% 仓位试探

### 趋势检测

| 方向 | 条件 | 确认 |
|------|------|------|
| 转跌 | close < MA20 | 连续 N 天（默认 1） |
| 转涨 | close ≥ MA20 且 MA20 向上 | 立即 |

MA20 向上条件避免"死猫反弹"。

### 信号优先级

1. 止损（无条件卖出）
2. 价格回归卡尔曼估计
3. 速度反转

## 执行时序

```
T日 20:00 收盘后
  ├── 📦 自动备份所有状态文件
  ├── 1. 数据校验（涨跌幅异常、停牌、数据滞后）
  ├── 2. 执行 T-1 日待执行订单（T 日开盘价成交）
  ├── 3. 下载当日数据，计算信号
  ├── 4. 买入信号 → pending_orders.json（T+1 日开盘价执行）
  │     涨停时跳过，顺延到下一个候选
  ├── 5. 卖出信号 → pending_orders.json（T+1 日开盘价执行）
  │     跌停时跳过
  ├── 6. 输出持仓表 + 交易汇总
  └── 7. 状态一致性检查 + 快照保存
```

## 资金池

股票和 ETF 分池管理，互不影响：

| 池 | 资金 | 最多持仓 | 单只上限 |
|----|------|---------|---------|
| 股票 | ¥200,000 | 5 只 | 20% |
| ETF | ¥100,000 | 5 只 | 20% |

买入时按池计算可用资金，超出上限自动跳过。卖出后自动补仓该池内最强的候选信号。

## 配置文件 `stocks.yaml`

```yaml
strategy:                     # 共享策略参数
  kalman_q_price: 0.0001
  kalman_r: 0.01
  entry_threshold: 0.02
  stop_loss_pct: 0.05
  trend_filter_enabled: true
  ...

stock:                        # 股票池
  initial_cash: 200000
  max_positions: 5
  single_position_pct: 0.20

etf:                          # ETF 池
  initial_cash: 100000
  max_positions: 5
  single_position_pct: 0.20

data_years: 2

watchlist:
  stocks:                     # 股票列表
    - symbol: "002594"
      name: "比亚迪"
    - symbol: "600519"
      name: "贵州茅台"

  etfs:                       # ETF 列表
    - symbol: "510050"
      name: "上证50ETF"
      type: etf
```

ETF 使用 Sina 数据源（`fund_etf_hist_sina`），全天候可用。

## 管理命令

```bash
# 查看持仓（含浮动盈亏、待执行、候选信号）
python manage.py show

# 手动管理持仓
python manage.py add 002594 比亚迪 800 95.20 2026-07-17
python manage.py remove 002594
python manage.py adjust 002594 500 90.00

# 备份与回滚
python manage.py backups         # 列出所有备份
python manage.py rollback        # 回滚到最新备份
python manage.py rollback 20260723_200000  # 回滚到指定时间

# 查看帮助
python manage.py --help
```

## 定时执行

```bash
# 设置定时任务（每个交易日 20:00）
crontab -e
# 添加: 0 20 * * 1-5 /home/renyu/project/opensrc/akquant/task/kalman/run_daily.sh

# 查看
crontab -l

# 取消
crontab -r

# 查看日志
ls -lt logs/
```

## 数据文件

| 文件 | 内容 | 更新时机 |
|------|------|---------|
| `positions.json` | 当前持仓 | 买入成交时写入，卖出时删除 |
| `pending_orders.json` | 待执行订单 | 买入/卖出信号生成时写入，执行后清空 |
| `trades.csv` | 已完成交易 | 卖出执行时写入 |
| `execution_log.csv` | 信号全生命周期 | 每个信号生成/执行/跳过时记录 |
| `signals.csv` | 历史信号快照 | 每次扫描时追加 |
| `state_snapshot.json` | 状态快照 | 每次扫描后更新 |
| `validation_log.csv` | 异常校验 | 发现数据异常时记录 |

## 数据校验

每次运行自动检查：

| 检查项 | 级别 | 说明 |
|--------|------|------|
| 涨跌幅超限（主板>10%/双创>20%/北交>30%） | ❌ 错误 | 可能数据错误 |
| 成交量为 0 | ⚠️ 警告 | 可能停牌 |
| 价格连续 3 天不变 | ⚠️ 警告 | 数据未更新 |
| 数据滞后 | ⚠️ 警告 | 最新日期 < 昨天 |
| 订单停滞 > 3 天 | ⚠️ 警告 | 待执行订单卡住 |

异常记录到 `validation_log.csv`。

## 备份与回滚

每次 `daily_signal.py` 运行前自动创建完整快照（positions.json, trades.csv, execution_log.csv, pending_orders.json, signals.csv），保留最近 30 个。出问题时：

```bash
python manage.py rollback    # 一键恢复到运行前状态
```

## 参数优化

回测验证策略参数：

```bash
# 快速网格搜索（405 组合，约 6 分钟）
python optimize.py --symbol 002594

# 使用最优参数回测
python main.py --symbol 002594 --use-best

# 趋势过滤回测
python main.py --symbol 002594 --trend-filter
```

## 注意事项

1. 买入信号 T+1 开盘价成交，涨停跳过
2. 当日数据未就绪时待执行订单保留，不会错误执行
3. 已修复的 bug：卖出信号不会被仓位上限拦截，不会被错误标记为买入
4. 股票和 ETF 资金独立，不会串池
5. 备份文件保留 30 天，自动清理

---

## 已修复问题

### 1. 卖出信号被错误拦截 (2026-07-24 修复)
**问题**: 当股票已有待执行买单时，新产生的卖出信号被拦截，错误记录为 `buy/skipped`，实际卖出原因（如止损、价格回归）被写入 `signal_reason` 字段。
**修复**: 卖出信号优先级最高，不受已有待执行订单拦截 (`daily_signal.py` 行 400)。

### 2. 仓位上限检查误拦卖出信号 (2026-07-24 修复)
**问题**: `已达最大持仓数(N)` 检查对买卖信号都生效，导致满仓时卖出信号也被拦截。
**修复**: 仓位上限检查仅对买入信号生效 (`daily_signal.py` 行 407)。

### 3. 待执行订单被同日新信号覆盖 (2026-07-24 修复)
**问题**: 同一股票再次触发同向信号时，`add_pending_order` 覆盖旧订单，导致 `signal_date` 更新为当天，执行日期被延后一天。
**修复**: 同向订单不覆盖，保留最早的信号日期 (`orders.py`)。

### 4. 数据新鲜度检查时机错误 (2026-07-24 修复)
**问题**: `_check_data_freshness` 在数据下载前执行，缓存清空后必然失败。且检查阈值使用 `>= 昨天` 而非 `>= 今天`，导致开盘前用旧数据错误执行订单。
**修复**: 检查改为 `>= 今天`，执行移到信号扫描之后（缓存已填充）(`daily_signal.py` 行 727, 637)。

### 5. 剩余资金计算跨池 (2026-07-24 修复)
**问题**: 买入数量计算时用全局持仓市值判断剩余资金，导致 ETF 池被股票市值占满而无法买入。
**修复**: 分池计算，只统计同类型持仓 (`daily_signal.py` 行 425)。

### 6. 涨跌幅校验未区分板块 (2026-07-24 修复)
**问题**: 统一使用 15% 阈值，科创板/创业板（±20%）和北交所（±30%）的涨跌停限制不同。
**修复**: 根据股票代码自动识别板块，主板 ±10%，双创 ±20%，北交所 ±30% (`validate.py`)。

### 7. 科创板最低买入单位 (2026-07-24 修复)
**问题**: 所有股票统一按 100 股/手计算，科创板实际为 200 股/手。
**修复**: 根据代码前缀（688xxx）使用 200 股 (`daily_signal.py` 行 432)。

### 8. 下跌趋势仓位无差异化 (2026-07-24 修复)
**问题**: `capped_pct = min(target_pct, max_pct)` 导致上涨（95%）和下跌（30%）都被截断到 `single_position_pct`（20%），没有区别。
**修复**: 改为 `capped_pct = target_pct × max_pct`，上涨 19%、下跌 6%（`daily_signal.py` 行 430）。

### 9. 买卖信号重复生成 (2026-07-23 修复)
**问题**: 同一天多次运行会产生重复的买卖信号，执行日志混乱。
**修复**: 已有待执行订单时不再生成同向信号，CSV 写入时去重。
