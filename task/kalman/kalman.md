# 卡尔曼滤波股票交易策略

## 文件结构

```
task/kalman/
├── kalman.md              # 本文件
├── .gitignore
│
├── stocks.yaml            # 监控配置（股票列表+策略参数+资金管理）
├── daily_signal.py        # 每日批量信号扫描
├── run_daily.sh           # 定时执行脚本（crontab）
├── manage.py              # 持仓手动管理工具
│
├── strategy.py            # KalmanStrategy（回测用）
├── kalman_filter.py       # 2状态卡尔曼滤波器
├── data_utils.py          # 数据下载+预处理（股票+ETF）
├── plot_kline.py          # K线图绘制
├── backtest.py            # 回测封装+指标+买卖点列表
├── main.py                # CLI回测入口
├── optimize.py            # 参数网格搜索
│
├── portfolio.py           # 持仓/交易持久化
├── orders.py              # 待执行订单管理（次日开盘价成交）
│
├── positions.json         # 当前持仓（自动生成）
├── pending_orders.json    # 待执行订单（自动生成）
├── trades.csv             # 已完成交易记录（自动生成）
├── signals.csv            # 历史信号日志（自动生成）
│
├── .cache/                # 数据缓存
└── logs/                  # 定时任务日志
```

---

## 策略架构

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

- **趋势判断**：决定能不能买、买多少（战略层）
- **卡尔曼滤波**：决定什么时候买（战术层）

### 信号优先级

1. 止损（无条件卖出）
2. 价格回归卡尔曼估计
3. 速度反转

---

## 每日信号系统

### 执行时序

```
T日 20:00 收盘后
  ├── 1. 执行 T-1 日的待执行订单（T日开盘价成交）
  ├── 2. 下载当日数据，计算信号
  ├── 3. 买入信号 → pending_orders.json（T+1日开盘价执行）
  ├── 4. 卖出信号 → 立即执行
  └── 5. 输出持仓表+交易汇总
```

### 使用方法

```bash
# 扫描全部股票
python daily_signal.py

# 静默模式
python daily_signal.py --quiet

# 定时执行（crontab）
crontab -e  # 添加: 0 20 * * 1-5 /path/to/run_daily.sh
crontab -l  # 查看
crontab -r  # 取消
```

### 输出示例

```
============================================================
  每日信号扫描  2026-07-21
  监控股票: 12 只
============================================================
  ✅ 成交 600900 长江电力: 3200股 @ ¥27.75 (信号日 07-20)
  ✅ 成交 000933 神火股份: 3700股 @ ¥24.43 (信号日 07-20)
  🟢 buy   300750 宁德时代    ¥376.43  仓位 30%  趋势=down
  ⚪ hold  002594 比亚迪     ¥93.92   仓位 95%  持仓中
  ⚪ hold  600519 贵州茅台    ¥1320.98  仓位 95%  持仓中

===========================================================================
                                 持仓总览
===========================================================================
  代码       名称       股数     成本        市值     浮动盈亏     收益率
  ------------------------------------------------------------------
  000933    神火股份    3700     24.43      90391        +0      +0.0%
  002594    比亚迪      1000     93.49      93920        +0      +0.0%
  600900    长江电力    3200     27.75      88800        +0      +0.0%
  ------------------------------------------------------------------
  持仓市值: ¥273,111    浮动盈亏: ¥+0

  待执行买单 (2 笔，次交易日开盘价成交):
    300750 宁德时代 100股  信号价 ¥376.43  日期 2026-07-21
    600519 贵州茅台 0股  信号价 ¥1320.98  日期 2026-07-21

  已完成交易: 0 笔
===========================================================================
```

### 信号含义

| 图标 | 信号 | 含义 |
|------|------|------|
| 🟢 buy | 买入 | 生成信号，次日开盘价成交（涨停除外） |
| 🔴 sell | 卖出 | 立即卖出全部持仓 |
| ⚪ hold | 持有 | 继续持有或等待信号 |
| ✅ 成交 | — | 待执行订单已成交 |
| ⏳ 待执行 | — | 等待次日数据后执行 |

---

## 持仓管理

### 手动管理

```bash
# 查看持仓
python manage.py show

# 新增持仓
python manage.py add 002594 比亚迪 800 95.20 2026-07-17

# 删除持仓
python manage.py remove 002594

# 调整持仓
python manage.py adjust 002594 500 90.00

# 从交易记录恢复
python manage.py restore 002594
```

### 持仓文件

`positions.json` 记录当前持仓（股数、成本、首次买入日期），买入信号次日成交后自动写入，卖出时自动删除。

`pending_orders.json` 记录待执行买单，下次运行时以开盘价成交。涨停时跳过执行，保留到下次。

`trades.csv` 记录已完成交易（入场/出场/盈亏）。

---

## 配置文件 stocks.yaml

```yaml
defaults:
  initial_cash: 100000           # 初始资金
  max_positions: 5               # 最多同时持有数
  single_position_pct: 0.20      # 单只最大仓位

  kalman_q_price: 0.0001
  kalman_q_vel: 0.00001
  kalman_r: 0.01

  entry_threshold: 0.02
  exit_threshold: 0.005
  stop_loss_pct: 0.05

  trend_filter_enabled: true
  trend_confirm_bars: 1
  trend_bear_pct: 0.30
  downtrend_entry: 0.03

data_years: 2

watchlist:
  - symbol: "002594"
    name: "比亚迪"
  - symbol: "600519"
    name: "贵州茅台"
  - symbol: "510050"             # ETF 示例
    name: "上证50ETF"
    type: etf
```

ETF 使用 `fund_etf_hist_sina` 接口（Sina 源），自动处理列名差异。

---

## 回测

### 命令行参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--symbol` | `002594` | 股票代码 |
| `--start` | `20200101` | 起始日期 |
| `--end` | `20241231` | 结束日期 |
| `--cash` | `100000` | 初始资金 |
| `--trend-filter` | — | 启用趋势过滤 |
| `--trend-confirm` | `1` | 下跌确认天数 |
| `--trend-bear-pct` | `0.30` | 下跌仓位比例 |
| `--downtrend-entry` | `0.03` | 下跌买入阈值 |
| `--use-best` | — | 加载最优参数 |
| `--kalman-q-price` | `1e-4` | 价格过程噪声 |
| `--kalman-r` | `1e-2` | 观测噪声 |
| `--entry-threshold` | `0.02` | 买入偏离阈值 |
| `--exit-threshold` | `0.005` | 卖出偏离阈值 |
| `--stop-loss` | `0.05` | 止损比例 |

### 回测结果（比亚迪 2020-2025）

| 配置 | 收益率 | Sharpe | 回撤 |
|------|--------|--------|------|
| 无趋势过滤 | 26.3% | 0.16 | 50.9% |
| 减仓30%+DT=0.03 | **42.6%** | **0.31** | **33.7%** |
| 硬封堵 | 20.4% | 0.15 | 44.4% |

### 参数优化

```bash
python optimize.py                     # 快速搜索（405组合）
python optimize.py --full              # 完整搜索
python main.py --use-best              # 应用最优参数
```

---

## Python API

```python
# 回测
from data_utils import download_stock_data, preprocess_data
from backtest import run_kalman_backtest, print_metrics

df = download_stock_data("002594", "20240101", "20250715")
df = preprocess_data(df)
result = run_kalman_backtest(df, symbol="002594", strategy_params={
    "trend_filter_enabled": True,
    "trend_bear_position_pct": 0.30,
    "downtrend_entry_threshold": 0.03,
})
print_metrics(result)

# 卡尔曼滤波器
from kalman_filter import KalmanFilter2D
kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)
filtered, velocity = kf.update(115.50)

# 持仓读取
from portfolio import load_positions, get_trade_summary
positions = load_positions()
summary = get_trade_summary()
```

---

## 各模块说明

| 模块 | 功能 |
|------|------|
| `daily_signal.py` | 每日批量信号扫描（待执行订单+信号生成+持仓表） |
| `manage.py` | 持仓手动管理（增删改查） |
| `portfolio.py` | 持仓/交易持久化读写 |
| `orders.py` | 待执行订单管理（涨停检查） |
| `strategy.py` | 回测策略（卡尔曼+趋势过滤） |
| `kalman_filter.py` | 2状态卡尔曼滤波器 |
| `data_utils.py` | 数据下载（股票+ETF）+预处理 |
| `backtest.py` | 回测封装+买卖点列表 |
| `main.py` | 回测CLI入口 |
| `optimize.py` | 参数网格搜索 |
| `plot_kline.py` | K线图绘制 |
| `stocks.yaml` | 监控配置 |
| `run_daily.sh` | 定时执行脚本 |

---

## 注意事项

1. 买入信号在当日收盘后生成，**次交易日开盘价成交**，涨停时跳过
2. 已有待执行订单时不重复生成买入信号
3. 趋势过滤默认启用，默认参数适合大多数场景
4. ETF 使用 Sina 数据源，代码以 `51/58/15/16` 开头
5. 环境：`conda activate akquant_test`
6. 定时任务：`crontab -l` 查看，`crontab -r` 取消
