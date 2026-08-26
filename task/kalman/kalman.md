# 卡尔曼滤波股票交易策略

基于 **2 状态卡尔曼滤波器**（价格 + 速度）+ **MA20 趋势过滤** 的 A 股 / ETF 量化交易系统。支持每日信号扫描、T+1 开盘价执行、持仓管理、回测、参数优化。

---

## 文件结构

```
task/kalman/
│
│  # 核心 — 每日实盘运行
├── daily_signal.py            # 每日批量信号扫描（主入口）
├── stocks.yaml                # 监控配置（股票列表 + 策略参数 + 资金池）
├── run_daily.sh               # crontab 定时执行脚本
│
│  # 信号引擎 — 策略核心（公共逻辑）
├── signal_engine.py           # TrendDetector + SignalEngine（趋势判断 + 买卖信号）
├── kalman_filter.py           # 2 状态卡尔曼滤波器（纯 numpy）
│
│  # 数据与持久化
├── data_utils.py              # 数据下载（股票 + ETF）+ 预处理
├── portfolio.py               # 持仓 / 交易记录读写
├── orders.py                  # 待执行订单管理（T+1 + 涨跌停保护）
├── exec_log.py                # 信号全生命周期日志
│
│  # 运维保障
├── backup.py                  # 运行前自动备份（保留 30 个快照）
├── validate.py                # 数据异常校验（涨跌幅 / 停牌 / 数据滞后）
├── state_check.py             # 状态一致性检查（前后快照对比）
├── manage.py                  # 手动管理工具（持仓 / 回滚 / 备份）
│
│  # 回测
├── strategy.py                # AKQuant 回测策略（委托 SignalEngine）
├── backtest.py                # 回测封装 + 指标 + 买卖点列表
├── main.py                    # 回测 CLI 入口
├── optimize.py                # 参数网格搜索
│
│  # 可视化
├── plot_kline.py              # K 线图（mplfinance + plotly）
│
│  # 组合回测
├── portfolio_backtest.py      # 全量投资组合回测（模拟 daily_signal 完整逻辑）
│
│  # 测试
├── tests/
│   ├── test_kalman_filter.py  # 卡尔曼滤波器单测
│   ├── test_signal_engine.py  # 信号引擎单测
│   ├── test_portfolio.py      # 持仓读写单测
│   ├── test_orders.py         # 订单管理单测
│   └── test_exec_log.py       # 执行日志单测
│
│  # 运行时数据（自动生成）
├── positions.json              # 当前持仓
├── pending_orders.json         # 待执行订单
├── trades.csv                  # 已完成交易
├── execution_log.csv           # 信号执行日志
├── signals.csv                 # 历史信号快照
├── state_snapshot.json         # 状态快照
├── validation_log.csv          # 异常校验日志
│
├── .cache/                     # 数据缓存（parquet）
├── backups/                    # 运行前备份（保留 30 个）
└── logs/                       # 定时任务日志
```

---

## 架构设计

### 信号引擎重构 (2026-07-25)

核心策略逻辑已从三处重复实现提取到 `signal_engine.py`：

```
              ┌─────────────────────────────────┐
              │       signal_engine.py           │
              │  ┌───────────┐ ┌──────────────┐ │
              │  │TrendDetector│ │SignalEngine │ │
              │  │ close<MA20  │ │ KF + 信号    │ │
              │  └───────────┘ └──────────────┘ │
              └──────┬───────┬───────┬──────────┘
                     │       │       │
              ┌──────┘  ┌────┘  ┌──────────┘
              ▼         ▼       ▼
        strategy.py  daily_    portfolio_
        (回测)       signal.py  backtest.py
                     (实盘)     (组合回测)
```

三个消费者现在复用同一套 `SignalEngine`，策略修改只需在一处进行。

### 参数命名兼容

`SignalEngine` 同时支持 `stocks.yaml` 和 `strategy.py` 两种命名约定：

| 功能 | stocks.yaml / daily_signal | strategy.py |
|------|---------------------------|-------------|
| 趋势确认天数 | `trend_confirm_bars` | `trend_filter_confirm_bars` |
| 下跌仓位比例 | `trend_bear_pct` | `trend_bear_position_pct` |
| 下跌买入阈值 | `downtrend_entry` | `downtrend_entry_threshold` |

两种命名均可使用。

---

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
 entry_threshold=2%        downtrend_entry=3%
 position=95%              position=30%
          │                         │
          └────────────┬────────────┘
                       ▼
              ┌──────────────────┐
              │ 卡尔曼信号（战术） │
              │ 价格偏离+速度反转  │
              └──────────────────┘
```

- **趋势判断**：决定能不能买、买多少（战略层）。`TrendDetector` 实现。
- **卡尔曼滤波**：决定什么时候买（战术层）。2 状态 `[价格, 速度]` 模型。

### 卡尔曼滤波器

```
状态向量:  x = [price, velocity]^T
状态转移:  x_k = F @ x_{k-1} + w_k      (F = [[1, 1], [0, 1]])
观测方程:  z_k = H @ x_k + v_k          (H = [1, 0])
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `Q_price` | 1e-4 | 价格过程噪声，越大越不信任模型 |
| `Q_vel` | 1e-5 | 速度过程噪声 |
| `R` | 1e-2 | 观测噪声，越大越不信任观测值 |

### 趋势检测

| 方向 | 条件 | 确认方式 |
|------|------|---------|
| 转跌 | close < MA20 | 连续 N 天确认（默认 1） |
| 转涨 | close ≥ MA20 且 MA20 向上 | 立即恢复 |

MA20 向上作为转涨条件避免了「死猫反弹」— MA20 仍向下时即使 close ≥ MA20 也不转涨。

### 信号优先级

1. **止损**（无条件卖出，最高优先级）
2. **价格回归**：`close < filtered * (1 - exit_threshold)`
3. **速度反转**：速度由正转负（卖出）或由负转正（买入）

---

## 每日信号系统

### 执行时序

```
T日 20:00
  ├── 📦 自动备份所有状态文件
  ├── 🔍 数据校验（涨跌幅异常 / 停牌 / 数据滞后 / 订单停滞）
  ├── ✅ 执行 T-1 日待执行订单（T 日开盘价成交）
  │     涨停买入 → 跳过，保留订单下次执行
  │     跌停卖出 → 跳过，保留订单下次执行
  ├── 📊 下载当日数据，逐只计算信号
  ├── 🟢 买入信号 → pending_orders.json（T+1 日开盘价执行）
  │     仓位满时自动跳过，按偏离度排序等待补仓
  ├── 🔴 卖出信号 → pending_orders.json（T+1 日开盘价执行）
  ├── 📋 输出持仓表 + 交易汇总 + 执行日志
  └── 📸 状态一致性检查 + 快照保存
```

### 资金池管理

股票和 ETF 分池独立，互不影响：

| 池 | 资金 | 最多持仓 | 单只上限 |
|----|------|---------|---------|
| 股票 | ¥300,000 | 5 只 | 20%（¥60,000） |
| ETF | ¥100,000 | 5 只 | 20%（¥20,000） |

> 股票池现金 2026-08-21 从 ¥200,000 调至 ¥300,000（备份 `stocks.yaml.bak_20260821`，变更记录见下方「配置变更记录」）。

- 买入时按池计算可用资金，超出上限自动跳过
- 卖出后自动补仓该池内偏离度最强的候选信号
- 资金不足时记录到 `execution_log.csv`（status=skipped）

### 用法

```bash
conda activate akquant_032
cd /home/renyu/project/opensrc/akquant/task/kalman

# 扫描全部股票
python daily_signal.py

# 静默模式（仅输出 CSV）
python daily_signal.py --quiet

# 不保存到 signals.csv（仅查看信号）
python daily_signal.py --no-save

# 定时执行（每个交易日 20:00）
crontab -e  # 添加: 0 20 * * 1-5 /path/to/run_daily.sh
```

### 信号格式

```
🟢 buy   300750 宁德时代    ¥376.43  仓位 30%  趋势=down  价格突破(偏离4.0%) | 下跌趋势(+3%阈值)
🔴 sell  000933 神火股份    ¥24.98   仓位  0%  趋势=up    价格回归(偏离-1.4%)
⚪ hold  600519 贵州茅台    ¥1320.98 仓位 95%  趋势=up    持仓中 | 趋势=up
✅ 成交  600900 长江电力: 3200股 @ ¥27.75 (信号日 07-20)
```

---

## 配置文件 `stocks.yaml`

```yaml
strategy:                     # 共享策略参数
  kalman_q_price: 0.0001
  kalman_q_vel: 0.00001
  kalman_r: 0.01
  entry_threshold: 0.02
  exit_threshold: 0.005
  stop_loss_pct: 0.05
  use_price_signal: true
  use_velocity_signal: true
  trend_filter_enabled: true
  trend_confirm_bars: 1
  trend_bear_pct: 0.30
  downtrend_entry: 0.03
  adx_filter_enabled: false   # ADX 趋势状态门控(2026-08-02新增,全池验证无增益)
  adx_filter_threshold: 20.0  # ADX<阈值视为震荡,锁定趋势方向(25 更严格)
  adx_filter_period: 14       # ADX 计算周期
  atr_adaptive_exit_enabled: false  # NATR自适应退出(2026-08-26 关闭: 修复引擎下 -34.5pp)
  exit_atr_factor: 1.0              # 退出阈值=max(0.5%, 1.0×NATR)(现关闭, 保留参数)
  sar_exit_enabled: false           # SAR跟踪止损(机制性失效: 追赶慢于回归阈值, 永不触发)
  mfi_filter_enabled: true          # MFI超买过滤(2026-08-25启用,修复超买后组合+57.1pp)
  mfi_overbought: 70.0              # MFI>70抑制买入(资金超买追高过滤)
  trailing_stop_pct: 0.05           # 峰值回撤止盈(2026-08-27启用,修复引擎下+24.1pp最优)

stock:                        # 股票池
  initial_cash: 300000
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

  etfs:                       # ETF 列表（自动使用 Sina 数据源）
    - symbol: "510050"
      name: "上证50ETF"
      type: etf
```

各股可覆盖策略参数：
```yaml
- symbol: "300750"
  name: "宁德时代"
  entry_threshold: 0.03       # 覆盖全局的 0.02
```

---

## 管理命令

```bash
# 查看持仓（含浮动盈亏、待执行订单、候选信号）
python manage.py show

# 手动管理
python manage.py add    002594 比亚迪 800 95.20 2026-07-17
python manage.py remove 002594
python manage.py adjust 002594 500 90.00
python manage.py restore 002594       # 从交易记录恢复

# 备份与回滚
python manage.py backups              # 列出备份
python manage.py rollback             # 回滚到最新
python manage.py rollback 20260723_200000
```

---

## 回测

### 命令行

```bash
# 单次回测
python main.py --symbol 002594
python main.py --symbol 600519 --start 20200101 --end 20251231
python main.py --symbol 002594 --trend-filter          # 启用趋势过滤
python main.py --symbol 002594 --atr-adaptive-exit     # NATR自适应退出宽度(实盘默认启用,回测需显式开启)

# 滚动窗口回测
python main.py --symbol 002594 --walk-forward

# 使用最优参数
python main.py --symbol 002594 --use-best
```

### 参数优化

```bash
# 快速搜索（486 组合，约 3 分钟）
python optimize.py --symbol 002594

# 完整搜索（20736 组合，用于最终优化）
python optimize.py --symbol 002594 --full

# 应用最优参数回测
python main.py --symbol 002594 --use-best
```

### 回测结果（比亚迪 2020-2025）

| 配置 | 收益率 | Sharpe | 最大回撤 |
|------|--------|--------|---------|
| 无趋势过滤 | 26.3% | 0.16 | 50.9% |
| 减仓 30% + DT=0.03 | **42.6%** | **0.31** | **33.7%** |
| 硬封堵 | 20.4% | 0.15 | 44.4% |

### 组合回测

按照 `stocks.yaml` 的完整 watchlist 和资金池配置，模拟 `daily_signal.py` 的 T+1 开盘价执行逻辑，对股票+ETF 组合进行全量历史回测，生成详细 HTML 报告。

```bash
# 默认参数（2020-01-01 ~ 2026-07-24，使用 stocks.yaml）
python portfolio_backtest.py

# 指定时间范围
python portfolio_backtest.py --start 20230101 --end 20260724

# 指定配置文件
python portfolio_backtest.py --config my_stocks.yaml

# 组合使用
python portfolio_backtest.py --start 20240101 --end 20260724 --config my_stocks.yaml
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start` | `20200101` | 起始日期 YYYYMMDD |
| `--end` | `20260724` | 结束日期 YYYYMMDD |
| `--config` | `stocks.yaml` | 配置文件路径 |

**输出文件**：

| 文件 | 说明 |
|------|------|
| `portfolio_report.html` | 交互式 HTML 报告（权益曲线、回撤、月度热力图、年度 TOP5、交易明细） |
| `portfolio_trades.csv` | 全部交易记录 |
| `portfolio_equity.csv` | 每日权益曲线 |

**报告内容**：核心指标、最终持仓、权益走势图、年度汇总、月度收益热力图、年度盈亏 TOP5（股票/ETF 分开）、标的盈亏汇总、交易分析（T+1 配对）、交易明细。

### Python API

```python
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

# 信号引擎
from signal_engine import SignalEngine
engine = SignalEngine(trend_filter_enabled=True, entry_threshold=0.02)
engine.set_position(False, 0.0)
engine.process_history(df)  # 预热
result = engine.update(close, ma20_cur, ma20_prev)
```

---

## 单元测试

### 运行测试

```bash
conda activate akquant_032

# 全部测试
python -m pytest task/kalman/tests/ -v

# 单个模块
python -m pytest task/kalman/tests/test_kalman_filter.py -v
python -m pytest task/kalman/tests/test_signal_engine.py -v
python -m pytest task/kalman/tests/test_portfolio.py -v
python -m pytest task/kalman/tests/test_orders.py -v
python -m pytest task/kalman/tests/test_exec_log.py -v

# 单个测试函数
python -m pytest task/kalman/tests/test_kalman_filter.py::TestKalmanFilter2D::test_initial_update -v
```

### 测试覆盖（68 个）

| 文件 | 覆盖模块 | 关键测试 |
|------|---------|---------|
| `test_kalman_filter.py` | `kalman_filter.py` | 初始值、收敛性、降噪、速度方向、reset、批量平滑、独立信号生成 |
| `test_signal_engine.py` | `signal_engine.py` | 趋势检测器状态转换、参数别名兼容、买卖/止损/持有信号、下跌阈值、预热 |
| `test_portfolio.py` | `portfolio.py` | 持仓增删查、合并加仓均价计算、交易记录读写、汇总统计 |
| `test_orders.py` | `orders.py` | 待执行增删去重、T+1 时序、涨跌停阻断、浮点容差 |
| `test_exec_log.py` | `exec_log.py` | 日志去重、状态转换（pending→executed/failed）、跳过记录 |

所有测试使用临时目录，不会触碰生产数据文件。

---

## 数据文件

| 文件 | 内容 | 更新时机 |
|------|------|---------|
| `positions.json` | 当前持仓 `{symbol: {name, shares, avg_cost, first_buy_date}}` | 买入成交写入，卖出时删除 |
| `pending_orders.json` | 待执行订单 `[{symbol, action, shares, signal_price, signal_date}]` | 信号生成写入，执行后移除 |
| `trades.csv` | 已完成交易（含盈亏） | 卖出执行时追加 |
| `execution_log.csv` | 信号全生命周期（pending→executed/failed/skipped） | 每步实时更新 |
| `signals.csv` | 历史信号快照 | 每次扫描追加（自动去重） |
| `state_snapshot.json` | 状态快照 | 每次扫描后更新 |
| `validation_log.csv` | 数据异常记录 | 发现异常时追加 |

---

## 数据校验

每次 `daily_signal.py` 运行前自动检查：

| 检查项 | 级别 | 说明 |
|--------|------|------|
| 单日涨跌幅超限 | ❌ 错误 | 主板 ±10%、双创 ±20%、北交所 ±30% |
| 成交量为零 | ⚠️ 警告 | 可能停牌 |
| 价格连续 3 天不变 | ⚠️ 警告 | 数据可能未更新 |
| 最新数据滞后 | ⚠️ 警告 | 最新交易日 < 昨天（且为工作日） |
| 订单停滞 > 3 天 | ⚠️ 警告 | 待执行订单未成交卡住 |

---

## 备份与回滚

每次 `daily_signal.py` 运行前自动创建完整快照（positions.json / trades.csv / execution_log.csv / pending_orders.json / signals.csv），保留最近 30 个。

```bash
python manage.py backups     # 列出所有备份
python manage.py rollback    # 一键恢复到最新备份
```

---

## 报告部署

### 服务器

腾讯云 `119.29.88.84:8888`，Nginx 静态文件服务。

| 页面 | URL |
|------|-----|
| 报告导航 | `http://119.29.88.84:8888/` |
| 实盘报告 | `http://119.29.88.84:8888/live/live_report.html` |
| 组合汇总 | `http://119.29.88.84:8888/backtest/portfolio/report_portfolio.html` |
| 股票池 | `http://119.29.88.84:8888/backtest/portfolio/report_stock.html` |
| ETF 池 | `http://119.29.88.84:8888/backtest/portfolio/report_etf.html` |

所有页面顶部有醒目的红色免责声明。

### 部署命令

```bash
# 一键服务器初始化（仅首次）
python setup_server.py         # SSH 密钥 + Nginx + 目录

# 手动部署
bash deploy.sh                 # 同步全部报告
bash deploy.sh --live-only     # 仅实盘报告
bash deploy.sh --backtest-only # 仅回测报告

# 自动部署（已集成）
python daily_signal.py --deploy   # 日频扫描后自动部署实盘报告
python main.py --portfolio --deploy  # 组合回测后自动部署
bash run_daily.sh                  # crontab 已默认开启 --deploy
```

### 页面内容

**导航页 (index.html)**：实盘报告 + 组合回测报告链接。

**实盘报告 (live_report.html)**：每次 `daily_signal.py` 运行后自动生成，包含：
- 核心指标卡片（总收益率、已实现/浮动盈亏、胜率等）
- 权益曲线 + 沪深300 对比 + 买卖点标记
- 月度收益柱状图
- 当前持仓（股票/ETF 分池）
- **待执行订单**（待买入/待卖出/被跳过信号，对齐 `manage.py show`）
- 已完成交易 + 交易明细

**组合回测报告 (report_portfolio.html)**：`main.py --portfolio` 生成，包含权益曲线 + 回撤 + 年度收益 + 月度热力图。

### 部署相关文件

| 文件 | 说明 |
|------|------|
| `deploy.sh` | rsync 同步脚本，含导航页自动生成 |
| `setup_server.py` | 一键服务器初始化（SSH + Nginx + 目录） |
| `nginx-kalman-reports.conf` | Nginx 配置模板（8888 端口 + autoindex） |

---

## 配置变更记录

### 峰值回撤止盈启用 (2026-08-27)
**变更**: 实盘 `stocks.yaml` 与无限仓位 `unlimited.yaml` 同步启用 `trailing_stop_pct: 0.05`（备份 `stocks.yaml.bak_20260827` / `unlimited.yaml.bak_20260827`）。

**第二批止盈方式研究(2026-08-27, 修复引擎)**: ①吊灯 Chandelier(峰值-k×NATR, k=2/3/4)与固定 5% 无差异(NATR×3≈5% 中位数, 自适应宽度大多数时刻重合)——固定 5% 是该族最优简化; ②ADX 衰减止盈触发极少(+21 笔), 被峰值回撤先覆盖; ③**均线止盈(MA10/MA20)**: 收益大幅提升(+61.6/+65.5pp: 169.6%→231.2%/235.1%)但风险显著恶化(夏普 1.57→1.25/1.13, 回撤 -9.6%→-19.0%/-29.2%)——年度稳健性: **MA10 5/7 年改善、无单年大幅回吐、最差年 +8.7%(优于基线 +5.0%)** = 收益-稳健平衡可选; **MA20 收益依赖 2023 单年(+41.5pp)、有负年(-5.7%)** = 不推荐。**MA10 作为可选注释参数加入配置(默认不启用)**, 需更高收益可启用(接受回撤 -19%)。
**依据**: 修复引擎下止盈方式系统研究——**峰值回撤止盈 5% 是唯一有效的止盈结构**（+24.1pp: 145.5%→169.6%, 夏普 1.36→1.57, 回撤 -11.4%→-9.6%, 两次复核一致; 年度 4/7 年改善、无单年大幅恶化）。参数 5% 为清晰拐点(3% 过紧 -40pp / 4-10% 改善 / 15% 回归基线, 非单调排除过拟合)。
**机制**: 持仓峰值回撤 ≥5% 卖出(移动止盈)——趋势上涨中不干预(让利润奔跑), 仅在趋势确认反转(回撤 5%)时锁定浮盈; 优于基线"滤波价回归"(滞后, 实际回撤 10-15% 才触发, 利润坐过山车)。
**其他止盈方式研究结论(均不采用)**: ①固定目标利润止盈(10/20/30%)大幅有害(-66~-90pp)——趋势中途截断右尾收益; ②时间止盈(最大持仓 N 根)温和中性(+5~8pp 但回撤差)——修正口径 bug 后结论从"灾难 -114pp"修正(初版用全局 `_bar_count` 在多 symbol 轮转下高估 ~10 倍, 改 `_hold_bars[symbol]` 后失真消除——回测代码教训又一例); ③trail+固定目标组合被目标止盈拖垮(-88pp)。**趋势策略正确止盈结构 = 移动止盈, 固定目标/时间止盈均截断右尾**。参数 `target_profit_pct`/`max_hold_bars` 保留(默认关闭)。

### NATR 自适应退出关闭 (2026-08-26)
**变更**: 实盘 `stocks.yaml` 与无限仓位 `unlimited.yaml` 同步关闭 `atr_adaptive_exit_enabled`（备份 `stocks.yaml.bak_20260826` / `unlimited.yaml.bak_20260826`）。
**依据**: 修复引擎（超买+预释放回退）下重测 ATR——**关 ATR（MFI-only）+145.5%/1.36/-11.4% vs 当前基线（MFI+ATR）+111.0%/1.04/-13.4%**——**ATR 为 -34.5pp 负贡献**（两次复核一致）。8-04 启用依据（旧超买引擎 +14.1pp）因回测代码缺陷失真——超买引擎持仓 10-19 只时 ATR 放宽退出的代价被分散稀释, 修复引擎（严格 5 只）下单笔权重高, ATR 的"亏损单持有更久"系统性拖累收益（交易 799 vs 1118, 紧阈值快速止损→快速轮换→小赢利累积）。
**SAR 同步验证**: `_update_sar` 调用 1550 次、SAR 卖出 0 次触发——机制性失效（AF 0.02 追赶慢于回归阈值, SAR 恒低于价格 5%+）, 维持关闭。
**教训**: 回测代码正确性直接影响策略决策——超买漏洞/预释放引擎语义先后使 MFI/RSI/ATR 结论反转, 机制结论落地前必须验证回测引擎与实盘语义一致（已入记忆 `backtest-code-correctness`）。

### MFI 超买过滤启用 (2026-08-25)
**变更**: 实盘 `stocks.yaml` 与无限仓位 `unlimited.yaml` 同步启用 `mfi_filter_enabled: true, mfi_overbought: 70`（备份 `stocks.yaml.bak_20260825` / `unlimited.yaml.bak_20260825`）。
**依据**: 修复回测超买漏洞后（2026-08-24）的**真实基线**组合回测——MFI 超买过滤 **+57.1pp（53.9%→111.0%）、夏普 0.59→1.04、回撤 -23.1%→-13.4%**（两次独立复核一致）。旧研究（2026-08-05，超买虚增基线）曾判定"组合层面 -23.2pp 有害"——负效应建立在虚假的超买持仓上（实盘从不超买，daily_signal 两阶段下单有 pending 名额检查）；修复后过滤掉的追高单被质量更高的候选替换，收益质量提升而非数量减少（交易 799 vs 783 基本不变）。
**机制**: MFI>70 视为资金超买（实证快速反转率 71.6% vs MFI<30 时 16.7%），抑制买入信号。
**其他机制复测**（修复后基线, 均维持不启用）: ADX 门控 +2.1pp/回撤 -18.3%（温和改善）、转涨确认2 +6.0pp（交易 +200 笔）、min_hold/SAR 完全持平（SAR 仍被 0.5% 回归罩住）、RSI -7.4pp、转涨确认3 -5.8pp、BBANDS 挤压 -2.1pp。

### 资金池调整 (2026-08-21)
**变更**: 股票池 `stock.initial_cash` ¥200,000 → **¥300,000**（备份 `stocks.yaml.bak_20260821`）。ETF 池不变（¥100,000）。
**原因**: 高价股（如新易盛 ¥442/股）在上涨趋势 95%×20% 口径下，目标金额 200000×0.19=¥38,000 不够 1 手（¥44,200）被跳过；30 万下目标 ¥57,000 可买 1 手。
**连带影响**:
- 单只上限 20% → ¥60,000（原 ¥40,000）；全池目标金额 3.8 万→5.7 万/只（买入量放大 1.5 倍）
- 报告口径: 总初始 ¥300,000→¥400,000，总收益率 = 总盈亏/¥400,000 被稀释（已实现盈亏为 20 万池真实交易所得）；权益曲线起点抬升；持仓占比缩小；组合回测（`run_portfolio_backtest` 读同一 config）口径同步变化
- **限制**: 下跌趋势 30% 仓位下目标金额 = 现金×6%，新易盛 1 手仍需 ¥736,667 现金（30 万买不起，属设计限制）；仅上涨趋势下可买
- 高价股买入能力上限: 30万×0.19=¥57,000 → 股价 ≤¥570/股（100股）可买

### 监控列表变更 (2026-08-14)
**变更**: 股票池 29 只人工 watchlist → **25 只**（科学选股双窗口筛选 + 行业分散，全链路由 screen_pool.py → diversify_recommend.py 完成；备份 `stocks.yaml.bak_20260814`；组合回测收益 +104.1%→+181.6% 全面碾压）。
**当前监控列表 (2026-08-21, 25 股 + 24 ETF)**:
- 股票: 新易盛/光智科技/天孚通信/光库科技/南亚新材/长川科技/香农芯创/天岳先进/中国巨石/佰维存储/迈为股份/胜宏科技/南网科技/深南电路/大族激光/菲利华/指南针/英维克/应流股份/天赐材料/融捷股份/赣锋锂业/铖昌科技/德福科技/长电科技
- ETF: 上证50/沪深300/中证500/中证1000/创业板50/科创板50/科创创业50/创业板/黄金/消费/证券/医疗/军工/光伏/新能源车/银行/新能源/半导体/煤炭/有色金属/恒生科技/纳斯达克/5G通信/红利
- 维护节奏: 季度动态重跑 screen_pool.py 全池筛选，踢垫底、补新候选（见 `task/select/overall_status.md`）

### 数据下载挂起 (2026-08-24)
**问题**: akshare(含 `akquant.utils.fetch_akshare_symbol`)底层 requests 无显式超时——网络挂起时进程无限等待, 8-24 daily_signal 在 20/49 处卡死 10 分钟(CPU 0.4%, 单独下载同一股票仅需 3.8s)。**修复**: `data_utils.py` 模块导入时 `socket.setdefaulttimeout(30)`——socket 级默认超时覆盖全部下载路径(股票/ETF/sina/em), 挂起 30s 抛 `socket.timeout` → `download_with_cache` 捕获后回退缓存数据(有缓存)或报错(无缓存), 不再无限等待。正常下载实测 ~4s, 30s 足够宽裕。新增 4 个测试(共 211 全绿)。

### 回测超买漏洞 (2026-08-24)
**问题**: `strategy.py` 的 `max_positions` 检查形同虚设——①`get_positions()` 的 ctx.positions 快照在多 symbol 回测下 stale(返回空), 同日多标的买入全部放行(T+1 全部成交, 组合回测单日曾 19 只入场、最大持仓 10-19 只); ②卖出下单时立即释放名额, 但卖出订单 T+1 才成交、被拒则持仓保留 → 名额提前释放 → 同日新买入 → 持仓净增。修复前组合回测 +222.2%/0.98/-31.4%/1852 笔均为超买虚增(实盘 daily_signal 自 8-12 两阶段下单起就有 pending 名额检查, 从不超买)。**修复**: ①自维护 `_opened_count`(下单 +1、卖出成交后 -1, 不依赖引擎快照); ②待成交买单用 `get_open_orders()` 查引擎实时状态判断成交/被拒(不能用 bar 窗口——多 symbol 事件流跨 bar 序号会误判超时释放名额); ③待成交卖单登记 `_pending_sells`, 卖出成交后(持仓消失)再释放名额。**验证**: 10 只同日触发集成测试恰好 5 只成交、同时持仓 ≤5; 真实回测 25 只股票池最大同时持仓 10→5、ETF 池 ≤3。修复后组合回测 +53.9%/0.59/-23.1%/783 笔——对齐实盘语义的真实表现, 历史超买口径报告均不可信。新增 9 个测试(共 256 全绿)。

### 名额预释放分析与回退 (2026-08-25/26)
**过程**: ①8-25 提出"名额预释放"修复(名额检查 `opened_count - len(pending_sells)` 计入待成交卖单, 允许卖出+买入同日下单)——基于实盘 8-20 迈为案例(实盘 8-21 买入、回测 8-24 才买), 认为回测"卖出成交 T+1 才释放名额"与实盘"先执行后扫描"不一致。该修复使组合回测 MFI 配置从 +111% 降至 +49.7%(交易 799→530)。②深入归因发现: 修复让"卖出后立即换仓"更顺滑, 在震荡/下行年份(2021-2022, -27pp)无缝换仓更频繁挨打; 同时预释放引擎下机制效果大洗牌(MFI +57pp→-12.7pp 反转为负、RSI -7.4pp→+37.7pp 转正)——均非真实策略变化而是引擎语义产物。③**关键否决(2026-08-26)**: A 股集合竞价挂买入单需资金预冻结——"同日卖出+买入开盘价"在真实交易中**不可行**(卖出未成交时无余额挂买单), 买入最早 T+2 开盘(卖出 T+1 成交资金到账 → T+1 收盘挂单)。**无预释放的"名额延迟"恰好对齐资金正确性(T+2 买入)**, 111% 是资金时序正确的可信基线; 预释放(49.7%)违反真实资金约束。**决策**: 回退名额预释放(名额检查恢复 8-24 版), 保留 `_clean_pending_sells` 被拒兜底(订单消失且持仓保留 → 移除登记不释放名额, 防被拒订单名额挂账)。**迈为 8-20 未成交根因排查**: 回测内 8-20 数据(196.04)迈为信号确为 buy +4.4%(与实盘一致), 未成交原因是回测 8-20 时 opened=5 满仓(回测 1-01 起自动交易新池、持仓结构与实盘旧池+人工不同, 清仓时序错位) → 名额拦截 → 8-24 数据再触发(+3.4%)超数据范围未成交——**非预释放机制问题, 是 watchlist/初始持仓错配的残留差异**。回退后组合回测恢复 +111.0%/1.04/-13.4%/799 笔(MFI 配置)。**配置决策随引擎语义恢复**: MFI 保持启用(无预释放下 +57.1pp 有效), RSI/转涨确认/ADX 的预释放引擎反转结论作废。

## 注意事项

1. 买入信号在当日收盘后生成，**次交易日开盘价成交**，涨停时跳过
2. 已有待执行订单时不重复生成买入信号（卖出信号不受此限制）
3. 趋势过滤默认启用，默认参数适用于大多数场景
4. 股票和 ETF 资金独立，不会跨池串用
5. ETF 使用 Sina 数据源，代码以 `51 / 15 / 58 / 56` 开头
6. 科创板（688）最低买入单位 200 股
7. 环境：`conda activate akquant_032`
8. 定时任务：`crontab -l` 查看，`crontab -r` 取消
9. **回测涨跌停保护(A 方案)**：KalmanStrategy 在 T日 on_bar 生成信号时检查 T日 close 涨跌停,涨停不买/跌停不卖(近似一字涨跌停,严格 close≥前收×1.1)。实盘 `orders.py` 检查 T+1 open 涨停(精确,prev_close=昨日收盘)。注:精确拦 T+1 一字涨跌停在 0.3.20 仍受限(open 强制 NextOpen + NextOpen pending 在 T+1 on_bar 无法 cancel),故用 T日 close 近似;A 方案改变回测(跳过涨停日,改变交易时序)。
10. **真实成交价回填**：人工实际下单后,用 `manage.py fill <symbol> <buy|sell> <shares> <price>` 记录实际价,`daily_signal` 下次执行时优先用此价(而非开盘价假设),使 `avg_cost`/`pnl` 反映真实成交。
11. **warmup 预热**：已设 `warmup_bars=40`,0.3.20 的 warmup 机制已完善,前40 bar 不调 on_bar。
12. **手续费**:回测(`run_backtest` 传 commission_rate=0.0003 / stamp_tax_rate=0.001 / transfer_fee_rate=0.00001 / min_commission=5.0)+ 实盘(`portfolio.calc_fee` 计算,`daily_signal` 买入 avg_cost 含费 / 卖出 pnl 减费)。A股:佣金万3双边最低5元、印花税千1仅卖出、过户费万0.1双边。
13. **ADX 趋势状态门控**:`adx_filter_enabled` 启用后,ADX<阈值视为震荡期,`TrendDetector` 锁定趋势方向(不翻转、确认计数清零),防止震荡期 MA20 方向抖动。**全池验证(2026-08-02, HS300 288 只)**:门控无系统增益——配对胜率 45%、收益差中位数 -0.86pp(接近随机);效果与股票基线特征强相关(相关 -0.58):基线亏损股 56% 改善(均值 +14pp),强势/大牛股 65-87% 恶化;低频交易股(≤30笔)大幅恶化(-32.5pp);ADX≥25 系统性有害(-7.6pp)。**建议不作为全局开关**;若用,仅对"基线回测亏损"的股票启用。验证脚本 `task/select/validate_adx.py`。
14. **震荡期与高频买卖研究(2026-08-03)**:全池信号实证——买入信号质量与 ADX/ER/带宽/NATR 分桶无关(亏损率 47-52% 无差异),"识别震荡期"路线证伪;48.7% 的交易 5 根内快速反转(往返 -3.07%、亏损率 81.7%),贡献 74.9% 总亏损;摩擦成本约 3.5-6pp。ADX 门控/min_hold/exit 放宽均无系统性增益。**BBANDS 挤压过滤**(`bbands_squeeze_enabled`, 带宽<历史均值×0.5 抑制买入)是唯一收益方向一致的机制:单标的全池均值 +2.4pp、中位 -1.4%、夏普中位 0.066;阈值敏感性:0.5 最优(0.3 不触发/0.7 过度抑制,非单调排除过拟合)。**但 HS300 实盘池组合回测(500k/10只/10%):收益 +8pp 但回撤 -18.7%→-23.4%、夏普 1.13→1.10,风险调整后不占优**。**建议默认关闭**,可选启用需接受回撤放大。综合结论:震荡期防高频买卖的结构性改进空间有限,更根本方向是退出机制(ATR 自适应宽度)。研究脚本 `task/select/study_oscillation.py`、`task/select/validate_adx.py`、`task/select/compare_portfolio_squeeze.py`。
15. **NATR 自适应退出宽度(2026-08-04, 已纳入实盘默认配置)**:`atr_adaptive_exit_enabled=true, exit_atr_factor=1.0`——退出阈值 = max(0.5%, 1.0×NATR),高波动期放宽避免被洗出。实证:快速反转亏损随 NATR 单调加深(-1.42%→-4.73%),0.5% 固定阈值在高波动期过紧。**全池(288只):收益均值 +8.5pp、中位数转正 +1.95%(唯一)、夏普中位 0.106、收益差中位 +0.68pp(唯一全正);组合(HS300实盘池):收益 +14.1pp、交易次数 -45%(4502→2481)、回撤 -18.7%→-18.1% 略改善**。代价:单标的全池回撤中位 +5.7pp(组合层不明显)。**已写入 stocks.yaml / stocks_hs300.yaml 默认启用**(SAR 保持关闭);回测 CLI: `--atr-adaptive-exit --exit-atr-factor 1.0`。**SAR 跟踪止损(`sar_exit_enabled`)完全无效(288只全持平)——0.5% 价格回归总是先触发,SAR 被罩住;除非关闭价格回归信号,否则无独立价值**。研究脚本 `task/select/validate_adx.py`、`task/select/compare_portfolio_squeeze.py`。
16. **量价确认研究:MFI 超买过滤(2026-08-05, 默认关闭)**:`mfi_filter_enabled`(MFI>70 抑制买入, `update` 新增 volume 参数)。实证:MFI>70 时买入信号快速反转率 71.6%(vs MFI<30 时 16.7%),资金超买追高质量最差。**单标的全池有效**(回撤最低 25.7%、收益差中位 +0.72pp, 与 atr 组合后均值 +21.8pp), **但组合实盘池有害**(收益 -23.2pp)——组合 max_positions 限制下被过滤信号换成其他标的, 而 MFI>70 恰是趋势最强阶段, 过滤掉组合右尾收益。**与 BBANDS 挤压过滤同模式: 单标的有效、组合负效果, 不纳入实盘默认配置**。四象限研究至此全部完成(趋势/震荡/风控/量价确认), 实盘仅落地 NATR 自适应退出。
17. **转涨确认参数(2026-08-07, 默认1=原行为)**:`trend_recover_confirm_bars`——转涨(下跌→上涨)需连续 M 天 close≥MA20 且 MA20 向上才切换,确认期内保持 down(隐式中间状态,无新增状态值)。全池回测(288只):rc2/rc3 单独**不建议启用**——收益均值 -8pp(延迟转涨错过趋势启动),配对胜率 38-45%,交易次数反增(59→74-80);rc3+ATR 收益中位 +7.6%(全池最高)但主要为 ATR 贡献。**保留为可选参数,不纳入默认配置**。
18. **无限仓位信号系统(2026-08-06~10)**:`task/select/unlimited/`——不限制资金与持股数量,买入数量=target_pct 映射[1,3]手(下跌1手/上涨3手,科创板200股),独立配置/状态/报告,数据缓存共用实盘 .cache。报告含搜索框、归一化收益(总利润/(3手×首笔买入价))、股票/基金分池、信号状态表(待成交置顶)。设计文档 `task/select/unlimited_position.md`,运行 `python unlimited_signal.py --deploy`。

---

## 已修复问题

### 升级到 akquant 0.3.20 (2026-07-25)
**升级**: task/kalman 从 akquant 0.2.22→0.3.20。策略参数声明从 `PARAM_MODEL` + `__init__` 改为类体内联字段(`FloatParam`/`IntParam`/`BoolParam`),兼容 `run_grid_search`。报告 API 从 `result.report()` 改为 `plot_report()`。`Bar.timestamp_str` 改为 `timestamp_iso`。加 `_flush_pending_order_events` 空实现(0.3.20 引擎要求)。实盘 `daily_signal` 同步升级(信号验证一致)。环境从 `akquant_test`(0.2.22)切换到 `akquant_032`(0.3.20,独立 conda 环境,与回退环境隔离)。78 单测全绿。

### 实盘 JSON 状态机增强 (2026-07-25)
**准确性**: 新增 `actual_fills.json` + `manage.py fill` 命令,人工实际成交价可回填,覆盖"按开盘价假设"记账。`orders.execute_pending_orders` 加 `actual_prices` 参数,优先用实际价(提供时跳过涨跌停检查,因人工已实际成交)。`daily_signal` 执行时读 `actual_fills.json`,执行后清理已用条目。向后兼容(无 actual_fills 时用 open,如现状)。
**可恢复性**: `save_positions`/`save_pending`/`save_trades`/`save_actual_fills` 改原子写入(临时文件 + `os.replace`,防中断损坏);`load_positions`/`load_pending` 加结构校验(损坏告警 + 提示 `manage.py rollback`,不静默丢失);`manage.py` add/remove/adjust/restore 操作前自动 `create_backup()`。新增 10 个单元测试(共 78)。

### 信号逻辑提取重构 (2026-07-25)
**重构**: 将 `KalmanStrategy`(strategy.py) / `SignalEvaluator`(daily_signal.py) / `BT_SignalEvaluator`(portfolio_backtest.py) 三份独立的趋势判断+买卖信号逻辑提取到 `signal_engine.py`。修复了 `save_snapshot()` 重复调用。新增 68 个单元测试。

### 卖出信号被错误拦截 (2026-07-24)
**问题**: 当股票已有待执行买单时，新产生的卖出信号被拦截。**修复**: 卖出信号优先级最高，不受已有待执行订单拦截。

### 仓位上限检查误拦卖出信号 (2026-07-24)
**问题**: `已达最大持仓数(N)` 检查对买卖信号都生效。**修复**: 仓位上限检查仅对买入信号生效。

### 待执行订单被同日新信号覆盖 (2026-07-24)
**问题**: 同向信号覆盖旧订单，signal_date 更新为当天，执行延后一天。**修复**: `orders.py` 中同向订单不覆盖，保留最早信号日期。

### 数据新鲜度检查时机错误 (2026-07-24)
**问题**: 检查在数据下载前执行，且阈值使用昨天而非今天，导致开盘前用旧数据错误执行。**修复**: 检查改为 `>= 今天`，执行移到信号扫描之后。

### 剩余资金计算跨池 (2026-07-24)
**问题**: 用全局持仓市值判断剩余资金，ETF 池被股票市值占满。**修复**: 分池统计。

### 涨跌幅校验未区分板块 (2026-07-24)
**问题**: 统一使用阈值，科创板/创业板/N板限制不同。**修复**: 根据代码前缀自动识别（主板 ±10%，双创 ±20%，北交所 ±30%）。

### 科创板最低买入单位 (2026-07-24)
**问题**: 统一按 100 股/手，科创板实际 200 股/手。**修复**: 代码 688 开头使用 200 股。

### 下跌趋势仓位无差异化 (2026-07-24)
**问题**: `min(target_pct, max_pct)` 导致上涨（95%）和下跌（30%）都被截断到 20%。**修复**: 改为 `target_pct × max_pct`。

### 状态一致性检查买入成交误报 (2026-08-11)
**问题**: `state_check.py` 股数变化检查只豁免卖出成交——昨日信号今日开盘成交(如赣锋锂业 200→700、新能源车ETF 3700→11300)被误报"无对应信号";unlimited 系统补仓成交(002460/516160/300274 100→300)同样误报。**修复**: 改为**窗口内买卖成交净变化比对**——从 execution_log 取 `executed` 记录按 `exec_date ≥ 上次快照日期` 过滤(缺失 exec_date 的历史回填/旧快照无 timestamp 时保守视为窗口内),买入股数和减卖出股数须与持仓净变化吻合才豁免;持仓消失检查同步加强(卖出股数 ≥ 原持仓才豁免)。防漏报: 买入 400 却多出 500 股、窗口外成交均仍警告。新增 6 个测试(共 165 全绿)。

### 两阶段下单: 补仓优先 + 新建按偏离度降序 (2026-08-12)
**问题**: 正常扫描按 watchlist 配置顺序先到先得分配名额——配置顺序无业务语义却决定买卖结果(如半导体ETF +9.9% 偏离被排位靠前的创业板50ETF +3.5% 挤掉名额)。**修复**: `evaluate_stock` 增加 `defer_orders` 参数, 主循环两阶段化——阶段 1 评估全部标的只收集候选(`_deferred`), 阶段 2 `_place_deferred_orders` 统一排序下单: 补仓(已持仓)优先于新建仓, 同类型按偏离度降序; 名额检查含已有 pending 新建单(防超买, 补仓单不占名额), 资金检查复用 allocated_cash 防超额。顺带修复: ①已有 pending 的持仓标的不再重复生成补仓候选(原 `add_pending_order` 去重兜底但打印/汇总误导) ②buy_qty=0(买不起 1 手)的标的 signal 改 hold(原保持 buy 落盘 → 报告误显 + state_check "昨日买入未执行" 误报)。新增 9 个测试(共 177 全绿)。unlimited 系统 `evaluate_stock` 默认参数不受影响。

### 月度收益历史月份漂移 (2026-08-13)
**问题**: 权益曲线历史点用"当前价"估算持仓市值——7 月收益隔日从 +0.4% 漂移到 -0.2%(今日价格变化污染历史月份)。**修复**: 新增 `_get_price_history()`(从 .cache 构建 {symbol: {date: 收盘价}}), `_build_equity_curve` 历史点用当日收盘价, 缺失回退当前价, 最终快照仍用当前价——历史月份收益固定, 只有当月随行情变化。新增 4 个测试(共 181 全绿)。

### 实盘报告移动端三处修复 (2026-08-19)
**问题**: ①月度收益热力表(年份+12月+全年共 14 列, 固有宽度 ~880px)是全页 8 张表中唯一未包滚动容器的——手机上被 `body overflow-x:clip` 直接裁剪, 5 月后月份及"全年"列不可见且无法滑到; ②主图 3 子图(权益/回撤/年度)固定 800px 高, 竖屏手机(375px 宽)上回撤/年度子图实际绘图区仅 ~150px, 比例失衡, 横屏手机(844×390)更甚; ③"已完成交易/交易明细"内联 `max-height:600px` 滚动盒与页面竖滚、表格横滚构成三层嵌套滚动, 触屏操作繁琐。**修复**: ①`_monthly_heatmap` 返回值包 `_wrap_table`(与其余 7 张表一致: 横向滚动+首列冻结+iOS 惯性滚动); ②新增 `_chart_box` 容器(data-dh/data-mh 传参)+`_ADAPTIVE_CHART_JS` 自适应脚本——窄屏(宽≤600px 或高≤500px, 后者覆盖横屏手机)时 `Plotly.relayout` 压缩图高(主图 800→560、月度图 300→260), resize 防抖 150ms 双向切换; 子图用分数域(domain)布局, 高度变化时比例自动保持——纯 CSS 缩放 div 只会等比缩小全部元素致文字不可读, 必须 relayout 重算布局; Plotly CDN 失败/图表未初始化/空容器均静默跳过; ③内联滚动样式改 `.vscroll` 类, 桌面保持 600px 滚动盒(补 iOS 惯性滚动), ≤600px 取消内层竖向滚动降为两层。验证: 新增 11 个测试(共 192 全绿, 含临时目录跑完整 `build_live_report` 的端到端冒烟), JS 逻辑经 node mock DOM/Plotly 12 项场景校验(竖屏/桌面/横屏/resize 双向切换/CDN 失败/未初始化图表), 修复前后报告数据内容归一化对比逐字节一致, HTML 标签配对栈式校验通过。已部署。
