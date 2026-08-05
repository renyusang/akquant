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
| 股票 | ¥200,000 | 5 只 | 20%（¥40,000） |
| ETF | ¥100,000 | 5 只 | 20%（¥20,000） |

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
  atr_adaptive_exit_enabled: true   # NATR自适应退出宽度(2026-08-04,默认启用)
  exit_atr_factor: 1.0              # 退出阈值=max(0.5%, 1.0×NATR),高波动期放宽
  sar_exit_enabled: false           # SAR跟踪止损(实证无效:0.5%回归先触发)

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
