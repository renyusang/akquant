# 每日信号系统 · 操作手册

基于卡尔曼滤波 + 趋势过滤的 A 股 / ETF 批量信号生成系统。每日收盘后自动扫描监控列表，生成买卖信号，T+1 开盘价执行。

完整架构和策略设计见 [kalman.md](kalman.md)。

---

## 快速开始

```bash
conda activate akquant_032   # 生产环境 (akquant 0.3.20; 旧环境 akquant_test 为 0.2.22 回退环境)
cd /home/renyu/project/opensrc/akquant/task/kalman

python daily_signal.py          # 扫描全部股票
python daily_signal.py --quiet  # 静默模式
python manage.py show           # 查看持仓
```

---

## 每日流程

```
20:00 收盘后 (crontab 触发)
  ├── 备份当前状态（保留 30 个快照）
  ├── 数据校验（涨跌幅异常、停牌、数据滞后）
  ├── 执行昨日待执行订单（今日开盘价成交）
  ├── 逐只扫描，生成买卖信号
  ├── 买入信号 → pending_orders.json → 次日开盘成交
  ├── 卖出信号 → pending_orders.json → 次日开盘成交
  └── 输出持仓表 + 交易汇总
```

**关键规则**：
- 当天生成的买入信号 **不会当天执行**，等次日开盘价
- 涨停买入 → 跳过，保留订单下次再试
- 跌停卖出 → 跳过，保留订单下次再试
- 满仓时新买入信号自动跳过，空出仓位后按偏离度自动补仓

---

## 信号输出

```
🟢 buy   300750 宁德时代    ¥376.43  仓位 30%  趋势=down  价格突破(偏离4.0%) | 下跌趋势(+3%阈值)
🔴 sell  000933 神火股份    ¥24.98   仓位  0%  趋势=up    价格回归(偏离-1.4%)
⚪ hold  600519 贵州茅台    ¥1320.98 仓位 95%  趋势=up    持仓中 | 趋势=up
```

| 图标 | 信号 | 含义 |
|------|------|------|
| 🟢 buy | 买入 | 已生成 pending order，次日开盘价成交 |
| 🔴 sell | 卖出 | 已生成 pending order，次日开盘价成交 |
| ⚪ hold | 持有 | 继续持有或等待信号 |
| ✅ 成交 | — | 待执行订单已成交 |
| ⏳ 待执行 | — | 订单等待次日数据后执行 |

---

## 配置文件

编辑 `stocks.yaml`：

```yaml
strategy:                     # 全局策略参数
  kalman_q_price: 0.0001
  entry_threshold: 0.02
  stop_loss_pct: 0.05
  trend_filter_enabled: true
  trend_confirm_bars: 1
  trend_bear_pct: 0.30
  downtrend_entry: 0.03

stock:                        # 股票池
  initial_cash: 200000
  max_positions: 5
  single_position_pct: 0.20

etf:                          # ETF 池
  initial_cash: 100000
  max_positions: 5
  single_position_pct: 0.20

watchlist:
  stocks:                     # 监控列表
    - symbol: "002594"
      name: "比亚迪"
  etfs:
    - symbol: "510050"
      name: "上证50ETF"
      type: etf
```

各股可单独覆盖策略参数：
```yaml
- symbol: "300750"
  name: "宁德时代"
  entry_threshold: 0.03       # 覆盖全局
```

---

## 管理命令

```bash
# 查看持仓
python manage.py show

# 手动管理持仓
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

## 定时执行

```bash
# 设置（每个交易日 20:00）
crontab -e
# 添加: 0 20 * * 1-5 /path/to/task/kalman/run_daily.sh

# 查看
crontab -l

# 取消
crontab -r

# 查看日志
ls -lt logs/
```

---

## 持仓总览示例

```
================================================================================
  持仓总览 (上限 20% = ¥40,000)
================================================================================
  代码       名称       股数     成本        市值     占比     浮动盈亏     收益率
  ------------------------------------------------------------------
  000933    神火股份    1600     24.85      39968  20.0%       +208    +0.5%
  300750    宁德时代     100    372.67      38301  19.2%      +1034    +2.8%
  600900    长江电力    1300     29.00      37570  18.8%       -130    -0.3%
  ------------------------------------------------------------------
  持仓市值: ¥131,786 / ¥200,000 = 65.9%    浮动盈亏: ¥+773

  执行日志: 已成交 5 | 未成交 0 | 待执行 2 | 已跳过 4
  已完成交易: 2 笔 | 盈利 0 | 亏损 2 | 累计盈亏: ¥-3,788
```

---

## 实盘报告

每次运行 `daily_signal.py` 后自动生成 `live_report.html`，基于实际交易数据（positions.json / trades.csv）：

| 板块 | 内容 |
|------|------|
| 核心指标 | 总收益率、已实现/浮动盈亏、现金/持仓/权益 |
| 权益走势 | 累积权益曲线 + 卖出点标记 |
| 月度收益 | 月度收益热力图 |
| 当前持仓 | 股票/ETF 分开，含成本/现价/浮动盈亏 |
| 已完成交易 | 全部平仓记录，含盈亏 |

---

## 组合回测

对 `stocks.yaml` 的全部标的进行历史回测，验证策略有效性：

```bash
# 默认参数
python portfolio_backtest.py

# 指定时间和配置
python portfolio_backtest.py --start 20230101 --end 20260724 --config stocks.yaml
```

生成的 `portfolio_report.html` 包含权益曲线、月度热力图、年度 TOP5 等详细分析。详见 [kalman.md](kalman.md#组合回测)。

---

## 故障排查

### 数据未就绪
当日数据还未更新时，待执行订单会保留不执行：
```
⚠️ 当日数据未就绪，跳过待执行订单（需 2026-07-25 数据）
⏳ 2 笔待执行订单等待数据更新后成交
```
这是正常行为，等数据更新后下次运行自动执行。

### 数据异常
以下情况会触发警告：

| 现象 | 原因 | 处理 |
|------|------|------|
| 涨跌幅异常 | 可能数据错误 | 检查 `validation_log.csv` |
| 成交量为 0 | 可能停牌 | 该股当日跳过 |
| 价格连续不变 | 数据未更新 | 等待数据源恢复 |
| 订单停滞 > 3 天 | 连续涨停/跌停 | 手动检查 `manage.py show` |

### 回滚状态
出问题时回滚到运行前状态：
```bash
python manage.py backups     # 查看可用备份
python manage.py rollback    # 恢复到最新
```

---

## 相关文档

- [kalman.md](kalman.md) — 完整架构、策略设计、回测、单元测试
- [策略参数](stocks.yaml) — 配置文件模板
