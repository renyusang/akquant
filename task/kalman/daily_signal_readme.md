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
  ├── 峰值台账回写（positions.json 的 peak_close）+ 影子对照（logs/peak_shadow.csv）
  ├── 消息面自动刷新（当日持仓/待买候选，写 score=0 / auto=True 的中性标题）
  ├── 买入信号 → pending_orders.json → 次日开盘成交
  ├── 卖出信号 → pending_orders.json → 次日开盘成交
  └── 输出持仓表 + 交易汇总 + 生成并部署报告

收盘后（人工，建议紧接在运行之后）
  └── 妙想复查 → 覆盖当晚的自动消息面记录 → 重新生成并部署报告
```

**关键规则**：
- 当天生成的买入信号 **不会当天执行**，等次日开盘价
- 涨停买入 → 跳过，保留订单下次再试
- 跌停卖出 → 跳过，保留订单下次再试
- 满仓时新买入信号自动跳过，空出仓位后按偏离度自动补仓
- **消息面复查必须在同一交易日做**：自动刷新只保留"当天"的记录，非当天（含已复查内容）
  次日一律被重抓的中性标题覆盖 —— 补做的复查不会出现在报告里（只留在 git 历史）。
  详见下节 [消息面复查（妙想）](#消息面复查妙想)。

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

## 消息面复查（妙想）

消息面数据在 `news_notes.json`，每条为 `{date, score, summary, auto?}`，由每日运行与人工复查共同维护。

**时序规则（必须遵守，否则白做）**

`news_notes.py` 的刷新策略是：**仅保留"当天"写入的记录，非当天（含此前已复查的评分）一律重抓为
中性新闻标题（`score=0, auto=True`）**。因此：

| 复查时点 | 结果 |
|---|---|
| **同一交易日、运行之后**（推荐） | 评分写入 `news_notes.json` → 重新生成报告即显示评分，可保留到当日结束 |
| 次日或非交易日补做 | 评分只在 git 历史里；次日运行时必然被自动记录覆盖，报告看不到 |

**覆盖范围**：当日**持仓** + **次日待买入候选**（即最近一次运行涉及的标的），不是整个监控池。

**推荐顺序**

```bash
# 1) 例行运行（crontab 20:00 自动，或手动）
bash run_daily.sh

# 2) 逐只妙想复查（对上述范围），写入 news_notes.json
#    格式: {symbol: {"date": "<交易日>", "score": -2..2, "summary": "..."}}  ← 复查记录不带 auto 字段

# 3) 让评分出现在报告页面上（运行时的报告用的是自动记录，需重新生成 + 部署）
python -c "from live_report import build_live_report; build_live_report()"
bash deploy.sh --live-only
```

**评分含义**（报告按此着色）

| 分值 | 含义 |
|---|---|
| +2 | 强烈利好 |
| +1 | 偏多 |
| 0 | 中性 / 多空交织 |
| -1 | 偏空 |
| -2 | 强烈利空（如重大风险警示、权威机构看空） |

**摘要写法**：以 `M-D 涨跌幅收X(浮盈+Y%)` 开头，随后 `;` 分隔要点，风险项前置 `⚠️`。
例：`10-9 **涨停 +9.15%**收35.68(浮盈+9.1%, 周一待加仓1000股); 当日主力净流入17.81亿; ⚠️股价曾自高位腰斩、应收+56%`

---

## 配置文件

编辑 `stocks.yaml`（**以文件本身为准**，下面是精简模板；完整参数与调参依据见 [kalman.md](kalman.md)）：

```yaml
strategy:                     # 全局策略参数
  kalman_q_price: 0.0001
  entry_threshold: 0.02
  exit_threshold: 0.005
  stop_loss_pct: 0.05
  trailing_stop_pct: 0.04     # 峰值回撤止盈(移动止盈; 见 kalman.md 批次3)
  trend_filter_enabled: true
  trend_confirm_bars: 1
  trend_bear_pct: 0.30
  downtrend_entry: 0.04       # 下跌趋势入场偏离(批次2 采纳 3%→4%)
  mfi_filter_enabled: true
  priority_mode: deviation    # 名额优先级(与实盘同口径)
  peak_persist_enabled: false # 峰值跨日持久化(实盘开关; 翻前先过影子期, 见研究计划)

stock:                        # 股票池
  initial_cash: 300000
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

**分池口径**（2026-09-29 修复：此前按"股票池单一口径"汇总全部持仓，ETF 会被算进股票池分母）：

```
================================================================================
           持仓总览 (股票上限 20% = ¥60,000 · ETF上限 20% = ¥20,000)
================================================================================
  代码       名称           股数       成本         市值     占比       浮动盈亏      收益率
  ------------------------------------------------------------------------
  002709   天赐材料        1500     34.69      52110  17.4%       +5618    +12.1%
  513100   纳斯达克ETF     8300      2.28      19920  19.9%       +1208     +6.5%
  512800   银行ETF       22200      0.85      19270  19.3%        -118     -0.6%
  ------------------------------------------------------------------------
  股票池: 市值 ¥169,236 / ¥300,000 = 56.4%    浮动盈亏 ¥-2,223
  ETF池: 市值 ¥39,190 / ¥100,000 = 39.2%    浮动盈亏 ¥+1,381
  两池合计: 市值 ¥208,426    浮动盈亏 ¥-842

  执行日志: 已成交 131 | 未成交(近30天) 0 | 待执行 4 | 已跳过 313
  已完成交易: 54 笔 | 盈利 15 | 亏损 39 | 累计盈亏: ¥-22,899
```

- **占比**按**各自池**的资金计算（ETF 行的分母是 ETF 池资金），`⚠️超标` 也按各自池的单只上限判定
- 配置缺某个池时显示"上限未配置"，不臆造分母
- `未成交(近30天)` 只统计近 30 天的失败单（历史失败行不再长期虚高计数）

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

## 回测

**组合回测（当前入口 = `main.py --portfolio`）**：按 `stocks.yaml` 的完整 watchlist 与资金池配置，模拟 `daily_signal.py` 的 T+1 开盘价执行逻辑，股票池与 ETF 池各跑一次后合并权益与交易。

```bash
# 默认（读 stocks.yaml，2020-01-01 起，至回测当天）
python main.py --portfolio

# 指定区间 + 跑完自动部署报告到服务器
python main.py --portfolio --start 20200101 --end 20261231 --deploy

# 换配置文件
python main.py --portfolio --config other.yaml
```

产物：`report_portfolio.html`（组合汇总：权益曲线 / 回撤 / 月度热力图 / 年度 TOP5 / 交易明细）、
`report_stock.html`、`report_etf.html`，以及 `portfolio_equity.csv` / `portfolio_trades.csv` / `portfolio_positions.csv`。

**单标的回测 / 参数优化 / 组合级扫描**

```bash
python main.py --symbol 002594 [--start 20200101 --end 20251231]   # 单标的(出 kline/report/trades CSV)
python main.py --symbol 002594 --backtest-only                     # 跳过绘图
python main.py --symbol 002594 --walk-forward                      # 滚动窗口
python main.py --symbol 002594 --use-best                          # 应用网格搜索结果
python optimize.py --symbol 002594 [--full]                        # 单标的参数网格(grid_search_*.csv)
python sweep_params.py --param <策略参数> --values v1,v2 \
       --period is,oos,full --pool stock|etf                       # 组合级参数扫描(研究用)
```

> `sweep_params.py` 的 `--param` 要用**策略参数名**（如 `trailing_stop_pct`、`downtrend_entry_threshold`），
> 不是 `stocks.yaml` 里的名字（两者不同名，写错会被静默忽略）；脚本自带"所有取值结果相同就报错"的自检。

**三个必须知道的点**

- 数据右端 = `.catalog` 末根 bar，`--end` 默认取**回测当天**；catalog **不会自动刷新**
  （`load_cached_data` 只在缺失时下载），需手动全历史重下 —— 刷新脚本见 [kalman.md](kalman.md#组合回测)。
  ⚠️ 全历史重下会按**当日 qfq 基准**重算复权，有除权的标的历史价格整体微调，刷新前后数字不完全可比。
- ⚠️ **`portfolio_backtest.py` 是旧脚本**（自造循环、无手续费/涨停保护等），**不参与任何机制结论**，
  仅留档可回滚；回测与参数研究一律用上面的入口。
- 研究结论的采纳门槛（IS/OOS 同向、跨池复现、年度无单年依赖、夏普不恶化、非单调排除过拟合等）
  与各参数的生产值依据见 [kalman.md](kalman.md#已修复问题)。

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
