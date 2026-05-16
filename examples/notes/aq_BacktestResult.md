# aq.BacktestResult 完整变量与方法参考

> 基于 `python/akquant/backtest/result.py` 和 `src/analysis/` (Rust) 源码分析。
> `BacktestResult` 由 `aq.run_backtest()` 返回，封装了回测的所有结果数据。

---

## 一、实例属性（构造函数注入）

| 属性 | 类型 | 用途 |
|------|------|------|
| `initial_cash` | `float` | 回测初始资金 |
| `strategy` | `Optional[Any]` | 策略实例引用（warm start 时可通过它访问策略状态） |
| `engine` | `Optional[Any]` | 引擎实例引用（可访问 `engine.portfolio.cash` 等内部状态） |
| `analyzer_outputs` | `dict[str, dict[str, Any]]` | 分析器插件输出字典 |
| `resolved_execution_policy` | `Optional[dict[str, Any]]` | 实际生效的执行策略配置 |

---

## 二、曲线数据（Property）

### 原始频率

| 属性 | 返回类型 | 用途 |
|------|---------|------|
| `equity_curve` | `pd.Series` | 权益曲线。Index 为带时区的 Datetime，Values 为总权益 |
| `cash_curve` | `pd.Series` | 现金曲线。Index 为带时区的 Datetime，Values 为可用现金 |
| `margin_curve` | `pd.Series` | 保证金曲线。Index 为带时区的 Datetime，Values 为已用保证金 |
| `daily_returns` | `pd.Series` | 日收益率序列（从 `equity_curve` 按日重采样并计算 `pct_change`） |

### 日频

| 属性 | 返回类型 | 用途 |
|------|---------|------|
| `equity_curve_daily` | `pd.Series` | 每日末权益曲线 |
| `cash_curve_daily` | `pd.Series` | 每日末现金曲线 |
| `margin_curve_daily` | `pd.Series` | 每日末保证金曲线 |

---

## 三、性能指标

### `result.metrics` — PerformanceMetrics 对象（属性访问）

包装了 Rust 侧的 `PerformanceMetrics`，可通过 `.` 访问各字段。`start_time` 和 `end_time` 自动转为带时区的 Datetime。

| 字段 | 类型 | 含义 |
|------|------|------|
| `total_return` | `float` | 总收益（数值），如 `0.15` = 15% |
| `total_return_pct` | `float` | 总收益率（百分比），如 `15.0` = 15% |
| `annualized_return` | `float` | 年化收益率 |
| `volatility` | `float` | 年化波动率 |
| `sharpe_ratio` | `float` | 夏普比率 |
| `sortino_ratio` | `float` | 索提诺比率 |
| `calmar_ratio` | `float` | 卡玛比率（年化收益/最大回撤） |
| `max_drawdown` | `float` | 最大回撤比率（小数） |
| `max_drawdown_value` | `float` | 最大回撤金额（正值表示回撤幅度） |
| `max_drawdown_pct` | `float` | 最大回撤百分比（正值，如 `15.0` = 15%） |
| `ulcer_index` | `float` | 溃疡指数 |
| `upi` | `float` | 溃疡绩效指数（Ulcer Performance Index） |
| `equity_r2` | `float` | 权益曲线 R²（线性拟合度） |
| `std_error` | `float` | 权益曲线线性回归标准误差 |
| `win_rate` | `float` | 胜率（百分比） |
| `initial_market_value` | `float` | 初始市值（默认 = `initial_cash`） |
| `end_market_value` | `float` | 结束市值 |
| `start_time` | `datetime` | 回测起始时间（带时区） |
| `end_time` | `datetime` | 回测结束时间（带时区） |
| `duration` | `timedelta` / `int` | 回测持续时间 |
| `total_bars` | `int` | 总 K 线数 |
| `exposure_time_pct` | `float` | 市场暴露时间百分比 |
| `var_95` | `float` | 95% 置信度 VaR（在险价值） |
| `var_99` | `float` | 99% 置信度 VaR |
| `cvar_95` | `float` | 95% 置信度 CVaR（条件在险价值） |
| `cvar_99` | `float` | 99% 置信度 CVaR |

### `result.trade_metrics` — TradePnL 对象（通过 `__getattr__` 委托到 `_raw.trade_metrics`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `gross_pnl` | `float` | 总毛利（扣除佣金前） |
| `net_pnl` | `float` | 总净利（扣除佣金后） |
| `total_commission` | `float` | 总佣金 |
| `total_closed_trades` | `int` | 总平仓交易次数 |
| `won_count` | `int` | 盈利次数 |
| `lost_count` | `int` | 亏损次数 |
| `won_pnl` | `float` | 盈利总额 |
| `lost_pnl` | `float` | 亏损总额 |
| `win_rate` | `float` | 胜率（百分比） |
| `loss_rate` | `float` | 亏损率（百分比） |
| `unrealized_pnl` | `float` | 未实现盈亏 |
| `avg_pnl` | `float` | 平均单笔盈亏 |
| `avg_return_pct` | `float` | 平均单笔收益率（百分比） |
| `avg_trade_bars` | `float` | 平均持仓 K 线数 |
| `avg_profit` | `float` | 平均单笔盈利 |
| `avg_profit_pct` | `float` | 平均单笔盈利百分比 |
| `avg_winning_trade_bars` | `float` | 平均盈利交易持仓 K 线数 |
| `avg_loss` | `float` | 平均单笔亏损 |
| `avg_loss_pct` | `float` | 平均单笔亏损百分比 |
| `avg_losing_trade_bars` | `float` | 平均亏损交易持仓 K 线数 |
| `largest_win` | `float` | 最大单笔盈利 |
| `largest_win_pct` | `float` | 最大单笔盈利百分比 |
| `largest_win_bars` | `float` | 最大单笔盈利持仓 K 线数 |
| `largest_loss` | `float` | 最大单笔亏损 |
| `largest_loss_pct` | `float` | 最大单笔亏损百分比 |
| `largest_loss_bars` | `float` | 最大单笔亏损持仓 K 线数 |
| `max_wins` | `int` | 最大连胜次数 |
| `max_losses` | `int` | 最大连败次数 |
| `profit_factor` | `float` | 盈亏比（总盈利/总亏损） |
| `total_profit` | `float` | 总盈利（同 `won_pnl`） |
| `total_loss` | `float` | 总亏损（同 `lost_pnl`） |
| `sqn` | `float` | 系统质量数（System Quality Number） |
| `kelly_criterion` | `float` | 凯利公式比例 |

> **兼容写法**：`result.metrics.sharpe_ratio` 和 `result.trade_metrics.sharpe_ratio` 均可访问。`metrics_df` 中包含两者的并集。

### `result.metrics_df` — 指标 DataFrame

```python
# 返回格式：
#                     value
# name
# start_time          datetime
# end_time            datetime
# duration            timedelta
# total_bars          1000
# trade_count         5
# initial_market_value 1000000.0
# end_market_value     1150000.0
# total_pnl           150000.0
# ...
```

按指标名索引，`value` 列为对应值。比 `metrics` 多了以下衍生指标：
- `max_leverage` — 最大杠杆
- `min_margin_level` — 最低保证金水平

---

## 四、交易数据

### `result.trades` — 平仓交易原始对象列表

`List[ClosedTrade]`，每个元素含以下属性：

| 属性 | 类型 | 含义 |
|------|------|------|
| `symbol` | `str` | 标的代码 |
| `entry_time` | `int` | 开仓时间（纳秒时间戳） |
| `exit_time` | `int` | 平仓时间（纳秒时间戳） |
| `entry_price` | `float` | 开仓均价 |
| `exit_price` | `float` | 平仓均价 |
| `quantity` | `float` | 交易数量 |
| `side` | `str` | 方向（`"long"` / `"short"`） |
| `pnl` | `float` | 毛利（Gross PnL） |
| `net_pnl` | `float` | 净利（扣除佣金后） |
| `return_pct` | `float` | 收益率（小数，`0.05` = 5%） |
| `commission` | `float` | 佣金 |
| `duration_bars` | `int` | 持仓 K 线数 |
| `duration` | `int` | 持仓时长（纳秒） |
| `mae` | `float` | 最大不利变动（MAE） |
| `mfe` | `float` | 最大有利变动（MFE） |
| `entry_tag` | `str` | 入场订单标签 |
| `exit_tag` | `str` | 出场订单标签 |
| `entry_portfolio_value` | `float` | 入场时组合总权益 |
| `max_drawdown_pct` | `float` | 持仓期间最大回撤百分比 |

### `result.trades_df` — 平仓交易 DataFrame

上表的 DataFrame 版本，时间列已转为带时区的 Datetime，`duration` 转为 `Timedelta`。

---

## 五、订单数据

### `result.orders` — 订单原始对象列表

`List[Order]`，通过 `__getattr__` 委托到 `_raw.orders`。

### `result.orders_df` — 订单 DataFrame（cached_property）

| 列名 | 类型 | 含义 |
|------|------|------|
| `id` | `str` | 订单 ID |
| `symbol` | `str` | 标的代码 |
| `side` | `str` | 方向（`"buy"` / `"sell"`） |
| `order_type` | `str` | 订单类型（`"market"` / `"limit"` / `"stop"` / `"stop_limit"` / `"stop_trail"`） |
| `quantity` | `float` | 委托数量 |
| `filled_quantity` | `float` | 已成交数量 |
| `limit_price` | `float` | 限价（NaN = 市价） |
| `stop_price` | `float` | 触发价（NaN = 无） |
| `avg_price` | `float` | 成交均价 |
| `commission` | `float` | 手续费 |
| `status` | `str` | 状态（`"filled"` / `"cancelled"` / `"rejected"` / `"open"` / `"pending"`） |
| `time_in_force` | `str` | 有效期（`"gtc"` / `"day"` / `"ioc"` / `"fok"`） |
| `created_at` | `datetime` | 创建时间 |
| `updated_at` | `datetime` | 更新时间 |
| `tag` | `str` | 订单标签 |
| `reject_reason` | `str` | 拒单原因（空字符串 = 未拒绝） |
| `owner_strategy_id` | `str` | 所属策略 ID（多策略 slot 场景） |

**衍生列**（Python 侧自动计算）：
- `filled_value` — 成交金额（`filled_quantity * avg_price`）
- `duration` — 存续时长（`updated_at - created_at`）

---

## 六、成交数据

### `result.executions_df` — 成交报告 DataFrame（cached_property）

| 列名 | 类型 | 含义 |
|------|------|------|
| `id` | `str` | 成交 ID |
| `order_id` | `str` | 关联订单 ID |
| `symbol` | `str` | 标的代码 |
| `side` | `str` | 方向 |
| `quantity` | `float` | 成交数量 |
| `price` | `float` | 成交价格 |
| `commission` | `float` | 手续费 |
| `timestamp` | `datetime` | 成交时间 |
| `bar_index` | `int` | 所在 Bar 序号 |
| `owner_strategy_id` | `str` | 所属策略 ID |

---

## 七、持仓数据

### `result.positions` — 持仓历史（宽表）

`pd.DataFrame`，Index 为带时区 Datetime，Columns 为各标的 Symbol，Values 为持仓数量。

### `result.positions_df` — 持仓明细 DataFrame（长表）

| 列名 | 类型 | 含义 |
|------|------|------|
| `symbol` | `str` | 标的代码 |
| `date` | `datetime` | 快照时间 |
| `long_shares` | `float` | 多头持仓数量 |
| `short_shares` | `float` | 空头持仓数量 |
| `close` | `float` | 收盘价 |
| `equity` | `float` | 账户权益 |
| `market_value` | `float` | 持仓市值 |
| `margin` | `float` | 占用保证金 |
| `unrealized_pnl` | `float` | 未实现盈亏 |
| `entry_price` | `float` | 开仓均价 |

---

## 八、强平审计

### `result.liquidation_audit_df` — 强平审计 DataFrame

| 列名 | 类型 | 含义 |
|------|------|------|
| `timestamp` | `datetime` | 强平时间 |
| `date` | `str` | 强平日期 |
| `daily_interest` | `float` | 当日利息 |
| `liquidated_count` | `int` | 强平数量 |
| `liquidated_symbols` | `str` | 被强平的标的（逗号分隔） |
| `priority` | `str` | 强平优先级（`"short_first"` / `"long_first"`） |

---

## 九、报告生成方法

### `result.report()`

```python
def report(
    self,
    title: str = "AKQuant 策略回测报告",
    filename: str = "akquant_report.html",
    show: bool = False,
    compact_currency: bool = True,
    market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
    plot_symbol: Optional[str] = None,
    include_trade_kline: bool = True,
    benchmark: Optional[Union[str, pd.Series]] = None,
    curve_freq: str = "raw",
) -> None
```

生成交互式 HTML 回测报告（需 `plotly`）。

| 参数 | 含义 |
|------|------|
| `title` | 报告标题 |
| `filename` | 输出文件路径 |
| `show` | 是否在浏览器中自动打开 |
| `compact_currency` | 金额使用 K/M/B 紧凑显示 |
| `market_data` | 行情数据，用于 K 线买卖点图。DataFrame 需含 `date/open/high/low/close/volume` 列 |
| `plot_symbol` | K 线复盘图标的代码（多标的时指定一个） |
| `include_trade_kline` | 是否包含 K 线交易回放图 |
| `benchmark` | 基准收益序列（`pd.Series`）或标的代码 |
| `curve_freq` | 曲线频率：`"raw"`（原始）或 `"D"`（日频） |

### `result.plot()`

```python
def plot(self, symbol=None, show=True, title="Backtest Result") -> Any
```

快速绘制 Plotly 仪表盘，返回 Figure 对象。

### `result.to_quantstats()`

将权益曲线转为 QuantStats 兼容的日收益序列（`pd.Series`，时区已剥离）。

### `result.report_quantstats()`

```python
def report_quantstats(
    self,
    benchmark: Optional[Union[str, pd.Series]] = None,
    title: str = "Strategy Report",
    filename: str = "quantstats-report.html",
    **kwargs,
) -> None
```

生成 QuantStats 格式的 HTML 报告（需 `quantstats`）。

---

## 十、分析输出方法

### 敞口分析

```python
result.exposure_df(freq: Optional[str] = "D") -> pd.DataFrame
```

| 列名 | 含义 |
|------|------|
| `date` | 日期 |
| `equity` | 账户权益 |
| `long_exposure` | 多头敞口 |
| `short_exposure` | 空头敞口 |
| `net_exposure` | 净敞口（多 − 空） |
| `gross_exposure` | 总敞口（多 + 空） |
| `net_exposure_pct` | 净敞口占比 |
| `gross_exposure_pct` | 总敞口占比 |
| `leverage` | 杠杆倍数 |

### 归因分析

```python
result.attribution_df(by="symbol", use_net=True, top_n=None) -> pd.DataFrame
```

| 参数 | 含义 |
|------|------|
| `by` | 分组维度：`"symbol"` / `"entry_tag"` / `"exit_tag"` / `"tag"` |
| `use_net` | 是否使用净 PnL（True）还是毛 PnL（False） |
| `top_n` | 取前 N 名（None = 全部） |

返回列：`group`、`trade_count`、`total_pnl`、`avg_return_pct`、`total_commission`、`contribution_pct`、`abs_contribution_pct`

### 容量分析

```python
result.capacity_df(freq: str = "D") -> pd.DataFrame
```

返回列：`date`、`order_count`、`filled_order_count`、`ordered_quantity`、`filled_quantity`、`ordered_value`、`filled_value`、`fill_rate_qty`、`fill_rate_value`、`equity`、`turnover`

### 订单/成交按策略聚合

```python
result.orders_by_strategy() -> pd.DataFrame       # 按 owner_strategy_id 聚合
result.executions_by_strategy() -> pd.DataFrame   # 按 owner_strategy_id 聚合
```

### 拒单分析

```python
result.top_reject_reasons(top_n=10) -> pd.DataFrame
# 列: reject_reason | count | ratio
```

```python
result.risk_rejections_by_strategy() -> pd.DataFrame
# 按策略分组的风险拒单明细：
# risk_reject_count / daily_loss_reject_count / drawdown_reject_count
# / reduce_only_reject_count / position_limit_reject_count
# / order_size_limit_reject_count / order_value_limit_reject_count
# / strategy_risk_budget_reject_count / portfolio_risk_budget_reject_count
# / other_risk_reject_count
```

```python
result.risk_rejections_trend(freq="D") -> pd.DataFrame
# 拒单时间趋势（按 freq 聚合）
```

```python
result.risk_rejections_trend_by_strategy(freq="D") -> pd.DataFrame
# 拒单时间趋势（按策略和 freq 聚合）
```

### 流式事件统计

```python
result.get_event_stats() -> dict[str, Any]
# 返回字段: processed_events / dropped_event_count / callback_error_count
# / backpressure_policy / stream_mode / sampling_enabled / sampling_rate / reason
```

---

## 十一、特殊机制

### `__repr__`

```python
print(result)
# 输出:
# BacktestResult:
#                     Value
# name
# total_bars          1000
# trade_count         5
# total_return_pct    15.0
# ...
```

### `__getattr__` 委托

访问 `BacktestResult` 不存在的属性时，会自动委托给 Rust 侧的 `_raw` 对象。因此可以直接访问 `result.trade_metrics`、`result.trades`、`result.orders`、`result.snapshots` 等 Rust 原生属性。

---

## 十二、常用代码片段

### 基本使用

```python
result = aq.run_backtest(strategy=MyStrategy, data=df, initial_cash=1000000)

# 查看指标
print(result.metrics.sharpe_ratio)
print(result.metrics.total_return_pct)
print(result.metrics.max_drawdown_pct)

# 查看 DataFrame
print(result.metrics_df)
print(result.trades_df.head())
print(result.orders_df[result.orders_df["status"] == "rejected"])

# 权益曲线
equity = result.equity_curve

# 生成报告
result.report(title="我的策略", filename="report.html", show=True)
```

### 访问策略内部状态

```python
result = aq.run_backtest(strategy=MyStrategy, data=df, initial_cash=1000000)
# 通过 result.strategy 访问策略实例
print(result.strategy.bars_held)      # 自定义状态
print(result.strategy.buy_count)      # 自定义计数器
```

### 访问引擎内部状态

```python
# 访问引擎和组合信息
print(result.engine.portfolio.cash)   # 最终现金
```

### 提取 QuantStats 序列

```python
returns = result.to_quantstats()
result.report_quantstats(benchmark="SPY", title="策略报告")
```

### 分析拒单原因

```python
# Top-10 拒单原因
print(result.top_reject_reasons(10))

# 按策略分组的拒单明细
print(result.risk_rejections_by_strategy())

# 拒单趋势
print(result.risk_rejections_trend("D"))
```

### 分析归因

```python
# 按标的归因
print(result.attribution_df(by="symbol"))

# 按入场标签归因
print(result.attribution_df(by="entry_tag", top_n=5))

# 敞口分析
print(result.exposure_df("D"))
```
