# aq.Strategy 完整变量与方法参考

> 基于 `python/akquant/strategy.py` 源码分析。`Strategy` 是 AKQuant 策略基类，所有用户策略均需继承此类。

---

## 一、用户应重写的回调方法（生命周期钩子）

### 核心策略入口

| 方法 | 签名 | 触发时机 |
|------|------|---------|
| `on_bar` | `(self, bar: Bar) -> None` | 每根 Bar 到达时，**最核心的策略逻辑入口** |
| `on_tick` | `(self, tick: Tick) -> None` | 每笔 Tick 到达时 |
| `on_timer` | `(self, payload: str) -> None` | 定时器触发，`payload` 为注册时传入的标识字符串 |

### 生命周期

| 方法 | 签名 | 触发时机 |
|------|------|---------|
| `on_start` | `(self) -> None` | 回测启动时（warm start 也会调用） |
| `on_resume` | `(self) -> None` | 从快照恢复时，仅在 warm start 模式下调用，在 `on_start` 之前 |
| `on_stop` | `(self) -> None` | 回测结束时 |

### 订单/成交回调

| 方法 | 签名 | 触发时机 |
|------|------|---------|
| `on_order` | `(self, order: Any) -> None` | 订单状态变更（提交/成交/取消/拒绝） |
| `on_trade` | `(self, trade: Any) -> None` | 订单成交时 |
| `on_reject` | `(self, order: Any) -> None` | 订单被风控拒绝时 |

### Session / 交易日钩子

| 方法 | 签名 | 触发时机 |
|------|------|---------|
| `on_session_start` | `(self, session: Any, timestamp: int) -> None` | 交易时段开始时 |
| `on_session_end` | `(self, session: Any, timestamp: int) -> None` | 交易时段结束时 |
| `on_before_trading` | `(self, trading_date: date, timestamp: int) -> None` | 每个交易日开盘前 |
| `on_pre_open` | `(self, event: Dict[str, Any]) -> None` | 开盘前一刻，此时下单在当日开盘价成交 |
| `on_daily_rebalance` | `(self, trading_date: date, timestamp: int) -> None` | 每日调仓时刻（每天最多一次） |
| `on_after_trading` | `(self, trading_date: date, timestamp: int) -> None` | 每个交易日收盘后 |

### 其他回调

| 方法 | 签名 | 触发时机 |
|------|------|---------|
| `on_portfolio_update` | `(self, snapshot: Dict[str, Any]) -> None` | 账户权益/现金发生变化时 |
| `on_error` | `(self, error: Exception, source: str, payload: Any = None) -> None` | 回调中抛出异常时（需 `error_mode="continue"`） |
| `on_expiry` | `(self, event: Dict[str, Any]) -> None` | 合约到期结算时（期货/期权） |
| `on_train_signal` | `(self, context: Any) -> None` | Walk-forward 训练窗口触发时 |
| `prepare_features` | `(self, df: pd.DataFrame, mode: str) -> Tuple[Any, Any]` | ML 特征工程，**必须自己实现**。`mode="training"` 返回 `(X, y)`，`mode="inference"` 返回 `(X_last_row, None)` |

---

## 二、交易 API（下单方法）

| 方法 | 签名 | 返回值 | 用途 |
|------|------|--------|------|
| `buy` | `(self, symbol=None, quantity=None, price=None, time_in_force=None, trigger_price=None, tag=None, fill_policy=None, slippage=None, commission=None) -> str` | `order_id` | 买入开多 |
| `sell` | `(self, symbol=None, quantity=None, price=None, time_in_force=None, trigger_price=None, tag=None, fill_policy=None, slippage=None, commission=None) -> str` | `order_id` | 卖出平多 |
| `short` | `(self, symbol=None, quantity=None, price=None, time_in_force=None, trigger_price=None, tag=None, fill_policy=None, slippage=None, commission=None) -> None` | — | 卖出开空 |
| `cover` | `(self, symbol=None, quantity=None, price=None, time_in_force=None, trigger_price=None, tag=None, fill_policy=None, slippage=None, commission=None) -> None` | — | 买入平空 |
| `close_position` | `(self, symbol=None) -> None` | — | 平仓，自动识别多/空方向 |
| `buy_all` | `(self, symbol=None) -> None` | — | 全仓买入（用完所有可用资金） |

**参数说明：**
- `symbol: Optional[str]` — 标的代码，默认使用当前 Bar/Tick 的 symbol
- `quantity: Optional[float]` — 数量。`buy`/`short` 默认使用 Sizer 计算；`sell`/`cover` 默认卖出/平掉当前所有持仓
- `price: Optional[float]` — 限价（`None` = 市价单）
- `time_in_force: Optional[TimeInForce]` — 订单有效期（`GTC` / `DAY` / `IOC` / `FOK`）
- `trigger_price: Optional[float]` — 触发价，非 None 则创建 Stop/StopLimit 订单
- `tag: Optional[str]` — 订单标签，便于在 `on_order` 中识别
- `fill_policy: Optional[Dict]` — 成交策略覆盖，如 `{"price_basis": "open", "temporal": "same_cycle"}`
- `slippage: Optional[Union[float, Dict]]` — 滑点配置
- `commission: Optional[Dict]` — 手续费覆盖

### 统一下单接口

| 方法 | 签名 | 用途 |
|------|------|------|
| `submit_order` | `(self, symbol=None, side="Buy", quantity=None, price=None, time_in_force=None, trigger_price=None, tag=None, client_order_id=None, order_type=None, extra=None, broker_options=None, trail_offset=None, trail_reference_price=None, fill_policy=None, slippage=None, commission=None) -> str` | 回测与实盘通用下单，实盘模式下由 LiveRunner 注入增强能力 |

### 目标仓位调仓

| 方法 | 签名 | 用途 |
|------|------|------|
| `order_target` | `(self, target: float, symbol=None, price=None, **kwargs) -> None` | 调仓到目标数量（正=多，负=空） |
| `order_target_percent` | `(self, target_percent: float, symbol=None, price=None, **kwargs) -> None` | 调仓到目标百分比（`0.33` = 33%） |
| `order_target_value` | `(self, target_value: float, symbol=None, price=None, **kwargs) -> None` | 调仓到目标市值 |
| `order_target_weights` | `(self, target_weights: Dict[str, float], price_map=None, liquidate_unmentioned=False, allow_leverage=False, rebalance_tolerance=0.0, **kwargs) -> None` | 多标的按权重调仓 |

### 订单管理

| 方法 | 签名 | 用途 |
|------|------|------|
| `cancel_order` | `(self, order_id: str) -> None` | 撤销指定订单 |
| `cancel_all_orders` | `(self, symbol=None) -> None` | 撤销所有未完成订单 |

### 订单查询

| 方法 | 签名 | 返回值 | 用途 |
|------|------|--------|------|
| `get_open_orders` | `(self, symbol=None) -> List[Any]` | 订单列表 | 当前未成交订单 |
| `get_order` | `(self, order_id: str) -> Optional[Any]` | Order / None | 指定订单详情 |

---

## 三、高级订单

| 方法 | 签名 | 用途 |
|------|------|------|
| `stop_buy` | `(self, symbol=None, trigger_price=0.0, quantity=None, price=None, time_in_force=None) -> None` | 止损买入单。价格上涨突破 `trigger_price` 时触发买入；`price=None` 为市价止损，`price` 非 None 为限价止损 |
| `stop_sell` | `(self, symbol=None, trigger_price=0.0, quantity=None, price=None, time_in_force=None) -> None` | 止损卖出单。价格下跌跌破 `trigger_price` 时触发卖出 |
| `place_trailing_stop` | `(self, symbol, quantity, trail_offset, side="Sell", trail_reference_price=None, time_in_force=None, tag=None) -> str` | 跟踪止损单（StopTrail）。价格向有利方向移动时自动调整止损价 |
| `place_trailing_stop_limit` | `(self, symbol, quantity, price, trail_offset, side="Sell", trail_reference_price=None, time_in_force=None, tag=None) -> str` | 跟踪止损限价单（StopTrailLimit） |
| `place_bracket_order` | `(self, symbol, quantity, entry_price=None, stop_trigger_price=None, take_profit_price=None, time_in_force=None, entry_tag=None, stop_tag=None, take_profit_tag=None) -> str` | 括号单：入场订单成交后自动挂出止损和止盈，止损止盈绑定为 OCO |
| `create_oco_order_group` | `(self, first_order_id, second_order_id, group_id=None) -> str` | 创建 OCO 组（一组中任一成交，另一自动撤销） |

---

## 四、状态查询方法

### 持仓查询

| 方法 | 返回值 | 用途 |
|------|--------|------|
| `get_position(symbol=None)` | `float` | 当前持仓数量（正=多头，负=空头） |
| `get_available_position(symbol=None)` | `float` | 可用持仓数量（考虑 T+1 限制后当日可卖数量） |
| `get_positions()` | `Dict[str, float]` | 所有持仓 `{symbol: quantity}` |
| `hold_bar(symbol=None)` | `int` | 当前持仓已持有的 Bar 数（引擎自动维护，建仓递增，平仓清零） |
| `position` (属性) | `Position` | 当前标的的 `Position` 对象，含 `.size` 和 `.avg_price` |

### 账户查询

| 方法 | 返回值 | 用途 |
|------|--------|------|
| `get_cash()` | `float` | 可用现金 |
| `get_portfolio_value()` | `float` | 总权益（现金 + 持仓市值） |
| `equity` (属性) | `float` | 等同于 `get_portfolio_value()` |
| `get_account()` | `Dict[str, Any]` | 账户完整快照，字段见下方 |
| `get_trades()` | `List[Any]` | 已平仓交易列表（ClosedTrade） |

**`get_account()` 返回的字典字段：**

| 字段 | 类型 | 含义 |
|------|------|------|
| `cash` | `float` | 可用资金 |
| `equity` | `float` | 总权益（现金 + 持仓市值） |
| `market_value` | `float` | 持仓总市值（equity - cash） |
| `frozen_cash` | `float` | 未完成订单预占资金 |
| `margin` | `float` | 当前仓位占用保证金 |
| `borrowed_cash` | `float` | 融资负债（现金为负时） |
| `short_market_value` | `float` | 空头市值 |
| `maintenance_ratio` | `float` | 维持担保比例 |
| `account_mode` | `str` | 账户模式（`"cash"` / `"margin"`） |
| `accrued_interest` | `float` | 累计计提利息 |
| `daily_interest` | `float` | 当日计提利息 |

### 历史数据查询

| 方法 | 返回值 | 用途 |
|------|--------|------|
| `get_history(count, symbol=None, field="close")` | `np.ndarray` | 历史字段数据（open/high/low/close/volume），不含当前 Bar |
| `get_history_df(count, symbol=None)` | `pd.DataFrame` | 历史 OHLCV DataFrame |
| `get_history_map(count, symbols, field="close")` | `Dict[str, np.ndarray]` | 批量获取多标的历史数据 |
| `set_history_depth(depth)` | — | 设置历史数据缓存深度（0 = 不保留） |
| `get_rolling_data(length=None, symbol=None)` | `(pd.DataFrame, Optional[pd.Series])` | 滚动训练数据（ML 用） |
| `set_rolling_window(train_window, step)` | — | 设置滚动训练窗口参数 |

### 标的信息查询

| 方法 | 返回值 | 用途 |
|------|--------|------|
| `get_instrument(symbol)` | `InstrumentSnapshot` | 标的属性快照（不可变对象） |
| `get_instruments(symbols=None)` | `Dict[str, InstrumentSnapshot]` | 批量获取标的属性 |
| `get_instrument_field(symbol, field)` | `Any` | 标的单个字段值 |
| `get_instrument_config(symbol, fields=None)` | `Any` | 标的配置，`fields=None` 返回完整快照 |

**`InstrumentSnapshot` 字段：**

| 字段 | 类型 | 含义 |
|------|------|------|
| `symbol` | `str` | 标的代码 |
| `asset_type` | `str` | 资产类型（`"STOCK"` / `"FUTURES"` / `"FUND"` / `"OPTION"`） |
| `multiplier` | `float` | 合约乘数 |
| `margin_ratio` | `float` | 保证金率 |
| `tick_size` | `float` | 最小价格变动单位 |
| `lot_size` | `float` | 每手数量 |
| `option_margin_model` | `Optional[str]` | 期权保证金模型 |
| `option_type` | `Optional[str]` | 期权类型（`"CALL"` / `"PUT"`） |
| `strike_price` | `Optional[float]` | 行权价 |
| `expiry_date` | `Optional[int]` | 到期日 |
| `underlying_symbol` | `Optional[str]` | 标的资产代码 |
| `implied_volatility` | `Optional[float]` | 隐含波动率 |
| `settlement_type` | `Optional[str]` | 结算方式（`"CASH"` / `"SETTLEMENT_PRICE"` / `"FORCE_CLOSE"`） |
| `settlement_price` | `Optional[float]` | 结算价 |

### ML 相关查询

| 方法 | 返回值 | 用途 |
|------|--------|------|
| `is_model_ready()` | `bool` | 模型是否已训练就绪 |
| `current_validation_window()` | `Optional[Dict]` | 当前 walk-forward 验证窗口状态 |

---

## 五、调度与时间方法

| 方法/属性 | 签名 | 用途 |
|-----------|------|------|
| `schedule` | `(self, trigger_time: Union[str, datetime, Timestamp], payload: str) -> None` | 注册单次定时任务 |
| `add_daily_timer` | `(self, time_str: str, payload: str) -> None` | 注册每日定时任务，`time_str` 如 `"14:55:00"` |
| `to_local_time` | `(self, timestamp: int) -> pd.Timestamp` | UTC 纳秒时间戳 → 本地时间 |
| `format_time` | `(self, timestamp: int, fmt="%Y-%m-%d %H:%M:%S") -> str` | UTC 纳秒时间戳 → 格式化字符串 |
| `now` | `-> Optional[pd.Timestamp]` (属性) | 当前回测时间的本地时间 |

---

## 六、指标注册

| 方法 | 签名 | 用途 |
|------|------|------|
| `register_indicator` | `(self, name: str, indicator: Indicator) -> None` | 注册预计算指标，本质调用 `register_precomputed_indicator` |
| `register_precomputed_indicator` | `(self, name: str, indicator: Indicator) -> None` | 注册预计算指标（需 `indicator_mode="precompute"`） |
| `register_incremental_indicator` | `(self, name, indicator=None, source="close", symbols=None, *, warmup_bars=0, indicator_factory=None, input_mode="source") -> None` | 注册增量指标（需 `indicator_mode="incremental"`） |
| `subscribe` | `(self, instrument_id: str) -> None` | 订阅标的数据（不订阅则 `on_bar` 收不到该标的） |

---

## 七、其他方法与属性

| 方法/属性 | 签名/类型 | 用途 |
|-----------|-----------|------|
| `set_sizer` | `(self, sizer: Sizer) -> None` | 设置仓位管理器 |
| `log` | `(self, msg: str, level: int = logging.INFO) -> None` | 输出带时间戳的策略日志 |
| `can_submit_client_order` | `(self, client_order_id: str) -> bool` | 检查 `client_order_id` 是否可再次提交（实盘用） |
| `get_execution_capabilities` | `(self) -> Dict[str, Any]` | 获取当前执行能力描述 |
| `rebalance_to_topn` | `(self, scores, top_n, *, weight_mode="equal", long_only=True, min_score=None, liquidate_unmentioned=True, allow_leverage=False, rebalance_tolerance=0.0, **kwargs) -> List[str]` | 根据打分选取 TopN 并执行调仓的便捷方法 |

**`rebalance_to_topn` 参数说明：**
- `scores: Dict[str, float]` — 打分字典
- `top_n: int` — 选取前 N 个
- `weight_mode: "equal" | "score"` — 等权或按分数归一化
- `long_only: bool` — 是否仅做多（负分不入选）
- `min_score: float` — 最低分数阈值（`long_only=True` 时默认 0）
- `liquidate_unmentioned: bool` — 是否清仓未入选标的
- `allow_leverage: bool` — 是否允许总权重超过 1.0
- `rebalance_tolerance: float` — 调仓容忍阈值（按组合市值比例）

### 属性（Property）

| 属性 | 类型 | 用途 |
|------|------|------|
| `symbol` | `str` | 当前 Bar/Tick 的标的代码 |
| `close` | `float` | 当前收盘价 / 最新价 |
| `open` | `float` | 当前开盘价（仅 Bar 模式有效） |
| `high` | `float` | 当前最高价（仅 Bar 模式有效） |
| `low` | `float` | 当前最低价（仅 Bar 模式有效） |
| `volume` | `float` | 当前成交量 |
| `position` | `Position` | 当前标的的 Position 对象 |
| `equity` | `float` | 当前总权益（= `get_portfolio_value()`） |
| `is_restored` | `bool` | 是否从快照恢复（warm start） |
| `now` | `Optional[pd.Timestamp]` | 当前回测的本地时间 |
| `runtime_config` | `StrategyRuntimeConfig` | 运行时行为配置 |

---

## 八、可配置的类/实例变量

### 费率与交易参数

| 变量 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `commission_rate` | `float` | `0.0` | 佣金费率（如 `0.0003` = 万三） |
| `min_commission` | `float` | `0.0` | 最低佣金（如 `5.0` = 最低 5 元） |
| `stamp_tax_rate` | `float` | `0.0` | 印花税率（A 股卖出 `0.001`） |
| `transfer_fee_rate` | `float` | `0.0` | 过户费率 |
| `lot_size` | `int` / `Dict[str, int]` | `1` | 手数。A 股需设为 `100`；多标的不同手数可用 `dict` |
| `sizer` | `Sizer` | `FixedSize(100)` | 仓位管理器（`FixedSize` / `PercentSizer` / `AllInSizer`） |

### 运行时配置

| 变量 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `timezone` | `str` | `"Asia/Shanghai"` | 时区 |
| `warmup_period` | `int` | `0` | 预热期，跳过前 N 根 Bar 不触发策略逻辑 |
| `runtime_config` | `StrategyRuntimeConfig` | `error_mode="raise"` | 运行时行为配置对象 |
| `model` | `Optional[QuantModel]` | `None` | ML 模型（`QuantModel` 包装器） |

---

## 九、StrategyRuntimeConfig 字段

| 字段 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `error_mode` | `"raise"` / `"continue"` / `"legacy"` | `"raise"` | `"raise"`——回调异常立即终止；`"continue"`——异常路由到 `on_error`，继续运行 |
| `enable_precise_day_boundary_hooks` | `bool` | `False` | 是否在精确 session 边界触发 `on_session_start/end` 等钩子 |
| `portfolio_update_eps` | `float` | `0.0` | 账户快照更新最小变化阈值（避免 `on_portfolio_update` 过于频繁） |
| `re_raise_on_error` | `bool` | `True` | `on_error` 回调后是否继续抛出异常 |
| `indicator_mode` | `"incremental"` / `"precompute"` | `"precompute"` | `"precompute"`——回测前一次性计算；`"incremental"`——每 Bar 增量更新 |

---

## 十、内部私有变量（仅了解，避免直接操作）

| 变量 | 类型 | 用途 |
|------|------|------|
| `ctx` | `Optional[StrategyContext]` | Rust 侧策略上下文句柄 |
| `current_bar` | `Optional[Bar]` | 当前处理的 Bar |
| `current_tick` | `Optional[Tick]` | 当前处理的 Tick |
| `_history_depth` | `int` | 历史数据缓存深度 |
| `_subscriptions` | `List[str]` | 已订阅的标的列表 |
| `_last_prices` | `Dict[str, float]` | 各标的最后成交价 |
| `_bar_count` | `int` | 全局 Bar 计数器 |
| `_start_initialized` | `bool` | 是否已完成启动初始化 |
| `_instrument_snapshots` | `Dict[str, InstrumentSnapshot]` | 标的属性快照缓存 |
| `_oco_groups` / `_oco_order_to_group` | `Dict` | OCO 组管理（Python 侧实现） |
| `_pending_brackets` | `Dict` | Bracket 订单待处理队列（Python 侧实现） |
| `_pending_schedules` / `_pending_daily_timers` | `List` | 待注册的定时任务队列 |
| `_precomputed_indicators` | `List[Indicator]` | 预计算指标列表 |
| `_incremental_indicators` | `Dict[str, IncrementalIndicatorRegistration]` | 增量指标注册表 |
| `_hold_bars` | `defaultdict[str, int]` | 引擎内置持仓 Bar 计数，通过 `hold_bar()` 查询 |
| `_known_orders` | `Dict[str, Order]` | 已知订单缓存 |
| `_seen_trade_keys` | `set` | 已处理成交的去重键集合 |

---

## 十一、VectorizedStrategy（子类）

`VectorizedStrategy(Strategy)` 用于预计算指标的高速回测模式。

| 构造函数参数 | 类型 | 用途 |
|-------------|------|------|
| `precalculated_data` | `Dict[str, Dict[str, np.ndarray]]` | `{symbol: {indicator_name: array}}` |

| 方法 | 签名 | 用途 |
|------|------|------|
| `get_value` | `(self, name, symbol=None) -> float` | 获取当前 Bar 对应游标的预计算指标值 |

**内部属性：**
- `precalc` — 预计算数据存储
- `cursors: defaultdict[str, int]` — 每标的的 Bar 游标（自动递增）

---

## 十二、常用代码片段

### 最小策略模板

```python
import akquant as aq
from akquant import Bar, Strategy

class MyStrategy(Strategy):
    warmup_period = 20  # 跳过前 20 根 Bar

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        closes = self.get_history(self.warmup_period, symbol, "close")
        if len(closes) < self.warmup_period:
            return

        ma5 = closes[-5:].mean()
        ma20 = closes[-20:].mean()
        pos = self.get_position(symbol)

        if ma5 > ma20 and pos == 0:
            self.order_target_percent(0.95, symbol)
        elif ma5 < ma20 and pos > 0:
            self.close_position(symbol)

result = aq.run_backtest(
    strategy=MyStrategy,
    data=df,
    initial_cash=1000000,
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    lot_size=100,
)
```

### 函数式策略模板

```python
def on_bar(ctx, bar):
    pos = ctx.get_position()
    if pos == 0:
        ctx.order_target_percent(0.95)

result = aq.run_backtest(strategy=on_bar, data=df, initial_cash=100000)
```

### 使用定时器做日度调仓

```python
class RotateStrategy(Strategy):
    def on_start(self):
        self.subscribe("600519")
        self.subscribe("000858")
        self.add_daily_timer("10:00:00", "rebalance")

    def on_bar(self, bar):
        pass  # 不使用 Bar 驱动

    def on_timer(self, payload):
        if payload != "rebalance":
            return
        closes = self.get_history_map(20, ["600519", "000858"], "close")
        scores = {s: closes[s][-1] / closes[s][0] - 1 for s in closes}
        self.rebalance_to_topn(scores, top_n=1)
```

### 使用 Bracket Order

```python
class BracketStrategy(Strategy):
    def on_bar(self, bar):
        if self.get_position() == 0:
            self.place_bracket_order(
                symbol=bar.symbol,
                quantity=100,
                stop_trigger_price=bar.close * 0.95,   # -5% 止损
                take_profit_price=bar.close * 1.10,     # +10% 止盈
            )
```
