# AKQuant Examples 详细说明文档

本文档对 `examples/` 目录下的所有示例脚本进行全面说明，涵盖核心功能、关键 API、适用场景及运行注意事项。

---

## 目录

- [快速入门路径](#快速入门路径)
- [基础与回测示例 (01-17)](#基础与回测示例)
- [高级回测示例 (18-24)](#高级回测示例)
- [流式回测与实时监控 (25-32)](#流式回测与实时监控)
- [报告与分析 (33)](#报告与分析)
- [多策略与经纪人 (34-35)](#多策略与经纪人)
- [订单与数据对齐 (36-37)](#订单与数据对齐)
- [实盘交易 (38-42)](#实盘交易)
- [权重调仓与动态加载 (43-44)](#权重调仓与动态加载)
- [指标与保证金 (45-48)](#指标与保证金)
- [生命周期钩子 (49-54)](#生命周期钩子)
- [函数式高级用法 (55-58)](#函数式高级用法)
- [策略集合 (strategies/)](#策略集合)
- [教科书示例 (textbook/)](#教科书示例)
- [工具文件](#工具文件)

---

## 快速入门路径

| 目标 | 推荐路径 |
|------|---------|
| 5 分钟入门 | `01_quickstart.py` → `17_readme_demo.py` |
| 参数优化 | `02_parameter_optimization.py` → `03_parameter_optimization_advanced.py` |
| 报告与分析 | `11_plot_visualization.py` → `33_report_and_analysis_outputs.py` |
| 流式监控 | `26_streaming_quickstart.py` → `27_streaming_monitoring_console.py` |
| 实时可视化 | `31_streaming_live_console.py` → `32_streaming_live_web.py` |
| 机器学习 | `09_ml_framework.py` → `10_ml_walk_forward.py` |
| 实盘交易 | `05_live_trading_ctp.py` → `38_live_functional_strategy_demo.py` |

---

## 基础与回测示例

### 01_quickstart.py — 多标的快速回测

**功能概述**：最基础的入门示例。从 AKShare 获取三只真实 A 股数据，运行简单策略（每只标的买入 33% 仓位，持有至 10% 止盈或 100 根 Bar 后平仓），并用手动计算验证引擎输出的性能指标。

**核心 API**：
- `aq.Strategy` — 策略基类，需实现 `on_bar(bar)` 方法
- `aq.run_backtest(data, strategy, ...)` — 启动回测的入口函数
- `result.metrics` — 回测结果指标（总收益、年化收益、波动率、最大回撤等）
- `result.orders_df` — 订单明细 DataFrame
- `result.liquidation_audit_df` — 强平审计数据
- `result.equity_curve` — 权益曲线数据
- `self.order_target_percent(0.33, symbol)` — 按百分比调仓
- `self.close_position(symbol)` — 平仓所有持仓

**关键特性**：
- 演示通过 `on_event` 回调收集 `BacktestStreamEvent` 流式事件
- 使用 `numpy.polyfit` 和 pandas 重采样验证 Rust 引擎计算指标的正确性
- 数据准备使用 `ak.stock_zh_a_daily()`

**适用场景**：初次接触 AKQuant，了解回测基本流程

---

### 02_parameter_optimization.py — 参数优化基础

**功能概述**：引入 `ParamModel`（基于 Pydantic）为策略定义类型化参数模式。实现双均线交叉策略，包含 `fast_period` 和 `slow_period` 两个 `IntParam` 参数，并通过 `run_grid_search()` 进行网格搜索优化。

**核心 API**：
- `aq.ParamModel` — 参数模型基类
- `aq.IntParam` — 整型参数，支持 `ge`/`le` 约束和 `title` 描述
- `aq.Indicator` — 滚动窗口指标计算类，配合 lambda 函数使用
- `aq.get_strategy_param_schema(StrategyClass)` — 导出策略参数 schema
- `aq.validate_strategy_params(StrategyClass, params)` — 校验参数合法性
- `aq.run_grid_search(strategy, data, param_grid, ...)` — 并行网格搜索优化

**关键特性**：
- 使用合成随机游走数据（两个标的不依赖外部数据源）
- 网格搜索结果按 `total_return` 排序
- `Indicator` 类支持通过 lambda 定义计算逻辑，自动处理滚动窗口

**适用场景**：学习参数化策略和网格搜索优化

---

### 03_parameter_optimization_advanced.py — 参数优化进阶

**功能概述**：更高级的参数优化示例。在网格搜索中增加 `warmup_calc`（预热计算）、`param_constraint`（参数约束，确保 `short_window < long_window`）和 `result_filter`（结果过滤，仅保留 `trade_count >= 2` 的结果）。同时演示 `run_walk_forward()` 滚动优化。

**核心 API**：
- `run_grid_search()` 参数：`warmup_calc`、`param_constraint`、`result_filter`
- `run_walk_forward(strategy, data, train_period, test_period, param_grid, ...)` — 滚动窗口优化
- `self.get_history(count, symbol)` — 获取历史数据

**关键特性**：
- `param_constraint` 回调在搜索前过滤不合法参数组合
- `result_filter` 回调在搜索后过滤不达标的回测结果
- WFO 使用 `train_period=100`、`test_period=50`，多指标排序（`sharpe_ratio`、`total_return`）

**适用场景**：需要约束化参数搜索和滚动优化的场景

---

### 04_mixed_assets.py — 混合资产回测

**功能概述**：演示使用 `InstrumentConfig` 和 `BacktestConfig` 配置包含不同资产类型（股票和期货）的投资组合，每种资产有不同的乘数、保证金率和最小价格变动单位。

**核心 API**：
- `aq.InstrumentConfig(symbol, asset_type, multiplier, margin_ratio, tick_size)` — 标的配置
- `aq.BacktestConfig(strategy_config, instruments_config)` — 回测配置对象
- `aq.StrategyConfig(initial_cash, commission_rate, ...)` — 策略账户配置
- `aq.run_backtest(config=config)` — 使用配置对象启动回测

**关键特性**：
- 股票类型使用默认设置（乘数=1.0，保证金=1.0）
- 期货标的 `FUTURE_B` 配置乘数=300.0，保证金率=0.1
- 使用面向对象的 `config` 参数替代平铺的 kwargs

**适用场景**：多资产类别组合回测

---

### 05_live_trading_ctp.py — CTP 实盘交易

**功能概述**：演示通过 `LiveRunner` 连接 CTP（综合交易平台）进行实盘/模拟交易。使用 `openctp-ctp` 连接 SimNow 模拟环境，以期货标的（黄金、螺纹钢、白银）运行实盘策略。

**核心 API**：
- `aq.Instrument(symbol, asset_type)` — 实盘标的定义
- `aq.AssetType.Futures` — 期货资产类型枚举
- `aq.live.LiveRunner(strategy_cls, instruments, md_front, ...)` — 实盘运行器
- `runner.run(cash=...)` — 启动实盘交易循环
- 生命周期回调：`on_start()`、`on_bar()`、`on_order()`、`on_trade()`

**关键特性**：
- 支持 `use_aggregator=True`（聚合 Tick 为 1 分钟 Bar）或 `use_aggregator=False`（每 Tick 视为一根 Bar）
- 实盘策略与回测策略代码完全一致
- 需要安装 `openctp-ctp` 包

**适用场景**：从回测到实盘的无缝迁移

---

### 06_complex_orders.py — 复杂订单（Bracket Order）

**功能概述**：演示 Bracket Order（括号单）—— 一个入场订单附带止盈和止损订单。策略在无持仓时买入，通过 Bracket 的止盈/止损自动退出。

**核心 API**：
- `self.place_bracket_order(symbol, quantity, side, entry_price, stop_trigger_price, take_profit_price)` — 下括号单
- `OrderStatus` — 订单状态枚举
- `self.on_trade(trade)` — 成交回调，跟踪入场订单 ID
- `self.on_order(order)` — 订单状态变更回调

**关键特性**：
- 入场订单类型支持市价（`entry_price=None`）或限价
- 使用合成正弦波价格数据以触发止盈和止损
- 演示了完整的订单生命周期管理

**适用场景**：需要止盈止损联动的高级订单场景

---

### 07_option_test.py — 期权测试

**功能概述**：测试期权到期行权和现金结算逻辑。策略买入一张看涨期权并持有至到期，验证期权是否正确按现金结算。

**核心 API**：
- `InstrumentConfig(symbol, asset_type="OPTION", multiplier=100, margin_ratio=..., option_type="CALL", strike_price=100.0, expiry_date=..., settlement_type="cash", underlying_symbol="UL")`
- `RiskConfig(safety_margin=0.0001)` — 风控配置
- `BacktestConfig` — 整合各类配置

**关键特性**：
- 期权乘数为 100
- 现金结算在到期日自动处理
- 断言最终组合价值约为 99,900（初始 100,000 减去权利金 100）

**适用场景**：期权回测和结算逻辑验证

---

### 08_event_callbacks.py — 事件回调综合演示

**功能概述**：全面演示框架中所有主要事件回调：`on_start`、`on_bar`、`on_timer`、`on_order`、`on_trade`、`on_reject`、`on_portfolio_update`、`on_stop`。策略刻意提交超大订单触发 `on_reject`，再提交正常订单，最后通过定时器触发退出。

**核心 API**：
- `self.add_daily_timer("14:55:00", "close_check")` — 添加每日定时器
- `self.subscribe(symbol)` — 订阅标的
- `self.on_reject(order)` — 订单拒绝回调
- `self.on_portfolio_update(cash, equity, ...)` — 组合更新回调
- `self.on_stop()` — 回测结束回调

**关键特性**：
- 日内数据使用 `Asia/Shanghai` 时区
- `fill_policy` 控制 Bar 成交方式（`same_cycle` + `close` 价格）
- `on_stop` 输出全部回调触发顺序的摘要

**适用场景**：全面了解策略生命周期回调机制

---

### 09_ml_framework.py — 机器学习框架验证

**功能概述**：独立验证（非回测环境）ML 适配器包装类 —— `SklearnAdapter` 和 `PyTorchAdapter`。展示模型的训练、预测、保存和加载往返过程。

**核心 API**：
- `aq.ml.SklearnAdapter(model)` — scikit-learn 模型适配器
- `aq.ml.PyTorchAdapter(network, criterion, optimizer_cls, lr, epochs, batch_size)` — PyTorch 模型适配器
- `adapter.fit(X, y)` / `adapter.predict(X)` / `adapter.save(path)` / `adapter.load(path)`

**关键特性**：
- sklearn 使用合成二分类数据
- PyTorch 使用合成回归数据（`nn.MSELoss` + `optim.Adam`，10 epochs）
- 保存后重新加载，验证预测结果一致性
- 不需要 `scikit-learn` 和 `torch`，无安装时跳过对应测试

**适用场景**：了解 ML 适配器的基本接口

---

### 10_ml_walk_forward.py — 机器学习 Walk-Forward

**功能概述**：将 ML 模型推理（通过 `SklearnAdapter` 的逻辑回归）集成到回测策略中，以 Walk-Forward 方式进行验证。框架自动处理数据分片和重训练。策略使用滞后收益作为特征预测价格方向。

**核心 API**：
- `self.model.set_validation(method="walk_forward", train_window=50, test_window=20, rolling_step=10)` — 配置 WFV
- `self.prepare_features(df, mode)` — **必须实现**的特征工程方法
- `self.current_validation_window()` — 获取当前 WFV 窗口信息
- `self.is_model_ready()` — 检查模型是否已训练就绪
- `self.get_history_df(count, symbol)` — 获取 DataFrame 格式的历史数据

**关键特性**：
- 特征 `ret1` 和 `ret2`（1 期和 2 期滞后收益）
- 预测阈值：0.55 买入、0.45 卖出
- 在 `on_train_signal` 中跳过预热期的训练
- 需要 `scikit-learn`

**适用场景**：ML 驱动的量化策略开发

---

### 11_plot_visualization.py — 可视化报告生成

**功能概述**：完整的可视化工作流 —— 获取真实 A 股数据，运行简单趋势跟随策略（阳线买入、阴线卖出），打印格式化指标，并生成包含交互式图表和 K 线交易回放标记的 HTML 报告。

**核心 API**：
- `result.report(title=..., filename=..., show=True, include_trade_kline=True)` — 生成 HTML 报告
- `aq.format_metric_value(value, ...)` — 格式化指标数值

**关键特性**：
- `include_trade_kline=True` 在 K 线图上标记买卖点
- 使用 `start_time` 和 `end_time` 限制回测日期范围
- 报告在浏览器中自动打开

**适用场景**：学习如何生成专业的回测分析报告

---

### 12_wfo_integrated.py — Walk-Forward 优化集成

**功能概述**：专用 WFO 示例，使用 `run_walk_forward` 对双均线交叉策略在两个标的上进行滚动优化。展示 `warmup_calc`、`param_constraint` 的使用，并检查各 WFO 窗口的参数选择变化。

**核心 API**：
- `run_walk_forward(strategy, data, train_period=250, test_period=60, param_grid=..., warmup_calc=..., param_constraint=..., ...)`

**关键特性**：
- 使用日线数据：训练 250 天（约 1 年），测试 60 天（约 3 个月）
- 优化指标为 `sharpe_ratio`
- 使用 `compounding=False`（简单加性 PnL）
- 按 WFO 窗口分组结果，展示最优参数随时间的演变

**适用场景**：理解滚动优化如何适配市场变化

---

### 13_quantstats_report.py — QuantStats 报告集成

**功能概述**：验证 QuantStats 集成 —— 测试 `result.to_quantstats()` 将权益曲线转换为 QuantStats 兼容的收益序列，以及 `result.report_quantstats()` 生成 QuantStats 格式的 HTML 报告。

**核心 API**：
- `result.to_quantstats()` — 返回带 DatetimeIndex 和时区信息的 pandas Series
- `result.report_quantstats(filename, title, benchmark=None)` — 生成 QuantStats 报告

**关键特性**：
- 策略为简单买入持有（首根 Bar 买入，永不卖出）
- 合成数据具有稳定上升趋势
- `benchmark=None` 避免网络下载基准数据

**适用场景**：需要 QuantStats 格式报告的场景

---

### 14_multi_frequency.py — 多频率回测

**功能概述**：演示多频率回测 —— 策略同时使用 1 分钟日内 Bar 进行信号执行，以及日线 Bar 进行趋势过滤。实现自定义 `MultiFrequencyAdapter`，通过 `BasePandasFeedAdapter.replay()` 将 1 分钟数据聚合为日线 Bar。

**核心 API**：
- `BasePandasFeedAdapter` — Pandas 数据馈送适配器基类
- `adapter.replay(freq="1D", align="session", session_windows=...)` — 数据聚合重放
- `aq.SMA(period)` — 增量型简单移动平均指标
- `self.register_incremental_indicator(name, factory, source, symbols)` — 注册增量指标
- `self.to_local_time(ts)` / `self.format_time(ts)` — 时间处理工具

**关键特性**：
- 两个逻辑标的：`"000001.SZ"`（分钟）和 `"000001.SZ_1D"`（日线）
- A 股交易时段（9:31-11:30，13:01-15:00）和时区（Asia/Shanghai）
- 日线 SMA 通过增量方式计算
- `indicator_mode="incremental"` 配置

**适用场景**：需要多时间框架信号确认的策略

---

### 15_plot_intraday.py — 日内绘图与回测

**功能概述**：演示分钟级别（日内）回测和可视化。获取日线 A 股数据，合成为 1 分钟 Bar（每日 240 根，覆盖上午和下午交易时段），运行快慢均线交叉策略，生成面向日内的 HTML 报告。

**核心 API**：
- `result.report(title=..., filename=..., show=True)` — 报告生成
- `format_metric_value(value, ...)` — 指标格式化

**关键特性**：
- 通过日线 OHLC 线性插值生成合成日内价格（含噪声）
- 覆盖 A 股交易时段（上午 + 下午）
- 数据列重命名（`timestamp` → `date`）

**适用场景**：日内策略的快速验证和可视化

---

### 16_adj_returns_signal.py — 复权收益信号

**功能概述**：演示使用前复权收盘价（`adj_close`）作为信号源，同时用原始收盘价进行撮合/估值。策略在复权收益为正时做多，为负时平仓。

**核心 API**：
- `bar.extra.get("adj_close")` — 访问 Bar 的额外数据字段
- `self.get_history(count, symbol, "adj_close")` — 获取历史复权价序列
- `akshare.stock_zh_a_daily(adjust="qfq")` — 获取前复权数据

**关键特性**：
- 双数据集模式：一个 DataFrame 含 OHLCV（撮合/估值），另一个含 `adj_close`（信号）
- 通过 `pd.merge` 合并原始和复权数据
- 预热期设为 5 根 Bar
- 避免前视偏差：历史查询不包含当前 Bar

**适用场景**：需要复权因子但不想修改撮合价格的场景

---

### 17_readme_demo.py — README 最小演示

**功能概述**：最小化的 README 风格演示。获取真实 A 股数据，运行基于阴阳线的策略（阳线买入、阴线卖出）。

**核心 API**：
- `aq.run_backtest(data, strategy, initial_cash=..., ...)` — 最简回测调用
- `self.get_position(symbol)` / `self.buy(symbol, quantity)` / `self.close_position(symbol)`
- `on_event` 回调收集流式事件

**关键特性**：
- 策略刻意保持极简（仅蜡烛线逻辑）
- 演示绝对最小样板代码：继承 `Strategy` → 实现 `on_bar` → 调用 `run_backtest`

**适用场景**：快速验证环境和最简策略原型

---

## 高级回测示例

### 18_benchmark_multisymbol.py — 多标的性能基准

**功能概述**：在三个合成标的上同时运行均线交叉策略，测量引擎吞吐量（Bars/秒），展示每标的独立指标管理、共享资金池和详细的交易报告。

**核心 API**：
- `aq.SMA(period)` — 简单移动平均
- `self.set_history_depth(0)` — 禁用 Python 侧历史缓存以提升性能
- `result.metrics`、`result.trade_metrics`、`result.trades`、`result.equity_curve`
- `aq.plot_result(result)` — 生成 HTML 图表
- `aq.get_logger()` — 获取日志记录器

**关键特性**：
- 使用 `Dict[str, DataFrame]` 格式的输入数据（引擎自动合并多标的 DataFrame）
- 每标的有独立的 SMA(5) 和 SMA(20) 指标（存储在 `dict` 中）
- `fill_policy={"price_basis": "open", "temporal": "same_cycle"}`
- 打印每标的 PnL 明细，验证 `net_pnl = gross_pnl - commission`
- 计算并显示吞吐量

**适用场景**：性能测试和多标的指标管理参考

---

### 19_factor_expression.py — 因子表达式引擎

**功能概述**：演示因子表达式引擎（`FactorEngine`）。下载真实 A 股数据到 Parquet 数据目录，运行 Alpha 因子风格表达式（时间序列和横截面），并验证输出。

**核心 API**：
- `aq.data.ParquetDataCatalog(root_path)` — 本地 Parquet 数据存储
- `catalog.write(symbol, df)` — 写入标的数据
- `aq.factor.FactorEngine(catalog)` — 因子计算引擎
- `engine.run(expression)` — 计算单条因子表达式
- `engine.run_batch(expressions)` — 批量计算多条表达式

**支持的表达式**：`Ts_Mean`（时序均值）、`Ts_Std`（时序标准差）、`Delta`（一阶差分）、`Rank`（横截面排名）、`Ts_ArgMax`（时序最大值位置）、`Ts_ArgMin`、`Ts_Rank`、`Ts_Corr`（时序相关系数）、`If`（条件表达式）

**关键特性**：
- 使用 `adjust="hfq"`（后复权）价格
- 在临时目录创建 Catalog，`finally` 块中清理
- 验证输出的 `factor_value`、`date`、`symbol` 列

**适用场景**：因子研究和 Alpha 表达式开发

---

### 20_risk_management_demo.py — 风险管理

**功能概述**：演示预交易风险管理规则，拒绝违反配置限制的订单。使用激进买入策略触发持仓规模限制和行业集中度限制。

**核心 API**：
- `aq.DataFeed` — 数据馈送
- `aq.Instrument(symbol, asset_type)` — 标的定义
- `aq.utils.fetch_akshare_symbol(symbol, start_date, end_date, adjust)` — 获取 AKShare 数据
- `aq.utils.load_bar_from_df(df)` — 将 DataFrame 转换为 Bar 列表
- `risk_config` 参数：`max_position_pct`（最大持仓比例）、`sector_concentration`（行业集中度）

**关键特性**：
- 两个行业：Consumer（600519，000858）和 Financial（601318）
- `max_position_pct=0.10` 限制单标的持仓不超过组合价值的 10%
- `sector_concentration=(0.15, sector_map)` 限制 Consumer 行业不超过 15%
- 通过 `orders_df["status"] == "rejected"` 和 `reject_reason` 列查看拒绝原因

**适用场景**：需要风控约束的回测策略

---

### 21_warm_start_demo.py — 热启动（断点续跑）

**功能概述**：演示热启动/Checkpoint-Resume 功能。将数据分为 2020-2021 和 2022-2023 两个阶段，第一阶段后保存快照，第二阶段从快照恢复。结果与完整单次回测对比验证一致性。

**核心 API**：
- `aq.save_snapshot(engine, strategy, path)` — 保存快照文件（`.pkl`）
- `aq.run_warm_start(checkpoint_path, data, ...)` — 从快照恢复并继续回测
- `self.register_indicator(name, indicator)` — 注册需要序列化的指标
- `self.is_restored` — 标记策略是否从快照恢复
- `self.on_resume()` — 恢复后触发的生命周期回调

**关键特性**：
- `run_warm_start` 无需传入策略类或 `initial_cash`（均从快照恢复）
- 验证 `buy_count` 和 `sell_count` 正确累积
- 验证最终权益与完整运行结果在容差范围内一致

**适用场景**：长时间回测的中断恢复、实盘状态持久化

---

### 22_strategy_runtime_config_demo.py — 策略运行时配置

**功能概述**：演示 `StrategyRuntimeConfig` 在回测时覆盖策略级设置的机制。三个场景：(1) 运行时配置覆盖策略默认值（`error_mode` 从 "raise" 改为 "continue"）；(2) `runtime_config_override=False` 阻止覆盖；(3) 热启动配合运行时配置覆盖。

**核心 API**：
- `aq.StrategyRuntimeConfig(error_mode="raise")` — 运行时配置类
- `run_backtest(strategy_runtime_config={"error_mode": "continue"})` — 传入运行时配置
- `run_backtest(runtime_config_override=False)` — 禁止配置覆盖
- `self.on_error(error, source, payload)` — error_mode 为 "continue" 时触发

**关键特性**：
- "raise" 模式下错误导致回测终止
- "continue" 模式下错误路由到 `on_error` 回调
- 警告去重测试

**适用场景**：不同环境下（回测/实盘）需要不同错误处理策略

---

### 23_functional_callbacks_demo.py — 函数式回调（类风格替代）

**功能概述**：演示函数式（非继承类）策略 API，使用普通 Python 函数作为回调。相比继承 `Strategy` 的方式更轻量。

**核心 API**：
- `run_backtest(strategy=on_bar, initialize=initialize, on_tick=..., on_order=..., on_trade=..., on_timer=...)` — 传入回调函数
- `ctx.buy()` / `ctx.sell()` / `ctx.get_position()` / `ctx.add_daily_timer()` — 上下文对象替代 `self`

**关键特性**：
- `ctx` 是策略上下文命名空间，同时持有用户状态和框架方法
- `initialize(ctx)` 在 `ctx` 上初始化状态
- 结果 `.strategy` 属性返回上下文对象，可从中读取状态

**适用场景**：偏函数式编程风格、快速原型开发

---

### 24_functional_tick_simulation_demo.py — 函数式 Tick 模拟

**功能概述**：演示使用 `FunctionalStrategy` 包装器直接手动分发 Tick 事件。不回测，而是手动构造 `Tick` 对象、设置订单和成交，调用 `_on_tick_event()` 验证回调链。

**核心 API**：
- `aq.akquant.StrategyContext` — 策略上下文
- `aq.akquant.Tick` — Tick 数据模型
- `aq.backtest.FunctionalStrategy` — 函数式策略包装器
- `strategy._on_tick_event(tick, ctx)` — 手动分发 Tick 事件
- `strategy._on_timer_event(payload, ctx)` — 手动分发定时器事件

**关键特性**：
- 使用自定义 `DemoContext` 类实现所需的 `get_position()` 方法
- Tick 时间戳带时区（Asia/Shanghai）
- 单元测试风格的验证示例

**适用场景**：理解内部事件分发机制、单元测试策略

---

## 流式回测与实时监控

### 25_streaming_backtest_demo.py — 流式回测错误模式

**功能概述**：演示流式回测的两种错误处理模式：`continue`（回调错误被记日志/容忍，在完成事件中报告）和 `fail_fast`（任何回调错误立即传播为异常）。

**核心 API**：
- `run_backtest(on_event=..., stream_progress_interval=..., stream_equity_interval=..., stream_batch_size=..., stream_max_buffer=..., stream_error_mode=...)`
- `akquant.BacktestStreamEvent` — 流式事件类型字典

**关键特性**：
- `continue` 模式下前两个事件有意触发 `RuntimeError`
- `fail_fast` 模式下首次回调错误立即抛出
- batch_size=8，max_buffer=64

**适用场景**：了解流式回测的容错机制

---

### 26_streaming_quickstart.py — 流式回测快速开始

**功能概述**：多标的流式回测快速入门。验证单调递增序列号、事件类型计数和每策略事件归属。策略使用 `order_target_percent` 分配 33% 仓位，实现 +10% 止盈和 100 根 Bar 时间止损。

**核心 API**：
- `BacktestConfig`、`StrategyConfig`、`RiskConfig`
- 流式参数：`stream_progress_interval`、`stream_equity_interval`、`stream_batch_size`、`stream_max_buffer`
- `fill_policy={"price_basis": "ohlc4", "temporal": "same_cycle"}`

**关键特性**：
- 三个真实 A 股标的
- 验证 `owner_strategy_id` 事件归属
- 验证序列号单调性

**适用场景**：流式回测的入门和理解

---

### 27_streaming_monitoring_console.py — 终端监控

**功能概述**：基于控制台的流式回测监控模式。运行多个策略配置（不同窗口大小的均线交叉），从流式回调捕获事件计数、进度事件和完成事件数据。

**核心 API**：
- `strategy_factory` 模式：用闭包创建带特定参数的策略类
- `BacktestStreamEvent` 字段：`event_type`、`seq`、`run_id`、`payload`

**关键特性**：
- 测试两组配置：(5, 20) 和 (8, 30)
- `StreamMonitor` dataclass 使用 `Counter` 跟踪事件计数
- 合成正弦波价格数据

**适用场景**：批量回测的进度监控

---

### 28_streaming_alerts_and_persist.py — 告警与事件持久化

**功能概述**：演示使用流式权益事件进行实时回撤监控和告警，附带 CSV 持久化。当回撤超过 3% 时生成告警并记录。

**核心 API**：
- 权益事件 payload：`payload.get("equity")` — 用于计算实时回撤
- `csv.DictWriter` 持久化事件到 CSV

**关键特性**：
- `AlertState` dataclass 跟踪 `peak_equity`、`max_drawdown` 和告警状态
- 回撤告警阈值为 -3%
- 事件输出到 `output/stream_alert_events.csv`
- 告警事件以合成 `alert_drawdown` 行注入

**适用场景**：实时风控告警和审计追踪

---

### 29_streaming_event_report.py — 事件报告生成

**功能概述**：加载示例 28 持久化的 CSV，使用 Plotly 生成交互式 HTML 报告。报告包含累积事件计数折线图和事件分布柱状图。

**核心 API**：
- `pandas.read_csv()` / `pivot_table` / `cumsum` — 事件数据聚合
- `plotly.graph_objects.Scatter` / `Bar` — 交互式图表
- `plotly.subplots.make_subplots` — 多子图布局

**关键特性**：
- 需要先运行示例 28 生成 CSV
- 累积图表展示事件计数随序列号的增长

**适用场景**：流式事件的离线分析和可视化

---

### 30_streaming_report_oneclick.py — 一键报告管线

**功能概述**：便捷启动器，依次运行示例 28 + 29（生成 CSV 后生成 HTML 报告），可选通过 HTTP 服务报告或浏览器打开。实现一键式管线。

**核心 API**：
- `subprocess.run()` — 调用其他示例脚本
- `http.server.ThreadingHTTPServer` — 本地 HTTP 服务
- `webbrowser.open()` — 浏览器预览
- `argparse` CLI：`--no-open`、`--serve`、`--port`、`--serve-seconds`

**关键特性**：
- HTTP 服务监听 `127.0.0.1`，默认端口 8765
- 纯编排脚本，不含回测逻辑

**适用场景**：自动化报告生成流程

---

### 31_streaming_live_console.py — 终端实时曲线

**功能概述**：终端实时仪表盘。使用 Unicode 迷你图（Sparkline）在终端内渲染权益曲线，并在出现回撤告警时打印。

**核心 API**：
- Unicode Sparkline 渲染（块字符 `▁▂▃▄▅▆▇█`）
- 使用 `\r` 回车符实现原地终端更新

**关键特性**：
- 42 字符宽度的 Sparkline，随权益事件更新
- 回撤告警阈值 -3%
- `flush=True` + 回车符实现平滑动画效果

**适用场景**：命令行环境下的实时回测监控

---

### 32_streaming_live_web.py — 浏览器实时曲线

**功能概述**：完整的实时网页仪表盘。在后台线程运行流式回测，通过 HTTP JSON 端点（`/state`）暴露状态。浏览器端通过 JavaScript 定期轮询并使用 Canvas 绘制实时权益曲线。

**核心 API**：
- `run_backtest` 在守护线程中运行
- `http.server.BaseHTTPRequestHandler` + `ThreadingHTTPServer`
- `threading.Lock` + `threading.Event` — 线程安全共享状态
- Canvas 渲染（客户端 JavaScript，260ms 轮询 / 90ms 渲染）

**关键特性**：
- 回测线程每次事件后休眠 `sleep_ms`（默认 20ms）以慢放视觉效果
- 权益曲线绘制在 HTML5 Canvas 上
- 回撤告警推送到浏览器
- `--port`、`--open`、`--sleep-ms`、`--keep-seconds` CLI 参数

**适用场景**：演示级别的实时可视化

---

## 报告与分析

### 33_report_and_analysis_outputs.py — 报告与分析输出

**功能概述**：演示回测后的报告和分析 API。在合成数据上运行简单策略后，生成 HTML 报告，并提取敞口、归因（按标的和标签）、容量、订单和成交明细 DataFrame。

**核心 API**：
- `result.report(filename, show=False, compact_currency=True)` — 生成 HTML 报告
- `result.exposure_df()` — 持仓敞口时序数据
- `result.attribution_df(by="symbol")` / `result.attribution_df(by="tag")` — PnL 归因分析
- `result.capacity_df()` — 容量分析
- `result.orders_by_strategy()` — 按策略分组订单
- `result.executions_by_strategy()` — 按策略分组成交

**关键特性**：
- 验证所有分析输出方法返回预期行数的 DataFrame

**适用场景**：了解回测结果的结构化分析能力

---

## 多策略与经纪人

### 34_multi_strategy_demo.py — 多策略 Slot

**功能概述**：演示多策略（多 Slot）功能 —— 多个策略共享相同标的，每个策略有独立的 `owner_strategy_id`、订单规模限制、风控后仅平仓标志和冷却期。对比单 Slot vs 多 Slot 运行，验证基于配置限制的订单拒绝。

**核心 API**：
- `StrategyConfig(strategies_by_slot={"beta": BetaStrategy})` — 注册辅助策略
- `strategy_max_order_size={"alpha": 5.0, "beta": 20.0}` — 每策略订单规模上限
- `strategy_reduce_only_after_risk={"alpha": True}` — 风控后仅允许平仓
- `strategy_risk_cooldown_bars={"alpha": 2}` — 风控后冷却期（Bar 数）

**关键特性**：
- 使用受控价格序列触发确定性行为
- 通过 `reject_reason` 列检查拒绝原因（"cooldown"、"order quantity"）

**适用场景**：多策略并行运行和策略级风控

---

### 35_custom_broker_registry_demo.py — 自定义经纪人注册

**功能概述**：演示自定义经纪人注册 API。创建最小化的 `_DemoMarketGateway` 和 `_DemoTraderGateway` 实现，注册构建函数，创建网关包，然后注销经纪人。

**核心 API**：
- `aq.gateway.register_broker(name, builder_fn)` — 注册自定义经纪人
- `aq.gateway.unregister_broker(name)` — 注销经纪人
- `aq.gateway.create_gateway_bundle(broker, feed, symbols, ...)` — 创建网关包
- `aq.gateway.GatewayBundle` — 包含 `market_gateway`、`trader_gateway`、`metadata`
- `aq.gateway.models.*` — 统一数据模型（`UnifiedAccount`、`UnifiedOrderRequest`、`UnifiedPosition` 等）

**关键特性**：
- 网关为最小存根实现
- `builder_fn` 接收 `feed`、`symbols`、`use_aggregator` 等参数
- 演示完整的注册→创建→注销流程

**适用场景**：集成自定义券商/交易接口

---

## 订单与数据对齐

### 36_trailing_orders.py — 跟踪止损订单

**功能概述**：在回测中演示跟踪止损/跟踪止损限价单。策略买入后立即使用 `place_trailing_stop` 放置跟踪止损单。

**核心 API**：
- `self.place_trailing_stop(symbol, quantity, side, reference_price, offset, ...)` — 放置跟踪止损单
- `self.on_order(order)` — 处理订单取消/拒单的清理

**关键特性**：
- 合成先涨后跌价格数据（含噪声）
- `trail_offset` 可配置
- 完整生命周期：入场 → 成交 → 跟踪止损 → 止损失效触发

**适用场景**：需要动态止损保护的策略

---

### 37_feed_replay_alignment_demo.py — 数据重放对齐

**功能概述**：展示使用 Feed 重放系统对不同对齐模式（session、global、day）和 session 窗口定义的 OHLCV 数据对齐和重采样。

**核心 API**：
- `BasePandasFeedAdapter` — 基类
- `adapter.replay(freq, align, emit_partial, session_windows, day_mode)` — 重放配置
- `normalize()` / `_clip_time_range()`

**关键特性**：
- 五种重放配置对比：
  1. session align（默认）
  2. session align + 自定义交易时段窗口（`09:30-11:30`、`13:00-15:00`）
  3. global align
  4. day align（trading 模式）
  5. day align（calendar 模式）
- 数据包含上午和下午时段，中间有间隔以演示 session 感知聚合

**适用场景**：理解数据时间对齐和多频率聚合的底层机制

---

## 实盘交易

### 38_live_functional_strategy_demo.py — 函数式实盘策略

**功能概述**：演示在 `LiveRunner` 中以模拟交易模式运行函数式（非继承类）策略。使用独立函数而非 `Strategy` 子类。

**核心 API**：
- `LiveRunner` 配合：`initialize()`、`on_bar()`、`on_order()`、`on_trade()`、`on_timer()`
- `ctx.add_daily_timer("14:50:00", "close_check")` — 添加每日定时器

**关键特性**：
- 配置 CTP 期货（IF2506），`trading_mode="paper"`
- `use_aggregator=True`
- 交替每根 Bar 买卖
- 使用 `tcp://127.0.0.1:12345` 作为占位网关地址

**适用场景**：函数式策略直接用于实盘模拟

---

### 39_live_broker_submit_order_demo.py — 实盘下单

**功能概述**：展示在 `broker_live` 模式下通过 `ctx.submit_order` 提交订单。`submit_order` 在交易网关连接后自动注入策略上下文。

**核心 API**：
- `ctx.submit_order(symbol, side, quantity, client_order_id, order_type)` — 实盘下单
- `trading_mode="broker_live"` — 真实交易模式

**关键特性**：
- 需要完整 CTP 凭证（broker_id、user_id、password、app_id、auth_code）
- 首根 Bar 提交一笔市价单
- 生成顺序递增的 `client_order_id`

**适用场景**：从模拟到实盘下单的迁移参考

---

### 40_functional_multi_slot_risk_demo.py — 函数式多 Slot 风控

**功能概述**：演示函数式回测中的多策略 Slot 和每 Slot 风控限制。两个函数式策略（alpha 和 beta）并行运行，alpha 配置了低的 `strategy_max_order_value` 导致订单被拒，beta 的订单通过。

**核心 API**：
- `strategies_by_slot` + `strategy_max_order_value`
- 流式事件检查风控事件

**关键特性**：
- alpha 的 `strategy_max_order_value=50.0` 拒单（10 单位 × ~10 价格 = 100 价值）
- beta 的 `200.0` 允许订单
- 通过 `reject_reason` 列和流式 `event_type == "risk"` 验证

**适用场景**：多策略场景下的差异化风控

---

### 41_live_multi_slot_orchestration_demo.py — 实盘多 Slot 编排

**功能概述**：在 `LiveRunner` 模拟模式中展示多 Slot 策略编排。主 Slot 使用函数式 `on_bar` 回调，辅助 Slot 使用类风格 `Strategy` 子类。

**核心 API**：
- `LiveRunner` + `strategy_id` + `strategies_by_slot`
- 混合使用函数式和类风格策略

**关键特性**：
- 主 slot（"alpha"）为函数式风格
- 辅助 slot（"beta"）为类风格 `SecondarySlotStrategy`
- 两个 slot 各买入 1 单位
- 连接 CTP 模拟交易

**适用场景**：混合编程风格的实盘编排

---

### 42_live_broker_event_audit_demo.py — 经纪人事件审计

**功能概述**：演示通过 `on_broker_event` 回调进行统一经纪人事件审计。所有订单/成交/报告事件路由到单一回调，通过 `owner_strategy_id` 实现多 Slot 归属。

**核心 API**：
- `LiveRunner` 的 `on_broker_event` 回调
- 事件 dict 包含 `event_type`、`owner_strategy_id`、`payload`

**关键特性**：
- 使用 `broker_live` 模式和占位 CTP 凭证
- 提交单笔市价买单

**适用场景**：多策略场景下的集中化监控和审计

---

## 权重调仓与动态加载

### 43_target_weights_rebalance.py — 目标权重调仓

**功能概述**：演示使用 `order_target_weights` 进行组合级目标权重再平衡。策略在三个标的的横截面中按 3 日动量选择前 2 名，以等权重再平衡。

**核心 API**：
- `self.order_target_weights(target_weights, liquidate_unmentioned=True, rebalance_tolerance=...)` — 按权重调仓
- `self.get_history(count, symbol, field)` — 获取历史数据计算动量

**关键特性**：
- `liquidate_unmentioned=True` 清零不在目标集中的持仓
- 使用 bucket 模式：在同一时间戳同步各标的后再进行横截面动量计算
- `history_depth` 参数支持 `get_history`

**适用场景**：动量轮动和组合再平衡策略

---

### 44_strategy_source_loader_demo.py — 策略动态加载

**功能概述**：演示使用 `strategy_source` 和 `strategy_loader` 参数动态加载策略。两个场景：(1) 从 Python 源文件加载（`python_plain` 加载器）；(2) 通过回调函数从加密外部源加载（`encrypted_external` 加载器）。

**核心 API**：
- `run_backtest(strategy=None, strategy_source=..., strategy_loader="python_plain", strategy_loader_options={"strategy_attr": "ClassName"})`
- `run_backtest(strategy=None, strategy_source=b"...", strategy_loader="encrypted_external", strategy_loader_options={"decrypt_and_load": callback_fn})`

**关键特性**：
- `python_plain` 加载器：运行时读取 `.py` 文件并实例化指定类名
- `encrypted_external` 加载器：将源字节传给用户提供的解密函数
- 两种场景 `strategy=None`（策略类来自加载器）

**适用场景**：策略代码保护和动态分发

---

## 指标与保证金

### 45_talib_indicator_playbook_demo.py — TA-Lib 指标组合

**功能概述**：演示组合多个 TA-Lib 指标（通过 `akquant.talib`）构建规则化交易系统。支持合成数据和真实 AKShare 数据两种数据源。

**核心 API**：
- `aq.talib.EMA`、`.ADX`、`.NATR`、`.BBANDS`、`.RSI`、`.MOM` — TA-Lib 指标
- `self.get_history(count, symbol, field)` — 获取历史数据
- `argparse` CLI：`--data-source synthetic|akshare`、`--symbol`、`--start-date`、`--end-date`

**关键特性**：
- 使用 `backend="rust"` 参数进行指标计算
- 组合**趋势跟随**信号（EMA 交叉 + ADX + NATR）和**均值回归**信号（布林带下轨 + RSI + MOM）
- 多种退出条件：上穿中轨、EMA 下穿、RSI 超 72
- 需要 90 根 Bar 的预热期

**适用场景**：多指标组合的策略模板和学习 TA-Lib 用法

---

### 46_broker_profile_demo.py — 经纪人费率模板

**功能概述**：演示使用 `broker_profile` 应用预配置费用结构（`cn_stock_t1_low_fee`）。策略在启动时打印解析的费率和手数参数。

**核心 API**：
- `run_backtest(broker_profile="cn_stock_t1_low_fee")`
- `self.commission_rate`、`self.stamp_tax_rate`、`self.transfer_fee_rate`、`self.min_commission`、`self.lot_size`

**关键特性**：
- `cn_stock_t1_low_fee` 是一个命名预设，自动设置佣金、印花税、过户费和最低佣金
- 无需手动传入费用参数

**适用场景**：快速应用标准费率配置

---

### 47_margin_liquidation_audit_demo.py — 保证金强平审计

**功能概述**：演示保证金账户的强制平仓机制。策略开立杠杆多头仓位，价格暴跌触发维持保证金阈值后由系统自动强平。

**核心 API**：
- `RiskConfig(account_mode="margin", enable_short_sell=True, initial_margin_ratio=..., maintenance_margin_ratio=..., allow_force_liquidation=True, liquidation_priority="short_first")`
- `self.get_account()` — 获取账户状态（`account_mode`、`cash`、`borrowed_cash`、`maintenance_ratio`、`accrued_interest`）
- `result.liquidation_audit_df` — 强平审计数据
- `result.report(filename=..., show=...)` — 生成 HTML 报告

**关键特性**：
- 初始现金 10,000；以 100 价格买入 150 单位（需要保证金）
- 第二根 Bar 价格跌至 20，触发 50% 维持保证金阈值
- 强制平仓优先级为 `"short_first"`

**适用场景**：信用交易和强制平仓审计

---

### 48_margin_liquidation_priority_compare.py — 强平优先级对比

**功能概述**：在持有双向仓位的保证金账户中对比 `short_first` vs `long_first` 强平优先级。展示不同优先级设置下哪个标的优先被强平。

**核心 API**：
- `RiskConfig` 的 `liquidation_priority` 参数
- `result.liquidation_audit_df` — 查看最后一行 `liquidated_symbols`

**关键特性**：
- 运行两次回测，仅 `liquidation_priority` 不同
- 维持保证金率设为 4.0 以强制立即强平
- `LONG` 买入 100 股，`SHORT` 卖出 50 股
- 输出最后一次审计的 `liquidated_symbols` 对比差异

**适用场景**：理解保证金强平顺序的影响

---

## 生命周期钩子

### 49_on_expiry_demo.py — 到期处理

**功能概述**：演示期货合约到期处理。使用 `on_expiry` 回调和 `BacktestConfig` + `InstrumentConfig`。策略买入一个后来到期的期货合约。

**核心 API**：
- `InstrumentConfig(asset_type="FUTURES", expiry_date=..., settlement_type="cash", settlement_price=...)`
- `self.on_expiry(symbol, expiry_date, quantity_closed, cash_flow, settlement_type)` — 到期回调
- `BacktestStreamEvent` 的 `event_type == "expiry"` 流式事件

**关键特性**：
- 期货合约 `FUT_EXP_DEMO` 在 20260131 到期，现金结算价 108.0
- 到期后持仓自动归零

**适用场景**：期货/期权到期处理

---

### 50_framework_hooks_demo.py — 框架级钩子

**功能概述**：演示类风格 `Strategy` 中所有框架级生命周期钩子。涵盖 session 开始/结束、交易前/后、每日再平衡、组合更新、订单拒绝和停止。

**核心 API**：
- `self.on_session_start()` / `self.on_session_end()` — 交易时段回调
- `self.on_before_trading()` / `self.on_after_trading()` — 交易前后回调
- `self.on_daily_rebalance()` — 每日再平衡回调
- `self.log(message)` — 策略日志
- `enable_precise_day_boundary_hooks = True` — 启用精确日内边界钩子

**关键特性**：
- 刻意提交超大订单（`strategy_max_position_size={"default": 10}`）以触发 `on_reject`
- 数据覆盖两个交易日，含上午和下午 Bar
- `on_stop` 打印完整事件序列

**适用场景**：全面了解策略生命周期钩子的触发顺序

---

### 51_class_tick_callbacks_demo.py — 类风格 Tick 回调

**功能概述**：演示类风格的 `on_tick` 回调处理，连同订单、成交和定时器回调。通过直接调用 `_on_tick_event()` 模拟 Tick 事件。

**核心 API**：
- `self.on_tick(tick)` — Tick 数据回调
- `akquant.akquant.Tick` — Tick 数据模型
- `strategy._on_tick_event(tick, ctx)` — 手动触发 Tick 事件

**关键特性**：
- 使用最小 `DemoContext` 满足内部回调机制
- 模拟两个 Tick 事件和一个定时器事件
- 验证 `on_tick` 分发与相邻回调正常工作

**适用场景**：Tick 级别策略开发

---

### 52_pre_open_demo.py — 盘前决策

**功能概述**：演示 `on_pre_open` 回调 —— 策略在开盘前下决策单，在即将到来的 Bar 的开盘价成交。

**核心 API**：
- `self.on_pre_open(event)` — 盘前回调，`event` 包含 `trading_date` 和 `expected_open_at`
- `self.format_time(timestamp)` — 格式化时间

**关键特性**：
- 使用 `self.submitted_dates` 跟踪已提交日期，避免重复下单
- `on_pre_open` 中的订单在下根 Bar 的开盘价执行
- 两根日线 Bar 演示重复模式

**适用场景**：盘前决策、次日开盘执行策略

---

### 53_timer_to_pre_open_demo.py — 定时器到盘前决策

**功能概述**：演示多日工作流 —— 15:00 定时器准备次日计划，`on_pre_open` 次日早上执行。

**核心 API**：
- `self.add_daily_timer("15:00:00", "prepare_next_day")` — 收盘前定时器
- `self.on_timer(payload)` — 定时器回调：检查收盘价，暂存次日计划
- `self.on_pre_open(event)` — 盘前回调：读取暂存计划并执行

**关键特性**：
- 收盘价 >= 10.5 时暂存 "buy" 计划
- 三根日线 Bar 覆盖买入和持有场景
- 真实工作流：当日收盘分析驱动次日交易决策

**适用场景**：T+1 市场中的日终分析+次日执行模式

---

### 54_functional_pre_open_demo.py — 函数式盘前决策

**功能概述**：盘前决策示例的函数式版本。使用函数式回调实现与示例 52 相同的 "盘前决策，开盘成交" 模式。

**核心 API**：
- `run_backtest(strategy=on_bar, on_pre_open=on_pre_open, ...)`
- `initialize(ctx)` 中初始化 `ctx.submitted_dates`

**关键特性**：
- 精确镜像示例 52 但使用函数式回调
- 展示函数式和类风格策略之间的特性对等

**适用场景**：偏好函数式风格的盘前策略开发者

---

## 函数式高级用法

### 55_functional_ml_walk_forward.py — 函数式 ML Walk-Forward

**功能概述**：演示函数式策略的 Walk-Forward 机器学习验证。`LogisticRegression` 模型在滚动窗口上训练，对样本外 Bar 进行推理。

**核心 API**：
- `run_backtest(on_train_signal=on_train_signal, ...)` — 传入训练信号回调
- `ctx.model.set_validation(method="walk_forward", train_window=50, test_window=20, rolling_step=10, frequency="1m")`
- `ctx.get_rolling_data()` — 获取滚动窗口数据
- `ctx.current_validation_window()` — 当前验证窗口信息
- `ctx.model.fit(X, y)` / `ctx.model.predict(X)` — 模型训练和预测
- `ctx.get_history_df(count)` — DataFrame 格式历史数据

**关键特性**：
- `train_window=50`、`test_window=20`、`rolling_step=10`，1 分钟频率
- 特征：`ret1`（1 期收益）、`ret2`（2 期收益）
- 标签：下一期收益是否为正
- 预测阈值：0.55 买入、0.45 卖出
- 生成 20,000 根合成随机游走 Bar

**适用场景**：函数式风格的 ML Walk-Forward 策略

---

### 56_functional_warm_start_demo.py — 函数式热启动

**功能概述**：演示函数式策略的热启动（Checkpoint/Resume）能力。分两阶段运行回测，使用 `save_snapshot` 保存和 `run_warm_start` 恢复。

**核心 API**：
- `save_snapshot(engine, strategy, path)` — 保存快照
- `run_warm_start(checkpoint_path, data, ...)` — 从快照恢复
- `ctx.is_restored` — 标记是否为热启动
- `on_resume(ctx)` — 恢复时触发

**关键特性**：
- 第一阶段处理 2 根 Bar，保存 checkpoint 到临时文件
- 第二阶段从 checkpoint 恢复，再处理 2 根 Bar
- `on_resume` 仅在 warm start 时触发
- 策略的 `processed_closes` 列表跨两阶段完整延续

**适用场景**：函数式策略的状态持久化

---

### 57_functional_multi_slot_warm_start_demo.py — 函数式多 Slot 热启动

**功能概述**：将热启动模式扩展到多 Slot 场景。演示两个策略 Slot（alpha 和 beta）的函数式 Checkpoint/Resume。

**核心 API**：
- `run_backtest(strategy=alpha_on_bar, strategy_id="alpha", strategies_by_slot={"beta": beta_on_bar})`
- `save_snapshot(...)` / `run_warm_start(...)`
- `alpha_strategy._slot_strategies` — 访问 slot 策略

**关键特性**：
- alpha 和 beta slot 各有独立的 `processed_closes`、`events`、`start_count`、`resume_count`
- checkpoint 保存和恢复所有 slot
- 恢复后两个 slot 的状态数组显示两阶段的完整数据

**适用场景**：多策略并行场景的断点续跑

---

### 58_incremental_bootstrap_demo.py — 增量指标引导

**功能概述**：演示增量指标的历史数据预热引导 —— 在回测的 `start_time` 前用历史数据预热指标，然后无缝切换到实时更新。

**核心 API**：
- `self.register_incremental_indicator(name, indicator_factory, source, symbols, warmup_bars)` — 注册增量指标
- `aq.SMA(period)` — 增量型指标
- `runtime_config = {"indicator_mode": "incremental"}`
- `run_backtest(start_time=..., end_time=...)`

**关键特性**：
- `warmup_bars=3` 加载足够的预 `start_time` 历史数据预热 SMA(3)
- `indicator_factory` lambda 为每个标的创建独立指标实例
- 从第一根活跃 Bar 起 SMA 值即已计算（非 NaN）

**适用场景**：需要指标在回测开始时就具有有效值的场景

---

## 策略集合

### strategies/01_stock_dual_moving_average.py — 双均线选股

**策略逻辑**：经典双均线（金叉/死叉）趋势跟随策略。快速均线上穿慢速均线 → 买入 95% 仓位；快速均线下穿慢速均线 → 平仓。

**核心 API**：`self.get_history(count, symbol, "close")`、`self.order_target_percent(0.95, symbol)`、`self.close_position(symbol)`

**关键特性**：
- A 股默认手数 100 股由引擎自动处理
- 佣金 0.03%（最低 5 元），印花税 0.1%（仅卖出）
- 使用 numpy 做均线计算

---

### strategies/02_stock_grid_trading.py — 网格交易

**策略逻辑**：在波动标的上做固定网格交易，当价格变动达到配置的百分比阈值（3%）时买/卖固定手数。实现 "跌买涨卖" 模式。

**核心 API**：`self.buy(symbol, 100)`、`self.sell(symbol, 100)`、`self.get_position(symbol)`

**关键特性**：
- 初始持仓 10 手（1000 股）
- 自行维护 `last_trade_price` 字典跟踪最近成交价
- 需要充足初始资金（500,000）应对多层网格

---

### strategies/03_stock_atr_breakout.py — ATR 突破

**策略逻辑**：平均真实波幅（ATR）通道突破策略。从 high/low/close 历史数据手工计算 ATR，价格突破上通道买入，跌破下通道卖出。

**核心 API**：`self.get_history(count, symbol, "high"/"low"/"close")`

**关键特性**：
- ATR 使用 Python 循环计算（简化版，非 EMA 平滑）
- 显式切片 `[:-1]` 排除当前 Bar 避免前视偏差
- 固定 500 股买入量

---

### strategies/04_stock_momentum_rotation.py — 动量轮动（Bar 级别）

**策略逻辑**：在两只白酒股间的动量轮动。计算 N 日动量，选择最强标的持有。

**核心 API**：`aq.run_backtest(data=dict_of_dataframes, ...)`、`self.order_target_percent(0.95, symbol)`

**关键特性**：
- 仅当 `bar.symbol == self.symbols[-1]`（当日最后标的的 Bar）时触发逻辑
- 全部负动量时清仓
- 多标的数据使用 `dict` 格式传入

---

### strategies/05_stock_momentum_rotation_timer.py — 动量轮动（日度再平衡）

**策略逻辑**：使用 `on_daily_rebalance` 回调实现动量轮动 —— 每个交易日触发一次，避免 Bar 到达顺序问题。

**核心 API**：`self.on_daily_rebalance(trading_date, timestamp)`、`self.get_history_map(count, symbols, "close")`、`self.rebalance_to_topn(scores, top_n, ...)`

**关键特性**：
- 使用合成数据，无外部依赖，快速确定
- `rebalance_to_topn` 自动处理所有下单和平仓

---

### strategies/06_stock_momentum_rotation_bucket.py — 动量轮动（Bucket 模式）

**策略逻辑**：收集同一时间戳所有标的的 Bar 后再执行再平衡 —— "bucket" 手动方法。

**核心 API**：`self._pending_by_ts`（用户管理的 `defaultdict(set)`）

**关键特性**：
- 三只标的（茅台、五粮液、中国平安）
- 手动跟踪每时间戳已到达的标的
- 官方推荐使用 `on_daily_rebalance` / `on_timer` 替代本模式

---

### strategies/07_stock_momentum_rotation_on_timer.py — 动量轮动（定时器）

**策略逻辑**：使用 `on_timer` 配合固定每日时间触发执行再平衡，解除再平衡逻辑与 Bar 到达顺序的耦合。

**核心 API**：`self.add_daily_timer("10:00:00", "rebalance")`、`self.on_timer(payload)`、`self.get_history_map(count, symbols, field)`、`self.rebalance_to_topn(...)`

**关键特性**：
- 使用合成数据，无外部依赖
- **推荐的时间驱动轮动方式**
- 在可预测的每日固定时间触发

---

## 教科书示例

`textbook/` 目录包含 16 章教学示例，涵盖从零基础到高级主题的完整学习路径：

### ch01_quickstart.py — 快速入门

最基础的入门示例。获取浦发银行日线数据，实现双均线（MA5 vs MA20）金叉死叉策略。覆盖完整回测流程：数据获取 → 策略定义 → 回测运行 → 结果输出。

---

### ch02_programming.py — Python 编程基础

纯编程/教学章节，不涉及 AKQuant 回测。覆盖 Pandas（DataFrame 创建、缺失值填充 `ffill`、滚动窗口 MA、重采样）、NumPy（向量化计算性能）、类型提示。为非编程背景的读者补 Python 基础。

---

### ch03_data.py — 数据处理流水线

完整的数据工程流水线：从 AKShare 获取数据 → 清洗（中英文列名转换、类型转换、NaN 处理、排序）→ Parquet 格式存储和加载。

**关键知识点**：
- AKQuant 标准列格式：`date, open, high, low, close, volume`
- 多标的回测需添加 `symbol` 列
- Parquet 优于 CSV 的性能推荐

---

### ch04_comparison.py — 框架对比

用**相同双均线策略**对比三种回测框架：
1. **Pandas 向量化** — 最快但交易模拟简化
2. **Backtrader** — 经典 Python 事件驱动框架，最慢
3. **AKQuant** — Rust 驱动的事件引擎

使用 3000 根合成 Bar 生成随机游走序列。AKQuant 使用 `closes[:-1]` 避免前视偏差。

---

### ch05_strategy.py — 策略生命周期

完整策略生命周期演示：`__init__` → `on_start` → `on_bar` → `on_stop`。实现带 **5% 止损**的双均线策略。引入 `self.log()` 结构化日志和 `self.entry_price` 记录入场价计算浮动 PnL。

---

### ch06_stock_a.py — A 股市场规则

专注中国 A 股市场规则：**T+1 交收**、**涨跌停**、**手数**。策略尝试每日买卖以演示 T+1 阻止当日卖出。

**核心知识点**：
- `self.get_available_position(symbol)` 返回当日可卖股数（买入当日返回 0）
- `t_plus_one=True` 参数启用 T+1 约束

---

### ch07_futures.py — 期货交易

期货交易演示：**保证金**、**杠杆**、**做空**（`short()`）、**合约乘数**。使用螺纹钢期货模拟合约（RB2310）。

**核心知识点**：
- 期货需显式配置 `InstrumentConfig`（`multiplier`、`margin_ratio`）
- `ChinaFuturesConfig` 提供合约模板
- `fill_policy` 设为 `same_cycle close`（期货策略典型配置）
- `slippage` 以百分比指定（0.02%）

---

### ch08_options.py — 期权策略

**备兑看涨（Covered Call）**策略演示：买入 1 手螺纹钢期货 + 卖出 1 张虚值看涨期权。

**核心知识点**：
- 期权合约代码格式：`RB2310-C-3800`（标的-行权价-类型）
- `ChinaOptionsConfig` 配置期权费率
- 买方保证金为 0，卖方保证金由引擎计算
- 数据以单 DataFrame 传入，通过 `symbol` 列区分

---

### ch09_funds.py — ETF 网格交易

ETF 网格交易策略：20 日均线为中枢，下跌 1% 买入、上涨 1% 卖出。

**核心知识点**：
- ETF 特有费率：低佣金（0.01%）、**零印花税**
- 使用正弦波价格模拟震荡市
- `last_buy_price` 动态更新实现浮动网格参考点

---

### ch09_portfolio.py — 60/40 股债配置

经典 **60/40 股债组合**再平衡策略。股票 ETF（510300）和债券 ETF（511010），每 20 个交易日（~月频）再平衡。

**核心知识点**：
- `self.get_portfolio_value()` 获取组合总市值
- `self.order_target_value(target_value, symbol)` 调整至目标市值
- 再平衡仅在股票标的的 Bar 上触发以避免重复执行

---

### ch10_analysis.py — 绩效分析

回测后绩效分析详解：从 `result.metrics_df` 提取关键指标，从 `result.trades_df` 分析逐笔交易（胜率、平均 PnL），生成 HTML 报告。

**核心 API**：`result.metrics_df`、`result.trades_df`、`result.equity_curve` / `equity_curve_daily`、`result.report(curve_freq="D")`

---

### ch11_optimization.py — 参数优化

通过 `aq.run_grid_search()` 对双均线策略参数进行网格搜索优化。优化参数 `short_window` [3,5,10] 和 `long_window` [15,20,30,60]，目标为最大化 Sharpe 比率。

**核心知识点**：
- `max_workers=4` 限制并行进程数
- `param_grid` 键名需匹配策略构造函数参数名
- `warmup_period` 动态设为 `long_window + 1`

---

### ch12_ml.py — 机器学习集成

将 **scikit-learn LogisticRegression** 集成入回测。特征：1 日收益、5 日收益、20 日均线偏离。模型每 20 根 Bar 在滚动 200 根窗口上重训练。

**核心知识点**：
- `self.get_history_df(count, symbol)` 返回 DataFrame 用于特征工程
- `self.calculate_features(df)` 用户自定义特征计算方法
- `StandardScaler` + `LogisticRegression` 流水线
- `HAS_SKLEARN` 守卫优雅处理环境缺失

---

### ch13_visualization.py — 可视化与基准对比

使用 `result.report()` 生成带基准对比的 HTML 报告。运行简单 MA 突破策略，生成含基准收益序列、市场数据 K 线图和交易标记的完整报告。

**核心知识点**：
- `result.report(benchmark=benchmark_series, market_data=df, plot_symbol=..., include_trade_kline=True)`
- 基准序列从收盘价百分比变化构建

---

### ch14_factor.py — 因子表达式

因子表达式引擎（`FactorEngine` + `ParquetDataCatalog`）演示。计算 Alpha 因子表达式：`Ts_Mean(Close, 5)`、`Ts_Std(Close, 20)`、`Rank(Volume)`、`Rank(Ts_Corr(Close, Volume, 10))`。

**核心知识点**：
- `catalog.write(symbol, df)` 写入标的数据
- `engine.run(expression)` 计算单条表达式
- `engine.run_batch(expressions)` 批量计算
- 因子的 DSL 类似 WorldQuant / DolphinDB 风格

---

### ch15_live_trading.py — 实盘交易配置

使用 `LiveRunner` + CTP 连接进行期货实盘交易。在螺纹钢期货上实现双均线策略。

**核心知识点**：
- SimNow CTP 测试环境地址
- 凭证为占位值（`YOUR_USER_ID` 等）
- 策略逻辑与回测完全一致，仅换运行器
- 需要 `akquant[ctp]` 扩展安装

---

### ch15_strategy_loader.py — 策略动态加载

演示两种策略加载方式：(1) `python_plain` — 从源文件路径加载策略类；(2) `encrypted_external` — 通过外部回调从加密载荷解析策略类。

**核心知识点**：
- 两种场景 `strategy=None`（类来自加载器）
- `result.strategy` 保存加载后的策略实例
- 使用 `TemporaryDirectory` 处理临时文件

---

## 工具文件

### benchmark_utils.py — 基准测试工具

**功能概述**：使用几何布朗运动生成大规模基准测试 OHLCV 数据。提供 `print_report` 打印标准化的性能指标。

**核心 API**：
- `get_benchmark_data(n, symbol, freq, start_time, seed)` — 返回合成 OHLCV DataFrame
- `print_report(engine_name, duration, total_bars, total_return_pct, total_trades)` — 标准基准输出

**关键特性**：
- 使用标的名称 MD5 哈希作为确定性种子
- 确保 OHLC 一致性（`high >= max(open, close)` 等）
- 时区为 UTC（15:00 偏移）
- 吞吐量以 Bars/秒计

---

### pb_mock.py — ML 策略示例

**功能概述**：高级 ML 策略示例。使用 `SklearnAdapter` 包装 scikit-learn `Pipeline`（StandardScaler + LogisticRegression）配合 Walk-Forward 验证。

**核心 API**：
- `SklearnAdapter(pipeline)` — 包装 scikit-learn pipeline
- `self.model.set_validation(method="walk_forward", train_window=50, rolling_step=10)`
- `self.prepare_features(df, mode)` — **必须实现**，处理 "training" 和 "inference" 两种模式
- `self.set_history_depth(60)` — 历史缓冲区大小需大于训练窗口
- `self.get_history_df(count)` — 获取近期数据进行推理

**关键特性**：
- `prepare_features` 是合约点：training 模式返回 `(X, y)`，inference 模式返回最后一行 `X`
- 框架自动处理数据分割和模型重训练
- 2 个特征：`ret1` 和 `ret2`
- 信号阈值：0.55 买入、0.45 卖出

---

## 数据说明

- **真实数据**：依赖 `akshare` 获取 A 股（日线/分钟线）、期货、期权等历史数据
- **合成数据**：使用随机游走（`np.cumsum(np.random.randn())`）、正弦波或几何布朗运动生成，适用于快速验证和单元测试
- **数据格式**：标准列名为 `date, open, high, low, close, volume`，多标的需包含 `symbol` 列
- **复权处理**：使用 AKShare 的 `adjust="qfq"`（前复权）或 `adjust="hfq"`（后复权）
- **文件存储**：推荐 Parquet 格式（性能优于 CSV），使用 `pd.to_parquet()` / `pd.read_parquet()`

## 运行要求

- Python >= 3.9
- akquant >= 0.2.x（`pip install akquant`）
- 可选依赖：
  - `akshare` — 真实数据获取
  - `scikit-learn` — ML 示例（09、10、12、55）
  - `torch` — PyTorch ML 示例（09）
  - `plotly` — 交互式图表（29、32）
  - `quantstats` — QuantStats 报告（13）
  - `openctp-ctp` — CTP 实盘交易（05、38-42）
