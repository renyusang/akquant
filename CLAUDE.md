# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AKQuant is a high-performance quantitative trading framework with a **Rust core engine** exposed to Python via **PyO3/maturin**. The Rust layer handles the event loop, order matching, portfolio accounting, and risk management. The Python layer provides the strategy API, configuration system, plotting, ML integration, TA-Lib indicators, and factor expressions.

## Build & Development Commands

```bash
# Install dev dependencies and compile the Rust extension (first time setup)
uv pip install -e ".[dev,ml,plot]"
uv run maturin develop

# Recompile after Rust changes (fast incremental)
uv run maturin develop

# Run all Python tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_engine.py

# Run a single test by name (keyword match)
uv run pytest tests/test_engine.py -k "test_name_pattern"

# Run Rust tests only
bash scripts/cargo-test.sh

# Lint & type-check
uv run ruff check python/akquant tests
uv run mypy python/akquant

# Run all pre-commit checks (ruff, mypy, docs links, docs API examples)
uv run pre-commit run --all-files

# Run standalone scripts from examples/
uv run python examples/01_quickstart.py
```

## Architecture

### Two-Layer Design

- **Rust core** (`src/`, `Cargo.toml`): The backtest engine, data structures, and all performance-critical logic. Compiled as a native Python extension module (`akquant.akquant`) via PyO3. All core types (Bar, Tick, Order, Trade, Instrument, Engine, Portfolio, etc.) are Rust structs exposed as Python classes via `#[pyclass]`.
- **Python layer** (`python/akquant/`): Wraps the Rust module with Pythonic APIs — the `Strategy` base class, `BacktestConfig`/`StrategyConfig` dataclasses, `BacktestResult` wrapper, plotting, feed adapters, parameter optimization, live trading gateways, and TA-Lib.

### Rust Core Structure (`src/`)

- **`engine/core.rs`** (~1240 lines): The main `Engine` struct. Orchestrates the event-driven backtest loop via a pipeline of processors. Holds all mutable state: instruments, orders, trades, clock, timers, strategy contexts, progress bar, and market/risk/settlement managers.
- **`pipeline/stages.rs`**: The pipeline processors that run each bar step: `ChannelProcessor` → `DataProcessor` → `StrategyProcessor` → `ExecutionProcessor` → `CleanupProcessor` → `StatisticsProcessor`. Each stage handles a specific phase of the event loop.
- **`pipeline/runner.rs`**: `PipelineRunner` drives the sequence of bar events from the data feed, dispatching each bar through the pipeline stages.
- **`context.rs`** (~950 lines): `StrategyContext` — the Python-visible context object passed to strategy callbacks. Also defines `EngineContext` (Rust-only internal context) and `ContextInit`/`ContextUpdate` messages for syncing state between Rust and Python.
- **`execution/`**: Execution simulation and order matching. `simulated.rs` (1319 lines) has the main simulation logic; `matcher.rs` handles order matching algorithms; execution modules exist per asset class (`stock.rs`, `futures.rs`, `crypto.rs`, `forex.rs`, `option.rs`).
- **`order_manager.rs`**: Order lifecycle management — creation, validation, status transitions, OCO/bracket order grouping, and the order event channel.
- **`portfolio.rs`**: Portfolio state tracking — positions, cash, margin, available positions, P&L.
- **`market/`**: Market models per asset class and region (`china.rs`, `stock.rs`, `futures.rs`, `option.rs`). `manager.rs` orchestrates market model selection.
- **`model/`**: Core data types — `instrument.rs`, `order.rs`, `market_data.rs` (Bar/Tick), `types.rs`, `corporate_action.rs`, `timer.rs`.
- **`data/`**: `DataFeed` (the Python-visible feed), `BarAggregator` (for multi-frequency resampling), `feed.rs`, `batch.rs`.
- **`risk/`**: Risk management — position size limits, sector concentration, reduce-only mode, cooldown periods.
- **`analysis/`**: Performance metrics, trade tracking, backtest result aggregation.
- **`statistics/`**: Per-bar statistics collection (equity curve, drawdowns, returns).
- **`settlement/`**: Daily settlement processing (margin, corporate actions, mark-to-market).
- **`indicators/`**: Built-in Rust indicator implementations (SMA, EMA, MACD, RSI, BollingerBands, ATR).
- **`margin/`**: Margin calculation models (futures margin, option margin).
- **`account.rs`**: Account-level metrics (total equity, market value, margin usage).

### Python Layer Structure (`python/akquant/`)

- **`strategy.py`**: The `Strategy` base class users subclass. Delegates to specialized modules for each concern:
  - `strategy_events.py` — `on_bar`, `on_tick`, `on_timer` event dispatch
  - `strategy_trading_api.py` — `buy()`, `sell()`, `close_position()`, `cancel_order()`, etc.
  - `strategy_order_events.py` — order fill/cancel/reject callbacks and trade deduplication
  - `strategy_scheduler.py` — `schedule()` and `add_daily_timer()`
  - `strategy_position.py` — `Position` class and `get_position()`
  - `strategy_history.py` — `get_history()`, `get_history_df()`, rolling window data
  - `strategy_framework_hooks.py` — pre-open timers, boundary callbacks, shutdown hooks
  - `strategy_ml.py` — ML model auto-configuration, walk-forward validation windows
  - `strategy_logging.py` / `strategy_time.py` — logging and time utilities
- **`backtest/engine.py`**: The `run_backtest()` and `run_warm_start()` entry points. Handles config resolution, data feed construction, strategy instantiation, and engine lifecycle.
- **`backtest/result.py`**: Python `BacktestResult` wrapper around the Rust `BacktestResult`, adding DataFrame properties (`.orders_df`, `.trades_df`, `.equity_curve`, `.metrics`).
- **`config.py`**: `BacktestConfig`, `StrategyConfig`, `InstrumentConfig`, and region-specific configs (`ChinaFuturesConfig`, `ChinaOptionsConfig`). Uses Python dataclasses.
- **`feed_adapter.py`**: Feed adapter system — `CSVFeedAdapter`, `ParquetFeedAdapter`, `ResampledFeedAdapter`, `ReplayFeedAdapter`. Convert pandas DataFrames into the Rust `DataFeed`.
- **`data.py`**: `DataLoader` and `ParquetDataCatalog` for loading/managing market data.
- **`optimize.py`**: `run_grid_search()` (multi-process parameter optimization) and `run_walk_forward()`.
- **`factor/`**: Polars-based factor expression engine (`engine.py`, `parser.py`, `ops.py`).
- **`talib/`**: Dual-backend (Python/Rust) TA-Lib implementation with 103 indicators.
- **`gateway/`**: Live trading gateways — CTP futures (`ctp_adapter.py`, `ctp_native.py`), MiniQMT, PTrade.
- **`plot/`**: Plotly-based visualization — equity curves, drawdowns, trade markers, dashboards.
- **`live.py`**: Live trading runner.
- **`sizer.py`**: Position sizing strategies (`FixedSize`, `PercentSizer`, `AllInSizer`).
- **`params.py`**: Declarative parameter system (`IntParam`, `FloatParam`, `BoolParam`, `ChoiceParam`, `DateRangeParam`) using Pydantic.
- **`analyzer_plugin.py`**: Plugin-based analysis system for custom post-backtest analysis.
- **`risk.py`**: Python-side risk configuration helpers.

### Data Flow

1. User provides a pandas DataFrame of OHLCV data → `run_backtest()` wraps it in a `DataFeedAdapter` → converts to Rust `DataFeed`.
2. `Engine.run()` iterates bars from the feed through the pipeline stages.
3. On each bar: strategy contexts are synced, user `on_bar()` is called (Python), user-submitted orders are processed through risk checks, matched against bar prices, and trades are recorded.
4. After the backtest completes, `BacktestResult` aggregates trades, orders, equity curve, and performance metrics.

## Key Conventions

- **Error handling**: The Rust core uses `anyhow::Result` and `thiserror` for error types. Python-facing errors use `PyValueError` / `PyRuntimeError` from PyO3.
- **Decimal precision**: All monetary values use `rust_decimal::Decimal` in Rust, not floats.
- **Timestamps**: Nanosecond Unix epoch (`i64`). Bar timestamps mark the **close** of the bar.
- **Timezones**: The engine uses offset-based timezone (`timezone_offset` in seconds). The Python layer converts named timezones to offsets before setting via `engine.set_timezone_name()`.
- **Asset types**: `AssetType` enum — Stock, Future, Crypto, Forex, Option. Each has dedicated execution and market model modules.
- **Order execution modes**: `ExecutionMode` enum — `CurrentClose`, `NextOpen`, `NextClose`, `NextAverage`, `NextHighLowMid`. Controls what price fills an order.
- **T+1 settlement**: Chinese stocks have T+1 (buy today, sell tomorrow). The engine tracks `available_positions` separately from total `positions`.
- **Multi-frequency**: The `BarAggregator` resamples a base-frequency feed into higher timeframes. Multi-frequency strategies receive bars from multiple aggregators.
- **Strategy slots**: The engine supports running multiple independent strategies in the same backtest, each with its own context and portfolio sub-account.
- **Warm-up period**: `warmup_bars` — bars consumed for indicator priming before strategy callbacks fire.
- **Pre-open timers**: Timers that fire before the market opens (for auction/opening strategies), registered via `register_pre_open_timers`.

## Testing

- Tests live in `tests/`. The largest test files are `test_engine.py` (~200K), `test_strategy_extras.py` (~208K), `test_talib_backend.py` (~38K), `test_talib_compat.py` (~31K), `test_live_runner_broker_bridge.py` (~49K).
- Golden/reference files are in `tests/golden/`.
- Rust tests are inline (`#[cfg(test)]` modules within source files) and run via `cargo test`.
- `test_examples_regression.py` validates that example scripts run without errors.

## Docs

- Documentation lives in `docs/` (zh/en) and is built with mkdocs + mkdocs-material.
- API examples in docs are validated by `scripts/check_docs_api_examples.py`.
- Doc links are validated by `scripts/check_docs_links.py`.
- `examples/README_detail.md` has a detailed catalog of all 60+ examples organized by topic.
