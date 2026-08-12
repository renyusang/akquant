"""
双均线交叉策略 (SMA Crossover Strategy).

用于 examples/02_parameter_optimization.py 的参数优化示例。

策略逻辑：
- 金叉（快线上穿慢线）→ 买入 100 股
- 死叉（快线下穿慢线）→ 卖出 100 股

注意：策略类必须定义在可导入的模块文件中（不能写在 __main__），
因为 run_grid_search 使用多进程并行，需要 pickle 序列化策略类。
"""

from typing import Any

from akquant import Indicator, IntParam, ParamModel, Strategy


# =============================================================================
# 参数模型
# =============================================================================


class SMACrossParams(ParamModel):
    """
    双均线策略的参数模型。

    ParamModel 是基于 Pydantic 的参数声明式定义，用于：
    1. 自动生成参数 schema（供 UI / 文档使用）
    2. 参数校验（类型、范围）
    3. 与 run_grid_search 的 param_grid 配合使用

    每个字段使用 IntParam / FloatParam 等声明，支持：
    - ge / le：取值范围约束
    - title：参数显示名称
    """

    # 快线周期：2 ~ 200，默认 10
    fast_period: int = IntParam(10, ge=2, le=200, title="快线周期")

    # 慢线周期：3 ~ 500，默认 30。
    # 注意：慢线周期必须大于快线周期才有意义，
    # 这个约束可以在 run_grid_search 的 param_constraint 回调中实现。
    slow_period: int = IntParam(30, ge=3, le=500, title="慢线周期")


# =============================================================================
# 策略类
# =============================================================================


class SMACrossStrategy(Strategy):
    """
    双均线交叉策略 (Double SMA Crossover)。

    金叉买入（快线 > 慢线），死叉卖出（快线 < 慢线）。
    每笔固定交易 100 股，仅持有单向多头仓位。

    参数:
        fast_period (int): 快线（短期均线）周期，默认 10
        slow_period (int): 慢线（长期均线）周期，默认 30
    """

    # 将参数模型绑定到策略类。
    # run_grid_search 会通过此属性自动发现可优化的参数。
    PARAM_MODEL = SMACrossParams

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        初始化策略。

        run_grid_search 会根据 param_grid 中的参数名，
        自动匹配构造函数的参数名并传入不同的值组合。

        :param fast_period: 快线周期
        :param slow_period: 慢线周期
        """
        # 必须调用父类 __init__，它会初始化 ctx、sizer、费率等基础属性
        super().__init__()

        # 保存参数到实例，供 on_bar 中使用
        self.fast_period = fast_period
        self.slow_period = slow_period

        # ---- 定义预计算指标 ----
        # Indicator 接受一个名称和一个计算函数（lambda）。
        # 计算函数接收整个标的的 DataFrame，返回一个与原索引对齐的 Series。
        # 框架会在回测开始前调用 _prepare_indicators() 批量计算所有指标，
        # 然后在 on_bar 中通过 get_value() 按时间戳查找对应值。
        #
        # 注意：lambda 中引用的 fast_period / slow_period 是闭包变量，
        # 在 __init__ 时就已经绑定为当前参数值。

        self.sma_fast = Indicator(
            "sma_fast",
            lambda df: df["close"].rolling(fast_period).mean(),
        )
        self.sma_slow = Indicator(
            "sma_slow",
            lambda df: df["close"].rolling(slow_period).mean(),
        )

        # ---- 注册指标 ----
        # register_indicator() 将指标加入 self._precomputed_indicators 列表。
        # 回测开始前框架会遍历该列表，对每个标的的数据调用 indicator(df, symbol)
        # 进行预计算。同时会设置 self.sma_fast / self.sma_slow 属性。
        #
        # 注意：只设置 self._indicators = [...] 是不会生效的，
        # 框架只认 _precomputed_indicators 列表。
        self.register_indicator("sma_fast", self.sma_fast)
        self.register_indicator("sma_slow", self.sma_slow)

    # -------------------------------------------------------------------------
    # 核心策略逻辑
    # -------------------------------------------------------------------------

    def on_bar(self, bar: Any) -> None:
        """
        K 线闭合回调 —— 策略核心逻辑入口。

        每根 Bar 结束时引擎调用此方法，传入当前 Bar 的数据。

        :param bar: 当前 Bar 对象，包含 symbol / open / high / low / close /
                    volume / timestamp 等字段
        """
        # 1. 获取当前 Bar 对应的指标值
        #    get_value(symbol, timestamp) 使用 pandas asof 查找，
        #    返回该时间戳之前（含）最近的指标值。
        #    预热期内指标数据不足时返回 NaN，此时不交易。
        fast = self.sma_fast.get_value(bar.symbol, bar.timestamp)
        slow = self.sma_slow.get_value(bar.symbol, bar.timestamp)

        # 2. 获取当前持仓数量
        #    get_position() 返回正数（多头）、0（空仓）或负数（空头）。
        #    不传 symbol 则默认使用当前 bar.symbol。
        qty = self.get_position(bar.symbol)

        # 3. 交叉信号判断
        #    金叉：快线 > 慢线 且 当前无多头持仓 → 买入
        if fast > slow and qty <= 0:
            # buy() 下市价买单，返回 order_id
            # quantity=100 表示买入 100 股
            self.buy(symbol=bar.symbol, quantity=100)

        #    死叉：快线 < 慢线 且 当前有多头持仓 → 卖出
        elif fast < slow and qty > 0:
            # sell() 下市价卖单，默认卖出全部持仓
            self.sell(symbol=bar.symbol, quantity=100)
