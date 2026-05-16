# run_walk_forward
run_walk_forward 实现的是滚动窗口前向优化（Walk-Forward Optimization）。

核心思路是这样的：

为什么不只用普通网格搜索

run_grid_search 在一个固定的数据集上找最优参数，存在过拟合风险——找到的参数可能只在历史数据上表
现好，实盘中失效。

WFO 的做法

将数据切分成多个「训练集 + 测试集」片段，滚动推进：

时间轴:  |---训练(100天)---|---测试(50天)---|
                        滚动 →  |---训练---|---测试---|
                                        滚动 →  |---训练---|---测试---|

在示例中（第 80-91 行）：

1. 训练窗口 train_period=100：用这 100 根 Bar 做网格搜索，选出最优参数
2. 测试窗口 test_period=50：用最优参数在接下来的 50 根 Bar 上实际跑回测
3. 窗口向前滚动 50 根 Bar，重复上述过程
4. 最终将所有测试窗口的资金曲线拼接起来，得到一条完整的样本外（OOS）资金曲线

和 run_grid_search 的关系

run_grid_search 回答的是「哪组参数在历史数据上最好」；run_walk_forward
回答的是「参数优化方法在样本外能不能赚钱」。WFO 是更接近实盘环境的评估方式。

滚动步长由 test_period 参数决定。核心逻辑在 optimize.py:891：

for i in range(0, total_len - train_period - test_period + 1, test_period):

range 的第三步就是滚动步长，值为 test_period（示例中为 50）。

窗口切分规则

对于第 n 轮（n 从 0 开始），起始位置 i = n * test_period：

i=0:   [0 ───── 训练100 ─────|100── 测试50 ──|150]
i=50:            [50 ───── 训练100 ─────|150── 测试50 ──|200]
i=100:                    [100 ─── 训练100 ───|200── 测试50 ──|250]

变量: train_start_idx
计算方式: i      
含义: 训练集起始位置
────────────────────────────────────────
变量: train_end_idx
计算方式: i + train_period
含义: 训练集结束（也是测试集起始）
────────────────────────────────────────
变量: oos_start_idx
计算方式: train_end_idx
含义: 测试集起始
────────────────────────────────────────
变量: oos_end_idx
计算方式: min(oos_start_idx + test_period, total_len)
含义: 测试集结束

关键点

- 步长 = test_period，意味着训练窗口每次前移 50 根 
Bar，相邻窗口之间训练数据有重叠（train_period - test_period = 50 根 Bar 重叠）
- 如果想让训练窗口不重叠，需要设置 test_period >= 
train_period，但那样测试窗口比训练窗口还长，一般不这么做
- 总轮次数 = (total_len - train_period - test_period) / test_period + 1