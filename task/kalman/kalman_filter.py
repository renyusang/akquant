"""
2 状态卡尔曼滤波器（价格 + 速度/趋势）。

纯 numpy 实现，无额外依赖。用于对股票价格序列进行降噪和趋势估计，
并根据滤波结果生成交易信号。

模型：
    状态向量:  x = [price, velocity]^T
    状态转移:  x_k = F @ x_{k-1} + w_k      (F = [[1, 1], [0, 1]])
    观测方程:  z_k = H @ x_k + v_k          (H = [1, 0])

用法:
    kf = KalmanFilter2D(Q_price=1e-4, Q_vel=1e-5, R=1e-2)
    for price in price_series:
        filtered_price, velocity = kf.update(price)
"""

from typing import Tuple

import numpy as np


class KalmanFilter2D:
    """2 状态卡尔曼滤波器（价格 + 速度）。

    参数:
        Q_price: 价格分量的过程噪声方差，越大表示越不相信模型预测 (默认 1e-4)
        Q_vel:   速度分量的过程噪声方差 (默认 1e-5)
        R:       观测噪声方差，越大表示越不相信观测值 (默认 1e-2)
        initial_price: 初始价格估计 (默认 0.0，首次 update 时自动设置)
    """

    def __init__(
        self,
        Q_price: float = 1e-4,
        Q_vel: float = 1e-5,
        R: float = 1e-2,
        initial_price: float = 0.0,
    ) -> None:
        # 状态向量 [price, velocity]
        self._x = np.array([initial_price, 0.0], dtype=np.float64)

        # 状态转移矩阵 F = [[1, 1], [0, 1]]
        self._F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        # 观测矩阵 H = [1, 0]
        self._H = np.array([[1.0, 0.0]], dtype=np.float64)

        # 过程噪声协方差 Q
        self._Q = np.array(
            [[Q_price, 0.0], [0.0, Q_vel]], dtype=np.float64
        )

        # 观测噪声协方差 R
        self._R = np.array([[R]], dtype=np.float64)

        # 误差协方差矩阵 P（初始为单位阵）
        self._P = np.eye(2, dtype=np.float64)

        # 上一次预测的观测值（用于计算残差）
        self._predicted_observation: float = 0.0

        # 上一步的速度（用于检测方向变化）
        self._prev_velocity: float = 0.0

        # 是否已初始化
        self._initialized: bool = False

        # 观测计数
        self._count: int = 0

    def update(self, observation: float) -> Tuple[float, float]:
        """输入观测价格，返回 (滤波价格, 估计速度)。

        首次调用时，自动用观测值初始化状态向量的价格分量。
        """
        observation = float(observation)
        self._count += 1

        if not self._initialized:
            # 首次观测：初始化价格，速度设为 0
            self._x[0] = observation
            self._x[1] = 0.0
            self._initialized = True
            self._predicted_observation = observation
            return observation, 0.0

        # 保存上一步速度
        self._prev_velocity = self._x[1]

        # ---------- 预测步骤 ----------
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # 预测的观测值
        self._predicted_observation = float((self._H @ x_pred)[0])

        # ---------- 更新步骤 ----------
        # 新息（innovation）
        innovation = observation - self._predicted_observation

        # 新息协方差 S = H @ P_pred @ H.T + R
        S = self._H @ P_pred @ self._H.T + self._R

        # 卡尔曼增益 K = P_pred @ H.T / S
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        # 状态更新
        self._x = x_pred + (K @ np.array([[innovation]])).flatten()

        # 协方差更新
        self._P = (np.eye(2) - K @ self._H) @ P_pred

        return float(self._x[0]), float(self._x[1])

    def get_filtered_price(self) -> float:
        """返回当前滤波后的价格估计。"""
        return float(self._x[0])

    def get_velocity(self) -> float:
        """返回当前速度估计（趋势方向和强度）。"""
        return float(self._x[1])

    def get_prev_velocity(self) -> float:
        """返回上一步的速度估计。"""
        return float(self._prev_velocity)

    def get_residual(self) -> float:
        """返回观测值与预测值之差（新息 / innovation）。"""
        return float(self._x[0] - self._predicted_observation)

    def reset(self, initial_price: float = 0.0) -> None:
        """重置滤波器状态。"""
        self._x = np.array([initial_price, 0.0], dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64)
        self._predicted_observation = 0.0
        self._prev_velocity = 0.0
        self._initialized = False
        self._count = 0


def kalman_smooth(
    prices: np.ndarray,
    Q_price: float = 1e-4,
    Q_vel: float = 1e-5,
    R: float = 1e-2,
) -> np.ndarray:
    """对完整价格序列进行卡尔曼滤波平滑，返回滤波后的价格数组。

    适用于离线分析、绘图等场景。
    """
    kf = KalmanFilter2D(Q_price=Q_price, Q_vel=Q_vel, R=R)
    filtered = np.empty_like(prices)
    for i, p in enumerate(prices):
        fp, _ = kf.update(float(p))
        filtered[i] = fp
    return filtered


def generate_signals(
    prices: np.ndarray,
    Q_price: float = 1e-4,
    Q_vel: float = 1e-5,
    R: float = 1e-2,
    entry_threshold: float = 0.02,
    exit_threshold: float = 0.005,
) -> np.ndarray:
    """对价格序列生成交易信号。

    信号值:
         1 → 买入（价格突破 / 速度转正）
        -1 → 卖出（价格回归 / 速度转负）
         0 → 持仓不变

    返回与 prices 等长的整数数组。
    """
    kf = KalmanFilter2D(Q_price=Q_price, Q_vel=Q_vel, R=R)
    signals = np.zeros(len(prices), dtype=int)
    in_position = False

    for i in range(len(prices)):
        fp, vel = kf.update(float(prices[i]))
        prev_vel = kf.get_prev_velocity()

        if i == 0:
            continue

        if not in_position:
            # 买入条件
            price_breakout = prices[i] > fp * (1.0 + entry_threshold)
            vel_turn = vel > 0.0 and prev_vel <= 0.0
            if price_breakout or vel_turn:
                signals[i] = 1
                in_position = True
        else:
            # 卖出条件
            price_reversion = prices[i] < fp * (1.0 - exit_threshold)
            vel_turn = vel < 0.0 and prev_vel >= 0.0
            if price_reversion or vel_turn:
                signals[i] = -1
                in_position = False

    return signals
