#!/usr/bin/env bash
# ----------------------------------------------------------------------
# 卡尔曼滤波策略 —— 隔离环境搭建脚本
#
# 支持两种方式:
#   方式1 (推荐): 使用已有的 conda 环境 (默认 akquant_032, 生产环境 0.3.20)
#     bash setup_env.sh --conda akquant_032
#
#   方式2: 创建独立 venv (需要 python3-venv)
#     bash setup_env.sh
#
# 本脚本不会修改本机已安装的系统级 Python 包，
# 不会影响父项目 akquant 的环境。
# ----------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-venv}"
CONDA_ENV="${2:-akquant_032}"

if [ "$MODE" = "--conda" ]; then
    echo "=== 使用 conda 环境: ${CONDA_ENV} ==="
    # conda 环境已存在，直接安装依赖
    if command -v conda &>/dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate "${CONDA_ENV}"
        echo "已激活 conda 环境: ${CONDA_ENV}"
        pip install matplotlib mplfinance 2>/dev/null || true
        echo "=== 环境就绪 ==="
        echo "激活命令: conda activate ${CONDA_ENV}"
    else
        echo "错误: 未找到 conda 命令"
        exit 1
    fi
else
    VENV_DIR="venv"
    echo "=== 创建隔离 Python 虚拟环境 ==="

    if python3 -m venv "${VENV_DIR}" 2>/dev/null; then
        source "${VENV_DIR}/bin/activate"
        echo "=== 升级 pip ==="
        pip install --upgrade pip
        echo "=== 安装依赖包 ==="
        pip install -r requirements.txt
        echo "=== 安装 akquant（本地 editable 模式）==="
        pip install -e /home/renyu/project/opensrc/akquant
        echo ""
        echo "=== 环境创建完成 ==="
        echo "虚拟环境路径: $(pwd)/${VENV_DIR}"
        echo "激活命令:     source ${VENV_DIR}/bin/activate"
        echo "退出命令:     deactivate"
    else
        echo "错误: python3-venv 未安装，无法创建 venv"
        echo "请尝试以下方法之一:"
        echo "  1. sudo apt install python3-venv"
        echo "  2. bash setup_env.sh --conda <env-name>"
        echo "  3. 手动在已有的 conda 环境中安装依赖:"
        echo "     conda activate <env-name>"
        echo "     pip install matplotlib mplfinance"
        exit 1
    fi
fi
