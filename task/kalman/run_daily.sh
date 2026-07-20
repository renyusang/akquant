#!/usr/bin/env bash
# =============================================================================
# 每日信号定时执行脚本
#
# 用法:
#   bash run_daily.sh              # 直接运行
#   bash run_daily.sh --quiet      # 静默模式
#
# crontab 配置 (每个交易日 20:00):
#   0 20 * * 1-5 /home/renyu/project/opensrc/akquant/task/kalman/run_daily.sh
# =============================================================================
set -euo pipefail

# ---- 路径配置 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="akquant_test"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/daily_$(date +%Y%m%d).log"
MAX_LOG_DAYS=30

# ---- 创建日志目录 ----
mkdir -p "${LOG_DIR}"

# ---- 激活 conda 环境 ----
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || {
        echo "[ERROR] 无法激活 conda 环境: ${CONDA_ENV}"
        exit 1
    }
else
    echo "[ERROR] 未找到 conda 命令"
    exit 1
fi

# ---- 执行信号扫描 ----
echo "============================================================" | tee -a "${LOG_FILE}"
echo "  每日信号扫描  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

cd "${SCRIPT_DIR}"
python daily_signal.py "$@" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "  完成 $(date '+%Y-%m-%d %H:%M:%S')  (exit=${EXIT_CODE})" | tee -a "${LOG_FILE}"
echo "  日志: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# ---- 清理过期日志 ----
find "${LOG_DIR}" -name "daily_*.log" -mtime +${MAX_LOG_DAYS} -delete 2>/dev/null || true

exit ${EXIT_CODE}
