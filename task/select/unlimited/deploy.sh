#!/usr/bin/env bash
# =============================================================================
# 无限仓位信号报告部署脚本
# 同步 unlimited_report.html 到服务器 /unlimited/
# 用法: bash deploy.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="119.29.88.84"
REMOTE_USER="ubuntu"
REMOTE_DIR="/var/www/reports"

cd "${SCRIPT_DIR}"

if [ ! -f unlimited_report.html ]; then
    echo "[ERROR] unlimited_report.html 不存在, 请先运行 unlimited_signal.py"
    exit 1
fi

# 确保远端目录存在
ssh -o ConnectTimeout=10 "${REMOTE_USER}@${REMOTE_HOST}" \
    "mkdir -p ${REMOTE_DIR}/unlimited" 2>/dev/null

echo "============================================"
echo "  部署无限仓位报告 → ${REMOTE_DIR}/unlimited/"
echo "============================================"
rsync -avz --progress \
    -e "ssh -o ConnectTimeout=10" \
    "unlimited_report.html" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/unlimited/"

echo ""
echo "  部署完成!"
echo "  访问地址: http://${REMOTE_HOST}:8888/unlimited/unlimited_report.html"
echo "============================================"
