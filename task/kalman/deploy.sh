#!/usr/bin/env bash
# =============================================================================
# 报告部署脚本 — 将 kalman 报告同步到腾讯云服务器
#
# 用法:
#   bash deploy.sh                        # 同步所有 HTML 报告
#   bash deploy.sh --live-only            # 仅同步实盘报告
#   bash deploy.sh --backtest-only        # 仅同步回测报告
#   bash deploy.sh --dry-run              # 预览将要同步的文件
#
# 依赖:
#   - SSH 免密登录已配置 (setup_server.py)
#   - rsync 已安装
# =============================================================================
set -euo pipefail

# ---- 配置 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="119.29.88.84"
REMOTE_USER="ubuntu"
REMOTE_DIR="/var/www/reports"
SSH_PORT="22"

# ---- 颜色 ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ---- 解析参数 ----
DRY_RUN=""
LIVE_ONLY=false
BACKTEST_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --live-only) LIVE_ONLY=true ;;
        --backtest-only) BACKTEST_ONLY=true ;;
    esac
done

# ---- 确保远程目录存在 ----
ensure_remote_dir() {
    ssh -o ConnectTimeout=10 "${REMOTE_USER}@${REMOTE_HOST}" \
        "mkdir -p ${REMOTE_DIR}/live ${REMOTE_DIR}/backtest/single ${REMOTE_DIR}/backtest/portfolio" 2>/dev/null
}

# ---- rsync 封装 ----
do_rsync() {
    local src="$1"
    local dst="$2"
    local desc="$3"

    if [ -n "$DRY_RUN" ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} ${desc}: ${src} → ${dst}"
    else
        echo -e "${BLUE}[SYNC]${NC} ${desc}"
    fi

    rsync -avz --progress \
        --checksum \
        --exclude='plotly.min.js' \
        ${DRY_RUN} \
        -e "ssh -o ConnectTimeout=10" \
        "${src}" "${REMOTE_USER}@${REMOTE_HOST}:${dst}"
}

# ---- 生成导航页 ----
generate_index() {
    # 生成一个简洁的文件索引页面 (无 plotly,纯 HTML)
    if [ -n "$DRY_RUN" ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would generate index.html"
        return
    fi

    echo -e "${BLUE}[INDEX]${NC} Generating navigation page..."

    local TMPDIR="$(mktemp -d)"
    local INDEX="${TMPDIR}/index.html"

    cat > "${INDEX}" << 'HTMLEOF'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>AKQuant Kalman 报告</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 0 auto; padding: 30px; background: #f5f5f5; color: #333; }
  h1 { color: #1a1a1a; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }
  h2 { color: #2c3e50; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
  a { color: #1f77b4; text-decoration: none; font-size: 15px; }
  a:hover { text-decoration: underline; }
  .card { background: white; border-radius: 8px; padding: 15px 20px; margin: 8px 0;
          box-shadow: 0 2px 4px rgba(0,0,0,0.06); display: flex; align-items: center; gap: 10px; }
  .card .icon { font-size: 24px; }
  .card .info { flex: 1; }
  .card .info .name { font-weight: bold; }
  .card .info .desc { font-size: 13px; color: #888; }
  .live-badge { background: #d62728; color: white; padding: 2px 8px; border-radius: 4px;
                font-size: 11px; font-weight: bold; }
  .section { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }
  footer { text-align: center; color: #999; margin-top: 40px; font-size: 12px; }
  .updated { font-size: 12px; color: #999; }
</style>
</head>
<body>
<div style="background:#fff5f5;border:3px solid #e74c3c;border-radius:8px;padding:20px 24px;margin-bottom:24px;font-size:16px;color:#721c24;line-height:1.8;text-align:center">
<div style="font-size:24px;margin-bottom:8px">⚠️</div>
<strong style="font-size:18px">免责声明</strong><br>
本网站所有内容仅为<u>个人量化策略研究记录</u>，<strong>不构成任何投资建议</strong>。<br>
报告中的信号和回测基于<strong>历史数据</strong>，过往表现<strong>不代表未来收益</strong>。<br>
股市有风险，投资需谨慎。使用者应<strong>独立判断并承担全部投资风险</strong>，<br>
作者不对因使用本网站信息产生的任何直接或间接损失承担责任。
</div>
<h1>📊 AKQuant Kalman 报告</h1>
<p>A股/ETF 卡尔曼滤波量化交易系统 — 回测 & 实盘报告</p>

<h2>🔴 实盘</h2>
<div class="section">
  <div class="card">
    <span class="icon">📈</span>
    <div class="info">
      <a href="live/live_report.html">实盘交易报告 <span class="live-badge">LIVE</span></a>
      <div class="desc">权益曲线 · 持仓 · 交易记录</div>
    </div>
  </div>
</div>

<h2>🟢 无限仓位信号</h2>
<div class="section">
  <div class="card">
    <span class="icon">🛰️</span>
    <div class="info">
      <a href="unlimited/unlimited_report.html">无限仓位信号报告</a>
      <div class="desc">1-3手信号 · 归一化收益排序 · 股票搜索</div>
    </div>
  </div>
</div>

<h2>🔵 组合回测</h2>
<div class="section">
  <div class="card">
    <span class="icon">📊</span>
    <div class="info">
      <a href="backtest/portfolio/report_portfolio.html">组合汇总报告</a>
      <div class="desc">股票池 + ETF 池合并权益</div>
    </div>
  </div>
  <div class="card">
    <span class="icon">📈</span>
    <div class="info">
      <a href="backtest/portfolio/report_stock.html">股票池报告</a>
      <div class="desc">股票池单独回测 + K 线复盘</div>
    </div>
  </div>
  <div class="card">
    <span class="icon">📉</span>
    <div class="info">
      <a href="backtest/portfolio/report_etf.html">ETF 池报告</a>
      <div class="desc">ETF 池单独回测 + K 线复盘</div>
    </div>
  </div>
</div>

<footer>
  <p>🤖 Generated by AKQuant Kalman Report System | <span class="updated" id="timestamp"></span></p>
</footer>
<script>
document.getElementById('timestamp').textContent = 'Updated: ' + new Date().toLocaleString('zh-CN');
</script>
</body>
</html>
HTMLEOF

    # 上传 index.html
    rsync -avz -e "ssh -o ConnectTimeout=10" "${INDEX}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/index.html" 2>/dev/null
    rm -rf "${TMPDIR}"
    echo -e "${GREEN}[OK]${NC} Navigation page updated"
}

# =============================================================================
# 主流程
# =============================================================================
echo "============================================"
echo "  AKQuant Kalman 报告部署"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 确保远程目录存在
if [ -z "$DRY_RUN" ]; then
    ensure_remote_dir
    echo -e "${GREEN}[OK]${NC} Remote directories ready"
fi

# ---- 1. 实盘报告 ----
if ! $BACKTEST_ONLY; then
    if [ -f "${SCRIPT_DIR}/live_report.html" ]; then
        do_rsync "${SCRIPT_DIR}/live_report.html" "${REMOTE_DIR}/live/" "实盘报告"
    else
        echo -e "${YELLOW}[SKIP]${NC} live_report.html 不存在"
    fi
fi

# ---- 2. 单股回测报告 ----
if ! $LIVE_ONLY; then
    REPORT_COUNT=0
    for f in "${SCRIPT_DIR}"/report_[0-9]*.html; do
        if [ -f "$f" ]; then
            do_rsync "$f" "${REMOTE_DIR}/backtest/single/" "单股回测: $(basename $f)"
            REPORT_COUNT=$((REPORT_COUNT + 1))
        fi
    done

    # K 线图
    for f in "${SCRIPT_DIR}"/kline_[0-9]*.html; do
        if [ -f "$f" ]; then
            do_rsync "$f" "${REMOTE_DIR}/backtest/single/" "K 线图: $(basename $f)"
        fi
    done

    # 网格搜索结果
    for f in "${SCRIPT_DIR}"/grid_search_*.csv; do
        if [ -f "$f" ]; then
            do_rsync "$f" "${REMOTE_DIR}/backtest/single/" "参数搜索: $(basename $f)"
        fi
    done

    # ---- 3. 组合回测报告 ----
    for f in report_stock.html report_etf.html report_portfolio.html portfolio_report.html; do
        if [ -f "${SCRIPT_DIR}/${f}" ]; then
            do_rsync "${SCRIPT_DIR}/${f}" "${REMOTE_DIR}/backtest/portfolio/" "组合回测: ${f}"
        fi
    done

    if [ $REPORT_COUNT -eq 0 ]; then
        echo -e "${YELLOW}[INFO]${NC} 未找到回测报告文件"
    fi
fi

# ---- 4. 生成/更新导航页 ----
if [ -z "$DRY_RUN" ]; then
    generate_index
fi

echo ""
echo -e "${GREEN}============================================"
echo "  部署完成! ${NC}"
echo "  访问地址: http://${REMOTE_HOST}:8888/"
echo "============================================"
