"""
数据备份与回滚模块。

每次 daily_signal.py 运行前自动备份所有状态文件，
提供回滚命令恢复任意历史快照。

备份文件:
    positions.json, trades.csv, execution_log.csv,
    pending_orders.json, signals.csv

目录结构:
    backups/20260723_200000/  (每次运行一个快照)
"""

import json
import os
import shutil
from datetime import datetime
from typing import List, Optional

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(TASK_DIR, "backups")
MAX_BACKUPS = 30  # 保留最近 30 个快照

FILES = [
    "positions.json",
    "trades.csv",
    "execution_log.csv",
    "pending_orders.json",
    "signals.csv",
]


def _ensure_dir() -> None:
    os.makedirs(BACKUP_DIR, exist_ok=True)


def create_backup() -> str:
    """创建当前状态的完整快照，返回快照目录名。"""
    _ensure_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(dest, exist_ok=True)

    for fname in FILES:
        src = os.path.join(TASK_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, fname))

    # 写入元信息
    meta = {"timestamp": timestamp, "files": [f for f in FILES if os.path.exists(os.path.join(dest, f))]}
    with open(os.path.join(dest, "meta.json"), "w") as f:
        json.dump(meta, f)

    # 清理旧快照
    _cleanup_old()

    return timestamp


def list_backups() -> List[str]:
    """列出所有快照，按时间降序。"""
    _ensure_dir()
    dirs = [d for d in os.listdir(BACKUP_DIR) if os.path.isdir(os.path.join(BACKUP_DIR, d))]
    dirs.sort(reverse=True)
    return dirs


def restore_backup(timestamp: Optional[str] = None) -> bool:
    """恢复到指定快照。不指定则使用最新的。

    返回 True 表示成功。
    """
    _ensure_dir()
    if timestamp is None:
        dirs = list_backups()
        if not dirs:
            print("没有可用的备份")
            return False
        timestamp = dirs[0]

    src_dir = os.path.join(BACKUP_DIR, timestamp)
    if not os.path.exists(src_dir):
        print(f"备份 {timestamp} 不存在")
        return False

    # 确认操作
    meta_path = os.path.join(src_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"恢复快照: {timestamp} ({len(meta.get('files',[]))} 个文件)")

    for fname in FILES:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(TASK_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {fname}")
        else:
            if os.path.exists(dst):
                os.remove(dst)
                print(f"  ✗ {fname} (已删除)")

    return True


def _cleanup_old() -> None:
    """清理超过 MAX_BACKUPS 的旧快照。"""
    dirs = list_backups()
    for d in dirs[MAX_BACKUPS:]:
        shutil.rmtree(os.path.join(BACKUP_DIR, d), ignore_errors=True)
