"""消息面笔记维护(2026-09-10, 修订: 每日统一更新)。

news_notes.json 格式:
    {symbol: {"date": "YYYY-MM-DD", "score": -2..2, "summary": "...",
              "auto": true?}}   # auto=True 为自动抓取(akshare 新闻标题)

更新策略(每日 daily_signal 运行时自动执行):
- **每天对全部标的刷新**(消息面每日必新, 无保留期豁免):
  仅**当天**写入的记录保留(妙想复查的评分/分析当天有效), 非当天
  (含此前的人工/自动记录)一律重抓为最新新闻标题(score=0, auto=True)。
- 评分生命周期: 评分只在当天妙想复查时存在; 次日自动刷新为中性标题记录。
- 网络失败静默跳过(保留旧记录), 不阻断主流程。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

NOTES_FILE = "news_notes.json"


def _notes_path(task_dir: str) -> str:
    return os.path.join(task_dir, NOTES_FILE)


def load_notes(task_dir: str) -> Dict[str, Any]:
    """读取 notes(容错: 缺失/损坏返回空 dict)。"""
    path = _notes_path(task_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_notes(task_dir: str, notes: Dict[str, Any]) -> None:
    path = _notes_path(task_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _is_fresh(note: Dict[str, Any], today: str) -> bool:
    """仅当天写入的记录算新鲜(次日全部重抓)。"""
    return str(note.get("date", "")) == today


def fetch_news_summary(symbol: str, max_items: int = 3) -> Optional[str]:
    """akshare 拉取个股/ETF 最新新闻标题, 返回摘要串(失败返回 None)。"""
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        df = ak.stock_news_em(symbol=str(symbol).zfill(6))
    except Exception:
        return None
    if df is None or df.empty or "新闻标题" not in df.columns:
        return None
    titles: List[str] = []
    for _, row in df.head(max_items * 2).iterrows():
        t = str(row.get("新闻标题", "")).strip()
        if t and t not in titles:
            titles.append(t)
        if len(titles) >= max_items:
            break
    if not titles:
        return None
    return "[自动] " + " | ".join(titles)


def auto_update_notes(task_dir: str, symbols: List[str],
                      today: Optional[str] = None) -> Dict[str, List[str]]:
    """对 symbols 每日刷新新闻并写回 notes。

    返回 {"updated": [...], "kept": [...], "failed": [...]}。
    仅**当天**记录保留(妙想复查的评分/分析当天有效); 非当天记录
    (含人工/自动)全部重抓为最新新闻标题。
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    notes = load_notes(task_dir)
    result: Dict[str, List[str]] = {"updated": [], "kept": [], "failed": []}
    changed = False
    for sym in dict.fromkeys(str(s).zfill(6) for s in symbols):
        note = notes.get(sym)
        if (isinstance(note, dict) and note.get("summary")
                and _is_fresh(note, today)):
            result["kept"].append(sym)
            continue
        summary = fetch_news_summary(sym)
        if summary is None:
            result["failed"].append(sym)
            continue
        notes[sym] = {"date": today, "score": 0, "summary": summary,
                      "auto": True}
        result["updated"].append(sym)
        changed = True
    if changed:
        try:
            _save_notes(task_dir, notes)
        except Exception:
            result["failed"].extend(result["updated"])
            result["updated"] = []
    return result
