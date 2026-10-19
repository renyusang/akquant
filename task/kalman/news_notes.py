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


# =============================================================================
# 妙想复查闭环(2026-10-19): 清单 → 写入 → 核验
#
# 背景: 复查必须在同一交易日、例行运行之后做, 否则次日运行时自动刷新会把
# 评分覆盖成中性标题(`auto=True`), 报告里看不到。原先"该复查哪些标的"只写在
# 文档里, 靠人记; 现在由运行流程落一份清单, 并可用 --status 核验完成度。
# =============================================================================

REVIEW_MARKER = "[自动]"


def is_reviewed(note: Any, today: str) -> bool:
    """当天写入、且非自动抓取的记录 = 已复查(带评分/分析)。"""
    if not isinstance(note, dict):
        return False
    if str(note.get("date", "")) != str(today):
        return False
    summary = str(note.get("summary", "")).strip()
    if not summary or summary.startswith(REVIEW_MARKER):
        return False
    return not bool(note.get("auto", False))


def review_status(task_dir: str, symbols: List[str],
                  today: Optional[str] = None) -> Dict[str, List[str]]:
    """当日复查完成情况: {'reviewed': [...], 'pending': [...]}(保持入参顺序)。"""
    today = today or datetime.now().strftime("%Y-%m-%d")
    notes = load_notes(task_dir)
    reviewed: List[str] = []
    pending: List[str] = []
    seen = set()
    for sym in symbols:
        s = str(sym).zfill(6)
        if not s or s in seen:
            continue
        seen.add(s)
        (reviewed if is_reviewed(notes.get(s), today) else pending).append(s)
    return {"reviewed": reviewed, "pending": pending}


def _read_state(task_dir: str, name: str, empty: Any) -> Any:
    path = os.path.join(task_dir, name)
    if not os.path.exists(path):
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return empty


def default_scope(task_dir: str) -> List[str]:
    """默认复查范围 = 当日持仓 + 次日待买候选(与例行运行口径一致)。"""
    syms: List[str] = []
    positions = _read_state(task_dir, "positions.json", {})
    if isinstance(positions, dict):
        syms.extend(str(k) for k in positions.keys())
    pending = _read_state(task_dir, "pending_orders.json", [])
    if isinstance(pending, list):
        syms.extend(
            str(o.get("symbol")) for o in pending
            if isinstance(o, dict) and o.get("action") == "buy" and o.get("symbol")
        )
    return list(dict.fromkeys(s for s in syms if s))


def rows_from_state(task_dir: str) -> Dict[str, Any]:
    """从持仓/待买 + 本地缓存末根收盘, 组装清单展示行(供 --queue 手工重生成)。

    例行运行路径由调用方直接传入 rows(带 results 的当日收盘), 此处只是
    无运行上下文时的兜底, 数据源与运行一致(.cache 末根 bar)。
    """
    positions = _read_state(task_dir, "positions.json", {})
    pending = _read_state(task_dir, "pending_orders.json", [])
    names: Dict[str, str] = {}
    if isinstance(pending, list):
        for o in pending:
            if isinstance(o, dict) and o.get("symbol"):
                names[str(o["symbol"]).zfill(6)] = str(o.get("name") or "")

    closes: Dict[str, float] = {}
    cache_dir = os.path.join(task_dir, ".cache")
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]
    if pd is not None and os.path.isdir(cache_dir):
        for sym in default_scope(task_dir):
            path = os.path.join(cache_dir, f"{sym}.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                if len(df):
                    closes[sym] = float(df["close"].iloc[-1])
            except Exception:
                pass

    rows: Dict[str, Any] = {}
    for sym in default_scope(task_dir):
        p = positions.get(sym) if isinstance(positions, dict) else None
        p = p if isinstance(p, dict) else {}
        cost = float(p.get("avg_cost") or 0)
        close = closes.get(sym) or 0.0
        rows[sym] = {
            "name": p.get("name") or names.get(sym, ""),
            "shares": p.get("shares", 0),
            "avg_cost": cost,
            "close": close,
            "pnl_pct": ((close / cost - 1) * 100 if cost > 0 and close > 0 else None),
        }
    return rows


def _fmt_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:,.{digits}f}{suffix}"


def build_review_queue(symbols: List[str], notes: Dict[str, Any],
                       status: Dict[str, List[str]], rows: Optional[Dict[str, Any]] = None,
                       today: Optional[str] = None) -> str:
    """生成复查清单 markdown(纯函数, 不落盘)。"""
    today = today or datetime.now().strftime("%Y-%m-%d")
    rows = rows or {}
    symbols = list(dict.fromkeys(str(s).zfill(6) for s in symbols))
    pending = set(status.get("pending", []))
    lines: List[str] = [
        f"# 妙想复查清单 · {today}",
        "",
        "范围 = 当日持仓 + 次日待买候选。**必须在同一交易日、例行运行之后完成**——",
        "次日运行会把评分重抓为中性标题(`auto=True`)，补做的复查不会进报告。",
        "",
        f"待复查 **{len(status.get('pending', []))}** 只 / "
        f"已复查 **{len(status.get('reviewed', []))}** 只",
        "",
        "| 代码 | 名称 | 股数 | 成本 | 收盘 | 浮动盈亏 | 状态 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for sym in symbols:
        row = rows.get(sym) or {}
        pnl = row.get("pnl_pct")
        pnl_txt = ("-" if pnl is None else f"{float(pnl):+.2f}%")
        state = "⏳ 待复查" if sym in pending else "✅ 已复查"
        lines.append(
            f"| {sym} | {row.get('name') or '-'} | "
            f"{_fmt_num(row.get('shares'), 0)} | "
            f"{_fmt_num(row.get('avg_cost'), 4)} | "
            f"{_fmt_num(row.get('close'), 4)} | {pnl_txt} | {state} |"
        )
    lines += ["", "## 底稿（自动抓取标题）", ""]
    for sym in symbols:
        row = rows.get(sym) or {}
        note = notes.get(sym) if isinstance(notes.get(sym), dict) else {}
        title = f"### {sym} {row.get('name') or ''}".rstrip()
        lines.append(title)
        lines.append("")
        lines.append(f"- 自动底稿：{note.get('summary') or '（无，抓取失败或未覆盖）'}")
        lines.append(
            f"- 写入：`python news_notes.py --set {sym} <score> \"<summary>\"`"
            "  （score: -2..2）"
        )
        lines.append("")
    lines += [
        "## 完成后核对",
        "",
        "```bash",
        "python news_notes.py --status        # 待复查 0 只即完成",
        'python -c "from live_report import build_live_report; build_live_report()"',
        "bash deploy.sh --live-only           # 让评分出现在报告页面上",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_review_queue(task_dir: str, symbols: List[str],
                       rows: Optional[Dict[str, Any]] = None,
                       today: Optional[str] = None,
                       queue_dir: Optional[str] = None) -> str:
    """写出当日复查清单(md + json 边车), 返回 md 路径。

    json 边车记录当次范围与状态, 供 `--status` 在范围含"被跳过候选"
    (不在 positions/pending 里)时仍能准确核验。
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    queue_dir = queue_dir or os.path.join(task_dir, "logs")
    os.makedirs(queue_dir, exist_ok=True)
    notes = load_notes(task_dir)
    status = review_status(task_dir, symbols, today=today)

    md = build_review_queue(symbols, notes, status, rows=rows, today=today)
    md_path = os.path.join(queue_dir, f"review_queue_{today}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    payload = {
        "date": today,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": list(symbols),
        "rows": rows or {},
        "status": status,
    }
    json_path = os.path.join(queue_dir, f"review_queue_{today}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return md_path


def record_review(task_dir: str, symbol: str, score: int, summary: str,
                  today: Optional[str] = None) -> Dict[str, Any]:
    """写入一条妙想复查记录(date=当天, 无 auto 标记), 返回该记录。"""
    score = int(score)
    if not -2 <= score <= 2:
        raise ValueError(f"score 必须在 -2..2, 收到 {score}")
    summary = str(summary).strip()
    if not summary:
        raise ValueError("summary 不能为空")
    today = today or datetime.now().strftime("%Y-%m-%d")
    notes = load_notes(task_dir)
    notes[str(symbol).zfill(6)] = {
        "date": today, "score": score, "summary": summary,
    }
    _save_notes(task_dir, notes)
    return notes[str(symbol).zfill(6)]


def _main(argv: Optional[List[str]] = None) -> int:
    """命令行入口: --status 核验 / --queue 重新生成清单 / --set 写入复查。"""
    import argparse

    parser = argparse.ArgumentParser(description="妙想复查清单与记录")
    parser.add_argument("--task-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--today", default=None, help="交易日(默认今天)")
    parser.add_argument("--status", action="store_true", help="核验当日复查完成度")
    parser.add_argument("--queue", action="store_true", help="按当前持仓/待买重新生成清单")
    parser.add_argument("--set", nargs=3, metavar=("SYMBOL", "SCORE", "SUMMARY"),
                        help="写入一条复查记录")
    args = parser.parse_args(argv)

    if args.set:
        sym, score, summary = args.set
        try:
            rec = record_review(args.task_dir, sym, int(score), summary, today=args.today)
        except ValueError as e:
            print(f"❌ {e}")
            return 2
        print(f"✅ 已写入 {sym}: score={rec['score']} date={rec['date']}")
        return 0

    today = args.today or datetime.now().strftime("%Y-%m-%d")
    if args.queue:
        scope = default_scope(args.task_dir)
        path = write_review_queue(args.task_dir, scope,
                                  rows=rows_from_state(args.task_dir), today=today)
        print(f"📝 清单: {path}（{len(scope)} 只）")
        return 0

    # --status(默认): 优先用当日清单边车的范围, 无清单则回退 持仓+待买
    scope = default_scope(args.task_dir)
    json_path = os.path.join(args.task_dir, "logs", f"review_queue_{today}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                scope = json.load(f).get("scope") or scope
        except Exception:
            pass
    status = review_status(args.task_dir, scope, today=today)
    if status["pending"]:
        print(f"⏳ 妙想复查未完: 待复查 {len(status['pending'])} 只 "
              f"({', '.join(status['pending'])}) / 已复查 {len(status['reviewed'])} 只")
        print("   清单: logs/review_queue_%s.md" % today)
        return 1
    print(f"✅ 妙想复查完成: {len(status['reviewed'])} 只"
          + (f" ({', '.join(status['reviewed'])})" if status["reviewed"] else ""))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
