"""news_notes 自动更新模块测试(2026-09-10)。

2026-10-19 新增: 妙想复查闭环(清单/写入/核验)测试。
"""

import ast
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import news_notes


def _today(offset=0):
    return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")


class TestAutoUpdate:
    def test_missing_fetched(self, monkeypatch, tmp_path):
        """缺失标的 → 抓取并写入(score=0, auto=True)。"""
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: f"[自动] {sym} 新闻标题")
        res = news_notes.auto_update_notes(str(tmp_path), ["600001"])
        assert res["updated"] == ["600001"]
        notes = json.loads((tmp_path / "news_notes.json").read_text(encoding="utf-8"))
        assert notes["600001"]["auto"] is True
        assert notes["600001"]["score"] == 0
        assert "新闻标题" in notes["600001"]["summary"]

    def test_same_day_manual_kept(self, monkeypatch, tmp_path):
        """当天妙想记录 → 保留(当天评分/分析有效, 不重复抓取)。"""
        (tmp_path / "news_notes.json").write_text(json.dumps({
            "600001": {"date": _today(0), "score": 2, "summary": "当天妙想分析"},
        }), encoding="utf-8")
        called = []
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: called.append(sym) or None)
        res = news_notes.auto_update_notes(str(tmp_path), ["600001"])
        assert res["kept"] == ["600001"]
        assert res["updated"] == []
        assert called == []          # 未触发抓取
        notes = json.loads((tmp_path / "news_notes.json").read_text(encoding="utf-8"))
        assert notes["600001"]["summary"] == "当天妙想分析"

    def test_yesterday_manual_refetched(self, monkeypatch, tmp_path):
        """人工/妙想记录昨天写的 → 次日重抓(每日统一更新核心行为)。"""
        (tmp_path / "news_notes.json").write_text(json.dumps({
            "600001": {"date": _today(-1), "score": 1, "summary": "昨日分析"},
        }), encoding="utf-8")
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: "[自动] 今日新标题")
        res = news_notes.auto_update_notes(str(tmp_path), ["600001"])
        assert res["updated"] == ["600001"]
        notes = json.loads((tmp_path / "news_notes.json").read_text(encoding="utf-8"))
        assert notes["600001"]["summary"] == "[自动] 今日新标题"
        assert notes["600001"]["score"] == 0     # 评分归 0(次日需妙想复查才有评分)

    def test_auto_note_stale_refetched(self, monkeypatch, tmp_path):
        """自动记录超 1 天 → 重抓。"""
        (tmp_path / "news_notes.json").write_text(json.dumps({
            "600001": {"date": _today(-2), "score": 0,
                       "summary": "[自动] 旧", "auto": True},
        }), encoding="utf-8")
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: "[自动] 新")
        res = news_notes.auto_update_notes(str(tmp_path), ["600001"])
        assert res["updated"] == ["600001"]

    def test_fetch_failure_silent(self, monkeypatch, tmp_path):
        """抓取失败 → failed 列表, 不写文件, 不报错。"""
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: None)
        res = news_notes.auto_update_notes(str(tmp_path), ["600001"])
        assert res["failed"] == ["600001"]
        assert not (tmp_path / "news_notes.json").exists()

    def test_dedup_symbols(self, monkeypatch, tmp_path):
        """重复代码去重(持仓+pending+skipped 来源可能重复)。"""
        calls = []
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: calls.append(sym) or "[自动] x")
        news_notes.auto_update_notes(str(tmp_path), ["600001", "600001", "600001"])
        assert calls == ["600001"]

    def test_corrupt_notes_recovers(self, monkeypatch, tmp_path):
        """损坏 JSON → 视为空, 正常抓取重建。"""
        (tmp_path / "news_notes.json").write_text("{broken", encoding="utf-8")
        monkeypatch.setattr(news_notes, "fetch_news_summary",
                            lambda sym, max_items=3: "[自动] 重建")
        res = news_notes.auto_update_notes(str(tmp_path), ["600001"])
        assert res["updated"] == ["600001"]


# =============================================================================
# 妙想复查闭环(2026-10-19)
# =============================================================================


def _write_notes(tmp_path, notes):
    (tmp_path / "news_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False), encoding="utf-8")


class TestIsReviewed:
    def test_manual_note_today_is_reviewed(self):
        note = {"date": _today(0), "score": 1, "summary": "10-19 **+1.5%**收0.877"}
        assert news_notes.is_reviewed(note, _today(0)) is True

    def test_auto_note_is_not_reviewed(self):
        note = {"date": _today(0), "score": 0, "summary": "[自动] 标题", "auto": True}
        assert news_notes.is_reviewed(note, _today(0)) is False

    def test_auto_prefix_without_flag_not_reviewed(self):
        """即便漏了 auto 标记, 以 [自动] 开头的仍是自动底稿。"""
        note = {"date": _today(0), "score": 0, "summary": "[自动] 标题"}
        assert news_notes.is_reviewed(note, _today(0)) is False

    def test_yesterday_manual_not_reviewed(self):
        note = {"date": _today(-1), "score": 1, "summary": "昨日分析"}
        assert news_notes.is_reviewed(note, _today(0)) is False


class TestReviewStatus:
    def test_split_reviewed_and_pending(self, tmp_path):
        _write_notes(tmp_path, {
            "600001": {"date": _today(0), "score": 1, "summary": "已复查"},
            "600002": {"date": _today(0), "score": 0,
                       "summary": "[自动] x", "auto": True},
        })
        st = news_notes.review_status(str(tmp_path), ["600001", "600002", "600003"])
        assert st["reviewed"] == ["600001"]
        assert st["pending"] == ["600002", "600003"]

    def test_dedup_keeps_order(self, tmp_path):
        _write_notes(tmp_path, {})
        st = news_notes.review_status(str(tmp_path), ["600002", "600001", "600002"])
        assert st["pending"] == ["600002", "600001"]

    def test_default_scope_positions_plus_pending_buys(self, tmp_path):
        (tmp_path / "positions.json").write_text(json.dumps({
            "512800": {"name": "银行ETF", "shares": 100},
            "159928": {"name": "消费ETF", "shares": 200},
        }), encoding="utf-8")
        (tmp_path / "pending_orders.json").write_text(json.dumps([
            {"symbol": "600001", "action": "buy"},
            {"symbol": "600002", "action": "sell"},
            {"symbol": "512800", "action": "buy"},   # 已持仓 → 去重
        ]), encoding="utf-8")
        scope = news_notes.default_scope(str(tmp_path))
        assert scope == ["512800", "159928", "600001"]


class TestReviewQueue:
    def test_writes_md_and_json(self, tmp_path):
        _write_notes(tmp_path, {
            "512800": {"date": _today(0), "score": 0,
                       "summary": "[自动] 工行历史新高", "auto": True},
        })
        rows = {"512800": {"name": "银行ETF", "shares": 22200, "avg_cost": 0.8523,
                           "close": 0.877, "pnl_pct": 2.9}}
        path = news_notes.write_review_queue(
            str(tmp_path), ["512800", "159928"], rows=rows, today=_today(0))
        md = Path(path).read_text(encoding="utf-8")
        assert "待复查 **2** 只" in md
        assert "512800" in md and "银行ETF" in md
        assert "[自动] 工行历史新高" in md          # 底稿带上自动标题
        assert "python news_notes.py --set 512800" in md
        side = json.loads((tmp_path / "logs" /
                           f"review_queue_{_today(0)}.json").read_text(encoding="utf-8"))
        assert side["scope"] == ["512800", "159928"]
        assert side["status"]["pending"] == ["512800", "159928"]
        assert side["rows"]["512800"]["close"] == 0.877

    def test_reviewed_symbol_marked_done(self, tmp_path):
        _write_notes(tmp_path, {
            "512800": {"date": _today(0), "score": 1, "summary": "已复查"},
        })
        path = news_notes.write_review_queue(
            str(tmp_path), ["512800"], today=_today(0))
        md = Path(path).read_text(encoding="utf-8")
        assert "已复查 **1** 只" in md
        assert "✅ 已复查" in md


class TestRecordReview:
    def test_set_then_reviewed(self, tmp_path):
        news_notes.record_review(str(tmp_path), "512800", 1, "10-19 复盘",
                                 today=_today(0))
        notes = json.loads((tmp_path / "news_notes.json").read_text(encoding="utf-8"))
        assert notes["512800"]["score"] == 1
        assert notes["512800"]["date"] == _today(0)
        assert "auto" not in notes["512800"]
        st = news_notes.review_status(str(tmp_path), ["512800"], today=_today(0))
        assert st["reviewed"] == ["512800"]

    def test_invalid_score_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            news_notes.record_review(str(tmp_path), "512800", 3, "x")

    def test_empty_summary_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            news_notes.record_review(str(tmp_path), "512800", 0, "   ")


class TestReviewCli:
    def test_status_pending_exit1(self, tmp_path, capsys):
        _write_notes(tmp_path, {"512800": {"date": _today(0), "score": 0,
                                           "summary": "[自动] x", "auto": True}})
        (tmp_path / "positions.json").write_text(
            json.dumps({"512800": {}}), encoding="utf-8")
        rc = news_notes._main(["--task-dir", str(tmp_path), "--status"])
        assert rc == 1
        assert "待复查 1 只" in capsys.readouterr().out

    def test_set_then_status_ok(self, tmp_path, capsys):
        (tmp_path / "positions.json").write_text(
            json.dumps({"512800": {}}), encoding="utf-8")
        assert news_notes._main(["--task-dir", str(tmp_path), "--set",
                                 "512800", "1", "要点"]) == 0
        rc = news_notes._main(["--task-dir", str(tmp_path), "--status"])
        assert rc == 0
        assert "复查完成" in capsys.readouterr().out

    def test_status_uses_queue_scope(self, tmp_path):
        """清单边车的范围优先于当前持仓(运行后被跳过候选仍需核验)。"""
        _write_notes(tmp_path, {})
        news_notes.write_review_queue(
            str(tmp_path), ["600001", "600002"], today=_today(0))
        (tmp_path / "positions.json").write_text(json.dumps({"512800": {}}),
                                                 encoding="utf-8")
        rc = news_notes._main(["--task-dir", str(tmp_path), "--status"])
        assert rc == 1


class TestRoutineWiring:
    """护栏: 例行运行必须真的写清单/核验(防"函数写了没接线")。"""

    def test_daily_signal_main_writes_review_queue(self):
        src = Path(__file__).resolve().parents[1] / "daily_signal.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = {n.func.id for n in ast.walk(main)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "write_review_queue" in calls
        assert "review_status" in calls
