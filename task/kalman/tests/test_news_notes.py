"""news_notes 自动更新模块测试(2026-09-10)。"""

import json
import os
import sys
from datetime import datetime, timedelta

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
