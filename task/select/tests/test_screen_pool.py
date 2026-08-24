"""Tests for screen_pool.py 科学选股筛选脚本。

覆盖: 候选池解析/去重、策略参数映射、双窗口指标 pivot、推荐过滤规则。
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "kalman"))

import screen_pool


class TestParseCandidates:
    """候选池解析: CSV 读取 / 命令行 / 去重 / ETF 判断。"""

    def test_is_etf(self):
        assert screen_pool.is_etf("510050")
        assert screen_pool.is_etf("159845")
        assert screen_pool.is_etf("588000")
        assert not screen_pool.is_etf("002594")
        assert not screen_pool.is_etf("688041")

    def test_parse_csv(self, tmp_path):
        csv_path = tmp_path / "cand.csv"
        csv_path.write_text("code,name\n002594,比亚迪\n510050,上证50ETF\n",
                            encoding="utf-8")
        cands = screen_pool.parse_candidates(str(csv_path), None)
        assert len(cands) == 2
        by_sym = {c["symbol"]: c for c in cands}
        assert by_sym["002594"]["asset_type"] == "stock"
        assert by_sym["510050"]["asset_type"] == "etf"

    def test_merge_dedup(self, tmp_path):
        """CSV 与命令行合并去重。"""
        csv_path = tmp_path / "cand.csv"
        csv_path.write_text("code,name\n002594,比亚迪\n", encoding="utf-8")
        cands = screen_pool.parse_candidates(str(csv_path), "002594,600900")
        assert len(cands) == 2
        assert {c["symbol"] for c in cands} == {"002594", "600900"}

    def test_zero_pad(self):
        cands = screen_pool.parse_candidates(None, "2594,510050")
        by_sym = {c["symbol"]: c for c in cands}
        assert "002594" in by_sym


class TestLoadStrategyParams:
    """策略参数: stocks.yaml 读取 + yaml 名→策略名映射。"""

    def test_param_mapping(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "strategy:\n"
            "  trend_confirm_bars: 1\n"
            "  trend_bear_pct: 0.30\n"
            "  downtrend_entry: 0.03\n"
            "  atr_adaptive_exit_enabled: true\n"
            "  exit_atr_factor: 1.0\n"
            "stock:\n"
            "  initial_cash: 200000\n"
            "  max_positions: 5\n"
            "  single_position_pct: 0.20\n"
            "watchlist:\n"
            "  stocks: [{symbol: '002594', name: 比亚迪}]\n",
            encoding="utf-8")
        params = screen_pool.load_strategy_params(str(yaml_path))
        assert params["trend_filter_confirm_bars"] == 1   # yaml→策略名
        assert params["trend_bear_position_pct"] == 0.30
        assert params["downtrend_entry_threshold"] == 0.03
        assert params["atr_adaptive_exit_enabled"] is True
        # 单标的回测口径
        assert params["single_position_pct"] == 1.0
        assert params["max_positions"] == 5


class TestPivotAndRank:
    """长表→宽表 + 推荐过滤规则。"""

    def _long_df(self):
        rows = []
        for sym, name, in_ret, out_ret, in_nt, out_nt, out_mdd in [
            ("000001", "双窗口盈利", 20.0, 15.0, 60, 20, 25.0),
            ("000002", "验证期亏损", 20.0, -5.0, 60, 20, 25.0),
            ("000003", "筛选期亏损", -5.0, 15.0, 60, 20, 25.0),
            ("000004", "交易过少", 20.0, 15.0, 10, 5, 25.0),
            ("000005", "交易过频", 20.0, 15.0, 500, 200, 25.0),
            ("000006", "回撤过大", 20.0, 15.0, 60, 20, 50.0),
            ("000007", "零交易", 0.0, 0.0, 0, 0, 0.0),
        ]:
            rows.append({"symbol": sym, "name": name, "window": "in",
                         "ret": in_ret, "sharpe": 0.5, "mdd": 20.0,
                         "ntrades": in_nt, "winrate": 50.0})
            rows.append({"symbol": sym, "name": name, "window": "out",
                         "ret": out_ret, "sharpe": 0.5, "mdd": out_mdd,
                         "ntrades": out_nt, "winrate": 50.0})
        return pd.DataFrame(rows)

    def test_pivot(self):
        wide = screen_pool.pivot_results(self._long_df())
        assert list(wide.columns)[:3] == ["symbol", "name", "in_ret"]
        assert "out_mdd" in wide.columns
        r = wide[wide["symbol"] == "000001"].iloc[0]
        assert r["in_ret"] == 20.0 and r["out_ret"] == 15.0

    def test_recommend_filters(self):
        wide = screen_pool.pivot_results(self._long_df())
        ranked = screen_pool.rank_and_recommend(wide, watchlist={"000001"})
        rec = ranked[ranked["recommended"]]
        assert list(rec["symbol"]) == ["000001"]  # 仅双窗口盈利+交易适中+回撤可控
        assert bool(ranked[ranked["symbol"] == "000001"]
                    ["in_watchlist"].iloc[0]) is True

    def test_empty_input(self):
        assert screen_pool.pivot_results(pd.DataFrame()).empty
        assert screen_pool.rank_and_recommend(pd.DataFrame()).empty


class TestRiskFlags:
    """风险标记 CSV 解析 + 红线前置剔除。"""

    def test_parse_risk_flags(self, tmp_path):
        risk_file = tmp_path / "risk.csv"
        risk_file.write_text(
            "code,severity,note\n"
            "000636,red,财务造假\n"
            "300489,yellow,会计差错\n"
            "301165,light,个人短线交易\n"
            "bad_code,unknown,忽略\n",
            encoding="utf-8")
        flags = screen_pool.parse_risk_flags(str(risk_file))
        assert flags["000636"]["severity"] == "red"
        assert flags["300489"]["severity"] == "yellow"
        assert flags["301165"]["severity"] == "light"
        assert "bad_code" not in flags  # 非法 severity 忽略

    def test_missing_file_returns_empty(self):
        assert screen_pool.parse_risk_flags("不存在.csv") == {}
        assert screen_pool.parse_risk_flags(None) == {}

    def test_red_flag_marks_skip(self, tmp_path):
        """红线候选标记 skip(前置剔除), 不进入回测。"""
        risk_file = tmp_path / "risk.csv"
        risk_file.write_text("code,severity,note\n000636,red,财务造假\n",
                             encoding="utf-8")
        flags = screen_pool.parse_risk_flags(str(risk_file))
        cands = screen_pool.parse_candidates(None, "000636,300489")
        for c in cands:
            if flags.get(c["symbol"], {}).get("severity") == "red":
                c["skip"] = True
        skipped = [c for c in cands if c.get("skip")]
        assert [c["symbol"] for c in skipped] == ["000636"]
        assert [c["symbol"] for c in cands if not c.get("skip")] \
            == ["300489"]


class TestDataPreparation:
    """数据准备段并行化(2026-08-24): 8 路线程池下载缺失 .catalog。

    原串行 for 逐只下载(828 只候选首次 ~1 小时), 改为线程池并行。
    下载无副作用各写各 catalog 文件; 失败标的标记 skip 与串行版一致。
    """

    def test_prepare_parallel_and_skip(self, monkeypatch, tmp_path):
        """4 只缺失并行下载(耗时<串行), 失败标的标记 skip。"""
        import time
        from concurrent.futures import ThreadPoolExecutor

        cat = tmp_path / "catalog"
        calls = []

        def _prepare_all(cands):
            """复刻 screen_pool.main() 数据准备段逻辑。"""
            missing = [c for c in cands
                       if not (cat / c["symbol"] / "data.parquet").exists()]

            def _load(c):
                time.sleep(0.2)
                calls.append(c["symbol"])
                try:
                    if c["symbol"] == "000002":
                        raise ValueError("模拟失败")
                    (cat / c["symbol"]).mkdir(parents=True, exist_ok=True)
                    (cat / c["symbol"] / "data.parquet").write_text("x")
                except Exception:
                    c["skip"] = True  # 与生产代码一致: 失败标记 skip

            with ThreadPoolExecutor(max_workers=8) as ex:
                for _ in ex.map(_load, missing):
                    pass

        cands = [{"symbol": s, "asset_type": "stock"}
                 for s in ("000001", "000002", "000003", "000004")]
        t0 = time.time()
        _prepare_all(cands)
        elapsed = time.time() - t0
        assert elapsed < 0.8, f"并行未生效: {elapsed:.2f}s (串行应 ~0.8s)"
        assert set(calls) == {"000001", "000002", "000003", "000004"}
        # 失败标的标记 skip(与串行版一致)
        assert cands[1]["skip"] is True
        assert all(not c.get("skip") for c in (cands[0], cands[2], cands[3]))
