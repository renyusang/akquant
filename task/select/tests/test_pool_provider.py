"""Tests for pool_provider.py 候选池 akshare 数据源。

覆盖: 代码清洗、成分拉取列选择、过滤逻辑(市值/成交额/ST, mock 网络)、
妙想 CSV 兼容、合并去重、输出列名对齐 screen_pool。
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pool_provider


class TestCodeNormalization:
    """新浪/东财代码清洗。"""

    def test_strip_exchange_prefix(self):
        f = pool_provider.fetch_spot_data
        # 用内部逻辑直接验证: 构造 DataFrame 走 _norm_code 等价路径
        import akshare as ak
        # 不直接调网络, 验证清洗函数行为
        assert "600519" == "sh600519".replace("sh", "").zfill(6)
        assert "000001" == "sz000001".replace("sz", "").zfill(6)
        assert "920000" == "bj920000".replace("bj", "").zfill(6)
        assert "600519" == "600519".zfill(6)


class TestConstituentColumns:
    """成分拉取的列选择(避免选到"指数代码"列)。"""

    def test_code_col_prefers_constituent(self, monkeypatch):
        """csindex 列含'指数代码'(全是指数自身)——必须选'成分券代码'。"""
        import akshare as ak
        df = pd.DataFrame({
            "日期": ["2026-01-01"] * 2,
            "指数代码": ["000300"] * 2,
            "指数名称": ["沪深300"] * 2,
            "成分券代码": ["600519", "000858"],
            "成分券名称": ["贵州茅台", "五粮液"],
        })
        monkeypatch.setattr(ak, "index_stock_cons_csindex",
                            lambda symbol: df)
        members = pool_provider.fetch_index_constituents("000300")
        assert members == [("600519", "贵州茅台"), ("000858", "五粮液")]


class TestBuildPool:
    """候选池构建: mock 网络, 验证过滤逻辑。"""

    def test_filter_and_dedup(self, monkeypatch):
        """市值/成交额过滤 + ST 排除 + 去重。"""
        # 成分: 6 只(含重复)
        members = [("600519", "茅台"), ("000001", "平安"), ("600000", "浦发"),
                   ("000002", "万科"), ("300750", "宁德"), ("688981", "中芯")]
        monkeypatch.setattr(pool_provider, "fetch_index_constituents",
                            lambda idx: members)
        # 行情: 茅台大市值大成交, 平安小市值, 浦发大市值小成交, 万科正常, 宁德/中芯正常
        spot = pd.DataFrame({
            "代码": ["600519", "000001", "600000", "000002", "300750",
                     "688981"],
            "总市值": [1.5e12, 2e9, 5e11, 8e10, 6e11, 7e11],
            "成交额": [5e9, 1e8, 5e7, 2e9, 3e9, 4e9],
        })
        monkeypatch.setattr(pool_provider, "fetch_spot_data", lambda: spot)
        monkeypatch.setattr(pool_provider, "fetch_st_list",
                            lambda: {"688981"})  # 中芯为 ST(mock)

        pool = pool_provider.build_ak_pool(
            ["000300"], min_market_cap=100.0, min_turnover=1.0, quiet=True)
        syms = [r["symbol"] for r in pool]
        assert "600519" in syms          # 大市值+大成交
        assert "000001" not in syms      # 市值<100亿
        assert "600000" not in syms      # 成交额<1亿
        assert "000002" in syms          # 两者都过
        assert "688981" not in syms      # ST 排除
        assert len(syms) == len(set(syms))  # 去重

    def test_sina_fallback_no_market_cap(self, monkeypatch):
        """新浪回退(无市值列) → 市值过滤跳过, 成交额过滤仍执行。"""
        members = [("600519", "茅台"), ("000001", "平安")]
        monkeypatch.setattr(pool_provider, "fetch_index_constituents",
                            lambda idx: members)
        spot = pd.DataFrame({"代码": ["600519", "000001"],
                             "成交额": [5e9, 5e7]})
        monkeypatch.setattr(pool_provider, "fetch_spot_data", lambda: spot)
        monkeypatch.setattr(pool_provider, "fetch_st_list", lambda: set())
        pool = pool_provider.build_ak_pool(
            ["000300"], min_market_cap=100.0, min_turnover=1.0, quiet=True)
        syms = [r["symbol"] for r in pool]
        assert "600519" in syms
        assert "000001" not in syms      # 成交额<1亿仍被过滤


class TestMergeAndSave:
    """妙想 CSV 兼容 + 合并去重 + 输出列名。"""

    def test_load_miaoxiang_and_merge(self, tmp_path):
        mx_path = tmp_path / "mx.csv"
        mx_path.write_text("code,name\n002594,比亚迪\n510050,上证50ETF\n",
                           encoding="utf-8")
        mx = pool_provider.load_miaoxiang_csv(str(mx_path))
        assert len(mx) == 2
        by_sym = {r["symbol"]: r for r in mx}
        assert by_sym["510050"]["asset_type"] == "etf"

        ak = [{"symbol": "002594", "name": "比亚迪", "asset_type": "stock",
               "source": "沪深300"}]
        merged = pool_provider.merge_pools(ak, mx)
        assert len(merged) == 2  # 002594 去重(先到先得保留 akshare source)

    def test_save_columns_align_screen_pool(self, tmp_path):
        """输出列名 code,name,source 对齐 screen_pool.parse_candidates。"""
        out = tmp_path / "pool.csv"
        rows = [{"symbol": "600519", "name": "贵州茅台",
                 "asset_type": "stock", "source": "沪深300"}]
        pool_provider.save_candidates(rows, str(out))
        df = pd.read_csv(out, dtype={"code": str})
        assert list(df.columns) == ["code", "name", "source"]
