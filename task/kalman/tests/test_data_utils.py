"""Tests for data_utils.py 网络超时兜底(2026-08-24)。

背景: akshare 底层 requests 无显式超时, 网络挂起时进程无限等待
(8-24 daily_signal 曾在 20/49 卡 10 分钟)。修复: 模块导入时设置
socket.setdefaulttimeout(30), 覆盖全部下载路径。
"""

import os
import socket
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data_utils
import daily_signal


class TestNetworkTimeout:
    """socket 级默认超时兜底。"""

    def test_default_timeout_set(self):
        """导入 data_utils 后 socket 默认超时 = 30s(全局生效)。"""
        assert socket.getdefaulttimeout() == pytest.approx(30.0)

    def test_constant_defined(self):
        """超时常量存在且为正数。"""
        assert data_utils._NETWORK_TIMEOUT > 0
        assert data_utils._NETWORK_TIMEOUT == 30.0

    def test_download_error_falls_back_to_cache(self, monkeypatch, tmp_path):
        """下载抛异常(含超时) → download_with_cache 用缓存数据。"""
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        df = pd.DataFrame({
            "date": ["2026-08-20", "2026-08-21"],
            "open": [10.0, 11.0], "high": [10.5, 11.5],
            "low": [9.5, 10.5], "close": [10.2, 11.2],
            "volume": [1000, 1200],
        })
        df.to_parquet(cache_dir / "000001.parquet", index=False)
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(cache_dir))

        def _fail(*_a, **_k):
            raise socket.timeout("模拟网络挂起超时")

        monkeypatch.setattr(data_utils, "download_data", _fail)
        out = daily_signal.download_with_cache(
            "000001", data_years=2, asset_type="stock")
        assert len(out) == 2
        assert float(out["close"].iloc[-1]) == 11.2

    def test_download_failure_without_cache_raises(self, monkeypatch, tmp_path):
        """下载失败且无缓存 → 抛异常(不静默返回空)。"""
        monkeypatch.setattr(daily_signal, "CACHE_DIR", str(tmp_path / "no_cache"))

        def _fail(*_a, **_k):
            raise ValueError("模拟下载失败")

        monkeypatch.setattr(data_utils, "download_data", _fail)
        with pytest.raises(ValueError):
            daily_signal.download_with_cache("000001", data_years=2)
