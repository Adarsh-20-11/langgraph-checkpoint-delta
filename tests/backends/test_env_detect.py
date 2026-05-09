"""
test_env_detect.py
------------------
Tests for environment detection and backend resolution.
All Spark/deltalake imports are mocked — no real backends needed.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langgraph_checkpoint_delta.env_detect import (
    detect_backend,
    has_active_spark_session,
    has_deltalake,
    resolve_backend,
)


class TestHasActiveSparkSession:
    def test_returns_false_when_pyspark_not_installed(self):
        with patch.dict("sys.modules", {"pyspark": None, "pyspark.sql": None}):
            assert has_active_spark_session() is False

    def test_returns_false_when_no_active_session(self):
        mock_spark = MagicMock()
        mock_spark.SparkSession.getActiveSession.return_value = None
        with patch.dict("sys.modules", {"pyspark": MagicMock(), "pyspark.sql": mock_spark}):
            assert has_active_spark_session() is False

    def test_returns_true_when_session_exists(self):
        mock_session = MagicMock()
        mock_spark = MagicMock()
        mock_spark.SparkSession.getActiveSession.return_value = mock_session
        with patch.dict("sys.modules", {"pyspark": MagicMock(), "pyspark.sql": mock_spark}):
            assert has_active_spark_session() is True


class TestHasDeltalake:
    def test_returns_false_when_not_installed(self):
        with patch.dict("sys.modules", {"deltalake": None}):
            assert has_deltalake() is False

    def test_returns_true_when_installed(self):
        with patch.dict("sys.modules", {"deltalake": MagicMock()}):
            assert has_deltalake() is True


class TestDetectBackend:
    def test_prefers_spark_when_session_exists(self):
        with (
            patch("langgraph_checkpoint_delta.env_detect.has_active_spark_session", return_value=True),
            patch("langgraph_checkpoint_delta.env_detect.has_deltalake", return_value=True),
        ):
            assert detect_backend() == "spark"

    def test_falls_back_to_deltalake_when_no_spark(self):
        with (
            patch("langgraph_checkpoint_delta.env_detect.has_active_spark_session", return_value=False),
            patch("langgraph_checkpoint_delta.env_detect.has_deltalake", return_value=True),
        ):
            assert detect_backend() == "deltalake"

    def test_raises_when_nothing_available(self):
        with (
            patch("langgraph_checkpoint_delta.env_detect.has_active_spark_session", return_value=False),
            patch("langgraph_checkpoint_delta.env_detect.has_deltalake", return_value=False),
        ):
            with pytest.raises(ImportError, match="No Delta Lake backend found"):
                detect_backend()


class TestResolveBackend:
    def test_auto_delegates_to_detect(self):
        with patch(
            "langgraph_checkpoint_delta.env_detect.detect_backend", return_value="deltalake"
        ):
            assert resolve_backend("auto") == "deltalake"

    def test_spark_explicit_succeeds_with_session(self):
        with patch(
            "langgraph_checkpoint_delta.env_detect.has_active_spark_session", return_value=True
        ):
            assert resolve_backend("spark") == "spark"

    def test_spark_explicit_fails_without_session(self):
        with patch(
            "langgraph_checkpoint_delta.env_detect.has_active_spark_session", return_value=False
        ):
            with pytest.raises(ImportError, match="no active SparkSession"):
                resolve_backend("spark")

    def test_deltalake_explicit_succeeds_when_installed(self):
        with patch(
            "langgraph_checkpoint_delta.env_detect.has_deltalake", return_value=True
        ):
            assert resolve_backend("deltalake") == "deltalake"

    def test_deltalake_explicit_fails_when_not_installed(self):
        with patch(
            "langgraph_checkpoint_delta.env_detect.has_deltalake", return_value=False
        ):
            with pytest.raises(ImportError, match="python-deltalake is not installed"):
                resolve_backend("deltalake")

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("foobar")