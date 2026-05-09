"""
env_detect.py
-------------
Runtime environment detection for backend auto-selection.

Isolated in its own module so tests can mock detection without
patching internals of the checkpointer or backends.

Decision logic for "auto" mode
-------------------------------
1. Is there an active SparkSession?  → use SparkBackend
2. Is `deltalake` importable?        → use DeltaLakeBackend
3. Neither available                 → raise ImportError with install hint
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

BackendName = Literal["spark", "deltalake"]


def has_active_spark_session() -> bool:
    """
    Returns True if pyspark is installed and an active SparkSession exists.

    Does NOT import pyspark at module level — this file must be importable
    even if neither backend is installed.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore[import]
        session = SparkSession.getActiveSession()
        return session is not None
    except ImportError:
        return False


def has_deltalake() -> bool:
    """Returns True if the python-deltalake (rust) library is importable."""
    try:
        import deltalake  # type: ignore[import]  # noqa: F401
        return True
    except ImportError:
        return False


def has_delta_spark() -> bool:
    """Returns True if delta-spark is importable (does not check for active session)."""
    try:
        import delta  # type: ignore[import]  # noqa: F401
        return True
    except ImportError:
        return False


def detect_backend() -> BackendName:
    """
    Auto-detect the appropriate backend for the current runtime environment.

    Resolution order:
      1. Active SparkSession present → "spark"
      2. python-deltalake importable → "deltalake"
      3. Neither                     → ImportError

    Returns:
        "spark" or "deltalake"

    Raises:
        ImportError: If no suitable backend is available, with install instructions.
    """
    if has_active_spark_session():
        return "spark"

    if has_deltalake():
        return "deltalake"

    raise ImportError(
        "No Delta Lake backend found. Install one of:\n"
        "  pip install langgraph-checkpoint-delta[deltalake]  "
        "# Rust-backed, no Spark required\n"
        "  pip install langgraph-checkpoint-delta[spark]      "
        "# PySpark + delta-spark\n"
        "  pip install langgraph-checkpoint-delta[all]        "
        "# Both backends\n\n"
        "Or pass backend='spark'|'deltalake' explicitly to DeltaCheckpointer "
        "to suppress auto-detection."
    )


def resolve_backend(backend: str) -> BackendName:
    """
    Validate and resolve a backend specifier to a canonical name.

    Args:
        backend: One of "auto", "spark", or "deltalake".

    Returns:
        "spark" or "deltalake"

    Raises:
        ValueError: If an unknown backend name is passed.
        ImportError: If "auto" is selected but no backend is available,
                     or if an explicit backend is not installed.
    """
    if backend == "auto":
        return detect_backend()

    if backend == "spark":
        if not has_active_spark_session():
            raise ImportError(
                "backend='spark' requested but no active SparkSession found. "
                "Ensure PySpark and delta-spark are installed and a SparkSession "
                "is running before instantiating DeltaCheckpointer."
            )
        return "spark"

    if backend == "deltalake":
        if not has_deltalake():
            raise ImportError(
                "backend='deltalake' requested but python-deltalake is not installed.\n"
                "  pip install langgraph-checkpoint-delta[deltalake]"
            )
        return "deltalake"

    raise ValueError(
        f"Unknown backend {backend!r}. Must be one of: 'auto', 'spark', 'deltalake'."
    )