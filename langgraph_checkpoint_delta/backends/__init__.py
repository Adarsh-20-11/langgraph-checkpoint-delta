"""
backends/__init__.py
--------------------
Factory for instantiating the correct backend at runtime.
"""

from __future__ import annotations

from langgraph_checkpoint_delta.backends.base import BaseDeltaBackend
from langgraph_checkpoint_delta.env_detect import BackendName


def make_backend(
    backend_name: BackendName,
    table_uri: str,
    include_finops: bool = False,
) -> BaseDeltaBackend:
    """
    Instantiate and return the appropriate backend.

    Args:
        backend_name:   "spark" or "deltalake" (already resolved by env_detect).
        table_uri:      Delta table URI or 3-part Unity Catalog name.
        include_finops: Whether to include FinOps cost-tracking columns.

    Returns:
        An initialised BaseDeltaBackend subclass.
    """
    if backend_name == "spark":
        from langgraph_checkpoint_delta.backends.spark_backend import SparkBackend
        backend = SparkBackend(table_uri=table_uri)
    elif backend_name == "deltalake":
        from langgraph_checkpoint_delta.backends.deltalake_backend import DeltaLakeBackend
        backend = DeltaLakeBackend(table_uri=table_uri)
    else:
        raise ValueError(f"Unknown backend: {backend_name!r}")

    backend.setup(include_finops=include_finops)
    return backend


__all__ = ["BaseDeltaBackend", "make_backend"]