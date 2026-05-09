"""
langgraph-checkpoint-delta
~~~~~~~~~~~~~~~~~~~~~~~~~~

A LangGraph checkpointer that persists agent state to Delta Lake tables,
designed for Databricks / Unity Catalog environments.

Quickstart
----------
    from langgraph_checkpoint_delta import DeltaCheckpointer

    checkpointer = DeltaCheckpointer(
        table_uri="catalog.schema.agent_checkpoints",
        backend="auto",   # "auto" | "spark" | "deltalake"
    )

The backend is selected at runtime:
- "auto"      → uses Spark if an active SparkSession exists, else python-deltalake
- "spark"     → forces delta-spark (requires pyspark + delta-spark installed)
- "deltalake" → forces python-deltalake (no Spark required)
"""

from langgraph_checkpoint_delta.checkpointer import DeltaCheckpointer

__all__ = ["DeltaCheckpointer"]
__version__ = "0.1.0"