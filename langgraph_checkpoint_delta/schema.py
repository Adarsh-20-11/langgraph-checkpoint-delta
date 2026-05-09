"""
schema.py
---------
Defines the Delta Lake table schema for checkpoint storage.

Columns are split into two categories:

1. **Queryable columns** — stored as typed Delta columns so that anyone
   with SQL access can inspect, filter, and join against checkpoint state
   without deserializing blobs. These also enable Time Travel queries like:
       SELECT * FROM checkpoints VERSION AS OF 42 WHERE thread_id = '...'

2. **Blob column** — the serialized checkpoint state + LangGraph metadata,
   stored as BINARY. Only the checkpointer deserializes this.

This separation is intentional: it preserves SQL observability (a core
architectural goal) while keeping the LangGraph internals opaque to
downstream consumers.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ColumnDef:
    """Lightweight column descriptor used to drive table creation in both backends."""
    name: str
    spark_type: str       # e.g. "StringType()"  — used by SparkBackend
    arrow_type: str       # e.g. "pa.string()"   — used by DeltaLakeBackend
    nullable: bool = True
    comment: str = ""


# ---------------------------------------------------------------------------
# Core checkpoint schema
# ---------------------------------------------------------------------------

CHECKPOINT_COLUMNS: List[ColumnDef] = [
    ColumnDef(
        name="thread_id",
        spark_type="StringType()",
        arrow_type="pa.string()",
        nullable=False,
        comment="LangGraph thread identifier. Groups all checkpoints for one conversation.",
    ),
    ColumnDef(
        name="checkpoint_ns",
        spark_type="StringType()",
        arrow_type="pa.string()",
        nullable=False,
        comment="Checkpoint namespace. Supports sub-graphs within a thread.",
    ),
    ColumnDef(
        name="checkpoint_id",
        spark_type="StringType()",
        arrow_type="pa.string()",
        nullable=False,
        comment="Unique ID for this checkpoint (UUID). Primary key within a thread.",
    ),
    ColumnDef(
        name="parent_checkpoint_id",
        spark_type="StringType()",
        arrow_type="pa.string()",
        nullable=True,
        comment="ID of the preceding checkpoint. NULL for the first checkpoint in a thread.",
    ),
    ColumnDef(
        name="type",
        spark_type="StringType()",
        arrow_type="pa.string()",
        nullable=True,
        comment="Serializer type identifier, e.g. 'msgpack'. Used during deserialization.",
    ),
    ColumnDef(
        name="checkpoint",
        spark_type="BinaryType()",
        arrow_type="pa.binary()",
        nullable=False,
        comment="Serialized checkpoint state blob.",
    ),
    ColumnDef(
        name="metadata",
        spark_type="BinaryType()",
        arrow_type="pa.binary()",
        nullable=False,
        comment="Serialized LangGraph checkpoint metadata blob.",
    ),
    ColumnDef(
        name="created_at",
        spark_type="TimestampType()",
        arrow_type="pa.timestamp('us', tz='UTC')",
        nullable=False,
        comment="UTC timestamp of checkpoint creation. Enables time-based filtering and TTL.",
    ),
]

# ---------------------------------------------------------------------------
# FinOps extension schema (optional, appended when cost tracking is enabled)
# These columns are intentionally separate so the core schema stays clean.
# ---------------------------------------------------------------------------

FINOPS_COLUMNS: List[ColumnDef] = [
    ColumnDef(
        name="dbu_cost_usd",
        spark_type="DoubleType()",
        arrow_type="pa.float64()",
        nullable=True,
        comment="Estimated Databricks Unit cost in USD for this checkpoint turn.",
    ),
    ColumnDef(
        name="token_input_count",
        spark_type="LongType()",
        arrow_type="pa.int64()",
        nullable=True,
        comment="Number of input tokens consumed to produce this checkpoint.",
    ),
    ColumnDef(
        name="token_output_count",
        spark_type="LongType()",
        arrow_type="pa.int64()",
        nullable=True,
        comment="Number of output tokens produced at this checkpoint.",
    ),
    ColumnDef(
        name="model_tier",
        spark_type="StringType()",
        arrow_type="pa.string()",
        nullable=True,
        comment="Model identifier used for this turn, e.g. 'claude-sonnet-4'. Used for cost lookup.",
    ),
]


def get_schema(include_finops: bool = False) -> List[ColumnDef]:
    """
    Return the full column list for table creation.

    Args:
        include_finops: If True, appends the FinOps cost-tracking columns.

    Returns:
        List of ColumnDef instances in column order.
    """
    columns = list(CHECKPOINT_COLUMNS)
    if include_finops:
        columns.extend(FINOPS_COLUMNS)
    return columns


# ---------------------------------------------------------------------------
# Partition strategy
# ---------------------------------------------------------------------------

#: Partitioning by thread_id keeps related checkpoints co-located on disk,
#: which speeds up get_tuple() and list() scans on a per-thread basis.
#: For very high thread-count workloads, consider hash-bucketing instead.
PARTITION_COLUMNS: List[str] = ["thread_id"]