"""
backends/spark_backend.py
--------------------------
Delta backend implementation using PySpark + delta-spark.

When to use this backend
------------------------
- You are running on Databricks (Jobs or interactive clusters)
- An active SparkSession already exists in your process
- You need Unity Catalog 3-part names (catalog.schema.table) as the
  table_uri, which requires Spark + UC runtime

Unity Catalog note
------------------
python-deltalake cannot write to Unity Catalog managed tables directly —
it lacks the UC credential vending integration. If your table_uri is a
3-part UC name (e.g. "main.agents.checkpoints"), you MUST use this backend.
The DeltaLakeBackend only works with raw ABFSS/S3/GCS paths or local paths.

UPSERT strategy
---------------
We use a MERGE INTO statement rather than overwrite to preserve ACID
guarantees. A blind append + dedup-on-read would work but would cause
unbounded table growth. MERGE ensures exactly one row per
(thread_id, checkpoint_ns, checkpoint_id).
"""

from __future__ import annotations

import datetime
from typing import Iterator, List, Optional

from langgraph_checkpoint_delta.backends.base import BaseDeltaBackend, Row
from langgraph_checkpoint_delta.schema import (
    PARTITION_COLUMNS,
    get_schema,
)


class SparkBackend(BaseDeltaBackend):
    """
    Delta backend that uses an active SparkSession for all I/O.

    The SparkSession is retrieved lazily via SparkSession.getActiveSession()
    at call time rather than at __init__ time. This makes the class safe to
    instantiate in test environments where the session may not exist yet.
    """

    def __init__(self, table_uri: str) -> None:
        """
        Args:
            table_uri: Either a 3-part Unity Catalog name
                       ("catalog.schema.table") or an absolute path
                       ("abfss://container@account.dfs.core.windows.net/path").
        """
        self.table_uri = table_uri

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spark(self):  # type: ignore[return]
        """Return the active SparkSession, raising clearly if absent."""
        from pyspark.sql import SparkSession  # type: ignore[import]
        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError(
                "SparkBackend requires an active SparkSession but none was found. "
                "Ensure Spark is initialised before using DeltaCheckpointer with backend='spark'."
            )
        return session

    def _build_spark_schema(self, include_finops: bool = False):
        """Build a PySpark StructType from our ColumnDef list."""
        from pyspark.sql import types as T  # type: ignore[import]

        type_map = {
            "StringType()": T.StringType(),
            "BinaryType()": T.BinaryType(),
            "TimestampType()": T.TimestampType(),
            "DoubleType()": T.DoubleType(),
            "LongType()": T.LongType(),
        }

        columns = get_schema(include_finops=include_finops)
        fields = [
            T.StructField(col.name, type_map[col.spark_type], col.nullable)
            for col in columns
        ]
        return T.StructType(fields)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, include_finops: bool = False) -> None:
        """Create the Delta table if it does not already exist."""
        spark = self._spark()
        schema = self._build_spark_schema(include_finops=include_finops)

        # DeltaTable.createIfNotExists handles the idempotency
        from delta import DeltaTable  # type: ignore[import]

        builder = (
            DeltaTable.createIfNotExists(spark)
            .tableName(self.table_uri)
            .addColumns(schema)
            .partitionedBy(*PARTITION_COLUMNS)
            .property("delta.autoOptimize.optimizeWrite", "true")
            .property("delta.autoOptimize.autoCompact", "true")
        )
        builder.execute()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, row: Row) -> None:
        """MERGE a single checkpoint row into the Delta table."""
        spark = self._spark()

        from delta import DeltaTable  # type: ignore[import]

        # Build a single-row DataFrame from the row dict.
        # Binary fields arrive as bytes; Spark wants bytearray for BinaryType.
        spark_row = {
            k: (bytearray(v) if isinstance(v, (bytes, bytearray)) else v)
            for k, v in row.items()
        }
        df = spark.createDataFrame([spark_row])  # type: ignore[arg-type]

        delta_table = DeltaTable.forName(spark, self.table_uri)

        (
            delta_table.alias("target")
            .merge(
                df.alias("source"),
                "target.thread_id = source.thread_id "
                "AND target.checkpoint_ns = source.checkpoint_ns "
                "AND target.checkpoint_id = source.checkpoint_id",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    # ------------------------------------------------------------------
    # Read — single row
    # ------------------------------------------------------------------

    def get_latest(
        self,
        thread_id: str,
        checkpoint_ns: str,
    ) -> Optional[Row]:
        spark = self._spark()
        df = (
            spark.read.format("delta")
            .table(self.table_uri)
            .filter(
                f"thread_id = '{thread_id}' AND checkpoint_ns = '{checkpoint_ns}'"
            )
            .orderBy("created_at", ascending=False)
            .limit(1)
        )
        rows = df.collect()
        if not rows:
            return None
        return self._spark_row_to_dict(rows[0])

    def get_by_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> Optional[Row]:
        spark = self._spark()
        df = (
            spark.read.format("delta")
            .table(self.table_uri)
            .filter(
                f"thread_id = '{thread_id}' "
                f"AND checkpoint_ns = '{checkpoint_ns}' "
                f"AND checkpoint_id = '{checkpoint_id}'"
            )
            .limit(1)
        )
        rows = df.collect()
        if not rows:
            return None
        return self._spark_row_to_dict(rows[0])

    # ------------------------------------------------------------------
    # Read — list
    # ------------------------------------------------------------------

    def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        limit: Optional[int] = None,
        before_checkpoint_id: Optional[str] = None,
    ) -> Iterator[Row]:
        spark = self._spark()

        predicate = (
            f"thread_id = '{thread_id}' AND checkpoint_ns = '{checkpoint_ns}'"
        )

        if before_checkpoint_id is not None:
            # Find created_at of the reference checkpoint, then filter < that
            ref_df = (
                spark.read.format("delta")
                .table(self.table_uri)
                .filter(
                    f"{predicate} AND checkpoint_id = '{before_checkpoint_id}'"
                )
                .select("created_at")
                .limit(1)
            )
            ref_rows = ref_df.collect()
            if ref_rows:
                ref_ts = ref_rows[0]["created_at"]
                predicate += f" AND created_at < '{ref_ts}'"

        df = (
            spark.read.format("delta")
            .table(self.table_uri)
            .filter(predicate)
            .orderBy("created_at", ascending=False)
        )

        if limit is not None:
            df = df.limit(limit)

        for spark_row in df.collect():
            yield self._spark_row_to_dict(spark_row)

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _spark_row_to_dict(spark_row) -> Row:  # type: ignore[return]
        """Convert a PySpark Row to a plain dict, converting bytearray → bytes."""
        d = spark_row.asDict()
        return {
            k: (bytes(v) if isinstance(v, bytearray) else v)
            for k, v in d.items()
        }