"""
backends/deltalake_backend.py
------------------------------
Delta backend implementation using python-deltalake (the Rust-backed library).

When to use this backend
------------------------
- No Spark available (local dev, CI, non-Databricks environments)
- You have a raw storage path (S3, ABFSS, GCS, local) for the Delta table
- You want fast, lightweight reads/writes without Spark startup overhead

Limitation vs SparkBackend
---------------------------
python-deltalake cannot write to Unity Catalog managed tables by name
("catalog.schema.table"). It needs a raw storage URI. If you're on
Databricks and need UC governance, use the SparkBackend instead.

UPSERT strategy
---------------
python-deltalake supports MERGE natively as of v0.17. We use it here
for the same reason as the Spark backend — to avoid unbounded table growth
from blind appends.

Arrow schema
------------
python-deltalake uses PyArrow for schema definition and data interchange.
All writes go through a PyArrow RecordBatch. This avoids pandas as a
dependency, keeping the install footprint lean.
"""

from __future__ import annotations

import datetime
from typing import Iterator, List, Optional

import pyarrow as pa  # type: ignore[import]

from langgraph_checkpoint_delta.backends.base import BaseDeltaBackend, Row
from langgraph_checkpoint_delta.schema import (
    PARTITION_COLUMNS,
    get_schema,
)


class DeltaLakeBackend(BaseDeltaBackend):
    """
    Delta backend using python-deltalake (Rust) for all I/O.

    Reads use DeltaTable.to_pyarrow_dataset() for predicate pushdown.
    Writes use write_deltalake() with merge semantics.
    """

    def __init__(self, table_uri: str) -> None:
        """
        Args:
            table_uri: Absolute storage path to the Delta table.
                       Examples:
                         "/tmp/checkpoints"                              (local)
                         "s3://my-bucket/agents/checkpoints"            (S3)
                         "abfss://container@account.dfs.core.windows.net/path"  (ADLS)
        """
        self.table_uri = table_uri

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_arrow_schema(self, include_finops: bool = False) -> pa.Schema:
        """Build a PyArrow schema from our ColumnDef list."""
        columns = get_schema(include_finops=include_finops)
        fields = []
        for col in columns:
            # eval is safe here — arrow_type values are hardcoded in schema.py
            arrow_type = eval(col.arrow_type, {"pa": pa})  # noqa: S307
            fields.append(pa.field(col.name, arrow_type, nullable=col.nullable))
        return pa.schema(fields)

    def _load_table(self):  # type: ignore[return]
        """Load an existing DeltaTable, raising clearly if not found."""
        from deltalake import DeltaTable  # type: ignore[import]
        return DeltaTable(self.table_uri)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, include_finops: bool = False) -> None:
        """Create the Delta table if it does not already exist."""
        from deltalake import DeltaTable  # type: ignore[import]
        from deltalake.exceptions import TableNotFoundError  # type: ignore[import]

        try:
            DeltaTable(self.table_uri)
            # Table exists — nothing to do (schema migration is out of scope for v0.1)
        except TableNotFoundError:
            schema = self._build_arrow_schema(include_finops=include_finops)
            # Write an empty RecordBatch to initialise the table with the correct schema
            empty_batch = pa.RecordBatch.from_pydict(
                {field.name: pa.array([], type=field.type) for field in schema},
                schema=schema,
            )
            from deltalake.writer import write_deltalake  # type: ignore[import]
            write_deltalake(
                self.table_uri,
                pa.Table.from_batches([empty_batch], schema=schema),
                partition_by=PARTITION_COLUMNS,
                mode="error",  # fail fast if something is wrong
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, row: Row) -> None:
        """MERGE a single checkpoint row into the Delta table."""
        from deltalake import DeltaTable  # type: ignore[import]
        from deltalake.writer import write_deltalake  # type: ignore[import]

        schema = self._build_arrow_schema()
        table = self._row_to_arrow_table(row, schema)

        dt = self._load_table()

        (
            dt.merge(
                source=table,
                predicate=(
                    "target.thread_id = source.thread_id "
                    "AND target.checkpoint_ns = source.checkpoint_ns "
                    "AND target.checkpoint_id = source.checkpoint_id"
                ),
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
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
        import pyarrow.dataset as ds  # type: ignore[import]

        dt = self._load_table()
        dataset = dt.to_pyarrow_dataset()

        result = dataset.to_table(
            filter=(
                (ds.field("thread_id") == thread_id)
                & (ds.field("checkpoint_ns") == checkpoint_ns)
            )
        )

        if result.num_rows == 0:
            return None

        # Sort by created_at descending, take first row
        import pyarrow.compute as pc  # type: ignore[import]
        sorted_table = result.sort_by([("created_at", "descending")])
        return self._arrow_row_to_dict(sorted_table, 0)

    def get_by_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> Optional[Row]:
        import pyarrow.dataset as ds  # type: ignore[import]

        dt = self._load_table()
        dataset = dt.to_pyarrow_dataset()

        result = dataset.to_table(
            filter=(
                (ds.field("thread_id") == thread_id)
                & (ds.field("checkpoint_ns") == checkpoint_ns)
                & (ds.field("checkpoint_id") == checkpoint_id)
            )
        )

        if result.num_rows == 0:
            return None

        return self._arrow_row_to_dict(result, 0)

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
        import pyarrow.dataset as ds  # type: ignore[import]

        dt = self._load_table()
        dataset = dt.to_pyarrow_dataset()

        predicate = (
            (ds.field("thread_id") == thread_id)
            & (ds.field("checkpoint_ns") == checkpoint_ns)
        )

        result = dataset.to_table(filter=predicate)

        if result.num_rows == 0:
            return

        sorted_table = result.sort_by([("created_at", "descending")])

        if before_checkpoint_id is not None:
            # Find the created_at of the reference checkpoint
            import pyarrow.compute as pc  # type: ignore[import]
            mask = pc.equal(sorted_table.column("checkpoint_id"), before_checkpoint_id)
            ref_indices = pc.list_flatten(
                pa.chunked_array([pa.array([mask.to_pylist().index(True)]
                                           if True in mask.to_pylist() else [])])
            )
            if len(ref_indices) > 0:
                ref_ts = sorted_table.column("created_at")[ref_indices[0].as_py()].as_py()
                ts_mask = pc.less(sorted_table.column("created_at"), pa.scalar(ref_ts))
                sorted_table = sorted_table.filter(ts_mask)

        if limit is not None:
            sorted_table = sorted_table.slice(0, limit)

        for i in range(sorted_table.num_rows):
            yield self._arrow_row_to_dict(sorted_table, i)

    # ------------------------------------------------------------------
    # Row conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_arrow_table(row: Row, schema: pa.Schema) -> pa.Table:
        """Convert a plain dict row to a single-row PyArrow Table."""
        arrays = {}
        for field in schema:
            val = row.get(field.name)
            arrays[field.name] = pa.array([val], type=field.type)
        return pa.table(arrays, schema=schema)

    @staticmethod
    def _arrow_row_to_dict(table: pa.Table, index: int) -> Row:
        """Extract row at `index` from a PyArrow Table as a plain dict."""
        return {
            col: table.column(col)[index].as_py()
            for col in table.schema.names
        }