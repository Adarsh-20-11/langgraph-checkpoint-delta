"""
checkpointer.py
---------------
The main DeltaCheckpointer class. Extends LangGraph's BaseCheckpointSaver
and implements the full checkpointer interface in both sync and async modes.

Architecture
------------
This class is deliberately thin. It:
  1. Resolves which backend to use (via env_detect)
  2. Translates between LangGraph's CheckpointTuple objects and the flat
     Row dicts that backends understand
  3. Delegates all Delta I/O to the backend

It knows nothing about Spark, pyarrow, or deltalake internals. That
separation means the checkpointer logic is fully testable with a fake backend.

LangGraph interface
-------------------
LangGraph calls these methods on the checkpointer:

  get_tuple(config)           → CheckpointTuple | None
  put(config, checkpoint, metadata, new_versions) → RunnableConfig
  put_writes(config, writes, task_id) → None   (pending writes — v0.2)
  list(config, *, filter, before, limit) → Iterator[CheckpointTuple]

And their async equivalents (aget_tuple, aput, alist).

CheckpointTuple structure
-------------------------
  CheckpointTuple(
      config:          RunnableConfig   — the config used to retrieve it
      checkpoint:      Checkpoint       — the actual state
      metadata:        CheckpointMetadata
      parent_config:   RunnableConfig | None  — config of the parent checkpoint
      pending_writes:  list | None
  )
"""

from __future__ import annotations

import datetime
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # type: ignore[import]
from typing_extensions import Self

from langgraph_checkpoint_delta.backends import make_backend
from langgraph_checkpoint_delta.backends.base import BaseDeltaBackend, Row
from langgraph_checkpoint_delta.env_detect import resolve_backend
from langgraph_checkpoint_delta.serialization import (
    deserialize_checkpoint,
    deserialize_metadata,
    get_default_serializer,
    serialize_checkpoint,
    serialize_metadata,
)


class DeltaCheckpointer(BaseCheckpointSaver):
    """
    A LangGraph checkpointer that persists agent state to Delta Lake.

    Supports both sync and async interfaces. Backend (Spark vs python-deltalake)
    is selected at instantiation time via the `backend` parameter.

    Example usage
    -------------
        # Auto-detect backend (Spark if available, else python-deltalake)
        cp = DeltaCheckpointer(table_uri="main.agents.checkpoints")

        # Force python-deltalake with a storage path (local dev / CI)
        cp = DeltaCheckpointer(
            table_uri="/tmp/my_checkpoints",
            backend="deltalake",
        )

        # Use with LangGraph
        graph = builder.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "thread-001"}}
        result = graph.invoke({"messages": [...]}, config)
    """

    backend: BaseDeltaBackend

    def __init__(
        self,
        table_uri: str,
        backend: str = "auto",
        include_finops: bool = False,
        serde: Optional[Any] = None,
    ) -> None:
        """
        Args:
            table_uri:      Delta table location. Either a 3-part Unity Catalog
                            name ("catalog.schema.table", Spark only) or an
                            absolute storage path ("/tmp/...", "s3://...", etc.).
            backend:        Backend selection: "auto" | "spark" | "deltalake".
                            "auto" uses Spark if an active SparkSession exists,
                            otherwise falls back to python-deltalake.
            include_finops: If True, adds DBU cost-tracking columns to the table
                            schema on first creation.
            serde:          Optional custom serializer. Defaults to LangGraph's
                            JsonPlusSerializer.
        """
        # BaseCheckpointSaver expects a serde to be passed via super().__init__
        _serde = serde or get_default_serializer()
        super().__init__(serde=_serde)

        resolved = resolve_backend(backend)
        self.backend = make_backend(
            backend_name=resolved,
            table_uri=table_uri,
            include_finops=include_finops,
        )
        self._include_finops = include_finops

    # ------------------------------------------------------------------
    # Context manager support (mirrors official LangGraph checkpointers)
    # ------------------------------------------------------------------

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        table_uri: str,
        backend: str = "auto",
        include_finops: bool = False,
    ) -> Iterator[Self]:
        """
        Sync context manager — mirrors the interface of langgraph-checkpoint-postgres.

        Example:
            with DeltaCheckpointer.from_conn_string("main.agents.cp") as cp:
                graph = builder.compile(checkpointer=cp)
        """
        cp = cls(table_uri=table_uri, backend=backend, include_finops=include_finops)
        try:
            yield cp  # type: ignore[misc]
        finally:
            pass  # Delta Lake has no persistent connections to close

    @classmethod
    @asynccontextmanager
    async def afrom_conn_string(
        cls,
        table_uri: str,
        backend: str = "auto",
        include_finops: bool = False,
    ) -> AsyncIterator[Self]:
        """
        Async context manager.

        Example:
            async with DeltaCheckpointer.afrom_conn_string("main.agents.cp") as cp:
                graph = builder.compile(checkpointer=cp)
        """
        cp = cls(table_uri=table_uri, backend=backend, include_finops=include_finops)
        try:
            yield cp  # type: ignore[misc]
        finally:
            pass

    # ------------------------------------------------------------------
    # Row <-> CheckpointTuple conversion
    # ------------------------------------------------------------------

    def _row_to_tuple(self, row: Row) -> CheckpointTuple:
        """Convert a backend Row dict to a LangGraph CheckpointTuple."""
        checkpoint = deserialize_checkpoint(
            self.serde,
            type_str=row["type"],
            data=row["checkpoint"],
        )
        metadata = deserialize_metadata(self.serde, row["metadata"])

        config = {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
            }
        }

        parent_config = None
        if row.get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }

        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=None,  # put_writes support is v0.2
        )

    def _make_row(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Row:
        """Build a backend Row dict from LangGraph objects."""
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = configurable.get("checkpoint_id")  # previous ID

        type_str, checkpoint_bytes = serialize_checkpoint(self.serde, checkpoint)
        metadata_bytes = serialize_metadata(self.serde, metadata)

        return {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": type_str,
            "checkpoint": checkpoint_bytes,
            "metadata": metadata_bytes,
            "created_at": datetime.datetime.now(tz=datetime.timezone.utc),
        }

    def _make_output_config(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
    ) -> Dict[str, Any]:
        """Build the RunnableConfig that put() returns."""
        return {
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint["id"],
            }
        }

    # ------------------------------------------------------------------
    # Sync interface — required by BaseCheckpointSaver
    # ------------------------------------------------------------------

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Return the CheckpointTuple for the given config, or None.

        If config contains a specific checkpoint_id, returns that exact
        checkpoint. Otherwise returns the most recent checkpoint for the thread.
        """
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")

        if checkpoint_id:
            row = self.backend.get_by_id(thread_id, checkpoint_ns, checkpoint_id)
        else:
            row = self.backend.get_latest(thread_id, checkpoint_ns)

        if row is None:
            return None

        return self._row_to_tuple(row)

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> Dict[str, Any]:
        """
        Persist a checkpoint and return an updated RunnableConfig.

        The returned config has checkpoint_id set to the ID of the
        checkpoint just written, so LangGraph can chain parent references.
        """
        row = self._make_row(config, checkpoint, metadata)
        self.backend.upsert(row)
        return self._make_output_config(config, checkpoint)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Persist intermediate writes for a task.

        Pending writes (used for fault tolerance in multi-step graphs) are
        planned for v0.2. For now this is a no-op, matching the behaviour
        of langgraph-checkpoint-sqlite in its initial release.
        """
        # TODO v0.2: persist to a separate `checkpoint_writes` Delta table
        pass

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        Yield CheckpointTuples for a thread, newest first.

        Args:
            config: Must contain configurable.thread_id.
            filter: Metadata filter dict (not yet implemented — v0.2).
            before: If set, return only checkpoints before this config's
                    checkpoint_id. Used by LangGraph for rewind/branching.
            limit:  Max number of results.
        """
        if config is None:
            return

        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        before_id = None
        if before is not None:
            before_id = before.get("configurable", {}).get("checkpoint_id")

        for row in self.backend.list_checkpoints(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            limit=limit,
            before_checkpoint_id=before_id,
        ):
            yield self._row_to_tuple(row)

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Async variant of get_tuple()."""
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")

        if checkpoint_id:
            row = await self.backend.aget_by_id(thread_id, checkpoint_ns, checkpoint_id)
        else:
            row = await self.backend.aget_latest(thread_id, checkpoint_ns)

        if row is None:
            return None

        return self._row_to_tuple(row)

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> Dict[str, Any]:
        """Async variant of put()."""
        row = self._make_row(config, checkpoint, metadata)
        await self.backend.aupsert(row)
        return self._make_output_config(config, checkpoint)

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async variant of put_writes(). No-op until v0.2."""
        pass

    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async variant of list()."""
        if config is None:
            return

        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        before_id = None
        if before is not None:
            before_id = before.get("configurable", {}).get("checkpoint_id")

        async for row in self.backend.alist_checkpoints(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            limit=limit,
            before_checkpoint_id=before_id,
        ):
            yield self._row_to_tuple(row)