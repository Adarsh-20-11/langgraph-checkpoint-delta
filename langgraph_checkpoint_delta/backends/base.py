"""
backends/base.py
----------------
Abstract interface that all Delta backend implementations must satisfy.

Why a separate backend abstraction?
------------------------------------
The checkpointer (checkpointer.py) knows nothing about Spark or
python-deltalake. It only talks to a BaseDeltaBackend. This means:

1. Tests can inject a FakeBackend without any Delta dependency installed.
2. Adding a third backend (e.g. a REST-based Delta Sharing backend) is
   just a new subclass — the checkpointer never changes.
3. The Spark and Deltalake backends can be tested independently from
   the LangGraph checkpoint logic.

Backend responsibilities
------------------------
A backend is responsible ONLY for reading and writing rows to a Delta table.
It knows nothing about LangGraph's CheckpointTuple, serialization, or
checkpoint IDs. That logic lives in the checkpointer.

Row format
----------
Backends exchange data as plain dicts matching the schema defined in
schema.py. All binary fields are raw `bytes`. Timestamps are
`datetime` objects (UTC, timezone-aware).
"""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional


# A row is just a plain dict — keys match schema.py column names.
Row = Dict[str, Any]


class BaseDeltaBackend(abc.ABC):
    """
    Abstract base class for Delta Lake backend implementations.

    Subclasses must implement all abstract methods in both sync and
    async variants. The async variants default to running the sync
    implementation in a thread pool executor — subclasses MAY override
    them with true async I/O if the underlying library supports it.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def setup(self, include_finops: bool = False) -> None:
        """
        Create the Delta table if it does not already exist.

        This is idempotent — calling it on an existing table is a no-op.
        Called once during DeltaCheckpointer initialisation.

        Args:
            include_finops: If True, add the FinOps cost-tracking columns.
        """

    async def asetup(self, include_finops: bool = False) -> None:
        """Async variant of setup(). Defaults to sync implementation."""
        import asyncio
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.setup(include_finops)
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def upsert(self, row: Row) -> None:
        """
        Write or overwrite a single checkpoint row.

        The row dict must contain all non-nullable columns defined in
        schema.py. The backend must perform an UPSERT (not a blind append)
        keyed on (thread_id, checkpoint_ns, checkpoint_id) to preserve
        Delta's ACID guarantees across retries.

        Args:
            row: A dict matching the checkpoint schema.
        """

    async def aupsert(self, row: Row) -> None:
        """Async variant of upsert(). Defaults to sync in executor."""
        import asyncio
        await asyncio.get_event_loop().run_in_executor(None, lambda: self.upsert(row))

    # ------------------------------------------------------------------
    # Read — single row
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_latest(
        self,
        thread_id: str,
        checkpoint_ns: str,
    ) -> Optional[Row]:
        """
        Return the most recent checkpoint row for a given thread.

        "Most recent" is defined as the checkpoint with the latest
        `created_at` timestamp within the given (thread_id, checkpoint_ns).

        Args:
            thread_id:     The thread to query.
            checkpoint_ns: The checkpoint namespace.

        Returns:
            A row dict, or None if no checkpoints exist for this thread.
        """

    async def aget_latest(
        self,
        thread_id: str,
        checkpoint_ns: str,
    ) -> Optional[Row]:
        """Async variant of get_latest()."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.get_latest(thread_id, checkpoint_ns)
        )

    @abc.abstractmethod
    def get_by_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> Optional[Row]:
        """
        Return a specific checkpoint row by its exact ID.

        Args:
            thread_id:      The thread to query.
            checkpoint_ns:  The checkpoint namespace.
            checkpoint_id:  The exact checkpoint UUID.

        Returns:
            A row dict, or None if not found.
        """

    async def aget_by_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> Optional[Row]:
        """Async variant of get_by_id()."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.get_by_id(thread_id, checkpoint_ns, checkpoint_id),
        )

    # ------------------------------------------------------------------
    # Read — list
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        limit: Optional[int] = None,
        before_checkpoint_id: Optional[str] = None,
    ) -> Iterator[Row]:
        """
        Yield checkpoint rows for a thread, newest first.

        Args:
            thread_id:             The thread to query.
            checkpoint_ns:         The checkpoint namespace.
            limit:                 Max rows to return. None = no limit.
            before_checkpoint_id:  If set, return only checkpoints created
                                   before this checkpoint_id (exclusive).
                                   Used by LangGraph's list() for pagination.

        Yields:
            Row dicts, ordered by created_at DESC.
        """

    async def alist_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        limit: Optional[int] = None,
        before_checkpoint_id: Optional[str] = None,
    ) -> AsyncIterator[Row]:
        """
        Async variant of list_checkpoints().

        Default implementation collects the sync iterator and yields
        results. Subclasses with true async I/O should override this.
        """
        import asyncio

        rows: List[Row] = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(
                self.list_checkpoints(
                    thread_id, checkpoint_ns, limit, before_checkpoint_id
                )
            ),
        )
        for row in rows:
            yield row