"""
conftest.py
-----------
Shared pytest fixtures for the test suite.

Key fixture: `fake_backend`
----------------------------
A fully in-memory implementation of BaseDeltaBackend. This lets us test
the checkpointer's logic (serialization, row conversion, config handling)
without any Delta Lake dependency installed.

All other fixtures build on top of `fake_backend`.
"""

from __future__ import annotations

import datetime
import uuid
from typing import AsyncIterator, Dict, Iterator, List, Optional
from unittest.mock import MagicMock

import pytest

from langgraph_checkpoint_delta.backends.base import BaseDeltaBackend, Row
from langgraph_checkpoint_delta.checkpointer import DeltaCheckpointer


# ---------------------------------------------------------------------------
# In-memory fake backend
# ---------------------------------------------------------------------------

class FakeBackend(BaseDeltaBackend):
    """
    Fully in-memory backend for unit testing.

    Stores rows in a plain list. No Delta, no Spark, no pyarrow.
    Ordering and filtering are done in pure Python.
    """

    def __init__(self) -> None:
        self.rows: List[Row] = []
        self.setup_called = False

    def setup(self, include_finops: bool = False) -> None:
        self.setup_called = True

    def upsert(self, row: Row) -> None:
        # Replace existing row if (thread_id, checkpoint_ns, checkpoint_id) matches
        for i, existing in enumerate(self.rows):
            if (
                existing["thread_id"] == row["thread_id"]
                and existing["checkpoint_ns"] == row["checkpoint_ns"]
                and existing["checkpoint_id"] == row["checkpoint_id"]
            ):
                self.rows[i] = row
                return
        self.rows.append(row)

    def get_latest(self, thread_id: str, checkpoint_ns: str) -> Optional[Row]:
        matching = [
            r for r in self.rows
            if r["thread_id"] == thread_id and r["checkpoint_ns"] == checkpoint_ns
        ]
        if not matching:
            return None
        return sorted(matching, key=lambda r: r["created_at"], reverse=True)[0]

    def get_by_id(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> Optional[Row]:
        for row in self.rows:
            if (
                row["thread_id"] == thread_id
                and row["checkpoint_ns"] == checkpoint_ns
                and row["checkpoint_id"] == checkpoint_id
            ):
                return row
        return None

    def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        limit: Optional[int] = None,
        before_checkpoint_id: Optional[str] = None,
    ) -> Iterator[Row]:
        matching = [
            r for r in self.rows
            if r["thread_id"] == thread_id and r["checkpoint_ns"] == checkpoint_ns
        ]
        sorted_rows = sorted(matching, key=lambda r: r["created_at"], reverse=True)

        if before_checkpoint_id is not None:
            ref = next(
                (r for r in sorted_rows if r["checkpoint_id"] == before_checkpoint_id),
                None,
            )
            if ref:
                sorted_rows = [
                    r for r in sorted_rows if r["created_at"] < ref["created_at"]
                ]

        if limit is not None:
            sorted_rows = sorted_rows[:limit]

        yield from sorted_rows


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def checkpointer(fake_backend: FakeBackend) -> DeltaCheckpointer:
    """
    A DeltaCheckpointer wired to the in-memory FakeBackend.

    We bypass __init__ to avoid triggering env_detect and make_backend,
    then manually inject the fake backend.
    """
    cp = object.__new__(DeltaCheckpointer)
    # Initialise the BaseCheckpointSaver parts
    from langgraph_checkpoint_delta.serialization import get_default_serializer
    BaseCheckpointSaver = DeltaCheckpointer.__bases__[0]
    BaseCheckpointSaver.__init__(cp, serde=get_default_serializer())
    cp.backend = fake_backend
    cp._include_finops = False
    return cp


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_config(
    thread_id: str = "thread-001",
    checkpoint_ns: str = "",
    checkpoint_id: Optional[str] = None,
) -> Dict:
    cfg: Dict = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


def make_checkpoint(checkpoint_id: Optional[str] = None) -> Dict:
    """Return a minimal LangGraph-compatible Checkpoint dict."""
    return {
        "v": 1,
        "id": checkpoint_id or str(uuid.uuid4()),
        "ts": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def make_metadata() -> Dict:
    """Return a minimal CheckpointMetadata dict."""
    return {"source": "input", "step": 0, "writes": {}, "parents": {}}