"""
serialization.py
----------------
Handles binary serialization and deserialization of LangGraph checkpoint
objects and their metadata.

Design notes
------------
- We delegate to LangGraph's own serializer registry rather than rolling
  our own. This ensures we stay compatible with whatever serialization
  LangGraph uses internally (currently msgpack-based).
- The serializer is injected into backends, not hardcoded, so it can be
  swapped in tests or if LangGraph changes its serialization strategy.
- Both `checkpoint` (the state blob) and `metadata` (LangGraph's internal
  bookkeeping) are serialized separately. This mirrors how the official
  LangGraph checkpointers (postgres, sqlite) handle it.
"""

from __future__ import annotations

from typing import Any, Tuple

from langgraph.checkpoint.base import (  # type: ignore[import]
    Checkpoint,
    CheckpointMetadata,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # type: ignore[import]


def get_default_serializer() -> SerializerProtocol:
    """
    Returns the default serializer.

    We use LangGraph's JsonPlusSerializer as the default — it handles
    Python objects that vanilla JSON can't (datetimes, UUIDs, etc.) and
    is the same serializer used by the official LangGraph checkpointers.

    If LangGraph ships a faster msgpack serializer in future versions,
    this is the one place to swap it in.
    """
    return JsonPlusSerializer()


def serialize_checkpoint(
    serde: SerializerProtocol,
    checkpoint: Checkpoint,
) -> Tuple[str, bytes]:
    """
    Serialize a LangGraph Checkpoint to (type_str, bytes).

    Args:
        serde:      The serializer to use.
        checkpoint: The LangGraph Checkpoint object.

    Returns:
        A (type, data) tuple where `type` is stored in the `type` column
        and `data` is stored in the `checkpoint` binary column.
    """
    type_str, bytes_data = serde.dumps_typed(checkpoint)
    return type_str, bytes_data


def deserialize_checkpoint(
    serde: SerializerProtocol,
    type_str: str,
    data: bytes,
) -> Checkpoint:
    """
    Deserialize a checkpoint blob back to a LangGraph Checkpoint.

    Args:
        serde:    The serializer to use.
        type_str: The type string stored in the `type` column.
        data:     The raw bytes from the `checkpoint` column.

    Returns:
        A LangGraph Checkpoint object.
    """
    return serde.loads_typed((type_str, data))


def serialize_metadata(
    serde: SerializerProtocol,
    metadata: CheckpointMetadata,
) -> bytes:
    """
    Serialize checkpoint metadata to bytes.

    Metadata is always serialized without a type prefix — we always use
    the same serializer for metadata, so the type is implicit.

    Args:
        serde:    The serializer to use.
        metadata: The LangGraph CheckpointMetadata object.

    Returns:
        Raw bytes to store in the `metadata` binary column.
    """
    return serde.dumps(metadata)


def deserialize_metadata(
    serde: SerializerProtocol,
    data: bytes,
) -> CheckpointMetadata:
    """
    Deserialize metadata bytes back to a CheckpointMetadata object.

    Args:
        serde: The serializer to use.
        data:  The raw bytes from the `metadata` column.

    Returns:
        A LangGraph CheckpointMetadata object.
    """
    return serde.loads(data)