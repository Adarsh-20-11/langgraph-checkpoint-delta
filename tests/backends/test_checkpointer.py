"""
test_checkpointer.py
--------------------
Unit tests for DeltaCheckpointer using the in-memory FakeBackend.

These tests cover the checkpointer's logic layer — config handling,
serialization round-trips, parent linkage, and list filtering.
They do NOT require Delta Lake, Spark, or pyarrow to be installed.
"""

from __future__ import annotations

import pytest

from conftest import make_checkpoint, make_config, make_metadata


class TestGetTuple:
    def test_returns_none_for_empty_thread(self, checkpointer):
        config = make_config(thread_id="no-such-thread")
        assert checkpointer.get_tuple(config) is None

    def test_returns_latest_when_no_checkpoint_id(self, checkpointer):
        cp1 = make_checkpoint()
        cp2 = make_checkpoint()
        meta = make_metadata()

        cfg1 = make_config(thread_id="t1")
        out1 = checkpointer.put(cfg1, cp1, meta, {})

        cfg2 = make_config(thread_id="t1", checkpoint_id=out1["configurable"]["checkpoint_id"])
        checkpointer.put(cfg2, cp2, meta, {})

        result = checkpointer.get_tuple(make_config(thread_id="t1"))
        assert result is not None
        assert result.checkpoint["id"] == cp2["id"]

    def test_returns_specific_checkpoint_by_id(self, checkpointer):
        cp1 = make_checkpoint()
        meta = make_metadata()
        out = checkpointer.put(make_config(thread_id="t2"), cp1, meta, {})

        result = checkpointer.get_tuple(
            make_config(thread_id="t2", checkpoint_id=out["configurable"]["checkpoint_id"])
        )
        assert result is not None
        assert result.checkpoint["id"] == cp1["id"]

    def test_returns_none_for_missing_checkpoint_id(self, checkpointer):
        result = checkpointer.get_tuple(
            make_config(thread_id="t3", checkpoint_id="does-not-exist")
        )
        assert result is None


class TestPut:
    def test_put_returns_config_with_checkpoint_id(self, checkpointer):
        cp = make_checkpoint()
        out = checkpointer.put(make_config(thread_id="t4"), cp, make_metadata(), {})
        assert "checkpoint_id" in out["configurable"]
        assert out["configurable"]["checkpoint_id"] == cp["id"]

    def test_put_idempotent_on_same_checkpoint_id(self, checkpointer, fake_backend):
        cp = make_checkpoint()
        meta = make_metadata()
        cfg = make_config(thread_id="t5")

        checkpointer.put(cfg, cp, meta, {})
        checkpointer.put(cfg, cp, meta, {})  # second put — same checkpoint

        # Should have exactly one row, not two
        assert len(fake_backend.rows) == 1

    def test_parent_checkpoint_id_linked(self, checkpointer, fake_backend):
        cp1 = make_checkpoint()
        meta = make_metadata()
        out1 = checkpointer.put(make_config(thread_id="t6"), cp1, meta, {})

        cp2 = make_checkpoint()
        checkpointer.put(
            make_config(thread_id="t6", checkpoint_id=out1["configurable"]["checkpoint_id"]),
            cp2,
            meta,
            {},
        )

        # The second row should have parent_checkpoint_id pointing to cp1
        rows = sorted(fake_backend.rows, key=lambda r: r["created_at"])
        assert rows[1]["parent_checkpoint_id"] == cp1["id"]


class TestList:
    def test_list_returns_newest_first(self, checkpointer):
        meta = make_metadata()
        cfg = make_config(thread_id="t7")
        cp1 = make_checkpoint()
        out1 = checkpointer.put(cfg, cp1, meta, {})

        cp2 = make_checkpoint()
        checkpointer.put(
            make_config(thread_id="t7", checkpoint_id=out1["configurable"]["checkpoint_id"]),
            cp2,
            meta,
            {},
        )

        results = list(checkpointer.list(make_config(thread_id="t7")))
        assert len(results) == 2
        assert results[0].checkpoint["id"] == cp2["id"]
        assert results[1].checkpoint["id"] == cp1["id"]

    def test_list_respects_limit(self, checkpointer):
        meta = make_metadata()
        cfg = make_config(thread_id="t8")
        prev_out = checkpointer.put(cfg, make_checkpoint(), meta, {})
        for _ in range(4):
            prev_out = checkpointer.put(
                make_config(thread_id="t8", checkpoint_id=prev_out["configurable"]["checkpoint_id"]),
                make_checkpoint(),
                meta,
                {},
            )

        results = list(checkpointer.list(make_config(thread_id="t8"), limit=2))
        assert len(results) == 2

    def test_list_empty_for_unknown_thread(self, checkpointer):
        results = list(checkpointer.list(make_config(thread_id="no-thread")))
        assert results == []


class TestAsyncInterface:
    @pytest.mark.asyncio
    async def test_aget_tuple_matches_sync(self, checkpointer):
        cp = make_checkpoint()
        checkpointer.put(make_config(thread_id="at1"), cp, make_metadata(), {})

        sync_result = checkpointer.get_tuple(make_config(thread_id="at1"))
        async_result = await checkpointer.aget_tuple(make_config(thread_id="at1"))

        assert sync_result is not None
        assert async_result is not None
        assert sync_result.checkpoint["id"] == async_result.checkpoint["id"]

    @pytest.mark.asyncio
    async def test_aput_persists(self, checkpointer):
        cp = make_checkpoint()
        await checkpointer.aput(make_config(thread_id="at2"), cp, make_metadata(), {})
        result = await checkpointer.aget_tuple(make_config(thread_id="at2"))
        assert result is not None
        assert result.checkpoint["id"] == cp["id"]