from __future__ import annotations

import asyncio
import logging
from typing import Iterator

logger = logging.getLogger(__name__)


class CollectionInitTracker:
    """
    A ``set``-compatible object that tracks initialised vector-DB collection
    names and guards concurrent initialisation with per-collection locks.

    Drop-in API surface (supports ``in``, ``add``, ``discard``, iteration)
    so existing callers can be updated incrementally.
    """

    def __init__(self) -> None:
        self._ready: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        # Master lock used only to create per-collection locks atomically.
        self._meta_lock = asyncio.Lock()

    #  set-compatible interface 

    def __contains__(self, name: object) -> bool:
        return name in self._ready

    def __iter__(self) -> Iterator[str]:
        return iter(self._ready)

    def __len__(self) -> int:
        return len(self._ready)

    def __repr__(self) -> str:
        return f"CollectionInitTracker({self._ready!r})"

    def add(self, name: str) -> None:
        """Mark *name* as initialised (idempotent)."""
        self._ready.add(name)

    def discard(self, name: str) -> None:
        """Remove *name* from the ready set (e.g. after a reset)."""
        self._ready.discard(name)
        self._locks.pop(name, None)

    #  async helpers ─

    async def get_lock(self, name: str) -> asyncio.Lock:
        """
        Return the per-collection lock, creating it atomically if absent.

        Usage::

            async with await tracker.get_lock(col_name):
                if col_name in tracker:
                    return          # already done by another coroutine
                # ... perform init ...
                tracker.add(col_name)
        """
        # Fast path — lock already exists.
        lock = self._locks.get(name)
        if lock is not None:
            return lock

        # Slow path — create under the master lock to avoid two coroutines
        # both deciding to create the same per-collection lock simultaneously.
        async with self._meta_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]



"""
utils/collection_tracker.py

Thread/coroutine-safe replacement for the bare ``set`` that tracked which
vector-DB collections have been initialised in the current worker process.

Why a dedicated class instead of ``set``?

- A plain ``set`` has no built-in locking.  Two coroutines that arrive
  simultaneously and both find the collection absent can both call
  ``create_collection``, causing redundant work and potential races.
- ``CollectionInitTracker`` pairs each collection name with an
  ``asyncio.Lock`` so that only one coroutine initialises it; all others
  await the lock and then find it already present.

Multi-worker note
─
Each Uvicorn worker process gets its own ``CollectionInitTracker`` instance
(they do NOT share memory).  The idempotent ``vectordb_client.is_collection_existed``
check that every caller performs before writing means cross-worker races are
harmless — the second worker's call simply finds the collection already there.
"""