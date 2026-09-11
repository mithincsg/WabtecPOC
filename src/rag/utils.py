"""Caches and the retrieval thread pool.

Both exist for the same reason: on a CPU-only box the model call is the
floor on latency, so nothing else may add to it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retrieval is a handful of independent blocking calls - a ChromaDB HNSW
# query, a BM25 pass, a sentence-transformers forward pass - and each
# releases the GIL inside C or torch, so threads give real overlap.
# Deliberately one flat pool used at a single level: nothing submitted to it
# waits on another task submitted to it, so it cannot deadlock itself.
_MAX_WORKERS = max(2, int(os.getenv("RAG_RETRIEVAL_WORKERS", "6")))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="rag")


def run_parallel(tasks: list[Callable[[], T]]) -> list[T]:
    """Runs zero-argument callables concurrently, returning results in order.

    A single task runs inline. An exception in any task propagates, as it
    would if the calls had been made in sequence.
    """
    if not tasks:
        return []
    if len(tasks) == 1:
        return [tasks[0]()]

    futures = [_EXECUTOR.submit(task) for task in tasks]
    return [future.result() for future in futures]


class LruTtlCache:
    """A small thread-safe LRU with an optional time-to-live.

    Every cache here is read from FastAPI's worker threads, so the lock is
    not optional. `ttl_seconds=None` means entries leave only by eviction.
    """

    def __init__(self, max_entries: int = 256, ttl_seconds: float | None = None):
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self.ttl_seconds is not None and time.monotonic() - stored_at > self.ttl_seconds:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


class EmbeddingCache:
    """An EmbeddingModel that remembers vectors it has already computed.

    BGE-M3 on CPU costs a few hundred milliseconds per text, and this
    pipeline embeds the same text repeatedly: the requirement query once per
    retrieval pass, and the retrieved reference test cases on every
    confidence-scoring run even though they never change.

    Misses within one call are batched into a single `encode`, so a
    partially warm batch still pays for only one forward pass.
    """

    def __init__(self, embedder, max_entries: int = 4096):
        self._embedder = embedder
        self._cache = LruTtlCache(max_entries=max_entries)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        # Deduplicate within the batch as well as against the cache: the same
        # description can legitimately appear twice in one scoring call.
        pending: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                vectors[index] = cached
            else:
                pending.setdefault(text, []).append(index)

        if pending:
            unique = list(pending)
            computed = self._embedder.embed(unique)
            for text, vector in zip(unique, computed):
                self._cache.put(text, vector)
                for index in pending[text]:
                    vectors[index] = vector

        return [v for v in vectors if v is not None]

    def __getattr__(self, name):
        # Keeps `model`, `model_name` and friends reachable, so the wrapper
        # is a drop-in for the embedder it holds.
        return getattr(self._embedder, name)


def stable_key(*parts: Any) -> str:
    """A short, stable cache key for arbitrary JSON-able request parts.
    Hashed rather than concatenated because one part is the full requirement
    text, which can be thousands of characters.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
