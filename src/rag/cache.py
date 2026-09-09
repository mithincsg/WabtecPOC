from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LruTtlCache:
    """A small thread-safe LRU with an optional per-cache time-to-live.

    Every cache in this project is read from FastAPI's worker threads, so the
    lock is not optional. `ttl_seconds=None` means entries only ever leave by
    being evicted as least-recently-used.
    """

    def __init__(self, max_entries: int = 256, ttl_seconds: float | None = None):
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if self.ttl_seconds is not None and time.monotonic() - stored_at > self.ttl_seconds:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get_or_call(self, key: Any, produce: Callable[[], Any]) -> Any:
        """Reads through to `produce` on a miss.

        Deliberately does *not* hold the lock across `produce`: the values
        cached here take seconds to compute, and blocking every other reader
        for that long would cost more than the occasional duplicated
        computation when two requests miss on the same key at once.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = produce()
        self.put(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class EmbeddingCache:
    """An EmbeddingModel that remembers vectors it has already computed.

    BGE-M3 on CPU costs a few hundred milliseconds per text, and this pipeline
    embeds the *same* text repeatedly: the requirement query is embedded once
    per retrieval pass (general, track data, API docs, reference scripts), and
    confidence scoring re-embeds the retrieved reference test cases on every
    request even though they never change. Caching by exact text removes all
    of that without changing a single vector.

    Misses within one call are batched into a single `encode`, so a partially
    warm batch still pays for only one forward pass.
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

        # Every slot is filled by construction; the cast keeps the type honest.
        return [v for v in vectors if v is not None]

    @property
    def wrapped(self):
        return self._embedder

    def __getattr__(self, name):
        # Keeps `model`, `model_name` and friends reachable, so the wrapper is
        # a drop-in for the embedder it holds.
        return getattr(self._embedder, name)


def stable_key(*parts: Any) -> str:
    """A short, stable cache key for arbitrary JSON-able request parts.

    Hashed rather than concatenated because one of the parts is the full
    requirement text, which can be thousands of characters.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
