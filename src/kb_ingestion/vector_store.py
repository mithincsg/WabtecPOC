from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import chromadb

_UPSERT_BATCH_SIZE = 100

# `count()` is a ChromaDB round trip, and the retriever calls it on every
# query (to clamp n_results) as does the keyword index (to detect staleness).
# The collection only changes during an ingestion run, which is a separate
# process, so a short-lived cached count is accurate for a serving process
# and removes several round trips per request.
_COUNT_TTL_SECONDS = 5.0


class VectorStore:
    """A single persistent ChromaDB collection holding every document_type,
    with document_type / subdivision / requirement_id as filterable
    metadata. One collection rather than several means the retriever can
    search across categories in one query plus a `where` filter, instead of
    fanning out and merging.
    """

    def __init__(self, persist_dir: Path, collection_name: str):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self._collection = self._get_or_create()
        self._count_lock = threading.Lock()
        self._count: int | None = None
        self._counted_at = 0.0

    def _get_or_create(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = self._get_or_create()
        self._invalidate_count()

    def count(self) -> int:
        with self._count_lock:
            fresh = self._count is not None and (
                time.monotonic() - self._counted_at < _COUNT_TTL_SECONDS
            )
            if fresh:
                return self._count
        # Counted outside the lock: the round trip should not serialise
        # concurrent readers, and a duplicated count is harmless.
        count = self._collection.count()
        with self._count_lock:
            self._count = count
            self._counted_at = time.monotonic()
        return count

    def _invalidate_count(self) -> None:
        with self._count_lock:
            self._count = None
            self._counted_at = 0.0

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for start in range(0, len(ids), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        self._invalidate_count()

    def get_file_state(self, source_path: str) -> tuple[str, str | None] | None:
        """(file_hash, pipeline_fingerprint) recorded on this file's existing
        chunks, or None if it has none. Every chunk from one extract/chunk
        pass carries the same pair, so the first is representative — this is
        what lets an unchanged file skip re-embedding with no side-file of
        state to keep in sync.
        """
        existing = self._collection.get(
            where={"source_path": source_path}, include=["metadatas"], limit=1
        )
        metadatas = existing.get("metadatas") or []
        if not metadatas:
            return None
        metadata = metadatas[0]
        file_hash = metadata.get("file_hash")
        if file_hash is None:
            return None
        return file_hash, metadata.get("pipeline_fingerprint")

    def delete_by_source_path(self, source_path: str) -> int:
        existing = self._collection.get(where={"source_path": source_path}, include=[])
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self._invalidate_count()
        return len(ids)

    def get_all_source_paths(self) -> set[str]:
        result = self._collection.get(include=["metadatas"])
        return {m["source_path"] for m in result["metadatas"] if "source_path" in m}

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Every chunk's (id, text, metadata). Used to build the BM25 keyword
        index, which — unlike the dense index — has to be constructed in
        process from the full corpus.
        """
        result = self._collection.get(include=["documents", "metadatas"])
        return (
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # n_results above the collection size makes Chroma warn and clamp;
        # clamping here keeps the logs clean on a small KB.
        n_results = max(1, min(n_results, self.count() or 1))
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
