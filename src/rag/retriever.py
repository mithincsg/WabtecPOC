from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from kb_ingestion.embeddings import EmbeddingModel
from kb_ingestion.vector_store import VectorStore

from .cache import LruTtlCache
from .concurrency import run_parallel
from .config import RetrievalConfig
from .keyword_index import KeywordHit, KeywordIndex
from .track_mapping import TrackMapping, TrackSelection

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    # Cosine similarity from the dense arm, or None when only the keyword arm
    # matched this chunk. Kept separate from `score` so confidence scoring can
    # reason about semantic closeness specifically.
    similarity: float | None = None
    bm25_score: float | None = None
    # Fused rank score; ordering key for the final list.
    score: float = 0.0
    matched_by: list[str] = field(default_factory=list)

    @property
    def source_label(self) -> str:
        """Short citation, e.g. "WEG-DDG-11078.pdf p.12 > 3.1 Record Header"
        or "L2R9479_A.xlsx > L2R9479_3".
        """
        parts = [
            str(
                self.metadata.get("source_file")
                or self.metadata.get("source_path")
                or "unknown source"
            )
        ]
        if self.metadata.get("page"):
            parts.append(f"p.{self.metadata['page']}")
        if self.metadata.get("section_path"):
            parts.append(str(self.metadata["section_path"]))
        if self.metadata.get("test_case_id"):
            parts.append(str(self.metadata["test_case_id"]))
        elif self.metadata.get("method_name"):
            parts.append(
                ".".join(
                    p
                    for p in (
                        self.metadata.get("class_name"),
                        self.metadata.get("method_name"),
                    )
                    if p
                )
            )
        elif self.metadata.get("table_title"):
            parts.append(str(self.metadata["table_title"]))
        return " > ".join(parts)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    context: str
    track_selection: TrackSelection
    dense_hits: int
    keyword_hits: int


@dataclass(frozen=True)
class _Pass:
    """One filtered search: the general pass, or the track-data pass."""

    top_k: int
    doc_types: list[str] | None
    subdivisions: list[str] | None = None
    exclude_doc_type: str | None = None


class HybridRetriever:
    """Requirement text -> two independent rankings -> one fused list.

    The dense arm (BGE-M3 cosine over ChromaDB) catches paraphrase: a
    requirement phrased as "most restrictive of" finds a test case written as
    "lowest applicable speed". The keyword arm (BM25) catches the exact
    identifiers dense retrieval blurs together — TBC137 vs TBC139, block 1015
    vs 1025, an API name like wcr_loco_sim.set_position. Test-case generation
    needs both: the wrong parameter number in a test case makes it useless.

    The two rankings are combined with reciprocal rank fusion, which needs no
    score calibration between arms — only their orderings — so the weights
    stay meaningful even though cosine similarity and BM25 scores are on
    completely different scales.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        config: RetrievalConfig,
        track_mapping: TrackMapping,
        keyword_index: KeywordIndex | None = None,
        cache_entries: int = 32,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config
        self.track_mapping = track_mapping
        self.keyword_index = keyword_index or KeywordIndex(vector_store)
        # Retrieval is deterministic for a given query and filter set, so the
        # repeat passes inside one request (and a re-run of the same
        # requirement) cost nothing after the first.
        self._results = LruTtlCache(max_entries=cache_entries)

    # --- public API -------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        *,
        requirement_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        if not query_text.strip():
            return RetrievalResult([], "", TrackSelection((), False), 0, 0)

        top_k = top_k if top_k is not None else self.config.top_k
        selection = self.track_mapping.selection_for(requirement_id)

        cache_key = (
            query_text,
            top_k,
            tuple(doc_types) if doc_types else None,
            tuple(selection.subdivisions),
            selection.search_all,
            selection.enabled,
        )
        cached = self._results.get(cache_key)
        if cached is not None:
            return cached

        passes = [
            _Pass(
                top_k=top_k,
                doc_types=self._general_doc_types(doc_types),
                exclude_doc_type=self.config.track_data_doc_type,
            )
        ]
        # Track data is retrieved as its own pass rather than being mixed into
        # the general one: its chunks are dense tables of track features that
        # would otherwise crowd out prose context on score alone, and it needs
        # the subdivision filter that only applies to it.
        if selection.enabled and self.config.track_data_top_k > 0:
            passes.append(
                _Pass(
                    top_k=self.config.track_data_top_k,
                    doc_types=[self.config.track_data_doc_type],
                    subdivisions=(
                        None if selection.search_all else list(selection.subdivisions)
                    ),
                )
            )

        chunks = [c for group in self._search_all(query_text, passes) for c in group]
        result = RetrievalResult(
            chunks=chunks,
            context=self.build_context(chunks),
            track_selection=selection,
            dense_hits=sum(1 for c in chunks if "dense" in c.matched_by),
            keyword_hits=sum(1 for c in chunks if "keyword" in c.matched_by),
        )
        self._results.put(cache_key, result)
        return result

    def build_context(self, chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
        """One citation-tagged context block, truncated to a character budget
        so the prompt stays inside the model's context window no matter how
        large the retrieved chunks are.
        """
        budget = max_chars if max_chars is not None else self.config.max_context_chars

        blocks: list[str] = []
        used = 0
        for index, chunk in enumerate(chunks, start=1):
            header = f"[{index}] {chunk.metadata.get('document_type', 'kb')} | {chunk.source_label}"
            block = f"{header}\n{chunk.text.strip()}"
            if blocks and used + len(block) > budget:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    def chunks_by_doc_type(
        self, chunks: list[RetrievedChunk], *doc_types: str
    ) -> list[RetrievedChunk]:
        wanted = set(doc_types)
        return [c for c in chunks if c.metadata.get("document_type") in wanted]

    # --- internals --------------------------------------------------------

    def _general_doc_types(self, override: list[str] | None) -> list[str] | None:
        requested = override if override else self.config.doc_types
        return list(requested) if requested else None

    def _search_all(
        self, query_text: str, passes: list[_Pass]
    ) -> list[list[RetrievedChunk]]:
        """Runs every arm of every pass concurrently, then fuses per pass.

        The two arms of a pass are independent, and so are the passes, so all
        of them are submitted as one flat batch rather than nested: a dense
        query and a BM25 scan overlap almost perfectly, one waiting on HNSW
        inside C and the other on NumPy. Fusion still happens per pass, so
        track-data results remain their own appended list exactly as before.
        """
        tasks: list = []
        for pass_ in passes:
            depth = max(pass_.top_k, self.config.candidates_per_arm)
            where = self._build_where(
                pass_.doc_types, pass_.subdivisions, pass_.exclude_doc_type
            )
            predicate = self._build_predicate(
                pass_.doc_types, pass_.subdivisions, pass_.exclude_doc_type
            )
            tasks.append(lambda d=depth, w=where: self._dense_search(query_text, d, w))
            tasks.append(
                lambda d=depth, f=predicate: self._keyword_search(query_text, d, f)
            )

        # Embed the query once, up front. Both dense arms want the same
        # vector, and the embedder is cached, so doing it here turns the
        # second arm's embedding into a dictionary lookup instead of a second
        # forward pass over identical text.
        if self.config.dense_weight > 0:
            self.embedder.embed([query_text])

        arms = run_parallel(tasks)
        return [
            self._fuse(arms[2 * i], arms[2 * i + 1], pass_.top_k)
            for i, pass_ in enumerate(passes)
        ]

    def _keyword_search(
        self, query_text: str, depth: int, predicate
    ) -> list[KeywordHit]:
        if self.config.keyword_weight <= 0:
            return []
        return self.keyword_index.search(query_text, depth, predicate)

    def _dense_search(
        self, query_text: str, depth: int, where: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        if self.config.dense_weight <= 0:
            return []

        query_embedding = self.embedder.embed([query_text])[0]
        result = self.vector_store.query(query_embedding, n_results=depth, where=where)

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - distance  # collection uses hnsw:space cosine
            if similarity < self.config.min_similarity:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata or {},
                    similarity=similarity,
                    matched_by=["dense"],
                )
            )
        return chunks

    def _fuse(
        self,
        dense: list[RetrievedChunk],
        keyword: list,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Weighted reciprocal rank fusion: each arm contributes
        weight / (rrf_k + rank) for the chunks it ranked. Rank-based rather
        than score-based, so a BM25 score of 14.2 and a cosine similarity of
        0.71 can be combined without pretending they're comparable numbers.
        """
        merged: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense, start=1):
            merged[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = self.config.dense_weight / (self.config.rrf_k + rank)

        for rank, hit in enumerate(keyword, start=1):
            contribution = self.config.keyword_weight / (self.config.rrf_k + rank)
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    metadata=hit.metadata,
                    bm25_score=hit.score,
                    matched_by=["keyword"],
                )
            else:
                existing.bm25_score = hit.score
                existing.matched_by.append("keyword")
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + contribution

        for chunk_id, chunk in merged.items():
            chunk.score = scores[chunk_id]

        ordered = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        return ordered[:top_k]

    # --- filters ----------------------------------------------------------
    # The dense arm filters inside Chroma with a `where` clause; the keyword
    # arm filters in Python. Both are built here from the same inputs so the
    # two arms can never end up searching different subsets.

    @staticmethod
    def _build_where(
        doc_types: list[str] | None,
        subdivisions: list[str] | None,
        exclude_doc_type: str | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if doc_types:
            clauses.append({"document_type": {"$in": list(doc_types)}})
        elif exclude_doc_type:
            clauses.append({"document_type": {"$ne": exclude_doc_type}})
        if subdivisions:
            clauses.append({"subdivision": {"$in": list(subdivisions)}})

        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _build_predicate(
        doc_types: list[str] | None,
        subdivisions: list[str] | None,
        exclude_doc_type: str | None,
    ):
        allowed = set(doc_types) if doc_types else None
        allowed_subdivisions = set(subdivisions) if subdivisions else None

        def predicate(metadata: dict[str, Any]) -> bool:
            document_type = metadata.get("document_type")
            if allowed is not None:
                if document_type not in allowed:
                    return False
            elif exclude_doc_type and document_type == exclude_doc_type:
                return False
            if allowed_subdivisions is not None:
                if metadata.get("subdivision") not in allowed_subdivisions:
                    return False
            return True

        return predicate
