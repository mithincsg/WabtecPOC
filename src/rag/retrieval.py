"""Everything that finds context for a requirement.

Three things live here, in this order:

- `KeywordIndex`, the BM25 arm, used over both the embedded collection and
  the static (never-embedded) sources.
- `HybridRetriever`, dense + keyword fused with reciprocal rank fusion over
  the ChromaDB knowledge base.
- `StaticContextProvider`, an in-process BM25-only index over
  data/python_apis and data/track_data plus the data/Examples templates.

python_apis and track_data are not embedded because they are one giant stub
file and thousands of near-identical per-subdivision rows - not prose an
embedding model gains from - and both are full of exact identifiers BM25
already handles better than dense search.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from kb_ingestion.config import PipelineConfig
from kb_ingestion.pipeline import IngestionPipeline, discover_files, extract_and_normalize
from kb_ingestion.storage import EmbeddingModel, VectorStore

from .config import RetrievalConfig
from .utils import LruTtlCache, run_parallel

logger = logging.getLogger(__name__)

# Tokens worth matching on here are not all plain words: "TBC137", "L2R9479",
# "08880", "wcr_loco_sim", "IV132.0". Splitting on non-alphanumerics but
# keeping digits attached to their letters preserves the identifiers keyword
# search exists to catch and that a dense embedding smears across neighbours.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\d+[A-Za-z]*)*|\d+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# --- BM25 ------------------------------------------------------------------


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class _Corpus:
    """One built index, and everything scored against it.

    A single immutable object rather than parallel attributes on the index:
    searches run on several threads, so a searcher takes one reference and a
    rebuild landing mid-search swaps the whole corpus instead of leaving
    that searcher reading new ids against old texts. The memoised scores
    live here too, so they are discarded with the corpus they describe.
    """

    bm25: BM25Okapi
    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    count: int
    scores: LruTtlCache = field(default_factory=LruTtlCache)


class KeywordIndex:
    """BM25 over every chunk in a store - the lexical half of hybrid search.

    Chroma stores dense vectors but no inverted index, so this is built in
    process and rebuilt when the chunk count changes; on a corpus this size
    the build is well under a second, so there is no persistence to sync.

    Scoring is the expensive part: `BM25Okapi.get_scores` walks every chunk
    in Python, and one request scores the same query several times over
    (general and track passes, then API and script passes) differing only in
    the metadata filter applied afterwards. Scores are therefore memoised
    per token set.
    """

    def __init__(self, vector_store, score_cache_entries: int = 64):
        self._vector_store = vector_store
        self._lock = threading.Lock()
        self._score_cache_entries = score_cache_entries
        self._corpus: _Corpus | None = None

    def _ensure_built(self) -> _Corpus | None:
        # count() is checked on every search: the store caches it behind a
        # short TTL and invalidates on write, so this is a dict lookup, and
        # an ingestion is still noticed as soon as it lands.
        count = self._vector_store.count()
        corpus = self._corpus
        if corpus is not None and corpus.count == count:
            return corpus

        with self._lock:
            # Re-check inside the lock: a concurrent request may have built
            # it while this one waited.
            corpus = self._corpus
            if corpus is not None and corpus.count == count:
                return corpus

            ids, texts, metadatas = self._vector_store.get_all_documents()
            tokenized = [tokenize(t) for t in texts]
            if not tokenized:
                # BM25Okapi divides by the average document length, so an
                # empty corpus is a ZeroDivisionError, not an empty result.
                self._corpus = None
                return None

            self._corpus = _Corpus(
                bm25=BM25Okapi(tokenized),
                chunk_ids=ids,
                texts=texts,
                metadatas=metadatas,
                count=count,
                scores=LruTtlCache(max_entries=self._score_cache_entries),
            )
            logger.info("Built BM25 keyword index over %d chunks", len(tokenized))
            return self._corpus

    def search(self, query: str, top_k: int, predicate=None) -> list[KeywordHit]:
        """Top BM25 matches for the query.

        `predicate(metadata) -> bool` filters candidates the way the dense
        arm's `where` clause does. It is applied in Python because BM25
        scores the whole corpus in one pass regardless - and because the
        score memo is keyed on the query alone, precisely so differently
        filtered passes share one scoring run.
        """
        corpus = self._ensure_built()
        if corpus is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = _scores_for(corpus, tokens)

        hits: list[KeywordHit] = []
        for index in _ranked_indices(scores):
            metadata = corpus.metadatas[index] or {}
            if predicate is not None and not predicate(metadata):
                continue
            hits.append(
                KeywordHit(
                    chunk_id=corpus.chunk_ids[index],
                    text=corpus.texts[index],
                    metadata=metadata,
                    score=float(scores[index]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def _scores_for(corpus: _Corpus, tokens: list[str]) -> np.ndarray:
    key = tuple(tokens)
    scores = corpus.scores.get(key)
    if scores is None:
        scores = np.asarray(corpus.bm25.get_scores(tokens), dtype=np.float64)
        corpus.scores.put(key, scores)
    return scores


def _ranked_indices(scores: np.ndarray) -> list[int]:
    """Corpus indices with a non-zero score, best first. A zero means the
    chunk shares no token with the query, so it can never be a hit however
    the caller's predicate filters.
    """
    positive = np.flatnonzero(scores > 0)
    if positive.size == 0:
        return []
    order = np.argsort(scores[positive])[::-1]
    return positive[order].tolist()


# --- Hybrid retrieval over the knowledge base ------------------------------


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    # Cosine from the dense arm, or None when only the keyword arm matched.
    # Kept separate from `score` so confidence scoring can reason about
    # semantic closeness specifically.
    similarity: float | None = None
    bm25_score: float | None = None
    # Fused rank score; the ordering key for the final list.
    score: float = 0.0
    matched_by: list[str] = field(default_factory=list)

    @property
    def source_label(self) -> str:
        """Short citation, e.g. "WEG-DDG-11078.pdf p.12 > 3.1 Record Header"."""
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
    track_subdivisions: list[str]
    dense_hits: int
    keyword_hits: int


@dataclass(frozen=True)
class _Pass:
    """One filtered search: the general pass, or the track-data pass."""

    top_k: int
    doc_types: list[str] | None
    exclude_doc_type: str | None = None


class HybridRetriever:
    """Requirement text -> two independent rankings -> one fused list.

    The dense arm (BGE-M3 cosine) catches paraphrase: "most restrictive of"
    finds "lowest applicable speed". The keyword arm (BM25) catches the exact
    identifiers dense retrieval blurs - TBC137 vs TBC139, block 1015 vs 1025.
    Test-case generation needs both: a wrong parameter number makes a test
    case useless.

    Reciprocal rank fusion combines them by rank, so the weights stay
    meaningful even though cosine and BM25 are on unrelated scales.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        config: RetrievalConfig,
        keyword_index: KeywordIndex | None = None,
        cache_entries: int = 32,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config
        self.keyword_index = keyword_index or KeywordIndex(vector_store)
        # Retrieval is deterministic for a given query and filter set, so the
        # repeat passes inside one request cost nothing after the first.
        self._results = LruTtlCache(max_entries=cache_entries)

    def retrieve(
        self,
        query_text: str,
        *,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        if not query_text.strip():
            return RetrievalResult([], "", [], 0, 0)

        top_k = top_k if top_k is not None else self.config.top_k

        cache_key = (query_text, top_k, tuple(doc_types) if doc_types else None)
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
        # Track data gets its own pass rather than being mixed into the
        # general one: its chunks are dense tables that would crowd out prose
        # on score alone. Every subdivision is searched - there is no
        # per-requirement mapping - so whichever subdivision's rows match
        # wins on relevance, same as any other retrieval.
        if self.config.track_data_top_k > 0:
            passes.append(
                _Pass(
                    top_k=self.config.track_data_top_k,
                    doc_types=[self.config.track_data_doc_type],
                )
            )

        chunks = [c for group in self._search_all(query_text, passes) for c in group]
        track_subdivisions = sorted(
            {
                c.metadata["subdivision"]
                for c in chunks
                if c.metadata.get("document_type") == self.config.track_data_doc_type
                and c.metadata.get("subdivision")
            }
        )
        result = RetrievalResult(
            chunks=chunks,
            context=self.build_context(chunks),
            track_subdivisions=track_subdivisions,
            dense_hits=sum(1 for c in chunks if "dense" in c.matched_by),
            keyword_hits=sum(1 for c in chunks if "keyword" in c.matched_by),
        )
        self._results.put(cache_key, result)
        return result

    def build_context(self, chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
        """One citation-tagged context block, inside a character budget so the
        prompt stays in the model's window however large the chunks are.
        """
        budget = max_chars if max_chars is not None else self.config.max_context_chars

        blocks: list[str] = []
        used = 0
        for index, chunk in enumerate(chunks, start=1):
            header = (
                f"[{index}] {chunk.metadata.get('document_type', 'kb')} | "
                f"{chunk.source_label}"
            )
            block = f"{header}\n{chunk.text.strip()}"
            if blocks and used + len(block) > budget:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    def _general_doc_types(self, override: list[str] | None) -> list[str] | None:
        requested = override if override else self.config.doc_types
        return list(requested) if requested else None

    def _search_all(self, query_text: str, passes: list[_Pass]) -> list[list[RetrievedChunk]]:
        """Runs every arm of every pass concurrently, then fuses per pass.

        The two arms of a pass are independent, and so are the passes, so all
        are submitted as one flat batch: a dense query and a BM25 scan
        overlap almost perfectly, one waiting on HNSW inside C and the other
        on NumPy.
        """
        tasks: list = []
        for pass_ in passes:
            depth = max(pass_.top_k, self.config.candidates_per_arm)
            where = self._build_where(pass_.doc_types, pass_.exclude_doc_type)
            predicate = self._build_predicate(pass_.doc_types, pass_.exclude_doc_type)
            tasks.append(lambda d=depth, w=where: self._dense_search(query_text, d, w))
            tasks.append(lambda d=depth, f=predicate: self._keyword_search(query_text, d, f))

        # Embed the query once up front. Both dense arms want the same
        # vector and the embedder is cached, so the second arm's embedding
        # becomes a dictionary lookup instead of a second forward pass.
        if self.config.dense_weight > 0:
            self.embedder.embed([query_text])

        arms = run_parallel(tasks)
        return [
            self._fuse(arms[2 * i], arms[2 * i + 1], pass_.top_k)
            for i, pass_ in enumerate(passes)
        ]

    def _keyword_search(self, query_text: str, depth: int, predicate) -> list[KeywordHit]:
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
        self, dense: list[RetrievedChunk], keyword: list, top_k: int
    ) -> list[RetrievedChunk]:
        """Weighted reciprocal rank fusion: each arm contributes
        weight / (rrf_k + rank) for the chunks it ranked. Rank-based, so a
        BM25 score of 14.2 and a cosine of 0.71 combine without pretending
        they are comparable numbers.
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

        return sorted(merged.values(), key=lambda c: c.score, reverse=True)[:top_k]

    # The dense arm filters inside Chroma with a `where` clause, the keyword
    # arm in Python. Both are built from the same inputs so the two arms can
    # never end up searching different subsets.

    @staticmethod
    def _build_where(
        doc_types: list[str] | None, exclude_doc_type: str | None
    ) -> dict[str, Any] | None:
        if doc_types:
            return {"document_type": {"$in": list(doc_types)}}
        if exclude_doc_type:
            return {"document_type": {"$ne": exclude_doc_type}}
        return None

    @staticmethod
    def _build_predicate(doc_types: list[str] | None, exclude_doc_type: str | None):
        allowed = set(doc_types) if doc_types else None

        def predicate(metadata: dict[str, Any]) -> bool:
            document_type = metadata.get("document_type")
            if allowed is not None:
                return document_type in allowed
            if exclude_doc_type and document_type == exclude_doc_type:
                return False
            return True

        return predicate


# --- Static context: python_apis, track_data, Examples ---------------------


class _StaticDocumentStore:
    """Just enough of VectorStore's interface for KeywordIndex to build a
    BM25 corpus over chunks read straight from disk.
    """

    def __init__(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]):
        self._chunk_ids = chunk_ids
        self._texts = texts
        self._metadatas = metadatas

    def count(self) -> int:
        return len(self._chunk_ids)

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        return self._chunk_ids, self._texts, self._metadatas


def _build_static_chunks(config: PipelineConfig) -> _StaticDocumentStore:
    static_config = dataclasses.replace(config, sources=config.static_sources)
    pipeline = IngestionPipeline(static_config)

    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for file in discover_files(static_config):
        try:
            _hash, chunks = pipeline.prepare_file(file)
        except Exception:
            logger.exception("Failed to parse static-context file %s", file.path)
            continue
        for chunk in chunks:
            chunk_ids.append(chunk.metadata.chunk_id)
            texts.append(chunk.text)
            metadatas.append(chunk.metadata.to_chroma_dict())

    logger.info("Built static (non-embedded) context over %d chunks", len(chunk_ids))
    return _StaticDocumentStore(chunk_ids, texts, metadatas)


@dataclasses.dataclass(frozen=True)
class _Example:
    """One requirement -> reference test cases -> reference script triple."""

    requirement_id: str
    requirement: str
    test_cases: str
    script: str


def _requirement_key(stem: str) -> str:
    return stem.split("_", 1)[0].strip().upper()


def _joined_text(path: Path, config: PipelineConfig) -> str:
    units = extract_and_normalize(path, config)
    return "\n\n".join(u.text.strip() for u in units if u.text.strip())


def _read_examples(config: PipelineConfig) -> list[_Example]:
    examples_dir = config.examples_dir
    if not examples_dir or not examples_dir.is_dir():
        return []

    requirement_texts = {
        _requirement_key(p.stem): p for p in sorted(examples_dir.glob("*.txt"))
    }
    test_case_files = {
        _requirement_key(p.stem): p
        for p in sorted((examples_dir / "reference_test_cases").glob("*.xlsx"))
    }
    script_files = {
        _requirement_key(p.stem): p
        for p in sorted((examples_dir / "reference_test_scripts").glob("*.txt"))
    }

    examples = [
        _Example(
            requirement_id=requirement_id,
            requirement=(
                _joined_text(requirement_texts[requirement_id], config)
                if requirement_id in requirement_texts
                else ""
            ),
            test_cases=(
                _joined_text(test_case_files[requirement_id], config)
                if requirement_id in test_case_files
                else ""
            ),
            script=(
                _joined_text(script_files[requirement_id], config)
                if requirement_id in script_files
                else ""
            ),
        )
        for requirement_id in sorted(
            set(requirement_texts) | set(test_case_files) | set(script_files)
        )
    ]
    logger.info("Loaded %d example(s) from %s", len(examples), examples_dir)
    return examples


def _budgeted(text: str, max_chars: int) -> str:
    """Trims to a whole line inside `max_chars`.

    Examples are shape templates, so a truncated one still does its job.
    Overrunning the context window does not: Ollama silently drops the
    *start* of the prompt, where the requirement, the retrieved facts and
    the rules that forbid copying example values live.
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars // 2:
        cut = cut[:newline]
    return cut.rstrip() + "\n[... example truncated - shape only, not a complete artefact]"


def _format_hits(hits: list[KeywordHit], max_chars: int = 0) -> str:
    """The matched chunks as one citation-tagged block, inside a character
    budget (0 = unbudgeted).

    The budget is latency, not book-keeping. Track-data chunks are dense rows
    of identifiers that tokenize at ~2.3 chars per token where prose runs at
    4: four unbudgeted track chunks measured 4,796 characters but 2,086
    tokens - larger than the retrieved knowledge base, the requirement and
    the system prompt combined, and nearly two minutes of every call at this
    CPU's prefill rate. `top_k` cannot bound it, because chunk size is a
    property of the subdivision.

    The first hit is always kept whole - a budget that returns nothing is
    worse than one that overruns slightly - and each later hit is added only
    if it fits.
    """
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source_file") or hit.metadata.get("source_path") or "unknown"
        locator = hit.metadata.get("method_name") or hit.metadata.get("section_path") or ""
        label = f"{source} > {locator}" if locator else source
        block = (
            f"[{index}] {hit.metadata.get('document_type', 'static')} | {label}\n"
            f"{hit.text.strip()}"
        )
        if max_chars > 0 and blocks and used + len(block) > max_chars:
            continue
        if max_chars > 0 and not blocks and len(block) > max_chars:
            block = _budgeted(block, max_chars)
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


class StaticContextProvider:
    """python_apis and track_data, keyword-searched from an in-process BM25
    index built straight off disk, plus the Examples folder as few-shot
    material under a character budget.

    Built once at construction - restart the server to pick up edits to
    data/python_apis, data/track_data or data/Examples.
    """

    def __init__(self, ingestion_config: PipelineConfig):
        self._config = ingestion_config
        self._keyword_index = KeywordIndex(_build_static_chunks(ingestion_config))
        self._examples = _read_examples(ingestion_config)

    def test_case_examples(self, max_chars: int) -> str:
        """Requirement -> reference test cases, for the datasheet prompt.

        Reference scripts are left out deliberately: they are the template
        for the script call, and here they would be tens of thousands of
        characters that teach nothing about datasheet phrasing while pushing
        the requirement out of the context window.
        """
        return self._render(max_chars, lambda e: e.test_cases, "Reference test case(s)")

    def script_examples(self, max_chars: int) -> str:
        """Requirement -> reference script, for the script prompt."""
        return self._render(max_chars, lambda e: e.script, "Reference script")

    def _render(self, max_chars: int, select, label: str) -> str:
        usable = [e for e in self._examples if select(e).strip()]
        if not usable or max_chars <= 0:
            return ""
        # An equal share each, so one long example cannot crowd the rest out.
        # The share covers the heading and the example's own requirement text
        # as well as the artefact, or max_chars is not a budget at all.
        share = max(200, max_chars // len(usable))
        blocks = []
        for example in usable:
            heading = f"### Example: {example.requirement_id}"
            sections = [heading]
            remaining = share - len(heading)
            if example.requirement:
                requirement = _budgeted(example.requirement, max(80, share // 4))
                sections.append("Requirement:\n" + requirement)
                remaining -= len(requirement) + len("Requirement:\n")
            sections.append(f"{label}:\n" + _budgeted(select(example), max(120, remaining)))
            blocks.append("\n\n".join(sections))
        return "\n\n---\n\n".join(blocks)

    def api_context(self, query: str, top_k: int, max_chars: int = 0) -> str:
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "python_apis"
        )
        return _format_hits(hits, max_chars)

    def track_context(self, query: str, top_k: int, max_chars: int = 0) -> str:
        """Searches every subdivision for whichever rows best match the query
        - there is no per-requirement subdivision mapping.
        """
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "track_data"
        )
        return _format_hits(hits, max_chars)
