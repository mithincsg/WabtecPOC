from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from kb_ingestion.config import PipelineConfig
from kb_ingestion.embeddings import LocalBGEM3Embedder
from kb_ingestion.vector_store import VectorStore
from rag.cache import EmbeddingCache, LruTtlCache
from rag.confidence import ConfidenceScorer
from rag.config import RAGSettings
from rag.generator import TestCaseGenerator, TestScriptGenerator
from rag.keyword_index import KeywordIndex
from rag.llm_client import OllamaClient
from rag.prompts import PromptLibrary
from rag.retriever import HybridRetriever
from rag.static_context import StaticContextProvider
from rag.track_mapping import TrackMapping

logger = logging.getLogger(__name__)


# Generated rows are held for this long, so re-submitting the same
# requirement (the reflex after a browser refresh, or while comparing two
# wordings) answers instantly instead of spending minutes reproducing a
# near-identical answer at temperature 0.2. Short enough that a knowledge-base
# re-ingestion is reflected within the hour without a restart.
_RESULT_CACHE_TTL_SECONDS = 60 * 60
_RESULT_CACHE_ENTRIES = 32


class Services:
    """Everything the API needs, built once and shared.

    The heavy pieces — the BGE-M3 weights and the BM25 index — are built on
    first use rather than at import, so the server starts immediately and
    `/api/health` answers even if the model hasn't loaded yet. Every lazy
    property is guarded by the same lock: two concurrent first requests would
    otherwise each load the embedding model, doubling peak memory on a
    machine that may not have the headroom, and each build their own BM25
    index over the whole corpus.
    """

    def __init__(self, repo_root: Path):
        load_dotenv(repo_root / ".env")

        self.repo_root = repo_root
        self.settings = RAGSettings.load(repo_root / "config" / "rag_config.yaml")

        # config/rag_config.yaml is the checked-in default; OLLAMA_HOST in
        # .env is the per-machine override (e.g. Ollama running on another
        # box), so it takes precedence when set.
        if os.getenv("OLLAMA_HOST"):
            self.settings.generation.ollama_host = os.environ["OLLAMA_HOST"]

        self.prompts = PromptLibrary(repo_root / "config" / "prompts.yaml")
        self.track_mapping = TrackMapping(repo_root / "config" / "track_mapping.yaml")
        self.ingestion_config = PipelineConfig.load(repo_root / "config" / "config.yaml")
        self.llm_client = OllamaClient(self.settings.generation)

        # Re-entrant: the lazy properties below are layered (retriever needs
        # the embedder, which needs the lock), and a plain Lock would
        # deadlock the first request that walks the whole chain.
        self._lock = threading.RLock()
        self._embedder = None
        self._vector_store: VectorStore | None = None
        self._keyword_index: KeywordIndex | None = None
        self._retriever: HybridRetriever | None = None
        self._static_context: StaticContextProvider | None = None
        self._test_case_generator: TestCaseGenerator | None = None
        self._script_generator: TestScriptGenerator | None = None

        self.result_cache = LruTtlCache(
            max_entries=_RESULT_CACHE_ENTRIES, ttl_seconds=_RESULT_CACHE_TTL_SECONDS
        )

    # --- lazily built ------------------------------------------------------

    @property
    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    retrieval = self.settings.retrieval
                    # Wrapped in a cache because the same texts are embedded
                    # over and over: the requirement query once per retrieval
                    # pass, and the retrieved reference test cases on every
                    # confidence-scoring run even though they never change.
                    self._embedder = EmbeddingCache(
                        LocalBGEM3Embedder(
                            model_name=retrieval.embedding_model,
                            device=retrieval.embedding_device,
                        )
                    )
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    retrieval = self.settings.retrieval
                    self._vector_store = VectorStore(
                        retrieval.chroma_persist_dir, retrieval.chroma_collection
                    )
        return self._vector_store

    @property
    def keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:
            with self._lock:
                if self._keyword_index is None:
                    self._keyword_index = KeywordIndex(self.vector_store)
        return self._keyword_index

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = HybridRetriever(
                        embedder=self.embedder,
                        vector_store=self.vector_store,
                        config=self.settings.retrieval,
                        track_mapping=self.track_mapping,
                        keyword_index=self.keyword_index,
                    )
        return self._retriever

    @property
    def static_context(self) -> StaticContextProvider:
        if self._static_context is None:
            with self._lock:
                if self._static_context is None:
                    self._static_context = StaticContextProvider(
                        self.ingestion_config, self.track_mapping
                    )
        return self._static_context

    @property
    def test_case_generator(self) -> TestCaseGenerator:
        if self._test_case_generator is None:
            with self._lock:
                if self._test_case_generator is None:
                    self._test_case_generator = TestCaseGenerator(
                        retriever=self.retriever,
                        llm_client=self.llm_client,
                        prompts=self.prompts,
                        scorer=ConfidenceScorer(
                            self.embedder, self.settings.confidence
                        ),
                        static_context=self.static_context,
                        static_track_top_k=self.settings.retrieval.static_track_top_k,
                        example_max_chars=self.settings.retrieval.example_max_chars,
                    )
        return self._test_case_generator

    @property
    def script_generator(self) -> TestScriptGenerator:
        if self._script_generator is None:
            with self._lock:
                if self._script_generator is None:
                    self._script_generator = TestScriptGenerator(
                        retriever=self.retriever,
                        llm_client=self.llm_client,
                        prompts=self.prompts,
                        max_tokens=self.settings.generation.script_max_tokens,
                        static_context=self.static_context,
                        static_api_top_k=self.settings.retrieval.static_api_top_k,
                        static_track_top_k=self.settings.retrieval.static_track_top_k,
                        kb_top_k=self.settings.retrieval.script_kb_top_k,
                        example_top_k=self.settings.retrieval.script_example_top_k,
                        example_max_chars=self.settings.retrieval.example_max_chars,
                    )
        return self._script_generator

    # --- cache keys ---------------------------------------------------------

    def prompt_revision(self) -> float:
        """The prompts file's modification time, so a cached generation is
        never replayed after someone edited the prompt that produced it.
        Prompt wording is what gets iterated on, and a stale cached answer
        would make an edit look like it had no effect.
        """
        try:
            return self.prompts.path.stat().st_mtime
        except OSError:
            return 0.0

    # --- startup -----------------------------------------------------------

    def warm_up(self) -> None:
        """Pays the two startup costs — loading the embedding model and
        loading the LLM weights into Ollama — while the user is still reading
        the page, rather than on their first request. Failures are logged and
        swallowed: a cold start is slow, not broken.
        """
        try:
            self.embedder.embed(["warm up"])
            logger.info("Embedding model ready")
        except Exception:  # noqa: BLE001
            logger.exception("Could not preload the embedding model")

        try:
            self.static_context  # noqa: B018 - property access builds the BM25 corpus
            logger.info("Static context (python_apis, track_data, Examples) ready")
        except Exception:  # noqa: BLE001
            logger.exception("Could not preload static context")

        if self.llm_client.warm_up():
            logger.info("%s loaded and resident", self.settings.generation.model)
