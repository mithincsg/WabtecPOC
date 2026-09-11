"""Request/response schemas, and the shared service container behind them."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from kb_ingestion.config import PipelineConfig
from kb_ingestion.storage import LocalBGEM3Embedder, VectorStore
from rag.config import PromptLibrary, RAGSettings
from rag.generator import TestCaseGenerator, TestScriptGenerator
from rag.llm_client import OllamaClient
from rag.retrieval import HybridRetriever, KeywordIndex, StaticContextProvider
from rag.schema import ConfidenceBreakdown, ConfidenceScorer, TestCase
from rag.utils import EmbeddingCache, LruTtlCache

logger = logging.getLogger(__name__)

# Generated rows are held this long, so re-submitting the same requirement
# (the reflex after a browser refresh) answers instantly instead of spending
# minutes reproducing a near-identical answer at temperature 0.15. Short
# enough that a re-ingestion is reflected within the hour without a restart.
_RESULT_CACHE_TTL_SECONDS = 60 * 60
_RESULT_CACHE_ENTRIES = 32


# --- Schemas ---------------------------------------------------------------


class ConfidenceOut(BaseModel):
    overall: float = 0.0
    retrieval: float = 0.0
    grounding: float = 0.0
    similarity_to_existing: float = 0.0
    closest_existing_id: str | None = None
    closest_existing_source: str | None = None
    needs_review: bool = False

    @classmethod
    def from_domain(cls, confidence: ConfidenceBreakdown) -> "ConfidenceOut":
        return cls(**confidence.__dict__)


class TestCaseOut(BaseModel):
    """One datasheet row. Field names match the schema's attribute names, so
    the React table and the Excel export are driven by the same keys.
    """

    s_no: int
    requirement: str
    description: str
    folder: str = ""
    optimization_technique: str = "Default"
    test_type: str = "Positive"
    test_technique: str = "Equivalence Partitioning"
    retired: str = "False"
    scorable: str = "Yes"
    comments: str = ""
    confidence: ConfidenceOut = Field(default_factory=ConfidenceOut)

    @classmethod
    def from_domain(cls, test_case: TestCase) -> "TestCaseOut":
        data = {
            attr: getattr(test_case, attr)
            for attr in (
                "s_no",
                "requirement",
                "description",
                "folder",
                "optimization_technique",
                "test_type",
                "test_technique",
                "retired",
                "scorable",
                "comments",
            )
        }
        return cls(**data, confidence=ConfidenceOut.from_domain(test_case.confidence))

    def to_domain(self) -> TestCase:
        return TestCase(
            s_no=self.s_no,
            requirement=self.requirement,
            description=self.description,
            folder=self.folder,
            optimization_technique=self.optimization_technique,
            test_type=self.test_type,
            test_technique=self.test_technique,
            retired=self.retired,
            scorable=self.scorable,
            comments=self.comments,
            confidence=ConfidenceBreakdown(**self.confidence.model_dump()),
        )


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    source: str
    document_type: str = ""
    similarity: float | None = None
    bm25_score: float | None = None
    # Which arm(s) found this chunk - shown in the UI, so it is visible when
    # keyword search is what surfaced a parameter table dense search ranked
    # too low.
    matched_by: list[str] = Field(default_factory=list)
    excerpt: str = ""


class GenerateTestCasesRequest(BaseModel):
    """There is deliberately no test-case count here. How many cases a
    requirement needs is a property of the requirement - one per verifiable
    behaviour - not something a user should guess before seeing the result.
    """

    requirement_text: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_types: list[str] | None = None
    # Set by a caller that wants the model re-run rather than the previous
    # identical result replayed. Not exposed in the UI; it exists so a
    # cached answer is never the only answer available.
    refresh: bool = False


class GenerateTestCasesResponse(BaseModel):
    requirement_id: str | None
    functional_area: str | None
    test_cases: list[TestCaseOut]
    retrieved: list[RetrievedChunkOut]
    track_subdivisions: list[str] = Field(default_factory=list)
    # Coverage is judged against behaviours, not row count, so a reviewer
    # needs the list the rows were written from.
    behaviours: list[str] = Field(default_factory=list)
    mean_confidence: float = 0.0
    review_threshold: float = 0.0
    elapsed_seconds: float = 0.0
    # True when these rows were replayed from an earlier identical request.
    cached: bool = False


class GenerateScriptRequest(BaseModel):
    requirement_text: str = Field(min_length=1)
    # The reviewed rows, not the ones first generated.
    test_cases: list[TestCaseOut] = Field(min_length=1)


class GenerateScriptResponse(BaseModel):
    requirement_id: str | None
    script: str
    elapsed_seconds: float = 0.0


class ExportTestCasesRequest(BaseModel):
    test_cases: list[TestCaseOut] = Field(min_length=1)
    requirement_id: str | None = None
    include_confidence: bool = True


class ExportScriptRequest(BaseModel):
    script: str = Field(min_length=1)
    requirement_id: str | None = None


class RequirementUploadResponse(BaseModel):
    filename: str
    requirement_text: str
    requirement_id: str | None = None


class HealthResponse(BaseModel):
    knowledge_base_chunks: int
    embedding_model: str
    llm_model: str
    llm_available: bool


# --- Service container -----------------------------------------------------


class Services:
    """Everything the API needs, built once and shared.

    The heavy pieces - the BGE-M3 weights and the BM25 index - are built on
    first use rather than at import, so the server starts immediately and
    /api/health answers even if the model hasn't loaded. Every lazy property
    is guarded by the same re-entrant lock: they are layered (the retriever
    needs the embedder), and two concurrent first requests would otherwise
    each load the embedding model and build their own BM25 index.
    """

    def __init__(self, repo_root: Path):
        load_dotenv(repo_root / ".env")

        self.repo_root = repo_root
        self.settings = RAGSettings.load(repo_root / "config" / "rag_config.yaml")

        # config/rag_config.yaml is the checked-in default; OLLAMA_HOST in
        # .env is the per-machine override, so it takes precedence.
        if os.getenv("OLLAMA_HOST"):
            self.settings.generation.ollama_host = os.environ["OLLAMA_HOST"]

        self.prompts = PromptLibrary(repo_root / "config" / "prompts.yaml")
        self.ingestion_config = PipelineConfig.load(repo_root / "config" / "config.yaml")
        self.llm_client = OllamaClient(self.settings.generation)

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

    @property
    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    retrieval = self.settings.retrieval
                    # Wrapped in a cache because the same texts are embedded
                    # over and over: the requirement query once per retrieval
                    # pass, and the retrieved reference cases on every
                    # confidence-scoring run.
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
                        keyword_index=self.keyword_index,
                    )
        return self._retriever

    @property
    def static_context(self) -> StaticContextProvider:
        if self._static_context is None:
            with self._lock:
                if self._static_context is None:
                    self._static_context = StaticContextProvider(self.ingestion_config)
        return self._static_context

    @property
    def test_case_generator(self) -> TestCaseGenerator:
        if self._test_case_generator is None:
            with self._lock:
                if self._test_case_generator is None:
                    retrieval = self.settings.retrieval
                    generation = self.settings.generation
                    self._test_case_generator = TestCaseGenerator(
                        retriever=self.retriever,
                        llm_client=self.llm_client,
                        prompts=self.prompts,
                        scorer=ConfidenceScorer(self.embedder, self.settings.confidence),
                        static_context=self.static_context,
                        static_track_top_k=retrieval.static_track_top_k,
                        examples_max_chars=retrieval.examples_test_case_max_chars,
                        static_track_max_chars=retrieval.static_track_max_chars,
                        batch_size=generation.test_case_batch_size,
                        plan_model=generation.plan_model,
                        plan_max_tokens=generation.plan_max_tokens,
                        plan_temperature=generation.plan_temperature,
                        plan_context_max_chars=retrieval.plan_context_max_chars,
                        parallelism=generation.writing_parallelism,
                    )
        return self._test_case_generator

    @property
    def script_generator(self) -> TestScriptGenerator:
        if self._script_generator is None:
            with self._lock:
                if self._script_generator is None:
                    retrieval = self.settings.retrieval
                    generation = self.settings.generation
                    self._script_generator = TestScriptGenerator(
                        retriever=self.retriever,
                        llm_client=self.llm_client,
                        prompts=self.prompts,
                        max_tokens=generation.script_max_tokens,
                        num_ctx=generation.script_num_ctx,
                        static_context=self.static_context,
                        static_api_top_k=retrieval.static_api_top_k,
                        static_track_top_k=retrieval.static_track_top_k,
                        examples_max_chars=retrieval.examples_script_max_chars,
                        static_api_max_chars=retrieval.static_api_max_chars,
                        static_track_max_chars=retrieval.static_track_max_chars,
                    )
        return self._script_generator

    def prompt_revision(self) -> float:
        """The prompts file's mtime, so a cached generation is never replayed
        after someone edited the prompt that produced it - otherwise an edit
        looks like it had no effect.
        """
        try:
            return self.prompts.path.stat().st_mtime
        except OSError:
            return 0.0

    def warm_up(self) -> None:
        """Pays the startup costs - the embedding model and the LLM weights -
        while the user is still reading the page. Failures are logged and
        swallowed: a cold start is slow, not broken.
        """
        try:
            self.embedder.embed(["warm up"])
            logger.info("Embedding model ready")
        except Exception:  # noqa: BLE001
            logger.exception("Could not preload the embedding model")

        try:
            self.static_context  # noqa: B018 - property access builds the corpus
            logger.info("Static context (python_apis, track_data, Examples) ready")
        except Exception:  # noqa: BLE001
            logger.exception("Could not preload static context")

        if self.llm_client.warm_up():
            logger.info("%s loaded and resident", self.settings.generation.model)
