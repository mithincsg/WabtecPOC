from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from kb_ingestion.config import PipelineConfig
from rag.cache import LruTtlCache
from rag.caf_mapping import CafMapping
from rag.config import RAGSettings
from rag.generator import TestCaseGenerator, TestScriptGenerator
from rag.llm_client import OllamaClient
from rag.prompts import PromptLibrary
from rag.static_context import StaticContextProvider

logger = logging.getLogger(__name__)


# Generated rows are held for this long, so re-submitting the same
# requirement (the reflex after a browser refresh, or while comparing two
# wordings) answers instantly instead of spending minutes reproducing a
# near-identical answer at temperature 0.2.
_RESULT_CACHE_TTL_SECONDS = 60 * 60
_RESULT_CACHE_ENTRIES = 32


class Services:
    """Everything the API needs, built once and shared.

    The one heavy piece — the BM25 index over every source folder — is built
    on first use rather than at import, so the server starts immediately and
    `/api/health` answers before it is ready. The lock is what keeps two
    concurrent first requests from each building their own index over the
    whole corpus.
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
        self.ingestion_config = PipelineConfig.load(repo_root / "config" / "config.yaml")
        # Requirement -> feature (the datasheet's Folder column). Cheap
        # enough to build eagerly: one small workbook, read on demand and
        # re-read only when it changes.
        self.caf_mapping = CafMapping(self.ingestion_config.caf_mapping_file)
        self.llm_client = OllamaClient(self.settings.generation)

        # Re-entrant: the generators below are built from the static context,
        # which needs the same lock, and a plain Lock would deadlock the first
        # request that walks the whole chain.
        self._lock = threading.RLock()
        self._static_context: StaticContextProvider | None = None
        self._test_case_generator: TestCaseGenerator | None = None
        self._script_generator: TestScriptGenerator | None = None

        self.result_cache = LruTtlCache(
            max_entries=_RESULT_CACHE_ENTRIES, ttl_seconds=_RESULT_CACHE_TTL_SECONDS
        )

    # --- lazily built ------------------------------------------------------

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
                    self._test_case_generator = TestCaseGenerator(
                        llm_client=self.llm_client,
                        prompts=self.prompts,
                        static_context=self.static_context,
                        static_parameter_top_k=self.settings.retrieval.static_parameter_top_k,
                        caf_mapping=self.caf_mapping,
                    )
        return self._test_case_generator

    @property
    def script_generator(self) -> TestScriptGenerator:
        if self._script_generator is None:
            with self._lock:
                if self._script_generator is None:
                    self._script_generator = TestScriptGenerator(
                        llm_client=self.llm_client,
                        prompts=self.prompts,
                        max_tokens=self.settings.generation.script_max_tokens,
                        static_context=self.static_context,
                        static_api_top_k=self.settings.retrieval.static_api_top_k,
                        static_track_top_k=self.settings.retrieval.static_track_top_k,
                        static_parameter_top_k=self.settings.retrieval.static_parameter_top_k,
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
        """Pays the two startup costs — building the BM25 index and loading
        the LLM weights into Ollama — while the user is still reading the
        page, rather than on their first request. Failures are logged and
        swallowed: a cold start is slow, not broken.
        """
        try:
            self.static_context  # noqa: B018 - property access builds the BM25 corpus
            logger.info("Static context (python_apis, track_data, parameter_config) ready")
        except Exception:  # noqa: BLE001
            logger.exception("Could not preload static context")

        if self.llm_client.warm_up():
            logger.info("%s loaded and resident", self.settings.generation.model)
