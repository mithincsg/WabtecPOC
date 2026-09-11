"""config/rag_config.yaml -> settings, and config/prompts.yaml -> prompts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import yaml

logger = logging.getLogger(__name__)


# --- Settings --------------------------------------------------------------


@dataclass
class RetrievalConfig:
    chroma_persist_dir: Path = Path("chroma_db")
    chroma_collection: str = "kb_collection"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"

    top_k: int = 10
    candidates_per_arm: int = 30
    dense_weight: float = 1.0
    keyword_weight: float = 0.6
    rrf_k: int = 60
    min_similarity: float = 0.0
    max_context_chars: int = 9000

    doc_types: list[str] = field(default_factory=list)
    track_data_doc_type: str = "track_data"
    track_data_top_k: int = 4

    static_api_top_k: int = 4
    static_track_top_k: int = 4
    # Character budgets on the two BM25-only blocks. Track chunks tokenize at
    # ~2.3 chars/token and were the largest block in the prompt when only
    # top_k bounded them - see retrieval._format_hits.
    static_track_max_chars: int = 1200
    static_api_max_chars: int = 3500
    # What the plan call sees of the knowledge base: much smaller, because
    # enumerating behaviours needs the requirement's vocabulary, not values.
    plan_context_max_chars: int = 1200

    # Character budgets for data/Examples/. Unbudgeted they are ~157k chars
    # against an 8k-token window, and Ollama then trims the prompt from the
    # start and drops the requirement itself.
    examples_test_case_max_chars: int = 1500
    examples_script_max_chars: int = 8000


@dataclass
class GenerationConfig:
    ollama_host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.15
    top_p: float = 0.9
    # Per writing call (test_case_batch_size behaviours), not per requirement.
    max_tokens: int = 700
    num_ctx: int = 8192
    request_timeout_seconds: int = 900
    keep_alive: str = "30m"
    # An int, or "auto" for one thread per logical processor.
    num_threads: int | str = "auto"
    # Hybrid-reasoning models (qwen3) emit a <think> block unless told not
    # to; it costs decode tokens for nothing this pipeline reads. Models
    # without the toggle ignore Ollama's `think` field.
    think: bool = False
    # A ceiling on how many behaviours the plan call may enumerate, not a
    # target number of test cases.
    max_test_cases: int = 20

    # Stage one, the plan call. Enumeration is the easier half of the job, so
    # it can run on a smaller model; None means "the same model".
    plan_model: str | None = None
    plan_max_tokens: int = 600
    # Hotter than the writing calls on purpose: enumeration wants breadth.
    plan_temperature: float = 0.35
    test_case_batch_size: int = 4
    # How many writing calls may be in flight at once; bounded by what Ollama
    # serves concurrently (OLLAMA_NUM_PARALLEL). 1 = strictly sequential.
    writing_parallelism: int = 2

    script_max_tokens: int = 1792
    # Kept equal to num_ctx by default: differing windows make Ollama reload
    # the weights on every switch between the two calls.
    script_num_ctx: int = 8192

    def resolved_num_threads(self) -> int:
        """The `num_thread` option to send Ollama, or 0 to let it choose.

        `auto` is one thread per *logical* processor, which is measured, not
        assumed: on the 6-core/12-thread box this was built on, 12 threads
        beat 6 by 28% on prefill, which is what dominates wall-clock here.
        The usual "SMT over-subscribes" advice is about memory-bound decode.
        Re-measure with scripts/bench_generation.py on other hardware.
        """
        value = self.num_threads
        if not isinstance(value, str):
            return max(0, int(value))
        if value.strip().lower() != "auto":
            raise ValueError(
                f"llm_num_threads must be an integer or 'auto', got {value!r}"
            )
        threads = os.cpu_count() or 0
        logger.info("llm_num_threads: auto -> %d thread(s)", threads)
        return threads


@dataclass
class ConfidenceConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "retrieval": 0.3,
            "grounding": 0.3,
            "similarity_to_existing": 0.4,
        }
    )
    review_threshold: float = 0.55


# Maps a YAML key to the attribute it sets, so one flat config file can feed
# three config objects without any of them knowing the file layout.
_RETRIEVAL_KEYS = {
    "chroma_persist_dir": "chroma_persist_dir",
    "chroma_collection": "chroma_collection",
    "embedding_model": "embedding_model",
    "embedding_device": "embedding_device",
    "retrieval_top_k": "top_k",
    "candidates_per_arm": "candidates_per_arm",
    "dense_weight": "dense_weight",
    "keyword_weight": "keyword_weight",
    "rrf_k": "rrf_k",
    "min_similarity": "min_similarity",
    "max_context_chars": "max_context_chars",
    "doc_types": "doc_types",
    "track_data_doc_type": "track_data_doc_type",
    "track_data_top_k": "track_data_top_k",
    "static_api_top_k": "static_api_top_k",
    "static_track_top_k": "static_track_top_k",
    "static_track_max_chars": "static_track_max_chars",
    "static_api_max_chars": "static_api_max_chars",
    "plan_context_max_chars": "plan_context_max_chars",
    "examples_test_case_max_chars": "examples_test_case_max_chars",
    "examples_script_max_chars": "examples_script_max_chars",
}
_GENERATION_KEYS = {
    "ollama_host": "ollama_host",
    "llm_model": "model",
    "llm_temperature": "temperature",
    "llm_top_p": "top_p",
    "llm_max_tokens": "max_tokens",
    "llm_num_ctx": "num_ctx",
    "llm_request_timeout_seconds": "request_timeout_seconds",
    "llm_keep_alive": "keep_alive",
    "llm_num_threads": "num_threads",
    "llm_think": "think",
    "max_test_cases": "max_test_cases",
    "plan_model": "plan_model",
    "plan_max_tokens": "plan_max_tokens",
    "plan_temperature": "plan_temperature",
    "test_case_batch_size": "test_case_batch_size",
    "writing_parallelism": "writing_parallelism",
    "script_max_tokens": "script_max_tokens",
    "script_num_ctx": "script_num_ctx",
}
_CONFIDENCE_KEYS = {
    "confidence_weights": "weights",
    "confidence_review_threshold": "review_threshold",
}


@dataclass
class RAGSettings:
    retrieval: RetrievalConfig
    generation: GenerationConfig
    confidence: ConfidenceConfig
    base_dir: Path

    @classmethod
    def load(cls, config_path: str | Path) -> "RAGSettings":
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        base_dir = config_path.resolve().parent.parent

        retrieval = RetrievalConfig()
        generation = GenerationConfig()
        confidence = ConfidenceConfig()

        for key, value in raw.items():
            if key in _RETRIEVAL_KEYS:
                setattr(retrieval, _RETRIEVAL_KEYS[key], value)
            elif key in _GENERATION_KEYS:
                setattr(generation, _GENERATION_KEYS[key], value)
            elif key in _CONFIDENCE_KEYS:
                setattr(confidence, _CONFIDENCE_KEYS[key], value)
            else:
                raise ValueError(f"Unknown config key: {key!r}")

        path = Path(retrieval.chroma_persist_dir)
        retrieval.chroma_persist_dir = (
            path if path.is_absolute() else (base_dir / path).resolve()
        )
        return cls(
            retrieval=retrieval,
            generation=generation,
            confidence=confidence,
            base_dir=base_dir,
        )


# --- Prompts ---------------------------------------------------------------


class PromptError(RuntimeError):
    """config/prompts.yaml is missing a prompt or a placeholder."""


@dataclass(frozen=True)
class PromptPair:
    system: str
    user_template: str

    def render_user(self, **values: str) -> str:
        """Fills the user template. `string.Template` ($name) rather than
        str.format, because these prompts contain literal JSON braces that
        would otherwise all need escaping in the YAML.
        """
        try:
            return Template(self.user_template).substitute(**values)
        except KeyError as exc:
            raise PromptError(
                f"Prompt template references ${exc.args[0]}, which was not supplied. "
                f"Available: {', '.join(sorted(values))}."
            ) from exc


class PromptLibrary:
    """Every prompt, read from config/prompts.yaml.

    Re-read on each access when `watch` is on (the default), so editing a
    prompt takes effect on the next request without restarting the server.
    """

    def __init__(self, path: str | Path, watch: bool = True):
        self.path = Path(path)
        self.watch = watch
        self._cache: dict | None = None
        self._mtime: float | None = None

    def _data(self) -> dict:
        if not self.path.is_file():
            raise PromptError(f"Prompt file not found: {self.path}")
        mtime = self.path.stat().st_mtime
        if self._cache is None or (self.watch and mtime != self._mtime):
            self._cache = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self._mtime = mtime
        return self._cache

    def pair(self, name: str) -> PromptPair:
        data = self._data()
        entry = data.get(name)
        if not isinstance(entry, dict):
            raise PromptError(
                f"{self.path} has no '{name}' prompt. Found: {', '.join(sorted(data))}."
            )
        for key in ("system", "user"):
            if not str(entry.get(key) or "").strip():
                raise PromptError(f"{self.path}: '{name}' is missing a non-empty '{key}'.")
        return PromptPair(system=entry["system"].strip(), user_template=entry["user"])

    @property
    def test_case_plan(self) -> PromptPair:
        """Stage one of test-case generation: the behaviour list."""
        return self.pair("test_case_plan")

    @property
    def test_cases(self) -> PromptPair:
        """Stage two: the datasheet rows for one batch of behaviours."""
        return self.pair("test_cases")

    @property
    def test_script(self) -> PromptPair:
        return self.pair("test_script")
