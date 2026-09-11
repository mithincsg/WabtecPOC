from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RetrievalConfig:
    chroma_persist_dir: Path = Path("chroma_db")
    chroma_collection: str = "kb_collection"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"

    top_k: int = 14
    candidates_per_arm: int = 30
    dense_weight: float = 1.0
    keyword_weight: float = 0.6
    rrf_k: int = 60
    min_similarity: float = 0.0
    max_context_chars: int = 14000

    doc_types: list[str] = field(default_factory=list)
    track_data_doc_type: str = "track_data"
    track_data_top_k: int = 4

    static_api_top_k: int = 4
    static_track_top_k: int = 4
    # Depth of the knowledge-base pass run for script generation, so the
    # script prompt carries the parameters and defaults its values must come
    # from rather than only the reference scripts. 0 disables it.
    script_kb_top_k: int = 6
    # Examples are template material, so they are budgeted rather than sent
    # whole — an over-long prompt is trimmed from the start by Ollama, which
    # is where the rules live.
    script_example_top_k: int = 1
    # Budget for the script prompt's reference script; divided across the
    # examples for the test-case prompt.
    example_max_chars: int = 12000


@dataclass
class GenerationConfig:
    ollama_host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    num_ctx: int = 24576
    # None = wait indefinitely (passed straight to requests as timeout=None).
    request_timeout_seconds: int | None = 900
    keep_alive: str = "30m"
    num_threads: int = 0
    # A ceiling, not a target: the model decides how many cases the
    # requirement needs, and this only bounds the response length.
    max_test_cases: int = 20
    # Test-case generation writes JSON for this many agreed behaviours per
    # LLM call (test_case_plan enumerates all of them first). Small enough
    # that one batch's JSON response never approaches llm_max_tokens, so
    # coverage can no longer be lost to truncation the way a single big call
    # could lose it.
    test_case_batch_size: int = 5
    script_max_tokens: int = 3072


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
# three separate config objects without any of them knowing the file layout.
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
    "script_kb_top_k": "script_kb_top_k",
    "script_example_top_k": "script_example_top_k",
    "example_max_chars": "example_max_chars",
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
    "max_test_cases": "max_test_cases",
    "test_case_batch_size": "test_case_batch_size",
    "script_max_tokens": "script_max_tokens",
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
