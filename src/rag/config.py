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
    static_parameter_top_k: int = 6


@dataclass
class GenerationConfig:
    ollama_host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    num_ctx: int = 8192
    # Two different failures, two different limits: request_timeout_seconds
    # caps the whole generation (at which point the partial answer is kept),
    # stall_timeout_seconds caps the gap between streamed tokens (at which
    # point the model is wedged and the request fails). 0 removes either
    # limit, which is the default — on CPU there is no wall-clock number that
    # tells a slow generation apart from a stuck one.
    request_timeout_seconds: int = 0
    stall_timeout_seconds: int = 0
    keep_alive: str = "30m"
    num_threads: int = 0
    # "auto" | "off" | "on". Reasoning models (qwen3, deepseek-r1, gpt-oss)
    # stream their chain of thought in a separate `thinking` field and only
    # then start the answer, so on "on" the whole of max_tokens can be spent
    # thinking and the request ends with empty content. This app wants strict
    # JSON and runnable Python, not reasoning prose, so "auto" turns thinking
    # off on every model that advertises the capability and leaves models
    # without it untouched (Ollama rejects the flag on those).
    think: str = "auto"
    # A ceiling, not a target: the model decides how many cases the
    # requirement needs, and this only bounds the response length.
    max_test_cases: int = 20
    script_max_tokens: int = 3072


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
    "static_parameter_top_k": "static_parameter_top_k",
}
_GENERATION_KEYS = {
    "ollama_host": "ollama_host",
    "llm_model": "model",
    "llm_temperature": "temperature",
    "llm_top_p": "top_p",
    "llm_max_tokens": "max_tokens",
    "llm_num_ctx": "num_ctx",
    "llm_request_timeout_seconds": "request_timeout_seconds",
    "llm_stall_timeout_seconds": "stall_timeout_seconds",
    "llm_keep_alive": "keep_alive",
    "llm_num_threads": "num_threads",
    "llm_think": "think",
    "max_test_cases": "max_test_cases",
    "script_max_tokens": "script_max_tokens",
}
@dataclass
class RAGSettings:
    retrieval: RetrievalConfig
    generation: GenerationConfig
    base_dir: Path

    @classmethod
    def load(cls, config_path: str | Path) -> "RAGSettings":
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        base_dir = config_path.resolve().parent.parent

        retrieval = RetrievalConfig()
        generation = GenerationConfig()

        for key, value in raw.items():
            if key in _RETRIEVAL_KEYS:
                setattr(retrieval, _RETRIEVAL_KEYS[key], value)
            elif key in _GENERATION_KEYS:
                setattr(generation, _GENERATION_KEYS[key], value)
            else:
                raise ValueError(f"Unknown config key: {key!r}")

        path = Path(retrieval.chroma_persist_dir)
        retrieval.chroma_persist_dir = (
            path if path.is_absolute() else (base_dir / path).resolve()
        )
        return cls(
            retrieval=retrieval,
            generation=generation,
            base_dir=base_dir,
        )
