from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceRoot:
    """One folder of source documents and the document_type its chunks get.

    Separate roots exist because each holds a different kind of thing parsed
    a different way — knowledge-base documents, Python API stubs, track-data
    HTML reports. Which is which comes from config/config.yaml, never from a
    hardcoded folder name in the code.
    """

    path: Path
    document_type: str
    # When true, an immediate subfolder name replaces document_type for files
    # under it (knowledge_base/reference_test_cases/x.xlsx ->
    # "reference_test_cases"), so documents can be grouped for filtering
    # without a config change per file.
    subfolders_as_type: bool = False

    def type_for(self, file_path: Path) -> str:
        if not self.subfolders_as_type:
            return self.document_type
        relative = file_path.relative_to(self.path)
        return relative.parts[0] if len(relative.parts) > 1 else self.document_type


@dataclass
class PipelineConfig:
    sources: list[SourceRoot] = field(default_factory=list)
    # Parsed the same way as `sources`, but never embedded/upserted — read
    # straight from disk into a BM25-only index by src/rag/static_context.py.
    static_sources: list[SourceRoot] = field(default_factory=list)
    # Few-shot example folder, always included in full rather than searched.
    examples_dir: Path | None = None

    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    xlsx_max_rows_per_chunk: int = 1

    pdf_heading_size_ratio: float = 1.15
    pdf_table_repeat_threshold: float = 0.5

    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    embedding_device: str = "auto"

    chroma_persist_dir: Path = Path("chroma_db")
    chroma_collection: str = "kb_collection"

    # Directory the repo root resolves to; relative paths above are resolved
    # against it so the pipeline works from any working directory.
    base_dir: Path = Path(".")

    @classmethod
    def load(cls, config_path: str | Path) -> "PipelineConfig":
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        base_dir = config_path.resolve().parent.parent

        cfg = cls(base_dir=base_dir)
        raw_sources = raw.pop("sources", None)
        raw_static_sources = raw.pop("static_sources", None)
        raw_examples_dir = raw.pop("examples_dir", None)

        for key, value in raw.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key: {key!r}")
            setattr(cfg, key, value)

        if not raw_sources:
            raise ValueError(f"{config_path} defines no `sources:` to ingest from.")
        cfg.sources = [cfg._build_source(entry) for entry in raw_sources]
        cfg.static_sources = [cfg._build_source(entry) for entry in raw_static_sources or []]
        cfg.examples_dir = cfg.resolve(raw_examples_dir) if raw_examples_dir else None
        cfg.chroma_persist_dir = cfg.resolve(cfg.chroma_persist_dir)
        return cfg

    def _build_source(self, entry: dict) -> SourceRoot:
        try:
            path = entry["path"]
            document_type = entry["document_type"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Each entry under `sources:` needs a `path` and a `document_type`."
            ) from exc
        return SourceRoot(
            path=self.resolve(path),
            document_type=document_type,
            subfolders_as_type=bool(entry.get("subfolders_as_type", False)),
        )

    def resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (self.base_dir / path).resolve()
