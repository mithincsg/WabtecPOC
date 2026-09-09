from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from kb_ingestion.config import PipelineConfig
from kb_ingestion.pipeline import IngestionPipeline, discover_files, extract_and_normalize

from .keyword_index import KeywordHit, KeywordIndex
from .track_mapping import TrackMapping

logger = logging.getLogger(__name__)


class _StaticDocumentStore:
    """Just enough of VectorStore's interface for KeywordIndex to build a
    BM25 corpus over — backed by chunks read straight from disk rather than
    a ChromaDB collection, since these sources are never embedded.
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


def _requirement_key(stem: str) -> str:
    return stem.split("_", 1)[0].strip().upper()


def _read_examples(config: PipelineConfig) -> str:
    examples_dir = config.examples_dir
    if not examples_dir or not examples_dir.is_dir():
        return ""

    requirement_texts: dict[str, Path] = {}
    for path in sorted(examples_dir.glob("*.txt")):
        requirement_texts[_requirement_key(path.stem)] = path

    test_case_files: dict[str, Path] = {}
    for path in sorted((examples_dir / "reference_test_cases").glob("*.xlsx")):
        test_case_files[_requirement_key(path.stem)] = path

    script_files: dict[str, Path] = {}
    for path in sorted((examples_dir / "reference_test_scripts").glob("*.txt")):
        script_files[_requirement_key(path.stem)] = path

    requirement_ids = sorted(
        set(requirement_texts) | set(test_case_files) | set(script_files)
    )

    blocks: list[str] = []
    for requirement_id in requirement_ids:
        sections = [f"### Example: {requirement_id}"]
        if requirement_id in requirement_texts:
            sections.append("Requirement:\n" + _joined_text(requirement_texts[requirement_id], config))
        if requirement_id in test_case_files:
            sections.append(
                "Reference test case(s):\n" + _joined_text(test_case_files[requirement_id], config)
            )
        if requirement_id in script_files:
            sections.append(
                "Reference script:\n" + _joined_text(script_files[requirement_id], config)
            )
        blocks.append("\n\n".join(sections))

    logger.info("Loaded %d example(s) from %s", len(blocks), examples_dir)
    return "\n\n---\n\n".join(blocks)


def _joined_text(path: Path, config: PipelineConfig) -> str:
    units = extract_and_normalize(path, config)
    return "\n\n".join(u.text.strip() for u in units if u.text.strip())


def _format_hits(hits: list[KeywordHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source_file") or hit.metadata.get("source_path") or "unknown"
        locator = hit.metadata.get("method_name") or hit.metadata.get("section_path") or ""
        label = f"{source} > {locator}" if locator else source
        blocks.append(f"[{index}] {hit.metadata.get('document_type', 'static')} | {label}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(blocks)


class StaticContextProvider:
    """python_apis and track_data, keyword-searched from an in-process BM25
    index built straight off disk (never embedded, never in ChromaDB); the
    Examples folder, always included in full as few-shot material.

    Built once, at construction — same lifetime as the running backend
    process. Restart the server to pick up edits to data/python_apis,
    data/track_data, or data/Examples.
    """

    def __init__(self, ingestion_config: PipelineConfig, track_mapping: TrackMapping):
        self._config = ingestion_config
        self._track_mapping = track_mapping
        self._keyword_index = KeywordIndex(_build_static_chunks(ingestion_config))
        self.examples_context = _read_examples(ingestion_config)

    def api_context(self, query: str, top_k: int) -> str:
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "python_apis"
        )
        return _format_hits(hits)

    def track_context(self, query: str, requirement_id: str | None, top_k: int) -> str:
        if top_k <= 0:
            return ""
        selection = self._track_mapping.selection_for(requirement_id)
        if not selection.enabled:
            return ""

        subdivisions = set(selection.subdivisions) if not selection.search_all else None

        def predicate(metadata: dict[str, Any]) -> bool:
            if metadata.get("document_type") != "track_data":
                return False
            if subdivisions is not None and metadata.get("subdivision") not in subdivisions:
                return False
            return True

        hits = self._keyword_index.search(query, top_k, predicate=predicate)
        return _format_hits(hits)
