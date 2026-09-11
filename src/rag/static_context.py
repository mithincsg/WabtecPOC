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


def _read_examples(config: PipelineConfig) -> dict[str, dict[str, str]]:
    """One entry per example requirement, with the requirement text, the
    reference test cases and the reference script kept apart.

    Kept apart because the two generators need different halves: writing
    datasheet rows needs the reference *test cases* for their phrasing, and
    writing a script needs the reference *script* for its layout. Handing
    each prompt all of both is what pushed these prompts past llm_num_ctx,
    at which point Ollama truncates from the start of the prompt — silently
    dropping the very rules that say not to copy values out of the examples.
    """
    examples_dir = config.examples_dir
    if not examples_dir or not examples_dir.is_dir():
        return {}

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

    examples: dict[str, dict[str, str]] = {}
    for requirement_id in requirement_ids:
        parts: dict[str, str] = {}
        if requirement_id in requirement_texts:
            parts["requirement"] = _joined_text(requirement_texts[requirement_id], config)
        if requirement_id in test_case_files:
            parts["test_cases"] = _joined_text(test_case_files[requirement_id], config)
        if requirement_id in script_files:
            parts["script"] = _joined_text(script_files[requirement_id], config)
        examples[requirement_id] = parts

    logger.info("Loaded %d example(s) from %s", len(examples), examples_dir)
    return examples


def _joined_text(path: Path, config: PipelineConfig) -> str:
    units = extract_and_normalize(path, config)
    return "\n\n".join(u.text.strip() for u in units if u.text.strip())


def _budgeted(text: str, max_chars: int | None) -> str:
    """Caps a context block at a character budget, cutting on a line boundary
    and saying so. These blocks are templates, so a trailing cut costs the
    model the end of an example rather than any fact it needs — and it is far
    better than letting the whole prompt overrun llm_num_ctx, which costs the
    *start* of the prompt, rules included.
    """
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars].rsplit("\n", 1)[0]
    return head + "\n# ... reference example truncated to fit the context window ..."


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
    Examples folder, served as few-shot *template* material only.

    Built once, at construction — same lifetime as the running backend
    process. Restart the server to pick up edits to data/python_apis,
    data/track_data, or data/Examples.
    """

    def __init__(self, ingestion_config: PipelineConfig, track_mapping: TrackMapping):
        self._config = ingestion_config
        self._track_mapping = track_mapping
        self._keyword_index = KeywordIndex(_build_static_chunks(ingestion_config))
        self._examples = _read_examples(ingestion_config)
        self._script_example_index = self._build_script_example_index()

    def _build_script_example_index(self) -> KeywordIndex:
        """A tiny BM25 index over the reference scripts, so script generation
        can include the one or two whose subject matter actually resembles the
        requirement instead of all of them. Every extra script in the prompt
        is both wasted context and one more set of foreign block numbers for
        the model to be tempted by.
        """
        ids, texts, metadatas = [], [], []
        for requirement_id, parts in self._examples.items():
            if not parts.get("script"):
                continue
            ids.append(requirement_id)
            texts.append("\n".join(filter(None, (parts.get("requirement"), parts["script"]))))
            metadatas.append({"requirement_id": requirement_id, "document_type": "example_script"})
        return KeywordIndex(_StaticDocumentStore(ids, texts, metadatas))

    def test_case_examples(self, max_chars: int | None = None) -> str:
        """Requirement text + reference test cases, no scripts: a datasheet
        row is written after the example rows' phrasing, and the automation
        scripts contribute nothing to that but bulk.

        The budget is divided per example rather than applied to the joined
        string, so a long first example cannot push the later ones out of the
        prompt entirely. A few rows from each of four requirements teaches the
        house phrasing better than every row of one of them.
        """
        usable = [
            (requirement_id, parts)
            for requirement_id, parts in self._examples.items()
            if parts.get("requirement") or parts.get("test_cases")
        ]
        per_example = (
            max_chars // len(usable) if max_chars and usable else max_chars
        )

        blocks = []
        for requirement_id, parts in usable:
            sections = [f"### Example: {requirement_id}"]
            if parts.get("requirement"):
                sections.append("Requirement:\n" + parts["requirement"])
            if parts.get("test_cases"):
                sections.append("Reference test case(s):\n" + parts["test_cases"])
            blocks.append(_budgeted("\n\n".join(sections), per_example))
        return "\n\n---\n\n".join(blocks)

    def script_examples(
        self, query: str, top_k: int = 1, max_chars: int | None = None
    ) -> str:
        """The most relevant reference script(s), for layout only."""
        if top_k <= 0:
            return ""
        hits = self._script_example_index.search(query, top_k)
        requirement_ids = [hit.metadata["requirement_id"] for hit in hits] or list(
            self._examples
        )[:top_k]

        blocks = []
        for requirement_id in requirement_ids:
            parts = self._examples.get(requirement_id, {})
            if not parts.get("script"):
                continue
            sections = [f"### Example: {requirement_id}"]
            if parts.get("requirement"):
                sections.append("Requirement:\n" + parts["requirement"])
            sections.append("Reference script:\n" + parts["script"])
            blocks.append("\n\n".join(sections))
        return _budgeted("\n\n---\n\n".join(blocks), max_chars)

    def api_context(self, query: str, top_k: int) -> str:
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "python_apis"
        )
        return _format_hits(hits)

    def track_context(
        self,
        query: str,
        requirement_id: str | None,
        top_k: int,
        subdivision_id: str | None = None,
    ) -> str:
        if top_k <= 0:
            return ""
        selection = self._track_mapping.selection_for(requirement_id, subdivision_id)
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
