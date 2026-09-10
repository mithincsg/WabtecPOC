from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from kb_ingestion.config import PipelineConfig
from kb_ingestion.pipeline import IngestionPipeline, discover_files, extract_and_normalize

from .keyword_index import KeywordHit, KeywordIndex

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


@dataclasses.dataclass(frozen=True)
class _Example:
    """One requirement -> reference test cases -> reference script triple."""

    requirement_id: str
    requirement: str
    test_cases: str
    script: str


def _read_examples(config: PipelineConfig) -> list[_Example]:
    examples_dir = config.examples_dir
    if not examples_dir or not examples_dir.is_dir():
        return []

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

    examples = [
        _Example(
            requirement_id=requirement_id,
            requirement=_joined_text(requirement_texts[requirement_id], config)
            if requirement_id in requirement_texts
            else "",
            test_cases=_joined_text(test_case_files[requirement_id], config)
            if requirement_id in test_case_files
            else "",
            script=_joined_text(script_files[requirement_id], config)
            if requirement_id in script_files
            else "",
        )
        for requirement_id in requirement_ids
    ]
    logger.info("Loaded %d example(s) from %s", len(examples), examples_dir)
    return examples


def _joined_text(path: Path, config: PipelineConfig) -> str:
    units = extract_and_normalize(path, config)
    return "\n\n".join(u.text.strip() for u in units if u.text.strip())


def _budgeted(text: str, max_chars: int) -> str:
    """Trims to a whole line inside `max_chars`.

    Examples are shape templates, so a truncated one still does its job:
    the first N lines of a datasheet or a script show the phrasing and the
    layout just as well as all of them. Overrunning the model's context
    window does not — Ollama silently drops the *start* of the prompt, which
    is where the requirement, the retrieved facts and the rules that forbid
    copying example values live.
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars // 2:
        cut = cut[:newline]
    return cut.rstrip() + "\n[... example truncated - shape only, not a complete artefact]"


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
    Examples folder, included as few-shot material under a character budget
    (see `_budgeted` for why a budget rather than the whole folder).

    Built once, at construction — same lifetime as the running backend
    process. Restart the server to pick up edits to data/python_apis,
    data/track_data, or data/Examples.
    """

    def __init__(self, ingestion_config: PipelineConfig):
        self._config = ingestion_config
        self._keyword_index = KeywordIndex(_build_static_chunks(ingestion_config))
        self._examples = _read_examples(ingestion_config)

    def test_case_examples(self, max_chars: int) -> str:
        """Requirement -> reference test cases, for the datasheet prompt.

        Reference *scripts* are deliberately left out: they are the shape
        template for the script call, and in the datasheet prompt they are
        tens of thousands of characters that teach nothing about datasheet
        phrasing while pushing the requirement out of the context window.
        """
        return self._render(max_chars, lambda e: e.test_cases, "Reference test case(s)")

    def script_examples(self, max_chars: int) -> str:
        """Requirement -> reference script, for the script prompt."""
        return self._render(max_chars, lambda e: e.script, "Reference script")

    def _render(self, max_chars: int, select, label: str) -> str:
        usable = [e for e in self._examples if select(e).strip()]
        if not usable or max_chars <= 0:
            return ""
        # An equal share each, so one long example (L2R7983's script is five
        # times the size of the others) cannot crowd the rest out entirely.
        share = max(200, max_chars // len(usable))
        blocks = []
        for example in usable:
            sections = [f"### Example: {example.requirement_id}"]
            if example.requirement:
                sections.append(
                    "Requirement:\n" + _budgeted(example.requirement, share // 4)
                )
            sections.append(f"{label}:\n" + _budgeted(select(example), share))
            blocks.append("\n\n".join(sections))
        return "\n\n---\n\n".join(blocks)

    def api_context(self, query: str, top_k: int) -> str:
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "python_apis"
        )
        return _format_hits(hits)

    def track_context(self, query: str, top_k: int) -> str:
        """Searches every subdivision in data/track_data/ for whichever rows
        best match the query — there is no per-requirement subdivision
        mapping, so any subdivision can surface values for any requirement.
        """
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "track_data"
        )
        return _format_hits(hits)
