from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from kb_ingestion.config import PipelineConfig
from kb_ingestion.extractors.html_extractor import normalize_subdivision
from kb_ingestion.pipeline import IngestionPipeline, discover_files

from .keyword_index import KeywordHit, KeywordIndex

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class Subdivision:
    """One subdivision folder under data/track_data, as the UI lists it."""

    id: str
    name: str
    chunks: int

    @property
    def label(self) -> str:
        return f"{self.id} {self.name}".strip()


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


def _list_subdivisions(store: _StaticDocumentStore) -> list[Subdivision]:
    _ids, _texts, metadatas = store.get_all_documents()
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for metadata in metadatas:
        if metadata.get("document_type") != "track_data":
            continue
        subdivision = str(metadata.get("subdivision") or "").strip()
        if not subdivision:
            continue
        counts[subdivision] = counts.get(subdivision, 0) + 1
        # The HTML report carries the display name ("Ginger"); its XML sibling
        # may not. Either file is enough to name the subdivision, so the first
        # one that has it wins for the whole folder.
        name = str(metadata.get("subdivision_name") or "").strip()
        if name and not names.get(subdivision):
            names[subdivision] = name

    found = [
        Subdivision(id=subdivision, name=names.get(subdivision, ""), chunks=count)
        for subdivision, count in sorted(counts.items())
    ]
    logger.info(
        "Track data covers %d subdivision(s): %s",
        len(found),
        ", ".join(s.label for s in found) or "none",
    )
    return found


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
    index built straight off disk (never embedded, never in ChromaDB).

    track_data is one folder per subdivision, each holding the HTML report and
    its `-subdiv.xml` sibling; both are indexed and both are stamped with the
    folder's subdivision, so filtering to a subdivision covers the pair.

    `data/Examples/` is deliberately *not* loaded here. Sending those files
    at request time cost ~8-10k characters of prompt on every generation —
    all of it prefill, all of it identical from one request to the next —
    to teach the model a shape that does not change. The shape has instead
    been distilled by hand into the house wording pattern and house
    skeleton in config/prompts.yaml. The folder stays in the repo as the
    source those templates were derived from and the material to re-derive
    them from when the house style changes; it is design-time input now,
    not runtime input.

    Built once, at construction — same lifetime as the running backend
    process. Restart the server to pick up edits to data/python_apis or
    data/track_data.
    """

    def __init__(
        self,
        ingestion_config: PipelineConfig,
    ):
        self._config = ingestion_config
        store = _build_static_chunks(ingestion_config)
        self._keyword_index = KeywordIndex(store)
        self._subdivisions = _list_subdivisions(store)

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
        top_k: int,
        subdivision: str | None,
    ) -> str:
        return self.track_context_for(query, top_k, subdivision)[0]

    def track_context_for(
        self,
        query: str,
        top_k: int,
        subdivision: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        """Track context for the subdivision picked in the UI, plus which
        subdivisions it was actually drawn from.

        The picked subdivision is the only thing that decides this. There is
        no requirement -> subdivision map any more: a stored map and a UI
        picker are two answers to one question, and the one the user just
        chose in front of the result has to win, so keeping both only created
        a way for the control to appear to do nothing.

        No subdivision means no track data, deliberately. Searching every
        subdivision instead would let BM25 return blocks and mileposts
        belonging to track this requirement is not tested on — a wrong value
        that looks right, which is worse than the `# TODO:` the generator
        writes when the track block is empty.

        The caller gets the subdivisions back rather than re-deriving them,
        because what was searched is reported in the UI and must be the same
        set that produced these hits.
        """
        if top_k <= 0 or not subdivision:
            return "", ()

        chosen = normalize_subdivision(subdivision)
        if not chosen:
            return "", ()

        def predicate(metadata: dict[str, Any]) -> bool:
            return (
                metadata.get("document_type") == "track_data"
                and metadata.get("subdivision") == chosen
            )

        hits = self._keyword_index.search(query, top_k, predicate=predicate)
        return _format_hits(hits), (chosen,)

    @property
    def subdivisions(self) -> list[Subdivision]:
        """Every subdivision the static index actually holds track data for —
        read off the indexed chunks rather than by listing the folder, so the
        dropdown can only ever offer track the search can really return.
        """
        return self._subdivisions
