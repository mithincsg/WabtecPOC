from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

from kb_ingestion.config import PipelineConfig
from kb_ingestion.extractors.xml_extractor import normalize_subdivision
from kb_ingestion.pipeline import ChunkingPipeline, discover_files

from .keyword_index import KeywordHit, KeywordIndex

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class Subdivision:
    """One subdivision folder under data/track_data, as the UI lists it.

    Identified by its number and nothing else: the display names the old HTML
    reports carried ("Ginger") are not in the XML export, and a picker that
    shows a name for some subdivisions and a bare number for others reads as
    a gap in the data rather than a choice.
    """

    id: str
    chunks: int


class _StaticDocumentStore:
    """The corpus KeywordIndex builds its BM25 index over: every chunk of
    every configured source, read straight from disk at startup.
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
    pipeline = ChunkingPipeline(config)

    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for file in discover_files(config):
        try:
            _hash, chunks = pipeline.prepare_file(file)
        except Exception:
            logger.exception("Failed to parse static-context file %s", file.path)
            continue
        for chunk in chunks:
            chunk_ids.append(chunk.metadata.chunk_id)
            texts.append(chunk.text)
            metadatas.append(chunk.metadata.to_dict())

    logger.info("Built static context over %d chunks", len(chunk_ids))
    return _StaticDocumentStore(chunk_ids, texts, metadatas)


def _list_subdivisions(store: _StaticDocumentStore) -> list[Subdivision]:
    _ids, _texts, metadatas = store.get_all_documents()
    counts: dict[str, int] = {}
    for metadata in metadatas:
        if metadata.get("document_type") != "track_data":
            continue
        subdivision = str(metadata.get("subdivision") or "").strip()
        if not subdivision:
            continue
        counts[subdivision] = counts.get(subdivision, 0) + 1

    found = [
        Subdivision(id=subdivision, chunks=count)
        for subdivision, count in sorted(counts.items())
    ]
    logger.info(
        "Track data covers %d subdivision(s): %s",
        len(found),
        ", ".join(s.id for s in found) or "none",
    )
    return found


# Identifiers as a requirement writes them: "TBC137", "CFG 65", "THE12".
_PARAMETER_ID_RE = re.compile(r"\b(TBC|CFG|THE)\s*(\d+)\b", re.IGNORECASE)


def _parameter_ids(text: str) -> list[str]:
    ids: list[str] = []
    for prefix, number in _PARAMETER_ID_RE.findall(text or ""):
        identifier = f"{prefix.upper()}{number}"
        if identifier not in ids:
            ids.append(identifier)
    return ids


def _index_parameter_records(store: _StaticDocumentStore) -> dict[str, KeywordHit]:
    """Parameter records by their identifier, for exact lookup.

    BM25 alone does not guarantee the parameter a requirement names comes
    back: "TBC137" is one rare token against a requirement's worth of common
    ones ("speed", "restricted", "enforce"), and a handful of unrelated
    records that match several of those outscore the one record that matters.
    The identifiers stated in the requirement are not a ranking question, so
    they are looked up directly instead.
    """
    ids, texts, metadatas = store.get_all_documents()
    records: dict[str, KeywordHit] = {}
    for chunk_id, text, metadata in zip(ids, texts, metadatas):
        if (metadata or {}).get("document_type") != "parameter_config":
            continue
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        found = _parameter_ids(first_line.split("|")[0])
        if len(found) == 1 and found[0] not in records:
            records[found[0]] = KeywordHit(
                chunk_id=chunk_id, text=text, metadata=metadata or {}, score=0.0
            )
    return records


def _format_hits(hits: list[KeywordHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source_file") or hit.metadata.get("source_path") or "unknown"
        locator = hit.metadata.get("method_name") or hit.metadata.get("section_path") or ""
        label = f"{source} > {locator}" if locator else source
        blocks.append(f"[{index}] {hit.metadata.get('document_type', 'static')} | {label}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(blocks)


class StaticContextProvider:
    """python_apis, track_data and parameter_config, keyword-searched from an
    in-process BM25 index built straight off disk. This is the whole of the
    app's retrieval — there is no dense arm and no vector store.

    parameter_config is the parameter configuration guide after
    `scripts/convert_parameter_guide.py` has turned its tables into one JSON
    record per parameter — near-identical rows keyed by exact identifiers,
    which is the same shape as track data and belongs here for the same
    reasons.

    track_data is one folder per subdivision, each holding that subdivision's
    `-subdiv.xml` export. Chunks are stamped with the folder's subdivision,
    which is what the UI's picker filters on.

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
    process. Restart the server to pick up edits to any source folder.
    """

    def __init__(
        self,
        ingestion_config: PipelineConfig,
    ):
        self._config = ingestion_config
        store = _build_static_chunks(ingestion_config)
        self._chunk_count = store.count()
        self._keyword_index = KeywordIndex(store)
        self._subdivisions = _list_subdivisions(store)
        self._parameter_records = _index_parameter_records(store)

    def api_context(self, query: str, top_k: int) -> str:
        if top_k <= 0:
            return ""
        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "python_apis"
        )
        logger.info("Static context (python_apis): %d/%d hit(s)", len(hits), top_k)
        return _format_hits(hits)

    def parameter_context(self, query: str, top_k: int) -> str:
        """TBC/CFG/THE records from the parameter configuration guide, formatted
        for a prompt. See `parameter_hits` for the records themselves.
        """
        return _format_hits(self.parameter_hits(query, top_k))

    def parameter_hits(self, query: str, top_k: int) -> list[KeywordHit]:
        """TBC/CFG/THE records from the parameter configuration guide, looked
        up by the identifiers the requirement states.

        Exact lookup is the whole of it: a requirement that names TBC137 or
        CFG16 has stated its own scope, so those records are the answer and
        `top_k` does not apply. A requirement that names none gets no
        records — see below for why there is no keyword fallback.

        Returned as raw hits, not just the formatted block, so a caller can
        report them as the retrieved context — these are the only chunks
        test-case generation has to show.
        """
        if top_k <= 0:
            return []
        named = [
            self._parameter_records[identifier]
            for identifier in _parameter_ids(query)
            if identifier in self._parameter_records
        ]
        if named:
            # A requirement that names its parameters has already said which
            # ones it is about, so nothing is padded in beside them. BM25
            # ranks the rest on shared prose, and the guide is 900 records of
            # the same prose: on a work-zone requirement naming TBC290 and
            # CFG22, TBC412 ("...calculated position uncertainty of the
            # leading edge of the train...") outscored TBC290's own record
            # and reached the prompt, where every record reads as in-scope.
            # A parameter the requirement never mentions is not a ranking
            # question to be answered less confidently — it is the wrong
            # answer, and one the reviewer cannot spot in a finished case.
            logger.info(
                "Static context (parameter_config): %d record(s) named in the requirement (%s); "
                "no keyword padding",
                len(named),
                ", ".join(_parameter_ids(query)),
            )
            return named

        # No identifier named, no records. BM25 always returns its top_k, so a
        # fallback here cannot answer "no parameter applies" — and the guide is
        # 900 records of one vocabulary ("on-board segment", "train", "track",
        # "message"), so the ranking is decided by shared boilerplate. On
        # L2R8289 (bulletin-dataset validation, which turns on no parameter at
        # all) the top four scored 324/323/319/318 — a 2% spread across four
        # unrelated records. The prompt then presents that block as the only
        # source of TBC/CFG/THE values, so the model writes a case around
        # whichever record the noise returned.
        logger.info(
            "Static context (parameter_config): no records "
            "(the requirement names no TBC/CFG/THE identifier)"
        )
        return []

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
            logger.info("Static context (track_data): skipped, no subdivision picked")
            return "", ()

        chosen = normalize_subdivision(subdivision)
        if not chosen:
            logger.info("Static context (track_data): subdivision %r did not normalize", subdivision)
            return "", ()

        def predicate(metadata: dict[str, Any]) -> bool:
            return (
                metadata.get("document_type") == "track_data"
                and metadata.get("subdivision") == chosen
            )

        hits = self._keyword_index.search(query, top_k, predicate=predicate)
        logger.info(
            "Static context (track_data): %d/%d hit(s) in subdivision %s",
            len(hits),
            top_k,
            chosen,
        )
        return _format_hits(hits), (chosen,)

    @property
    def chunk_count(self) -> int:
        """How much the index holds, for the health endpoint."""
        return self._chunk_count

    @property
    def subdivisions(self) -> list[Subdivision]:
        """Every subdivision the static index actually holds track data for —
        read off the indexed chunks rather than by listing the folder, so the
        dropdown can only ever offer track the search can really return.
        """
        return self._subdivisions
