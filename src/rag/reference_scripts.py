from __future__ import annotations

import logging
import re
from pathlib import Path

from .keyword_index import KeywordHit, KeywordIndex

logger = logging.getLogger(__name__)

# Same normalization requirement IDs need everywhere they're used as a join
# key: parse_requirement's ID pattern is generic and can capture trailing
# noise ("L2R9479_A"), and the example filenames themselves carry a "_A"
# suffix. Bare "L2R<digits>" is the only thing both sides are guaranteed to
# share.
_BARE_ID_RE = re.compile(r"(L2R\d+)", re.IGNORECASE)

# How much of a matched reference script to inject. The choreography and
# call-shape patterns a script needs to learn live in the setup/pre-condition
# block and one full branch - the remaining branches in a real script are
# largely repetitions of that same pattern over different values, which cost
# prompt tokens without teaching anything new. Cutting a fixed budget of the
# most valuable prefix, rather than the whole file, is the deliberate
# token/accuracy trade-off requested when this was implemented.
_MAX_CHARS = 3000


def _bare_id(text: str) -> str | None:
    match = _BARE_ID_RE.search(text or "")
    return match.group(1).upper() if match else None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    return text[:cut] + "\n# [reference truncated here - remaining branches omitted]"


class _TinyStore:
    """Just enough of VectorStore's interface for KeywordIndex to build a
    BM25 corpus over a handful of in-memory documents, mirroring
    `_StaticDocumentStore` in static_context.py but without reading anything
    from disk itself - the 3 requirement texts are already loaded by the
    caller.
    """

    def __init__(self, ids: list[str], texts: list[str]):
        self._ids = ids
        self._texts = texts

    def count(self) -> int:
        return len(self._ids)

    def get_all_documents(self):
        return self._ids, self._texts, [{} for _ in self._ids]


class ReferenceScriptProvider:
    """A real, approved script for the most relevant past requirement in
    `data/Examples/reference_test_scripts/`, for teaching the script-
    generation call how a real script sequences calls together - the
    choreography no amount of retrieved API stubs alone can show, since a
    stub documents one call in isolation, never how several are chained.

    Exact-ID lookup first (the common case: re-generating a script for the
    same requirement after test-case edits), then a small BM25 fallback over
    the 3 requirement texts for a similar-but-different requirement. Below a
    minimum relevance floor, returns nothing rather than forcing an
    unrelated example into the prompt - a mismatched reference risks the
    model imitating a pattern that doesn't apply, which is worse than no
    reference at all.

    Deliberately not routed through kb_ingestion: with 3 files, chunking
    would fragment a whole script into disconnected prose pieces, the
    opposite of what's needed. This mirrors CafMapping's shape instead - a
    small, non-chunked, exact-keyed side table read once at construction.
    """

    def __init__(self, examples_dir: Path | None):
        self._scripts: dict[str, str] = {}
        summaries_ids: list[str] = []
        summaries_texts: list[str] = []

        if examples_dir is not None and examples_dir.is_dir():
            scripts_dir = examples_dir / "reference_test_scripts"
            for path in sorted(scripts_dir.glob("*.txt")) if scripts_dir.is_dir() else []:
                req_id = _bare_id(path.stem)
                if req_id:
                    self._scripts[req_id] = path.read_text(encoding="utf-8", errors="replace")

            for path in sorted(examples_dir.glob("L2R*.txt")):
                req_id = _bare_id(path.stem)
                if req_id and req_id in self._scripts and req_id not in summaries_ids:
                    summaries_ids.append(req_id)
                    summaries_texts.append(path.read_text(encoding="utf-8", errors="replace"))

        self._index = KeywordIndex(_TinyStore(summaries_ids, summaries_texts))
        logger.info(
            "Reference-script provider: %d example script(s) available (%s)",
            len(self._scripts),
            ", ".join(sorted(self._scripts)) or "none",
        )

    def best_match(self, query: str, requirement_id: str) -> tuple[str, str] | None:
        exact = _bare_id(requirement_id or "")
        if exact and exact in self._scripts:
            return exact, _truncate(self._scripts[exact], _MAX_CHARS)

        hits: list[KeywordHit] = self._index.search(query, top_k=1, predicate=None)
        if not hits or hits[0].score <= 0:
            return None
        matched_id = hits[0].chunk_id
        script = self._scripts.get(matched_id)
        if script is None:
            return None
        return matched_id, _truncate(script, _MAX_CHARS)
