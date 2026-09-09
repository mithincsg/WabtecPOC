from __future__ import annotations

import re
from dataclasses import dataclass

from kb_ingestion.chunking import normalize_text

# A requirement ID as used across this project's data ("L2R9479",
# "L2R1145424", "L2R9479_A"): a short letter prefix, then digits, then
# optional trailing word characters.
_ID_PATTERN = r"([A-Za-z]{1,6}\d[\w.-]*)"

# The common case: the user pastes "<ID> <text>".
_ID_AT_START_RE = re.compile(rf"^{_ID_PATTERN}\b")

# Fallback for a pasted requirement document, which leads with section
# headings ("15 Speed Enforcement", "15.1.1 Track Based Speed Restrictions")
# before a line holding just the ID, sometimes with list numbering
# ("2.     L2R7983"), then the requirement body. Searched anywhere in the
# text so preamble doesn't hide it. Some requirement files carry the ID only
# in the filename, so finding nothing here is a legitimate result rather
# than a parse failure.
_ID_OWN_LINE_RE = re.compile(rf"(?m)^\s*(?:\d+[.)]\s+)?{_ID_PATTERN}\s*$")

# The functional area a requirement belongs to, which becomes the datasheet's
# "Folder" column. Requirement documents name it in a heading
# ("15 Speed Enforcement"), so it's read out rather than configured.
_AREA_HEADING_RE = re.compile(
    r"(?m)^\s*\d+(?:\.\d+)*\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s*$"
)


@dataclass(frozen=True)
class ParsedRequirement:
    requirement_id: str | None
    raw_text: str
    query_text: str
    functional_area: str | None


def parse_requirement(raw_text: str) -> ParsedRequirement:
    """The "requirement understanding" step ahead of retrieval: pulls out the
    requirement ID (for citation, for the track-data mapping, and for the
    datasheet's Requirement column) and the functional area, then normalizes
    the body the same way ingested documents were normalized so the query is
    embedded consistently with the knowledge base.
    """
    cleaned = raw_text.strip()

    start_match = _ID_AT_START_RE.match(cleaned)
    if start_match:
        requirement_id = start_match.group(1)
        body = cleaned[start_match.end():].strip() or cleaned
    else:
        own_line = _ID_OWN_LINE_RE.search(cleaned)
        requirement_id = own_line.group(1) if own_line else None
        # Leading heading text stays in the query — it's domain context for
        # the embedding, not noise.
        body = cleaned

    return ParsedRequirement(
        requirement_id=requirement_id,
        raw_text=raw_text,
        query_text=normalize_text(body),
        functional_area=_detect_functional_area(cleaned),
    )


def _detect_functional_area(text: str) -> str | None:
    """The first numbered title-case heading, e.g. "15 Speed Enforcement" ->
    "Speed Enforcement". The top-level heading is the functional area; deeper
    ones name sub-behaviours, so the shallowest match wins.
    """
    best: tuple[int, str] | None = None
    for match in _AREA_HEADING_RE.finditer(text):
        depth = match.group(0).strip().split()[0].count(".")
        if best is None or depth < best[0]:
            best = (depth, match.group(1).strip())
        if depth == 0:
            break
    return best[1] if best else None
