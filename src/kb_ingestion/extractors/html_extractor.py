from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from . import ExtractedUnit

# The reports identify their subdivision three ways: a body line
# ("... subdivision 8880, revision 803"), the document title
# ("UP.08880.803.opk"), and a quoted display name ("Spokane"). The numeric ID
# is the one that matters — it's what the requirement -> track map is keyed
# on and what the automation scripts pass as subdivID — so it's read first
# and the display name is kept alongside it for the citation label.
_SUBDIVISION_ID_RE = re.compile(r"subdivision\s+(\d{3,6})\b", re.IGNORECASE)
_TITLE_ID_RE = re.compile(r"<title>[^<]*?\.?(\d{4,6})\.\d+", re.IGNORECASE)
_SUBDIVISION_NAME_RE = re.compile(r'Subdivision\s*"([^"]+)"')
_STEM_ID_RE = re.compile(r"(\d{3,6})")
_WHITESPACE_RE = re.compile(r"\s+")
_MIN_TABLE_ROWS = 2


class _TrackReportParser(HTMLParser):
    """Parses a Wabtec 'HTML Track by Group' subdivision report: a sequence
    of <a name="Section"></a><table>...</table> blocks, each table's first
    row being column labels and the rest being track-feature records
    (track name, milepost, block, signal/switch, speed restriction, etc.).

    Some cells (e.g. a switch's "Turnout Leg" Facing/Normal/Reverse block
    references) render their content as a nested <table> rather than plain
    text. `_table_depth` tracks that nesting so only the outermost table for
    a named section drives row/column extraction - a nested table's rows
    are flattened into the text of whichever outer cell contains them,
    instead of being mistaken for a new top-level table (which would both
    corrupt the outer table's accumulated rows and desync _in_table/_in_row,
    silently dropping every real row that follows).
    """

    def __init__(self):
        super().__init__()
        self.sections: list[tuple[str, list[list[str]]]] = []
        self._pending_section_name: str | None = None
        self._table_depth = 0
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            name = dict(attrs).get("name")
            if name:
                self._pending_section_name = name
            return

        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
            return

        if tag == "tr":
            if self._table_depth == 1:
                self._in_row = True
                self._current_row = []
            elif self._table_depth > 1 and self._in_cell and self._has_cell_content():
                self._current_cell_parts.append("; ")
            return

        if tag in ("td", "th"):
            if self._table_depth == 1 and self._in_row:
                self._in_cell = True
                self._current_cell_parts = []
            elif self._table_depth > 1 and self._in_cell and self._has_cell_content():
                self._current_cell_parts.append(" ")
            return

        if tag == "br" and self._in_cell:
            self._current_cell_parts.append("; ")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            if self._table_depth == 1 and self._in_cell:
                text = _WHITESPACE_RE.sub(" ", "".join(self._current_cell_parts)).strip()
                self._current_row.append(text)
                self._in_cell = False
            return

        if tag == "tr":
            if self._table_depth == 1 and self._in_row:
                if any(self._current_row):
                    self._current_table.append(self._current_row)
                self._in_row = False
            return

        if tag == "table":
            if self._table_depth == 1:
                if self._pending_section_name and len(self._current_table) >= _MIN_TABLE_ROWS:
                    self.sections.append((self._pending_section_name, self._current_table))
                self._pending_section_name = None
                self._current_table = []
            if self._table_depth > 0:
                self._table_depth -= 1

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell_parts.append(data)

    def _has_cell_content(self) -> bool:
        # Whitespace-only text nodes (HTML indentation) between an outer
        # cell's open tag and a nested table's first row would otherwise
        # make _current_cell_parts non-empty without any real content yet,
        # producing a spurious leading separator.
        return bool("".join(self._current_cell_parts).strip())


def _rows_to_text(rows: list[list[str]]) -> str:
    lines = []
    for row in rows:
        cells = [cell.strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


class HTMLTrackDataExtractor:
    """Wabtec 'HTML Track by Group' subdivision reports — the readable half
    of a subdivision export, indexed alongside the machine-serialized
    `-subdiv.xml` in the same folder (see xml_extractor.py: the report is
    grouped and labelled, the XML carries fields the report never prints).
    Preserves the real hierarchy: subdivision
    -> named section (Blocks, Switches, Signals, Speed Restrictions, ...) ->
    track-feature records. Each section becomes one ExtractedUnit
    (unit_type="table") - identical shape to a PDF table unit, so it flows
    through the same oversized-table line-splitting in chunking with no
    special-casing needed.
    """

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        raw_html = file_path.read_text(encoding="utf-8", errors="replace")

        subdivision = _subdivision_id(raw_html, file_path)
        name_match = _SUBDIVISION_NAME_RE.search(raw_html)
        subdivision_name = name_match.group(1).strip() if name_match else ""
        label = f"{subdivision} {subdivision_name}".strip()

        parser = _TrackReportParser()
        parser.feed(raw_html)

        units: list[ExtractedUnit] = []
        for index, (section_name, rows) in enumerate(parser.sections, start=1):
            text = _rows_to_text(rows)
            if not text.strip():
                continue
            units.append(
                ExtractedUnit(
                    text=text,
                    unit_type="table",
                    locator={
                        "section_path": f"{label} > {section_name}",
                        "table_title": section_name,
                        "table": index,
                        # Filterable on its own, so the requirement ->
                        # subdivision picked in the UI can
                        # restrict retrieval to the track a requirement is
                        # actually tested on.
                        "subdivision": subdivision,
                        "subdivision_name": subdivision_name or None,
                    },
                )
            )
        return units


def _subdivision_id(raw_html: str, file_path: Path) -> str:
    """The subdivision's numeric ID, normalised to the zero-padded form used
    in filenames and the track map ("08880").

    The containing folder wins when there is one: track data is filed as
    `data/track_data/<subdivision>/`, and that folder is what the UI's
    subdivision picker lists, so a report filed under 08101 must be findable
    under 08101 even if whoever exported it left a different ID in the body.
    Failing that, tried in order of reliability: the body line that states it,
    the document title, then the filename — any one of those can be missing or
    reformatted, so all are checked rather than assuming a single layout.
    """
    from_folder = subdivision_from_folder(file_path)
    if from_folder:
        return from_folder

    for pattern, source in (
        (_SUBDIVISION_ID_RE, raw_html),
        (_TITLE_ID_RE, raw_html),
        (_STEM_ID_RE, file_path.stem),
    ):
        match = pattern.search(source)
        if match:
            return normalize_subdivision(match.group(1))
    return file_path.stem


def subdivision_from_folder(file_path: Path) -> str:
    """The subdivision ID from the folder a track file sits in, or "" if that
    folder isn't named after one.

    Track data is organised one folder per subdivision
    (`data/track_data/08101/08101.803.html`, `.../08101-subdiv.xml`), which
    makes the folder the single most reliable statement of which subdivision a
    file belongs to — and the one thing the HTML report and its XML sibling
    are guaranteed to agree on.
    """
    name = file_path.parent.name.strip()
    return normalize_subdivision(name) if name.isdigit() else ""


def normalize_subdivision(value: str) -> str:
    """"8880", "08880" and " 8880 " all name the same subdivision. Padding to
    five digits gives one canonical form, so a track map entry written either
    way still matches what was ingested.
    """
    digits = "".join(c for c in str(value) if c.isdigit())
    return digits.zfill(5) if digits else str(value).strip()
