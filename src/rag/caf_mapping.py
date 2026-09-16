from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The Change Approval Form workbook's header row, wherever it sits: the sheet
# opens with a title line above it, so the header is found by name rather
# than assumed to be row 1.
_SECTION_HEADER = "section"
_FEATURE_HEADER = "feature"
_REQUIREMENT_HEADER = "requirement no"

# A feature name is written once against the first requirement it covers and
# left blank on the rows below it (merged-looking cells). Carrying the last
# seen value down is what makes those rows mapped rather than unmapped.
_BLANKISH = {"", "-", "n/a", "na"}


@dataclass(frozen=True)
class CafEntry:
    """One requirement's row in the Change Approval Form."""

    requirement_id: str
    feature: str
    section: str


class CafMapping:
    """Requirement number -> feature, read from config/caf_mapping.json.

    The datasheet's `Folder` column is the feature a requirement belongs to.
    The Change Approval Form workbook (`data/CAF.xlsx`) is where that
    association is maintained by hand, but it is not read directly at
    request time — `scripts/convert_caf_mapping.py` converts it once into
    `config/caf_mapping.json`, and this class only ever reads that JSON file.
    That keeps a request-time lookup to a small, dependency-free JSON parse
    instead of an openpyxl workbook read, and makes the mapping a normal
    config file: diffable, and reloadable without a restart the same way
    `config/caf_mapping.json` is.

    Whenever `data/CAF.xlsx` changes, re-run the conversion script — this
    class does not watch the workbook, only the JSON file it produces.
    Reloaded when the JSON file's mtime changes.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._entries: dict[str, CafEntry] = {}
        self._mtime: float | None = None

    # --- lookup ------------------------------------------------------------

    def entry_for(self, requirement_id: str | None) -> CafEntry | None:
        if not requirement_id:
            return None
        self._load()

        key = _normalize_id(requirement_id)
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        # "L2R9479_A" is the same requirement as "L2R9479" with a revision
        # suffix — the same fallback the track mapping makes.
        base = key.split("_")[0]
        return self._entries.get(base)

    def folder_for(self, requirement_id: str | None) -> str:
        entry = self.entry_for(requirement_id)
        return entry.feature if entry else ""

    @property
    def known_requirement_ids(self) -> list[str]:
        self._load()
        return sorted(self._entries)

    # --- loading -----------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            # No mapping file is a valid setup — it just means no requirement
            # has a feature mapped yet, and the folder is left empty.
            self._entries, self._mtime = {}, None
            return

        mtime = self.path.stat().st_mtime
        if self._mtime == mtime:
            return

        self._entries = _read_json(self.path)
        self._mtime = mtime
        logger.info("Loaded %d CAF requirement mapping(s) from %s", len(self._entries), self.path)


def _read_json(path: Path) -> dict[str, CafEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, CafEntry] = {}
    for requirement_id, row in (raw.get("requirements") or {}).items():
        key = _normalize_id(requirement_id)
        if key:
            entries[key] = CafEntry(
                requirement_id=key,
                feature=str(row.get("feature", "")),
                section=str(row.get("section", "")),
            )
    return entries


# --- Change Approval Form workbook -> JSON conversion -----------------------
#
# The parsing logic below reads data/CAF.xlsx itself. It lives here so it has
# one home, but at request time only `scripts/convert_caf_mapping.py` calls
# it — CafMapping above never touches the workbook.


def read_caf_workbook(path: str | Path) -> dict[str, CafEntry]:
    """Parses the Change Approval Form workbook into requirement -> feature.

    Used only by `scripts/convert_caf_mapping.py` to produce
    `config/caf_mapping.json`; the running app never calls this.
    """
    import openpyxl

    path = Path(path)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    entries: dict[str, CafEntry] = {}

    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        columns = _find_header(rows)
        if columns is None:
            continue

        section = ""
        feature = ""
        for row in rows:
            section = _carry(row, columns.get(_SECTION_HEADER), section)
            feature = _carry(row, columns.get(_FEATURE_HEADER), feature)

            raw_id = _cell(row, columns.get(_REQUIREMENT_HEADER))
            if not raw_id or not feature:
                continue
            # One cell occasionally lists more than one identifier.
            for requirement_id in re.split(r"[,;/]| and ", raw_id):
                key = _normalize_id(requirement_id)
                if key:
                    entries.setdefault(key, CafEntry(key, feature, section))

    workbook.close()
    return entries


def _find_header(rows) -> dict[str, int] | None:
    """Consumes rows until the header row, returning header name -> index.

    Scans a few rows rather than assuming row 1: the workbook starts with a
    title ("Requirement Numbers From Change Approval Form") above the
    headers.
    """
    for _ in range(10):
        row = next(rows, None)
        if row is None:
            return None
        labels = {
            str(value).strip().lower(): index
            for index, value in enumerate(row)
            if value is not None
        }
        if _REQUIREMENT_HEADER in labels and _FEATURE_HEADER in labels:
            return labels
    return None


def _cell(row: tuple, index: int | None) -> str:
    if index is None or index >= len(row) or row[index] is None:
        return ""
    return _clean(str(row[index]))


def _carry(row: tuple, index: int | None, previous: str) -> str:
    value = _cell(row, index)
    return previous if value.lower() in _BLANKISH else value


def _clean(value: str) -> str:
    # Feature names are typed into the form over wrapped lines, and the line
    # breaks would otherwise reach the Folder column verbatim.
    return re.sub(r"\s+", " ", value).strip()


def _normalize_id(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).upper()
