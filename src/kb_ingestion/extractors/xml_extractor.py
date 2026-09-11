from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from . import ExtractedUnit
from .html_extractor import normalize_subdivision

_MIN_TABLE_ROWS = 1
# Container elements below this size hold no real data (e.g. an unused
# SpeedEnforcementContainer / LicenseContainer) and produce nothing but noise.
_ACRONYM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z]{2,})")
_WORD_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


def _local(tag: str) -> str:
    """Strips the Wabtec XML namespace (`{http://...}Tag` -> `Tag`)."""
    return tag.rsplit("}", 1)[-1]


def _is_leaf(el: ET.Element) -> bool:
    return len(el) == 0


def _humanize(name: str) -> str:
    """"BlockSpeedRestrictions" -> "Block Speed Restrictions",
    "DOTNumbers" -> "DOT Numbers" (acronym runs stay together).
    """
    spaced = _ACRONYM_BOUNDARY_RE.sub(r"\1 \2", name)
    spaced = _WORD_BOUNDARY_RE.sub(r"\1 \2", spaced)
    return spaced


def _pluralize(word: str) -> str:
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _singular(tag: str) -> str:
    return tag[: -len("Feature")] if tag.endswith("Feature") else tag


def _scalar_fields(el: ET.Element) -> list[tuple[str, str]]:
    return [(_local(c.tag), (c.text or "").strip()) for c in el if _is_leaf(c)]


def _group_children(el: ET.Element) -> list[ET.Element]:
    return [c for c in el if not _is_leaf(c)]


class _Section:
    def __init__(self, title: str, ancestor_cols: list[str]):
        self.title = title
        self.ancestor_cols = ancestor_cols
        self.value_cols: list[str] = []
        self._seen_cols: set[str] = set(ancestor_cols)
        self.rows: list[dict[str, str]] = []

    def add_row(self, ancestors: dict[str, str], scalars: dict[str, str]) -> None:
        for col in scalars:
            if col not in self._seen_cols:
                self._seen_cols.add(col)
                self.value_cols.append(col)
        row = {c: ancestors.get(c, "") for c in self.ancestor_cols}
        row.update(scalars)
        self.rows.append(row)

    def to_text(self) -> str:
        columns = self.ancestor_cols + self.value_cols
        lines = [" | ".join(columns)]
        for row in self.rows:
            lines.append(" | ".join(row.get(c, "") for c in columns))
        return "\n".join(lines)


class _SubdivisionXMLParser:
    """Walks a Wabtec subdivision XML export, turning every group of
    repeated sibling elements (a "*Container" holding one element per
    feature) into one section, aggregated across the whole file the same
    way `_TrackReportParser` aggregates one HTML table per named section.

    A group is identified generically — by same-tag siblings — rather than
    by a fixed list of container names, since the exact set of containers
    (Blocks, Switches, Speed Restrictions, Signals, DOT Numbers, ...) is a
    property of the Wabtec schema, not something this parser should hard-code
    and have silently go stale if a future export adds or renames one.

    Nesting is preserved by threading the enclosing row's own id field (e.g.
    a BlockFeature's `BlockId`) down as an extra leading column on every
    section reached through it, so a block's speed restrictions are still
    traceable to the block they belong to once flattened into a flat table.
    """

    def __init__(self):
        # Keyed by the repeated child's tag, so occurrences reached through
        # different parent instances (every block's own speed-restriction
        # list) accumulate into the same section instead of one per parent.
        self._sections: dict[str, _Section] = {}
        self._order: list[str] = []

    def parse(self, root: ET.Element) -> list[tuple[str, str]]:
        self._walk(root, ancestors={})
        return [(self._sections[key].title, self._sections[key].to_text()) for key in self._order]

    def _walk(self, el: ET.Element, ancestors: dict[str, str]) -> None:
        groups = _group_children(el)
        if not groups:
            return

        tags = {_local(c.tag) for c in groups}
        if len(tags) == 1 and len(groups) >= _MIN_TABLE_ROWS:
            self._process_feature_list(groups, ancestors)
        else:
            for child in groups:
                self._walk(child, ancestors)

    def _process_feature_list(self, rows: list[ET.Element], ancestors: dict[str, str]) -> None:
        child_tag = _local(rows[0].tag)
        singular = _singular(child_tag)
        id_field = f"{singular}Id"

        section = self._sections.get(child_tag)
        if section is None:
            title = _humanize(_pluralize(singular))
            section = _Section(title, list(ancestors.keys()))
            self._sections[child_tag] = section
            self._order.append(child_tag)

        for row in rows:
            scalars = dict(_scalar_fields(row))
            section.add_row(ancestors, scalars)

            own_id = scalars.get(id_field)
            child_ancestors = ancestors
            if own_id:
                child_ancestors = {**ancestors, id_field: own_id}
            for nested in _group_children(row):
                self._walk(nested, child_ancestors)


class XMLTrackDataExtractor:
    """Wabtec subdivision export (`<subdivision>-subdiv.xml`), the
    machine-serialized source the paired `.html` "Track by Group" report is
    generated from. Same subdivision -> section -> feature-record shape as
    `HTMLTrackDataExtractor`, so both land in the vector/keyword store with
    identical `ExtractedUnit`/locator structure and are retrieved together.
    """

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        # A parse failure (e.g. an illegal control character from whatever
        # produced the export) is left to propagate rather than swallowed
        # here: both the ingestion pipeline and the static-context builder
        # already catch and log a failing file per-file, same as every other
        # extractor — silently returning no units here would instead read as
        # "this subdivision has no track data", which is the wrong signal.
        root = ET.parse(file_path).getroot()

        subdivision, subdivision_name = _subdivision_identity(root, file_path)
        label = f"{subdivision} {subdivision_name}".strip()

        header_fields = _header_fields(root)
        sections: list[tuple[str, str]] = []
        if header_fields:
            sections.append(("Header", _rows_to_text_from_pairs(header_fields)))

        parser = _SubdivisionXMLParser()
        sections.extend(parser.parse(root))

        units: list[ExtractedUnit] = []
        for index, (section_name, text) in enumerate(sections, start=1):
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
                        "subdivision": subdivision,
                        "subdivision_name": subdivision_name or None,
                    },
                )
            )
        return units


def _header_fields(root: ET.Element) -> list[tuple[str, str]]:
    for el in root:
        if _local(el.tag) == "FileHeader":
            return _scalar_fields(el)
    return []


def _rows_to_text_from_pairs(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"{key} | {value}" for key, value in pairs if key)


def _subdivision_identity(root: ET.Element, file_path: Path) -> tuple[str, str]:
    """Mirrors `HTMLTrackDataExtractor`'s subdivision-id resolution so a
    paired html/xml file for the same subdivision are tagged identically:
    the `<Subdivision><SubdivisionID>` field is authoritative here (this
    format states it directly, unlike the HTML report's free-text body
    line), the filename stem is the fallback.
    """
    subdivision_name = ""
    for subdivision_el in root.iter():
        if _local(subdivision_el.tag) != "Subdivision":
            continue
        raw_id = None
        for child in subdivision_el:
            tag = _local(child.tag)
            if tag == "SubdivisionID" and child.text and child.text.strip():
                raw_id = child.text.strip()
            elif tag == "SubdivisionName" and child.text:
                subdivision_name = child.text.strip()
        if raw_id:
            return normalize_subdivision(raw_id), subdivision_name
        break

    return normalize_subdivision(file_path.stem), subdivision_name
