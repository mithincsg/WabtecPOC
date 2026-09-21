from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from . import ExtractedUnit

# Every element in these files is namespaced
# ("{http://www.wabtec.com/WRE/SystemTest/Track}BlockFeature"). Nothing here
# needs to tell two namespaces apart — there is only ever one — so the prefix
# is stripped and tags are handled by their local name.
_NAMESPACE_RE = re.compile(r"^\{[^}]*\}")

# A record's own identity field, tried in order and matched
# case-insensitively — the same field is spelled `blockId` in a BlockVersion
# record and `BlockId` in the BlockFeature it refers to. Used to prefix the
# records nested underneath it, so a heading or speed-restriction line still
# says which block it belongs to after the tree is flattened into text.
_ID_FIELDS = (
    "blockid",
    "switchid",
    "signalid",
    "deviceid",
    "wiuid",
    "trackname",
    "id",
)

# Elements that carry no track facts: file-format bookkeeping (offsets, CRCs,
# the exporting tool's local path) that would only add noise to a keyword
# index full of block and device numbers.
_SKIPPED_TAGS = frozenset({"FileHeader"})

_MAX_CONTEXT_DEPTH = 2

# Numeric character references, decimal or hex.
_CHAR_REF_RE = re.compile(r"&#(?:[xX]([0-9a-fA-F]+)|([0-9]+));")
# Raw bytes that no character reference produced but that are still illegal.
_INVALID_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitized(file_path: Path) -> str:
    """The file's text with XML-illegal characters removed.

    These exports pad fixed-width fields with NULs and emit them literally as
    `&#x0;` — `<DataSourceId>80020&#x0;&#x0;...</DataSourceId>`. That is not
    well-formed XML at any version, so a strict parser rejects the entire
    file, and one padded field would otherwise cost the whole subdivision. The
    padding carries no information, so it is dropped and the real value kept,
    rather than failing a 2 MB file over it.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")

    def replace(match: re.Match[str]) -> str:
        hex_digits, decimal_digits = match.groups()
        try:
            code = int(hex_digits, 16) if hex_digits else int(decimal_digits)
        except ValueError:
            return ""
        return match.group(0) if _is_legal_xml_char(code) else ""

    return _INVALID_CHARS_RE.sub("", _CHAR_REF_RE.sub(replace, text))


def _is_legal_xml_char(code: int) -> bool:
    """The Char production from the XML 1.0 spec."""
    return (
        code in (0x9, 0xA, 0xD)
        or 0x20 <= code <= 0xD7FF
        or 0xE000 <= code <= 0xFFFD
        or 0x10000 <= code <= 0x10FFFF
    )


def _header_line(label: str, record_type: str, first_line: str) -> str:
    """The label row for a record type, naming the subdivision, the record
    type and its fields.

    It earns its place twice over. Chunking repeats a table unit's first line
    on every chunk it splits the unit into, so this is what keeps a chunk from
    the middle of 2,400 heading records still saying which subdivision and
    which kind of record it holds.

    And it is what makes these records findable at all by a requirement's own
    words. The XML spells its fields in camel case — `SignalId`,
    `CabSignalCutoutSpeedLimit` — which BM25 tokenizes whole, so a requirement
    asking about a "signal" or a "speed limit" matches none of them. Splitting
    the names into words here puts those terms in the chunk exactly once,
    rather than restating them on every row.
    """
    fields = [pair.split("=", 1)[0] for pair in first_line.split(" | ")]
    words = [_split_camel(record_type), *(_split_camel(f) for f in fields)]
    return f"{label} > " + " | ".join(w for w in words if w)


def _split_camel(name: str) -> str:
    """`CabSignalCutoutSpeedLimit` -> `Cab Signal Cutout Speed Limit`."""
    cleaned = name.rsplit(" > ", 1)[-1].strip()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", cleaned)


def _local(tag: str) -> str:
    return _NAMESPACE_RE.sub("", tag)


def _is_leaf(element: ET.Element) -> bool:
    return len(element) == 0


def _text_of(element: ET.Element) -> str:
    return (element.text or "").strip()


class XMLTrackDataExtractor:
    """Wabtec `<subdivision>-subdiv.xml` track files — the only track source
    the app reads.

    It is the full serialized track database, and carries every field a test
    script needs: WIU security keys and addresses, per-block
    element/heading/elevation series, device status table indices,
    acquisition records. The human-readable HTML report that used to sit
    beside it printed a strict subset of this and was dropped.

    The XML has no labels of its own beyond tag names, so the shape is
    recovered from the tree: every repeated feature record becomes one line of
    `field=value` pairs, records of the same type are collected together, and
    each type becomes one ExtractedUnit (unit_type="table"), which chunking
    splits on line boundaries when a record type runs long.

    Records nested under a parent (a heading inside a block) are prefixed with
    the parent's identifier, because flattening otherwise strips the one thing
    that makes the line answerable: which block the heading belongs to.
    """

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        root = ET.fromstring(_sanitized(file_path))

        subdivision = _subdivision_id(root, file_path)
        subdivision_name = _first_text(root, "SubdivisionName")
        label = f"{subdivision} {subdivision_name}".strip()
        # The export pads this fixed-width field with trailing spaces
        # ("UP  "); stripped so it matches the literal value a script must
        # write (`{"scac": "UP"}`), not a value with invisible padding that
        # would silently fail equality checks downstream.
        railroad_scac = _first_text(root, "RailroadSCAC").strip() or None

        track_names = _track_name_lookup(root)

        # dict preserves insertion order, so records appear in document order
        # — a block's features stay near the block's own record.
        records: dict[str, list[str]] = {}
        _collect(root, (), records, track_names)

        units: list[ExtractedUnit] = []
        for index, (record_type, lines) in enumerate(records.items(), start=1):
            if not lines:
                continue
            text = "\n".join([_header_line(subdivision, record_type, lines[0]), *lines])
            if not text.strip():
                continue
            units.append(
                ExtractedUnit(
                    text=text,
                    unit_type="table",
                    locator={
                        "section_path": f"{subdivision} > {record_type}",
                        "table_title": record_type,
                        "table": index,
                        # What the UI's picker filters on.
                        "subdivision": subdivision,
                        "subdivision_name": subdivision_name or None,
                        "railroad_scac": railroad_scac,
                    },
                )
            )
        return units


def _collect(
    element: ET.Element,
    context: tuple[str, ...],
    records: dict[str, list[str]],
    track_names: dict[str, str],
) -> None:
    """Walks the tree, appending one text line per record to its type's list.

    An element's leaf children are its own fields; its non-leaf children are
    nested records, visited with this element's identifier added to their
    context.
    """
    for child in element:
        tag = _local(child.tag)
        if tag in _SKIPPED_TAGS:
            continue
        if _is_leaf(child):
            # A leaf directly under a container with no record wrapper —
            # already emitted as part of its parent's field line.
            continue

        fields = [
            (_local(grandchild.tag), _text_of(grandchild))
            for grandchild in child
            if _is_leaf(grandchild)
        ]
        fields = [(name, value) for name, value in fields if value]
        fields = _with_track_name(fields, track_names)

        if fields:
            prefix = " > ".join(context)
            line = " | ".join(f"{name}={value}" for name, value in fields)
            records.setdefault(tag, []).append(f"{prefix} > {line}" if prefix else line)

        nested = [grandchild for grandchild in child if not _is_leaf(grandchild)]
        if nested:
            _collect(child, _extend_context(context, tag, fields), records, track_names)


def _extend_context(
    context: tuple[str, ...], tag: str, fields: list[tuple[str, str]]
) -> tuple[str, ...]:
    """Adds this record's identifier to the context its nested records carry.

    Capped at `_MAX_CONTEXT_DEPTH`: the containers nest several levels deep,
    and repeating the whole ancestry on every line would cost more index space
    than it buys — the nearest identified ancestors are what disambiguate a
    line, the ones above that are the same for a whole section.
    """
    identifier = _identifier(fields)
    if not identifier or len(context) >= _MAX_CONTEXT_DEPTH:
        return context
    return context + (f"{tag} {identifier}",)


def _with_track_name(
    fields: list[tuple[str, str]], track_names: dict[str, str]
) -> list[tuple[str, str]]:
    """Resolves a record's `TrackValue` code into its `TrackName`, inline.

    `BlockFeature` records carry a bare numeric `TrackValue` (e.g. `13`) and
    nothing else naming the physical track it runs on — the name lives only
    in the file's separate `TrackNamesContainer` legend, a record with none of
    a requirement's wording in it, so a keyword search has nothing to match it
    on and it goes unretrieved: the block's own chunk then reaches the model
    with a code and no way to resolve it, e.g. block 13022 was written into a
    generated script under `TrackValue=13` with no indication that means
    Siding1, and the script paired it with the wrong track. Resolving the
    join here, once, at parse time means every block's chunk is
    self-contained and needs no second record fetched alongside it.
    """
    resolved: list[tuple[str, str]] = []
    already_named = any(name.lower() == "trackname" for name, _ in fields)
    for name, value in fields:
        resolved.append((name, value))
        if name.lower() == "trackvalue" and not already_named:
            track_name = track_names.get(value)
            if track_name:
                resolved.append(("TrackName", track_name))
    return resolved


def _track_name_lookup(root: ET.Element) -> dict[str, str]:
    """`{TrackValue: TrackName}` for every `TrackNameFeature` in the file.

    Built once per file, before the main walk, so `_collect` can resolve a
    `BlockFeature`'s `TrackValue` as it emits that record's line rather than
    leaving the join to whatever later fetches the two records separately.
    """
    lookup: dict[str, str] = {}
    for element in root.iter():
        if _local(element.tag) != "TrackNameFeature":
            continue
        name = value = ""
        for child in element:
            child_tag = _local(child.tag)
            if child_tag == "TrackName":
                name = _text_of(child)
            elif child_tag == "TrackValue":
                value = _text_of(child)
        if value and name:
            lookup[value] = name
    return lookup


def _identifier(fields: list[tuple[str, str]]) -> str:
    values = {name.lower(): value for name, value in fields}
    for name in _ID_FIELDS:
        if values.get(name):
            return values[name]
    # No recognised ID field means no useful context to pass down. Falling
    # back to "first field" instead would stamp something like the exporter's
    # local .opk path onto every line beneath it — thousands of repetitions of
    # a token that identifies nothing and dilutes every BM25 score in the
    # section.
    return ""


def _first_text(root: ET.Element, tag: str) -> str:
    for element in root.iter():
        if _local(element.tag) == tag:
            text = _text_of(element)
            if text:
                return text
    return ""


def _subdivision_id(root: ET.Element, file_path: Path) -> str:
    """The subdivision this file describes, zero-padded to five digits.

    The containing folder is trusted first: `data/track_data/08101/` is how
    the files are organised and how the UI's subdivision picker lists them, so
    a file whose internal ID disagrees with the folder it was filed under
    should still be searchable under that folder. `<SubdivisionID>` is the
    fallback, then the filename.
    """
    from_folder = subdivision_from_folder(file_path)
    if from_folder:
        return from_folder

    stated = _first_text(root, "SubdivisionID")
    if stated:
        return normalize_subdivision(stated)

    match = re.search(r"(\d{3,6})", file_path.stem)
    return normalize_subdivision(match.group(1)) if match else file_path.stem


def subdivision_from_folder(file_path: Path) -> str:
    """The subdivision ID from the folder a track file sits in, or "" if that
    folder isn't named after one.

    Track data is organised one folder per subdivision
    (`data/track_data/08101/08101-subdiv.xml`), which makes the folder the
    single most reliable statement of which subdivision a file belongs to.
    """
    name = file_path.parent.name.strip()
    return normalize_subdivision(name) if name.isdigit() else ""


def normalize_subdivision(value: str) -> str:
    """"8880", "08880" and " 8880 " all name the same subdivision. Padding to
    five digits gives one canonical form, so a subdivision written either way
    still matches what was indexed.
    """
    digits = "".join(c for c in str(value) if c.isdigit())
    return digits.zfill(5) if digits else str(value).strip()
