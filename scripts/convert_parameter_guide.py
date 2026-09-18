"""Parameter Configuration Guide (PDF) -> one JSON record per parameter.

The guide is the authority for TBC/CFG/THE values, their units, valid ranges
and which railroad owns them. Ingested as a PDF it becomes prose chunks, and a
requirement asking about TBC137 retrieves a page of neighbouring parameters
around it. Converted here, each parameter is its own record, so the static
BM25 index can return exactly the one asked for.

    python scripts/convert_parameter_guide.py

Re-run after dropping a new revision of the guide into data/knowledge_base/.
The backend reads the JSON, not the PDF — data/knowledge_base/ holds the
source PDFs and is not read at request time — so a new revision has no effect
until this runs and the server restarts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_GLOB = "data/knowledge_base/*Parameter Configuration Guide*.pdf"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "parameter_config" / "parameter_config_guide.json"

_SECTION_RE = re.compile(r"^(?:///\s*)?(\d+(?:\.\d+)*)\s+([A-Z]\S*.*)$")
_TABLE_CAPTION_RE = re.compile(r"^(Table\s+\d+\.\d+:\s*\S.*)$")
_ID_RE = re.compile(r"^\[?[A-Z]{2,4}\d+[A-Z]?\]?$")

# Longer than any real column label in this guide. find_tables occasionally
# reads a whole page of text as one leading column, and that blob arrives as a
# header name; anything this long is that, not a label.
_MAX_HEADER_LENGTH = 120

# This is the North America guide, so mph is the unit a test case is written
# in; the Canada km/h equivalent is out of scope. The guide prints it as extra
# trailing cells (its common value, "km/h", its range), which find_tables hands
# back as a row short of its identifier — the same shape as a description
# wrapped across a column break, so it lands in the previous record's
# description. Stripped here rather than in the wrapping rule: the two are
# indistinguishable by shape, but a km/h figure trailing a description is
# always the Canada column.
# Not anchored to the end: a description that also wrapped over a page break
# has the Canada cells joined into the middle of it, ahead of its own tail.
# The unit followed by a numeric range is specific enough that prose which
# merely mentions km/h is not matched.
_CANADA_UNITS_RE = re.compile(
    r"\s*(?:-?[\d.]+\s+)?km/h\s*-?[\d.]+\s*(?:-|to)\s*-?[\d.]+", re.IGNORECASE
)

# The page footer, which leaks into a description whenever a table runs to the
# bottom of a page. Bounded at the footer's own last words rather than run to
# the end of the string: the description continuing on the next page is
# appended after the footer, so an unbounded match deletes the very tail this
# is here to preserve.
_FOOTER_RE = re.compile(
    r"\s*(?:Wabtec Corporation\s*)?This document contains legally privileged"
    r".*?strictly prohibited\.?",
    re.IGNORECASE | re.DOTALL,
)

# The same footer cut off mid-sentence by the column it landed in, with no
# closing words to bound the match.
_PARTIAL_FOOTER_RE = re.compile(
    r"\s*(?:Wabtec Corporation\s*)?This document contains legally privileged.*$",
    re.IGNORECASE | re.DOTALL,
)

_FIELD_NAMES = {
    "tbc#": "id",
    "cfg#": "id",
    "the#": "id",
    "msg#": "id",
    "number": "id",
    "message number": "id",
    "parameter": "id",
    "onboard component": "id",
    "troubleshooting guide files": "id",
    # railroad.cfg lists every parameter of a railroad under that railroad's
    # reporting mark, which is a grouping column, not the row's identifier.
    "rr": "group",
    "description": "description",
    "units": "units",
    "common": "common_value_north_america",
    "common value north america": "common_value_north_america",
    "railroad": "railroad",
    "value": "value",
    "msg name": "name",
    "messaging name": "name",
    "safety message": "name",
}

# Page furniture: the admin table at the top of every page, the confidentiality
# block, and the boxed NOTE callouts that find_tables reports as one-cell tables.
_SKIP_FIRST_CELL_PREFIXES = ("Document Type", "Wabtec Corporation", "NOTE")


def _clean(cell: str | None) -> str:
    if not cell:
        return ""
    return re.sub(r"[ \t]+", " ", cell.replace(" ", " ")).strip()


def _collapse(row: list[str | None]) -> list[str]:
    """A row's cells with the empties dropped.

    Every table in this guide is drawn with merged cells, so one logical
    column arrives as two to four physical ones and which of them carries the
    text varies by row. Dropping the empties puts the values back in column
    order, which is what lets header and data rows be zipped.
    """
    return [c for c in (_clean(cell) for cell in row) if c]


def _field_name(header: str) -> str:
    # Underlined header text comes back with the rule itself as runs of "_",
    # on their own lines ("Msg name\n_").
    header = re.sub(r"(?m)^\s*_+\s*$", " ", header)
    key = re.sub(r"\s+", " ", header.replace("-\n", "").replace("\n", " ")).strip().lower()
    if key in _FIELD_NAMES:
        return _FIELD_NAMES[key]
    # The min/max column is labelled four different ways across the guide
    # ("Valid", "Valid Range", "Valid Range (min-max)", the last of them split
    # over two lines), and a parameter's range has to be one field name.
    if key.startswith("valid"):
        return "valid_range"
    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return slug or "field"


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _looks_like_id(value: str) -> bool:
    return bool(_ID_RE.match(value))


def _looks_like_prose(value: str) -> bool:
    """A cell that landed in the identifier column but is a sentence.

    A wrapped description continuing onto the next page sometimes arrives
    with exactly one cell fewer than the header, which is the same shape as a
    grouped table's non-first row, so it is read as a record whose identifier
    is a paragraph. The identifiers in this guide are all short — TBC137,
    [02020], a display string, a component name — so length is what separates
    them; the threshold is well clear of the longest real one.

    Length alone misses the short fragments a sentence can break into
    ("intermodal train."), so a trailing full stop counts too: no identifier
    in this guide ends in one, and a cell that does is the end of a sentence
    belonging to the record above.
    """
    return len(value) > 60 or len(value.split()) > 8 or value.rstrip().endswith(".")


def _split_description(text: str, italic_lines: set[str]) -> tuple[str, str]:
    """A description cell's leading lines are the parameter's title, the rest
    its explanation.

    The split is not "first line, then the rest": a long title wraps, and
    nothing in the text says where it stopped. The guide sets every title in
    regular type and every explanation in italics, so the page's italic lines
    are what decide — the title is the lines above the first italic one.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return "", ""

    body_start = len(lines)
    for index, line in enumerate(lines):
        if line.lower() in italic_lines:
            body_start = index
            break

    if body_start == 0:
        return lines[0], _join_lines(lines[1:])
    return _join_lines(lines[:body_start]), _join_lines(lines[body_start:])


def _join_lines(lines: list[str]) -> str:
    """Unwrap lines, respecting the hyphen a wrap broke a word on
    ("Data Non-" / "Comm Area" is one word, not two).
    """
    joined = ""
    for line in lines:
        if not joined:
            joined = line
        elif joined.endswith("-"):
            joined += line
        else:
            joined = f"{joined} {line}"
    return joined


def _italic_lines(page: pymupdf.Page) -> set[str]:
    """Every wholly italic line on the page, to test description lines against.

    Whole lines rather than a blob of the page's italic text: a title line
    like "Comm Area" also occurs inside its own italic explanation, and a
    substring test would read the title as the start of the body.
    """
    found = set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            if spans and all("italic" in span["font"].lower() for span in spans):
                text = _clean("".join(span["text"] for span in spans))
                if text:
                    found.add(text.lower())
    return found


def _page_headings(page: pymupdf.Page) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Section headings and table captions with the y they sit at, so a table
    can be matched to the ones above it rather than to the page's last.
    """
    sections: list[tuple[float, str]] = []
    captions: list[tuple[float, str]] = []
    for block in page.get_text("blocks"):
        top = block[1]
        for raw in block[4].split("\n"):
            line = raw.strip()
            caption = _TABLE_CAPTION_RE.match(line)
            if caption:
                captions.append((top, re.sub(r"\s*\.{3,}.*$", "", caption.group(1)).strip()))
                continue
            section = _SECTION_RE.match(line)
            # The table of contents is dotted leaders; a real heading is not.
            if section and "...." not in line:
                sections.append((top, f"{section.group(1)} {section.group(2).strip()}"))
    return sections, captions


def _last_above(items: list[tuple[float, str]], y: float) -> str | None:
    above = [text for top, text in items if top < y]
    return above[-1] if above else None


def _is_page_furniture(table_rows: list[list[str | None]]) -> bool:
    first = _clean(table_rows[0][0] if table_rows[0] else "")
    if any(first.startswith(prefix) for prefix in _SKIP_FIRST_CELL_PREFIXES):
        return True
    # find_tables occasionally reports a whole page as a single cell when the
    # ruling lines confuse it. The same page's real table is reported too, so
    # dropping the blob loses nothing.
    return len(table_rows) == 1 and len(_collapse(table_rows[0])) <= 1


def _header_labels(table: Any) -> list[str]:
    return [
        name.strip()
        for name in (table.header.names or [])
        if name and name.strip() and len(name.strip()) <= _MAX_HEADER_LENGTH
    ]


def _read_header(table: Any, rows: list[list[str | None]]) -> tuple[list[str], int]:
    """Field names, and the index of the first data row.

    Header text in this guide is split across rows as often as across columns
    ("Valid" / "Range" / "(min-max)" stacked on three lines, "priority at" /
    "fast poll" / "Range" on the QoS matrix). PyMuPDF already resolves that
    into one label per column, so the names come from it; what is left is
    working out how many of the extracted rows those labels consumed, which is
    every leading row whose text the labels already contain.
    """
    labels = _header_labels(table)
    if not labels:
        return [], 0

    header_text = " ".join(labels).lower()
    start = 0
    for row in rows:
        cells = _collapse(row)
        if cells and all(cell.lower() in header_text for cell in cells):
            start += 1
            continue
        break

    return _dedupe([_field_name(label) for label in labels]), start


def _column_signature(table: Any) -> tuple[str, ...]:
    """The table's columns as resolved field names.

    Resolved rather than raw, because a table running over fifteen pages is
    not drawn identically on each of them — the min/max column alone is
    labelled four different ways — and a signature that changed on those would
    read one table as a dozen.
    """
    return tuple(_dedupe([_field_name(label) for label in _header_labels(table)]))


# The fields a wrapped row can add to. Units, ranges and defaults are single
# values that never wrap, so text arriving for one of them on a continuation
# row is the Canada column drawn alongside — appending it would produce
# "mph km/h" and "0-125 0-201", which is exactly the ambiguity a test case
# must not be written from.
_CONTINUABLE_FIELDS = ("description", "name", "railroad")


def _physical_columns(header_row: list[str | None], names: list[str]) -> dict[int, str]:
    """Which physical cell index each logical column is drawn in.

    Every table here is drawn with merged cells, so a five-column table
    arrives as seven physical ones and a row's values cannot be matched to
    columns by counting them. The header row is drawn on the same grid, so
    the indices its labels sit at are the indices its columns sit at.
    """
    positions = [index for index, cell in enumerate(header_row) if _clean(cell)]
    if len(positions) != len(names):
        return {}
    return dict(zip(positions, names))


def _continue_record(
    previous: dict[str, Any] | None,
    row: list[str | None],
    cells: list[str],
    columns: dict[int, str],
) -> None:
    """Appends a wrapped row's text to the record it belongs to.

    A continuation row is still drawn on the table's grid, so each piece of
    text says which column it belongs to and is carried into that one: the
    tail of a long description and the tail of a long railroad name arrive on
    the same row, and joining both onto the description is how "signal
    controlling movement in the block" ended up inside TBC283's explanation.
    Only where the grid cannot be read does everything go to the description,
    which is the field they almost always belong to.
    """
    if previous is None:
        return

    if columns:
        for index, cell in enumerate(row):
            name = columns.get(index)
            value = _clean(cell).replace("\n", " ")
            if value and name in _CONTINUABLE_FIELDS:
                previous[name] = " ".join(part for part in (previous.get(name), value) if part)
        return

    previous["description"] = " ".join(
        part for part in (previous.get("description"), *cells) if part
    ).replace("\n", " ")


def _records_from_table(
    table: Any,
    rows: list[list[str | None]],
    *,
    page_number: int,
    section: str,
    caption: str,
    italic_lines: set[str],
    continues: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """`continues` is the last record of the same table on the page before.

    A description wrapping over a page break arrives as the first row of the
    next page, one cell short of its identifier, and without this it has no
    record to attach to and is dropped — which is how TBC283's explanation
    ended at "When the on-board segment's delayed in".
    """
    names, start = _read_header(table, rows)
    if not names or "id" not in names:
        return []

    columns = _physical_columns(rows[start - 1], names) if start else {}
    grouped = names[0] == "group"
    records: list[dict[str, Any]] = []
    group = ""
    for row in rows[start:]:
        cells = _collapse(row)
        if not cells:
            continue

        row_names = names
        if grouped:
            # Only the first parameter of a railroad sits on the row naming it;
            # the rest are one cell narrower and belong to the railroad above.
            if len(cells) == len(names):
                group = cells[0]
            elif len(cells) == len(names) - 1:
                row_names = names[1:]
        elif len(cells) < len(names) and (records or continues) and not _looks_like_id(cells[0]):
            # A row short of its identifier is the tail of the one above,
            # wrapped across a page or column break.
            _continue_record(records[-1] if records else continues, row, cells, columns)
            continue

        record: dict[str, Any] = {}
        for name, value in zip(row_names, cells):
            if name == "description":
                title, body = _split_description(value, italic_lines)
                record["name"] = title
                record["description"] = body
            else:
                record[name] = value.replace("\n", " ")

        identifier = record.get("id", "")
        if not identifier:
            continue
        if _looks_like_prose(identifier) and not _looks_like_id(identifier):
            _continue_record(records[-1] if records else continues, row, cells, columns)
            continue
        if group:
            record["group"] = group
        record["table"] = caption
        record["section"] = section
        record["page"] = page_number
        records.append(record)

    return records


def _document_metadata(doc: pymupdf.Document) -> dict[str, str]:
    cover = doc[0].get_text()
    fields = {}
    for key, label in (
        ("document_number", "Document Number:"),
        ("revision", "Revision:"),
        ("date", "Date:"),
    ):
        match = re.search(rf"{re.escape(label)}\s*(.+)", cover)
        if match:
            fields[key] = match.group(1).strip()

    # The cover sets the title over two lines under a "///" rule, so it runs
    # to the blank line rather than to the end of the first one.
    title = re.search(r"///\s*(.+?)\n\s*\n", cover, re.DOTALL)
    if title:
        fields["title"] = " ".join(title.group(1).split())
    return fields


def convert(pdf_path: Path) -> dict[str, Any]:
    doc = pymupdf.open(pdf_path)
    parameters: list[dict[str, Any]] = []
    heading = ""
    section = ""
    caption = ""

    signature: tuple[str, ...] = ()

    for page_index in range(doc.page_count):
        page = doc[page_index]
        sections, captions = _page_headings(page)
        italic_lines = _italic_lines(page)
        carried_section = heading

        for table in page.find_tables(strategy="lines_strict").tables:
            rows = table.extract()
            if not rows or _is_page_furniture(rows):
                continue

            # Most of these tables run for pages, so a page's caption does not
            # belong to whichever table is first on it — a page can carry the
            # tail of one table above the start of the next, and the tail has
            # no caption of its own. The columns are what say which: the same
            # columns as the table before means the same table continuing,
            # under the caption and heading it started beneath.
            # A caption of its own says "new table" even where the columns
            # match the one before, which is how Table 3.4 and Table 3.5 are
            # told apart: consecutive, and both an identifier and a
            # description.
            columns = _column_signature(table)
            top = table.bbox[1]
            own_caption = _last_above(captions, top)
            same_table = True
            if columns != signature or (own_caption and own_caption != caption):
                signature = columns
                caption = own_caption or ""
                section = _last_above(sections, top) or carried_section
                same_table = False

            parameters.extend(
                _records_from_table(
                    table,
                    rows,
                    page_number=page_index + 1,
                    section=section,
                    caption=caption,
                    italic_lines=italic_lines,
                    # Only within one table: a short first row under a table
                    # that has just started belongs to that table, not to
                    # whatever record the previous one ended on.
                    continues=parameters[-1] if same_table and parameters else None,
                )
            )

        if sections:
            heading = sections[-1][1]

    for record in parameters:
        description = _clean_description(record.get("description") or "")
        if description:
            record["description"] = description
        else:
            record.pop("description", None)

    return {
        "document": _document_metadata(doc) | {"source_file": pdf_path.name},
        "parameters": parameters,
    }


def _clean_description(description: str) -> str:
    without_footer = _PARTIAL_FOOTER_RE.sub("", _FOOTER_RE.sub(" ", description))
    cleaned = _CANADA_UNITS_RE.sub(" ", without_footer)
    return re.sub(r"\s+([.,;:])", r"", re.sub(r"[ 	]+", " ", cleaned)).strip()


def _find_pdf() -> Path:
    matches = sorted(REPO_ROOT.glob(DEFAULT_PDF_GLOB))
    if not matches:
        sys.exit(f"No parameter configuration guide found at {DEFAULT_PDF_GLOB}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pdf_path = args.pdf or _find_pdf()
    payload = convert(pdf_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{pdf_path.name} -> {args.output} ({len(payload['parameters'])} parameters)")


if __name__ == "__main__":
    main()
