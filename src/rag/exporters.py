from __future__ import annotations

import io
import re
from datetime import datetime, timezone

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .schema import DATASHEET_COLUMNS, TestCase

SHEET_NAME = "Datasheet"

# Column widths in the delivered datasheets: a wide wrapped Description with
# narrow fixed columns either side. Keyed by attribute so reordering
# DATASHEET_COLUMNS doesn't misalign them.
_COLUMN_WIDTHS = {
    "s_no": 6,
    "requirement": 13,
    "description": 105,
    "folder": 20,
    "optimization_technique": 22,
    "test_type": 11,
    "test_technique": 24,
    "retired": 9,
    "scorable": 10,
    "comments": 14,
}

_HEADER_FONT = Font(bold=True)
_COMMENTS_FILL = PatternFill("solid", fgColor="FFC000")
_SCENARIO_LINE_RE = re.compile(r"^\s*test\s*scenario\s*:", re.IGNORECASE)


def datasheet_to_xlsx(
    test_cases: list[TestCase], include_confidence: bool = True
) -> bytes:
    """The datasheet, as an .xlsx byte string ready to stream to the browser.

    Columns A–J are exactly the delivered format, so the file can be dropped
    into the existing test-execution flow unchanged. The confidence columns
    are appended after them (K onwards) rather than inserted, so anything
    reading the sheet by column position still works; drop them with
    include_confidence=False if the consumer is strict about column count.
    """
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET_NAME

    labels = [label for _, label in DATASHEET_COLUMNS]
    confidence_labels = [
        "Confidence",
        "Confidence_Retrieval",
        "Confidence_Grounding",
        "Similarity_To_Existing",
        "Closest_Existing_Case",
        "Needs_Review",
    ]
    if include_confidence:
        labels += confidence_labels

    sheet.append(labels)
    for index, label in enumerate(labels, start=1):
        cell = sheet.cell(row=1, column=index)
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        if label == "Comments":
            cell.fill = _COMMENTS_FILL

    for test_case in test_cases:
        row = test_case.to_row()
        if include_confidence:
            confidence = test_case.confidence
            row += [
                confidence.overall,
                confidence.retrieval,
                confidence.grounding,
                confidence.similarity_to_existing,
                confidence.closest_existing_id or "",
                "Yes" if confidence.needs_review else "No",
            ]
        sheet.append(row)

    _apply_layout(sheet, len(labels))
    _bold_scenario_lines(sheet, len(test_cases))

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _apply_layout(sheet, column_count: int) -> None:
    for index, (attr, _label) in enumerate(DATASHEET_COLUMNS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = _COLUMN_WIDTHS.get(attr, 16)
    for index in range(len(DATASHEET_COLUMNS) + 1, column_count + 1):
        sheet.column_dimensions[get_column_letter(index)].width = 14

    description_column = next(
        i for i, (attr, _) in enumerate(DATASHEET_COLUMNS, start=1) if attr == "description"
    )
    for row in sheet.iter_rows(min_row=2, max_col=column_count):
        for cell in row:
            wrap = cell.column == description_column
            cell.alignment = Alignment(vertical="top", wrap_text=wrap)

    sheet.freeze_panes = "A2"


def _bold_scenario_lines(sheet, row_count: int) -> None:
    """openpyxl can't style part of a cell's text, so the whole Description
    cell can't carry the "Test Scenario:" line in bold the way the
    hand-written sheets do. Rather than fake it, the cell is left unstyled
    and the row height is set to auto — a reviewer's own formatting survives
    when they edit it.
    """
    description_column = next(
        i for i, (attr, _) in enumerate(DATASHEET_COLUMNS, start=1) if attr == "description"
    )
    for row_index in range(2, row_count + 2):
        cell = sheet.cell(row=row_index, column=description_column)
        if _SCENARIO_LINE_RE.match(str(cell.value or "")):
            # Height None lets Excel size the row to the wrapped text.
            sheet.row_dimensions[row_index].height = None


def script_to_text(script: str, requirement_id: str | None) -> bytes:
    """The automation script as a downloadable .txt, with a provenance header.

    The header is a comment block naming the requirement, the generation
    time and that this is a generated draft, so a script that reaches a
    reviewer out of context can't be mistaken for a hand-written one.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"# Generated test script draft for requirement "
        f"{requirement_id or '(not stated)'}\n"
        f"# Generated {stamp} — review before execution.\n"
    )
    body = script if script.endswith("\n") else script + "\n"
    # CRLF to match the existing scripts in the repository, which are
    # Windows-authored; a mixed-ending diff would obscure real changes.
    return (header + body).replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def download_name(kind: str, requirement_id: str | None, extension: str) -> str:
    """A filename a reviewer can identify without opening it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", requirement_id or "requirement")
    return f"{safe_id}_{kind}_{stamp}.{extension}"
