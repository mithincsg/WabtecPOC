from __future__ import annotations

from pathlib import Path

import openpyxl

from . import ExtractedUnit

_TEST_CASE_ID_ALIASES = {"test case id", "test_case_id", "testcaseid", "tc id", "tcid", "case id"}
_REQUIREMENT_ID_ALIASES = {"requirement", "requirement id", "requirement_id", "req id", "reqid"}
_STEPS_ALIASES = {"steps", "test steps", "test_steps", "step"}
_EXPECTED_RESULT_ALIASES = {"expected result", "expected_result", "expected", "result"}
_SEQUENCE_ALIASES = {"s_no", "s.no", "sno", "sr no", "sl no", "seq", "id", "#"}


def _cell_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _match_column(header: list[str], aliases: set[str]) -> int | None:
    for index, name in enumerate(header):
        if name.strip().lower() in aliases:
            return index
    return None


class XLSXExtractor:
    """One ExtractedUnit per non-empty data row (unit_type="test_case"),
    with the sheet's header row prefixed as ``"Header: value"`` pairs so each
    row is self-describing once separated from its sheet. Header columns
    matching known aliases (requirement id, test case id, ...) are also
    pulled out into `extra` as structured fields, since a test-case row is a
    structured knowledge unit, not just flattened text.
    """

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        units: list[ExtractedUnit] = []
        workbook = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        try:
            for sheet in workbook.worksheets:
                units.extend(self._extract_sheet(sheet))
        finally:
            workbook.close()
        return units

    def _extract_sheet(self, sheet) -> list[ExtractedUnit]:
        rows = sheet.iter_rows(values_only=True)
        try:
            header = [_cell_str(v) for v in next(rows)]
        except StopIteration:
            return []

        columns = {
            "requirement_id": _match_column(header, _REQUIREMENT_ID_ALIASES),
            "test_case_id": _match_column(header, _TEST_CASE_ID_ALIASES),
            "steps": _match_column(header, _STEPS_ALIASES),
            "expected_result": _match_column(header, _EXPECTED_RESULT_ALIASES),
            "sequence": _match_column(header, _SEQUENCE_ALIASES),
        }

        units: list[ExtractedUnit] = []
        for row_index, row in enumerate(rows, start=2):
            values = [_cell_str(v) for v in row]
            unit = self._build_row_unit(sheet.title, row_index, header, values, columns)
            if unit:
                units.append(unit)
        return units

    @staticmethod
    def _build_row_unit(
        sheet_title: str,
        row_index: int,
        header: list[str],
        values: list[str],
        columns: dict[str, int | None],
    ) -> ExtractedUnit | None:
        if not any(values):
            return None

        pairs = [f"{h}: {v}" for h, v in zip(header, values) if h and v]
        if not pairs:
            return None

        def get(col_name: str) -> str | None:
            index = columns[col_name]
            return values[index] if index is not None and values[index] else None

        requirement_id = get("requirement_id")
        test_case_id = get("test_case_id")
        if not test_case_id and requirement_id:
            seq = get("sequence")
            if seq:
                test_case_id = f"{requirement_id}_{seq}"

        return ExtractedUnit(
            text=" | ".join(pairs),
            unit_type="test_case",
            locator={"sheet": sheet_title, "row": row_index},
            extra={
                "header": header,
                "requirement_id": requirement_id,
                "test_case_id": test_case_id,
                "steps": get("steps"),
                "expected_result": get("expected_result"),
            },
        )
