from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ExtractedUnit

# Rendered first and in this order, so the identifier and the title lead the
# chunk and the values a test case needs follow in a stable place. Anything
# else the record carries is rendered after these, under its own key.
_LEADING_FIELDS = ("units", "valid_range", "common_value_north_america", "value", "railroad")

_LABELS = {
    "units": "Units",
    "valid_range": "Valid range",
    "common_value_north_america": "Common value (North America)",
    "value": "Value",
    "railroad": "Railroad",
}

_LOCATOR_FIELDS = ("table", "section", "page", "group")


def _label(field: str) -> str:
    return _LABELS.get(field, field.replace("_", " ").capitalize())


def _render(record: dict[str, Any]) -> str:
    identifier = str(record.get("id") or "").strip()
    name = str(record.get("name") or "").strip()
    lines = [" | ".join(part for part in (identifier, name) if part)]

    rendered = {"id", "name", "description", *_LOCATOR_FIELDS}
    for field in _LEADING_FIELDS:
        value = record.get(field)
        if value:
            lines.append(f"{_label(field)}: {value}")
        rendered.add(field)

    for field, value in record.items():
        if field not in rendered and value:
            lines.append(f"{_label(field)}: {value}")

    description = str(record.get("description") or "").strip()
    if description:
        lines.append(description)

    group = record.get("group")
    if group:
        lines.insert(1, f"Railroad section: {group}")

    return "\n".join(lines)


class ParameterJSONExtractor:
    """The parameter configuration guide, after
    `scripts/convert_parameter_guide.py` has turned its tables into records.

    One parameter is one unit, so TBC137 is retrievable on its own rather
    than as whatever share of a page of neighbouring parameters a prose chunk
    happened to catch. Each unit leads with the identifier and title because
    that is what a requirement names, and carries the units, valid range,
    default and owning railroad that a test case has to assert against.
    """

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        records = payload.get("parameters") or []

        units = []
        for record in records:
            if not record.get("id"):
                continue
            units.append(
                ExtractedUnit(
                    text=_render(record),
                    unit_type="parameter",
                    locator={
                        "section_path": record.get("section") or "",
                        "table_title": record.get("table") or "",
                        "page": record.get("page"),
                    },
                    extra={"parameter_id": str(record["id"])},
                )
            )
        return units
