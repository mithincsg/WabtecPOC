from __future__ import annotations

from typing import Any

from . import ExtractedUnit

# A field only earns a chunk when it says what a value means or where its
# limits are. "Employee Identifier, Printable ASCII" tells a test case nothing
# it could assert, and ~900 such rows would only dilute the ranking.
_DECODING_KEYS = ("enumeration", "range", "unit", "encoding")


def _is_metric(message: dict[str, Any]) -> bool:
    # Every message the ICD lists twice has a metric twin that states its
    # speeds in km/h (Bulletin Dataset v135 beside the mph v134). That is the
    # Canadian variant; speeds here are North American mph only.
    return any(field.get("unit") == "km/h" for field in message.get("fields") or [])


def _body(field: dict[str, Any]) -> str:
    name = field["field"]
    lines = [f"Type: {field.get('type', '')}, size {field.get('size_bytes', '')} byte(s)"]
    if field.get("unit"):
        lines.append(f"Unit: {field['unit']}")
    if field.get("encoding"):
        lines.append(f"Encoding: {field['encoding']}")
    if field.get("range"):
        lines.append(f"Range: {field['range'].get('min')} to {field['range'].get('max')}")
    if field.get("description"):
        lines.append(field["description"])
    # Written the way a test case states it, so the model copies the form
    # and every code line carries the field name it belongs to.
    for code, meaning in (field.get("enumeration") or {}).items():
        lines.append(f"{name}={code} ({meaning})")
    return "\n".join(lines)


def icd_units(messages: list[dict[str, Any]]) -> list[ExtractedUnit]:
    """One unit per decoded field of the office/locomotive ICD, so a
    requirement saying "freight" or "train type" retrieves the code table a
    test case has to state.

    A field decoded identically in several messages (Bulletin Type is the
    same table in four bulletin messages) is one unit naming all of them,
    so one code table never fills several retrieval slots.

    Design notes are left out: the long ones quote a dozen fields each, and
    attached to every field they quote they outrank the field names BM25
    should be matching on.
    """
    merged: dict[tuple[str, str], list[str]] = {}
    for message in messages:
        if _is_metric(message):
            continue
        for field in message.get("fields") or []:
            if "field" not in field or not any(field.get(k) for k in _DECODING_KEYS):
                continue
            used_in = merged.setdefault((field["field"], _body(field)), [])
            label = f"{message['msg_id']} {message['name']}"
            if label not in used_in:
                used_in.append(label)

    return [
        ExtractedUnit(
            text=f"ICD field {name} (message {'; '.join(used_in)})\n{body}",
            unit_type="icd_field",
            locator={"section_path": f"{used_in[0]} > {name}"},
        )
        for (name, body), used_in in merged.items()
    ]
