"""Build the synthetic test data the script prompt falls back on.

    python scripts/build_synthetic_data.py

Some values a test script needs exist in no delivered source: the TGT_* keys
`wcr_ptc.verify_target_details` takes, where a bulletin's limits land as
target block + offset, the limits themselves, and wait/effective timings.
This writes them to data/synthetic_data/synthetic_test_data.json, derived
from each subdivision's real track XML wherever the track can answer:

- bulletin segments are placed inside real blocks, and their target
  locations are the segment mileposts interpolated over that block's
  BlockMilepostMeasureFeature (offset <-> milepost) pairs;
- track speed restriction targets are real BlockSpeedRestrictionFeature rows
  reshaped into target form.

Everything else (key meanings, TGT_TYPE strings, speeds, timings) is a fixed
synthetic default below. Re-run after track data changes, then restart the
server.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb_ingestion.extractors.xml_extractor import (  # noqa: E402
    normalize_subdivision,
    sanitized_xml,
    track_name_lookup,
)

DEFAULT_TRACK_DATA = REPO_ROOT / "data" / "track_data"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "synthetic_data" / "synthetic_test_data.json"

# The same window static_track_block_window sends as the prompt's block
# table, so every block named here is also one the prompt shows.
BLOCK_WINDOW = 8
MAX_TRACKS = 2
MAX_SPEED_RESTRICTIONS = 2

TARGET_KEYS = {
    "TGT_TYPE": "target type, one of the TGT_TYPE values",
    "TGT_START_LOC_BLOCKID": "block the target starts in",
    "TGT_START_LOC_OFFSET": "start offset into that block, feet",
    "TGT_START_LOC_DISTID": "PTC subdivision/district id of the start block",
    "TGT_END_LOC_BLOCKID": "block the target ends in",
    "TGT_END_LOC_OFFSET": "end offset into that block, feet",
    "TGT_END_LOC_DISTID": "PTC subdivision/district id of the end block",
    "TGT_SPEED_LIMIT": "target speed, mph",
    "TGT_SPEED_APPLICABILITY": "speed applicability code",
    "TGT_RESTRICTED_SPEED": "restricted speed flag",
}

TARGET_TYPES = {
    "work zone bulletin": "WORK ZONE",
    "PTC suspension bulletin": "PTC SUSPENSION",
    "block speed restriction (track)": "BLOCK SPEED RESTRICTION",
    "subdivision speed restriction (track)": "SUBDIV SPEED RESTRICTION",
    "signal becoming located": "SIGNAL-BECOMING LOCATED",
    "unknown signal": "UNKNOWN SIGNAL",
    "signal, train orientation change": "SIGNAL-TRAIN ORIENT CHANGE",
}

TARGET_DEFAULTS = {
    "work_zone_speed_mph": 0,
    "ptc_suspension_speed_mph": 0,
    "bulletin_speed_restriction_mph": 25,
    "speed_applicability": 0,
}

TIMINGS = {
    "target_wait_seconds": 60,
    "long_target_wait_seconds": 120,
    "ack_wait_seconds": 10,
    "effective_in_effect_now_minutes": 0,
    "effective_becoming_soon_minutes": 1,
}


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _leaves(element: ET.Element) -> dict[str, str]:
    return {_tag(child): (child.text or "").strip() for child in element if len(child) == 0}


def _read_blocks(root: ET.Element, track_names: dict[str, str]) -> list[dict]:
    blocks = []
    for element in root.iter():
        if _tag(element) != "BlockFeature":
            continue
        fields = _leaves(element)
        try:
            block = {
                "block_id": int(fields["BlockId"]),
                "sub_id": int(fields["SubId"]),
                "track": track_names.get(fields.get("TrackValue", ""), ""),
                "length": int(fields["Length"]),
                "start": int(fields["StartMilepost"]),
                "end": int(fields["EndMilepost"]),
            }
        except (KeyError, ValueError):
            continue
        pairs = []
        restrictions = []
        for child in element.iter():
            tag = _tag(child)
            if tag == "BlockMilepostMeasureFeature":
                measure = _leaves(child)
                pairs.append((int(measure["Offset"]), int(measure["Number"])))
            elif tag == "BlockSpeedRestrictionFeature":
                restrictions.append({key: int(value) for key, value in _leaves(child).items() if value.lstrip("-").isdigit()})
        block["pairs"] = sorted(pairs)
        block["restrictions"] = restrictions
        if block["track"]:
            blocks.append(block)
    return blocks


def _measured(block: dict) -> bool:
    """At least two offset <-> milepost pairs, mileposts rising with offset,
    so any milepost between the first and last pair maps to one offset."""
    numbers = [number for _offset, number in block["pairs"]]
    return len(numbers) >= 2 and all(a < b for a, b in zip(numbers, numbers[1:]))


def _milepost_at(block: dict, fraction: float) -> int:
    first, last = block["pairs"][0][1], block["pairs"][-1][1]
    return round(first + (last - first) * fraction)


def _offset_at(block: dict, milepost: int) -> int:
    pairs = block["pairs"]
    for (o1, n1), (o2, n2) in zip(pairs, pairs[1:]):
        if n1 <= milepost <= n2:
            offset = o1 + (milepost - n1) * (o2 - o1) / (n2 - n1)
            return max(0, min(block["length"], round(offset)))
    raise ValueError(f"milepost {milepost} is outside block {block['block_id']}'s measure pairs")


def _miles(raw: int) -> float:
    return round(raw / 10000, 4)


def _segment(start_block: dict, start_fraction: float, end_block: dict, end_fraction: float) -> dict:
    start = _milepost_at(start_block, start_fraction)
    end = _milepost_at(end_block, end_fraction)
    return {
        "start": start,
        "end": end,
        "start_miles": _miles(start),
        "end_miles": _miles(end),
        "target": _target(start_block, start, end_block, end),
    }


def _target(start_block: dict, start: int, end_block: dict, end: int) -> dict:
    return {
        "TGT_START_LOC_BLOCKID": start_block["block_id"],
        "TGT_START_LOC_OFFSET": _offset_at(start_block, start),
        "TGT_START_LOC_DISTID": start_block["sub_id"],
        "TGT_END_LOC_BLOCKID": end_block["block_id"],
        "TGT_END_LOC_OFFSET": _offset_at(end_block, end),
        "TGT_END_LOC_DISTID": end_block["sub_id"],
    }


def _approach_point(block: dict) -> dict:
    return {"point_miles": _miles((block["start"] + block["end"]) // 2), "block": block["block_id"]}


def _bulletin_segments(a: dict, b: dict) -> dict:
    contiguous = [_segment(a, 0.20, a, 0.50), _segment(a, 0.50, a, 0.80)]
    overlapping = [_segment(a, 0.20, a, 0.60), _segment(a, 0.40, a, 0.80)]
    return {
        "single_block": [_segment(a, 0.25, a, 0.75)],
        "spanning": [_segment(a, 0.50, b, 0.50)],
        "non_contiguous": [_segment(a, 0.25, a, 0.45), _segment(b, 0.55, b, 0.75)],
        "contiguous": contiguous,
        "overlapping": overlapping,
        # Both pairs cover 20-80% of the block, so each merges to this one target.
        "merged_target": _target(a, _milepost_at(a, 0.20), a, _milepost_at(a, 0.80)),
    }


def _predecessor(window: list[dict], block: dict) -> dict | None:
    # Blocks on one track can overlap (a parallel branch starting at the same
    # milepost), so "the block before" is the one ending where this one
    # starts, not the previous one in milepost order.
    return next((other for other in window if other["end"] == block["start"]), None)


def _unambiguous(window: list[dict], *blocks: dict) -> bool:
    """No other block on the track overlaps these, so a position inside one
    of them resolves to that block and not to a parallel branch."""
    return all(
        not (other is not block and other["start"] < block["end"] and other["end"] > block["start"])
        for block in blocks
        for other in window
    )


def _speed_restriction_targets(window: list[dict]) -> list[dict]:
    """One restriction per block, so the targets cover different blocks, and
    only blocks with a block before them for the train to start in."""
    targets = []
    for block in window:
        restriction = next((r for r in block["restrictions"] if "SpeedLimit" in r), None)
        previous = _predecessor(window, block)
        if not (restriction and previous and _unambiguous(window, previous, block)):
            continue
        target = {
            "TGT_TYPE": TARGET_TYPES["block speed restriction (track)"],
            "TGT_START_LOC_BLOCKID": block["block_id"],
            "TGT_START_LOC_OFFSET": restriction.get("StartOffset", 0),
            "TGT_START_LOC_DISTID": block["sub_id"],
            "TGT_END_LOC_BLOCKID": block["block_id"],
            "TGT_END_LOC_OFFSET": restriction.get("EndOffset", block["length"]),
            "TGT_END_LOC_DISTID": block["sub_id"],
            "TGT_SPEED_LIMIT": restriction["SpeedLimit"],
        }
        targets.append(
            {
                "track_train_type": restriction.get("TrainType"),
                "track_direction": restriction.get("Direction"),
                "approach": _approach_point(previous),
                "target": target,
            }
        )
        if len(targets) == MAX_SPEED_RESTRICTIONS:
            break
    return targets


def _track_entry(blocks: list[dict]) -> dict | None:
    window = sorted(blocks, key=lambda block: block["start"])[:BLOCK_WINDOW]
    for a in window:
        previous = _predecessor(window, a)
        b = next((other for other in window if other["start"] == a["end"]), None)
        if previous and b and _measured(a) and _measured(b) and _unambiguous(window, previous, a, b):
            return {
                "sub_id": a["sub_id"],
                "direction": "increasing",
                "approach": _approach_point(previous),
                "bulletin_segments": _bulletin_segments(a, b),
                "speed_restriction_targets": _speed_restriction_targets(window),
            }
    return None


def build_subdivision(xml_path: Path) -> dict:
    root = ET.fromstring(sanitized_xml(xml_path))
    track_names = track_name_lookup(root)
    blocks = _read_blocks(root, track_names)

    tracks = {}
    names = list(dict.fromkeys(track_names.values()))
    for track in sorted(names, key=lambda name: not name.lower().startswith("main")):
        entry = _track_entry([block for block in blocks if block["track"] == track])
        if entry:
            tracks[track] = entry
        if len(tracks) == MAX_TRACKS:
            break
    return {"tracks": tracks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-data", default=str(DEFAULT_TRACK_DATA), help="Track data folder.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the JSON.")
    args = parser.parse_args()

    track_data = Path(args.track_data)
    output = Path(args.output)

    subdivisions = {}
    for xml_path in sorted(track_data.glob("*/*-subdiv.xml")):
        subdivision = normalize_subdivision(xml_path.parent.name)
        subdivisions[subdivision] = build_subdivision(xml_path)
        print(f"{subdivision}: {len(subdivisions[subdivision]['tracks'])} track(s)")

    payload = {
        "synthetic": True,
        "note": (
            "Synthetic test data from scripts/build_synthetic_data.py, not a delivered source. "
            "Segments and targets are derived from data/track_data; keys, types, speeds and "
            "timings are fixed defaults."
        ),
        "generated_from": "data/track_data",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_keys": TARGET_KEYS,
        "target_types": TARGET_TYPES,
        "target_defaults": TARGET_DEFAULTS,
        "timings": TIMINGS,
        "subdivisions": subdivisions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
