from __future__ import annotations

import json
import logging
from pathlib import Path

from kb_ingestion.extractors.xml_extractor import normalize_subdivision

logger = logging.getLogger(__name__)

_SEGMENT_SHAPES = ("single_block", "spanning", "non_contiguous", "contiguous", "overlapping")


def _pairs(values: dict) -> str:
    return "; ".join(f"{key}={value}" for key, value in values.items())


def _dict(values: dict) -> str:
    return json.dumps(values, separators=(", ", ": "))


class SyntheticTestData:
    """The synthetic values `scripts/build_synthetic_data.py` writes for
    what no delivered source holds: the TGT_* keys `verify_target_details`
    takes, bulletin segment limits placed on real blocks with the target
    location each produces, and wait/effective timings.

    A lookup by subdivision rather than a BM25 source: the picked
    subdivision decides which segments apply, and ranked into the static
    index these rows would compete with real track and parameter records.
    Read once; rebuild the file and restart to pick up a change.
    """

    def __init__(self, path: Path | None):
        self._data: dict = {}
        if not path or not path.is_file():
            logger.warning("Synthetic test data %s not found -- script prompts get none", path)
            return
        self._data = json.loads(path.read_text(encoding="utf-8"))
        logger.info(
            "Synthetic test data covers %d subdivision(s)", len(self._data.get("subdivisions", {}))
        )

    def context(self, subdivision: str | None, include_bulletin_segments: bool) -> str:
        """`include_bulletin_segments` is most of this block's size, and only a
        requirement that sends a 01041 bulletin can use it, so every other
        requirement is spared the prefill."""
        if not self._data:
            return ""
        lines = [
            "Synthetic values (scripts/build_synthetic_data.py, not a delivered source). "
            "Segments and targets are derived from this subdivision's track data; the rest "
            "are defaults.",
            "verify_target_details keys: " + _pairs(self._data.get("target_keys", {})),
            "TGT_TYPE values: "
            + "; ".join(f'{what} -> "{value}"' for what, value in self._data.get("target_types", {}).items()),
            "Target defaults: " + _pairs(self._data.get("target_defaults", {})),
            "Timings: " + _pairs(self._data.get("timings", {})),
        ]

        chosen = normalize_subdivision(subdivision) if subdivision else ""
        tracks = self._data.get("subdivisions", {}).get(chosen, {}).get("tracks", {})
        if tracks and include_bulletin_segments:
            lines.append(
                f"Subdivision {chosen} bulletin segments: 01041 set_segment start/end (raw) "
                f"-> the verify_target_details location that segment produces."
            )
        for track, entry in tracks.items():
            lines.append(f"{track} (sub_id {entry['sub_id']}, direction {entry['direction']}):")
            if include_bulletin_segments:
                approach = entry["approach"]
                lines.append(
                    f"  approach set_position point {approach['point_miles']} (block {approach['block']})"
                )
                segments = entry["bulletin_segments"]
                for shape in _SEGMENT_SHAPES:
                    parts = segments.get(shape, [])
                    for number, segment in enumerate(parts, 1):
                        label = shape if len(parts) == 1 else f"{shape} #{number}"
                        lines.append(
                            f"  {label}: start={segment['start']} end={segment['end']} -> {_dict(segment['target'])}"
                        )
                if segments.get("merged_target"):
                    lines.append(
                        f"  contiguous or overlapping, as one target: {_dict(segments['merged_target'])}"
                    )
            for restriction in entry.get("speed_restriction_targets", []):
                approach = restriction["approach"]
                lines.append(
                    f"  track speed restriction, approach point {approach['point_miles']} "
                    f"(block {approach['block']}), track train type "
                    f"{restriction['track_train_type']}, direction {restriction['track_direction']}: "
                    f"{_dict(restriction['target'])}"
                )
        logger.info(
            "Synthetic test data: %s",
            f"{len(tracks)} track(s) in subdivision {chosen}, bulletin segments "
            f"{'included' if include_bulletin_segments else 'skipped (no 01041 message object)'}"
            if tracks
            else "keys and timings only",
        )
        return "\n".join(lines)
