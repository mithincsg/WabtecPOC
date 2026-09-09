from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from kb_ingestion.extractors.html_extractor import normalize_subdivision

# What to do with a requirement that has no row in the mapping file.
_EXCLUDE = "exclude"
_ALL = "all"
_VALID_UNMAPPED = (_EXCLUDE, _ALL)


@dataclass(frozen=True)
class TrackSelection:
    """Which track-data subdivisions apply to a requirement.

    `subdivisions` empty with `search_all` false means: this requirement has
    no track mapping, so skip track data. Empty with `search_all` true means:
    search every subdivision.
    """

    subdivisions: tuple[str, ...]
    search_all: bool

    @property
    def enabled(self) -> bool:
        return self.search_all or bool(self.subdivisions)


class TrackMapping:
    """The requirement -> subdivision map from config/track_mapping.yaml.

    The track_data folder holds one HTML report per subdivision, each listing
    thousands of track features. Searching all of them for every requirement
    swamps retrieval with track that the requirement is not tested on, so a
    requirement only ever sees the subdivisions listed against it. The file
    is reloaded when it changes, so adding a row takes effect on the next
    request.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mappings: dict[str, tuple[str, ...]] = {}
        self._unmapped = _EXCLUDE
        self._mtime: float | None = None

    def _load(self) -> None:
        if not self.path.is_file():
            # No mapping file is a valid setup — it just means no requirement
            # has track data associated with it yet.
            self._mappings, self._unmapped, self._mtime = {}, _EXCLUDE, None
            return

        mtime = self.path.stat().st_mtime
        if self._mtime == mtime:
            return

        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        mappings: dict[str, tuple[str, ...]] = {}
        for req_id, subdivisions in (raw.get("mappings") or {}).items():
            if isinstance(subdivisions, (str, int)):
                subdivisions = [subdivisions]
            # Normalised the same way the HTML extractor normalises what it
            # ingested, so a map entry written as 8880 still matches chunks
            # stored under 08880.
            mappings[str(req_id).strip().upper()] = tuple(
                normalize_subdivision(s) for s in subdivisions if str(s).strip()
            )

        unmapped = str(raw.get("unmapped", _EXCLUDE)).strip().lower()
        if unmapped not in _VALID_UNMAPPED:
            raise ValueError(
                f"{self.path}: `unmapped` must be one of {_VALID_UNMAPPED}, got {unmapped!r}."
            )

        self._mappings, self._unmapped, self._mtime = mappings, unmapped, mtime

    def selection_for(self, requirement_id: str | None) -> TrackSelection:
        self._load()

        if requirement_id:
            key = requirement_id.strip().upper()
            if key in self._mappings:
                return TrackSelection(self._mappings[key], search_all=False)
            # "L2R9479_A" is the same requirement as "L2R9479" with a
            # revision suffix, so fall back to the base ID before treating it
            # as unmapped.
            base = key.split("_")[0]
            if base in self._mappings:
                return TrackSelection(self._mappings[base], search_all=False)

        return TrackSelection((), search_all=self._unmapped == _ALL)

    @property
    def known_requirement_ids(self) -> list[str]:
        self._load()
        return sorted(self._mappings)
