from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from kb_ingestion.extractors.html_extractor import normalize_subdivision


@dataclass(frozen=True)
class TrackSelection:
    """Which track-data subdivisions apply to a requirement.

    `subdivisions` empty with `search_all` false and `needs_subdivision`
    false means: this requirement has no track relevance, so skip track
    data. `needs_subdivision` true means: the requirement text matched a
    track keyword, but no subdivision has been chosen yet, so track data is
    still skipped until one is. `subdivisions` non-empty means a subdivision
    was chosen (by the user), so only it is searched.
    """

    subdivisions: tuple[str, ...]
    search_all: bool
    needs_subdivision: bool = False

    @property
    def enabled(self) -> bool:
        return self.search_all or bool(self.subdivisions)


class TrackMapping:
    """The track-relevance keywords from config/track_mapping.yaml.

    The track_data folder holds one HTML report per subdivision, each listing
    thousands of track features. Searching all of them for every requirement
    swamps retrieval with track that the requirement is not tested on, and a
    requirement's text says whether it cares about track data at all, but
    never which subdivision — that's for the user to pick. So a requirement
    only sees track data once its text matches a configured keyword *and* a
    subdivision has been supplied by the caller. The file is reloaded when it
    changes, so adding a keyword takes effect on the next request.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._keywords: tuple[str, ...] = ()
        self._mtime: float | None = None

    def _load(self) -> None:
        if not self.path.is_file():
            # No mapping file is a valid setup — it just means no requirement
            # is ever flagged as track-relevant.
            self._keywords, self._mtime = (), None
            return

        mtime = self.path.stat().st_mtime
        if self._mtime == mtime:
            return

        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._keywords = tuple(
            str(keyword).strip() for keyword in (raw.get("keywords") or []) if str(keyword).strip()
        )
        self._mtime = mtime

    def matches(self, requirement_text: str | None) -> bool:
        """Whether the requirement's text mentions a track-feature keyword."""
        self._load()
        if not requirement_text or not self._keywords:
            return False
        haystack = requirement_text.lower()
        return any(keyword.lower() in haystack for keyword in self._keywords)

    def selection_for(
        self, requirement_text: str | None, subdivision: str | None = None
    ) -> TrackSelection:
        if subdivision and str(subdivision).strip():
            return TrackSelection((normalize_subdivision(subdivision),), search_all=False)

        if self.matches(requirement_text):
            return TrackSelection((), search_all=False, needs_subdivision=True)

        return TrackSelection((), search_all=False)
