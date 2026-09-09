from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml


class PromptError(RuntimeError):
    """Raised when config/prompts.yaml is missing a prompt or a placeholder."""


@dataclass(frozen=True)
class PromptPair:
    system: str
    user_template: str

    def render_user(self, **values: str) -> str:
        """Fills the user template. `string.Template` ($name) rather than
        str.format, because these prompts contain literal JSON braces that
        would otherwise all need escaping in the YAML.
        """
        try:
            return Template(self.user_template).substitute(**values)
        except KeyError as exc:
            raise PromptError(
                f"Prompt template references ${exc.args[0]}, which was not supplied. "
                f"Available: {', '.join(sorted(values))}."
            ) from exc


class PromptLibrary:
    """Every prompt, read from config/prompts.yaml.

    Re-read on each access when `watch` is on (the default), so editing a
    prompt takes effect on the next request without restarting the server —
    prompt wording is the thing that gets iterated on most, and a reload
    cycle per edit is the main friction in tuning it.
    """

    def __init__(self, path: str | Path, watch: bool = True):
        self.path = Path(path)
        self.watch = watch
        self._cache: dict | None = None
        self._mtime: float | None = None

    def _data(self) -> dict:
        if not self.path.is_file():
            raise PromptError(f"Prompt file not found: {self.path}")
        mtime = self.path.stat().st_mtime
        if self._cache is None or (self.watch and mtime != self._mtime):
            self._cache = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self._mtime = mtime
        return self._cache

    def pair(self, name: str) -> PromptPair:
        data = self._data()
        entry = data.get(name)
        if not isinstance(entry, dict):
            raise PromptError(
                f"{self.path} has no '{name}' prompt. Found: {', '.join(sorted(data))}."
            )
        for key in ("system", "user"):
            if not str(entry.get(key) or "").strip():
                raise PromptError(f"{self.path}: '{name}' is missing a non-empty '{key}'.")
        return PromptPair(system=entry["system"].strip(), user_template=entry["user"])

    @property
    def test_cases(self) -> PromptPair:
        return self.pair("test_cases")

    @property
    def test_script(self) -> PromptPair:
        return self.pair("test_script")
