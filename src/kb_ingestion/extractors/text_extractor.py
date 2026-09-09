from __future__ import annotations

import ast
from pathlib import Path

from . import ExtractedUnit
from .python_extractor import PythonExtractor

# Enough of a signal to be worth trying `ast.parse`. Checked before parsing
# because a prose document occasionally parses as valid Python by accident
# (a single bare word is a valid expression statement), and that would send
# it down the code path.
_PYTHON_MARKERS = ("import ", "def ", "class ", "elif ", "self.")
_MIN_PYTHON_MARKERS = 2


def looks_like_python(text: str) -> bool:
    """Whether a .txt file is actually Python source.

    The reference automation scripts in this project are Python saved with a
    .txt extension. Treated as prose they get sentence-split and space-
    collapsed, which destroys the indentation and branch structure that make
    them useful as examples — so the content decides the parser, not the
    extension.
    """
    if sum(marker in text for marker in _PYTHON_MARKERS) < _MIN_PYTHON_MARKERS:
        return False
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


class TextExtractor:
    """Plain-text files. Prose becomes a single text unit, chunked downstream
    like PDF section text; a file that turns out to be Python is handed to
    the ast-based extractor instead, so it is split on function and class
    boundaries with its layout intact.
    """

    def __init__(self):
        self._python_extractor = PythonExtractor()

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        if looks_like_python(text):
            return self._python_extractor.extract(file_path)
        return [ExtractedUnit(text=text, unit_type="text", locator={})]
