from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ExtractedUnit:
    """One logical unit pulled out of a source document, before chunking.

    A "unit" is the natural granularity of its format: a PDF section or
    table, a spreadsheet row (one test case), a Python function/class, an
    HTML track-report section. Chunking may split a unit that is too large,
    or pack several small ones together — it never mixes units of different
    unit_types, which is what keeps a test case from being glued to a
    paragraph of prose.
    """

    text: str
    # section_text | table | test_case | function | class | module_docstring
    # | raw_python | text
    unit_type: str
    locator: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class Extractor(Protocol):
    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        ...


from .html_extractor import HTMLTrackDataExtractor  # noqa: E402
from .pdf_extractor import PDFExtractor  # noqa: E402
from .python_extractor import PythonExtractor  # noqa: E402
from .text_extractor import TextExtractor  # noqa: E402
from .xlsx_extractor import XLSXExtractor  # noqa: E402

_python_extractor = PythonExtractor()
_xlsx_extractor = XLSXExtractor()

# One parser per format, as the brief specifies: PyMuPDF for PDF, openpyxl
# for spreadsheets, stdlib ast for Python stubs, stdlib html.parser for the
# track reports.
EXTRACTORS_BY_SUFFIX: dict[str, Extractor] = {
    ".pdf": PDFExtractor(),
    ".xlsx": _xlsx_extractor,
    ".xlsm": _xlsx_extractor,
    ".py": _python_extractor,
    ".pyi": _python_extractor,
    ".txt": TextExtractor(),
    ".html": HTMLTrackDataExtractor(),
    ".htm": HTMLTrackDataExtractor(),
}

__all__ = [
    "EXTRACTORS_BY_SUFFIX",
    "ExtractedUnit",
    "Extractor",
    "HTMLTrackDataExtractor",
    "PDFExtractor",
    "PythonExtractor",
    "TextExtractor",
    "XLSXExtractor",
]
