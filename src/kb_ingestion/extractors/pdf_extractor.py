from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf
except ImportError:  # PyMuPDF < 1.24.3 only ships the legacy `fitz` alias
    import fitz as pymupdf

from . import ExtractedUnit

_HEADING_SIZE_RATIO_DEFAULT = 1.15
_TABLE_REPEAT_THRESHOLD_DEFAULT = 0.5
_LINE_REPEAT_THRESHOLD = 0.5
_FALLBACK_HEADING_MAX_WORDS = 12
_MIN_TABLE_ROWS = 2

_DIGITS_RE = re.compile(r"\d+")
_HEADING_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+){0,5})\.?\s+\S")
_DOT_LEADER_RE = re.compile(r"\.{4,}")
_DECORATIVE_PREFIX_RE = re.compile(r"^[/\\|•·\s]+")


# --- Heading/section-hierarchy detection -----------------------------------
# (Line grouping, heading classification, and the section stack used to
# build a document -> section -> subsection -> content hierarchy as the PDF
# is read page by page.)


@dataclass
class Line:
    text: str
    top: float
    bottom: float
    size: float
    bold: bool
    page: int


@dataclass
class HeadingCandidate:
    level: int
    title: str


def group_words_into_lines(words: list[dict], page: int, tolerance: float = 2.0) -> list[Line]:
    """Cluster words into visual lines by vertical position, tolerant of
    the small sub-pixel 'top' jitter between words on the same line. Takes
    the format-neutral dicts produced by `_page_words` (x0/x1/top/bottom/
    text/size/fontname), so the line-grouping, heading-classification and
    section-stack logic below is independent of which PDF library read the
    page.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    clusters: list[list[dict]] = []
    for word in ordered:
        if clusters and abs(word["top"] - clusters[-1][-1]["top"]) <= tolerance:
            clusters[-1].append(word)
        else:
            clusters.append([word])

    lines: list[Line] = []
    for cluster in clusters:
        cluster.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in cluster)
        size = max(w.get("size", 0) for w in cluster)
        bold = any("bold" in str(w.get("fontname", "")).lower() for w in cluster)
        top = min(w["top"] for w in cluster)
        bottom = max(w["bottom"] for w in cluster)
        lines.append(Line(text=text, top=top, bottom=bottom, size=size, bold=bold, page=page))
    return lines


def strip_decorative_prefix(text: str) -> str:
    return _DECORATIVE_PREFIX_RE.sub("", text).strip()


def compute_body_size(all_line_sizes: list[float]) -> float:
    """The most common line font size in the document, used as the body-text
    baseline that headings are measured against.
    """
    if not all_line_sizes:
        return 0.0
    counts: dict[float, int] = {}
    for size in all_line_sizes:
        rounded = round(size, 1)
        counts[rounded] = counts.get(rounded, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def build_size_rank(candidate_sizes: list[float]) -> dict[float, int]:
    """Maps distinct heading-candidate font sizes to a level, largest first,
    for headings with no numbering to key off of.
    """
    distinct = sorted({round(s, 1) for s in candidate_sizes}, reverse=True)
    return {size: min(i + 1, 4) for i, size in enumerate(distinct)}


class HeadingClassifier:
    def __init__(self, body_size: float, size_rank: dict[float, int], size_ratio: float = 1.15):
        self.body_size = body_size
        self.size_rank = size_rank
        self.size_ratio = size_ratio

    def classify(self, line: Line) -> HeadingCandidate | None:
        if _DOT_LEADER_RE.search(line.text):
            return None

        title = strip_decorative_prefix(line.text)
        if not title:
            return None

        match = _HEADING_NUMBER_RE.match(title)
        if match and (line.bold or line.size > self.body_size * 1.05):
            level = len(match.group(1).split("."))
            return HeadingCandidate(level=level, title=title)

        is_large = self.body_size and line.size > self.body_size * self.size_ratio
        is_short = len(title.split()) <= _FALLBACK_HEADING_MAX_WORDS
        if is_large and line.bold and is_short:
            level = self.size_rank.get(round(line.size, 1), 1)
            return HeadingCandidate(level=level, title=title)

        return None


class SectionStack:
    def __init__(self):
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))

    @property
    def path(self) -> str:
        return " > ".join(title for _, title in self._stack)

    @property
    def current_title(self) -> str:
        return self._stack[-1][1] if self._stack else ""


# --- Extraction --------------------------------------------------------


def _word_center_in_bbox(word: dict, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _table_to_text(table: list[list[str | None]]) -> str:
    lines = []
    for row in table:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _digit_collapsed(text: str) -> str:
    """Digit-insensitive signature for repeat detection - a running admin
    header table often repeats a label like "Page 10 of 35" with only the
    page number changing, which would otherwise defeat exact-text matching.
    """
    return _DIGITS_RE.sub("#", text)


# --- PyMuPDF page reading --------------------------------------------------
# Everything above this point is format-neutral; these two functions are the
# only place the PDF library itself appears. They flatten a PyMuPDF page into
# the same word/table dicts the line grouper expects.

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for synthetic/real bold


def _page_words(page) -> list[dict]:
    """Words with their font size and boldness, from PyMuPDF's structured
    text. `get_text("words")` is faster but drops font attributes, and
    heading detection needs them, so spans are split on whitespace instead
    and each word inherits its span's font.
    """
    words: list[dict] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span.get("text") or ""
                if not span_text.strip():
                    continue
                x0, top, x1, bottom = span["bbox"]
                size = span.get("size", 0.0)
                is_bold = bool(span.get("flags", 0) & _BOLD_FLAG) or "bold" in str(
                    span.get("font", "")
                ).lower()
                fontname = span.get("font", "") + ("-Bold" if is_bold else "")

                # Span bboxes cover the whole span, so per-word x extents are
                # apportioned by character count. Only used to order words
                # within a line and to test table containment, so approximate
                # widths are fine; the y extents (what line grouping keys off)
                # are exact.
                total_chars = max(len(span_text), 1)
                char_width = (x1 - x0) / total_chars
                cursor = 0
                for token in span_text.split():
                    start = span_text.index(token, cursor)
                    cursor = start + len(token)
                    words.append(
                        {
                            "text": token,
                            "x0": x0 + start * char_width,
                            "x1": x0 + cursor * char_width,
                            "top": top,
                            "bottom": bottom,
                            "size": size,
                            "fontname": fontname,
                        }
                    )
    return words


def _looks_tabular(rows: list[list[str | None]]) -> bool:
    """Rejects false-positive tables.

    PyMuPDF's finder keys off ruling lines, so a decorative box or a framed
    "this page intentionally left blank" notice comes back as a 2x2 table
    whose cells are fragments of one sentence. Emitting those as table units
    both adds noise and removes their words from the prose stream, so a
    detected table has to actually look tabular: at least two columns, and
    at least two rows that are themselves multi-column, with enough filled
    cells to be a grid rather than a split sentence.
    """
    cells = [[(c or "").strip() for c in row] for row in rows]
    if max((len(row) for row in cells), default=0) < 2:
        return False

    filled = sum(1 for row in cells for c in row if c)
    multi_column_rows = sum(1 for row in cells if sum(1 for c in row if c) >= 2)
    return multi_column_rows >= _MIN_TABLE_ROWS and filled >= 6


def _page_tables(page) -> list[dict]:
    """Tables via PyMuPDF's table finder. Returns {bbox, text} so a table can
    both be emitted as its own unit and have its words excluded from the
    prose stream.
    """
    tables: list[dict] = []
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001 - table finding is best-effort
        return tables

    for table in found.tables:
        try:
            data = table.extract()
        except Exception:  # noqa: BLE001
            continue
        if len(data) < _MIN_TABLE_ROWS or not _looks_tabular(data):
            continue
        text = _table_to_text(data)
        if text.strip():
            tables.append({"bbox": tuple(table.bbox), "text": text})
    return tables


class PDFExtractor:
    """Structure-aware PDF extraction (PyMuPDF): builds a heading hierarchy (numbering
    patterns + font-size/bold tiers) as it walks the document, emits prose as
    per-section `section_text` units (never spanning two sections) and
    tables as their own units tagged with the active section and a detected
    title - never merging table content into the prose stream.
    """

    def __init__(
        self,
        heading_size_ratio: float = _HEADING_SIZE_RATIO_DEFAULT,
        table_repeat_threshold: float = _TABLE_REPEAT_THRESHOLD_DEFAULT,
    ):
        self.heading_size_ratio = heading_size_ratio
        self.table_repeat_threshold = table_repeat_threshold

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        pages_data = self._read_pages(file_path)
        if not pages_data:
            return []

        classifier = self._build_classifier(pages_data)
        noisy_table_signatures = self._detect_repeated_tables(pages_data)
        noisy_line_signatures = self._detect_repeated_lines(pages_data)
        return self._build_units(pages_data, classifier, noisy_table_signatures, noisy_line_signatures)

    def _read_pages(self, file_path: Path) -> list[tuple[int, list[Line], list[dict]]]:
        pages_data: list[tuple[int, list[Line], list[dict]]] = []
        with pymupdf.open(file_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                tables = _page_tables(page)
                words = [
                    w
                    for w in _page_words(page)
                    if not any(_word_center_in_bbox(w, t["bbox"]) for t in tables)
                ]
                lines = group_words_into_lines(words, page_index)
                pages_data.append((page_index, lines, tables))
        return pages_data

    def _build_classifier(
        self, pages_data: list[tuple[int, list[Line], list[dict]]]
    ) -> HeadingClassifier:
        all_sizes = [line.size for _, lines, _ in pages_data for line in lines]
        body_size = compute_body_size(all_sizes)

        candidate_sizes = [
            line.size
            for _, lines, _ in pages_data
            for line in lines
            if line.bold
            and body_size
            and line.size > body_size * self.heading_size_ratio
            and len(line.text.split()) <= _FALLBACK_HEADING_MAX_WORDS
        ]
        size_rank = build_size_rank(candidate_sizes)
        return HeadingClassifier(body_size, size_rank, self.heading_size_ratio)

    def _detect_repeated_tables(
        self, pages_data: list[tuple[int, list[Line], list[dict]]]
    ) -> set[str]:
        """Tables whose (digit-insensitive) text repeats across most pages
        are running boilerplate (e.g. a per-page admin header table that
        only differs by a "Page N of M" cell), not real content.
        """
        signature_pages: dict[str, set[int]] = {}
        for page_index, _, tables in pages_data:
            for t in tables:
                sig = _digit_collapsed(t["text"])
                signature_pages.setdefault(sig, set()).add(page_index)

        num_pages = len(pages_data)
        threshold = max(2, round(num_pages * self.table_repeat_threshold))
        return {sig for sig, pages in signature_pages.items() if len(pages) >= threshold}

    def _detect_repeated_lines(
        self, pages_data: list[tuple[int, list[Line], list[dict]]]
    ) -> set[str]:
        """Lines that repeat identically across most pages (running
        headers/footers, page numbers) are noise, not section content - the
        per-page-blob equivalent of this used to run after extraction; doing
        it here, on individual lines, means a footer never leaks into a
        section's text and never gets mistaken for a heading.
        """
        if len(pages_data) < 3:
            return set()

        line_pages: dict[str, set[int]] = {}
        for page_index, lines, _ in pages_data:
            texts = {line.text.strip() for line in lines if line.text.strip()}
            for sig in {_digit_collapsed(text) for text in texts}:
                line_pages.setdefault(sig, set()).add(page_index)

        threshold = max(2, round(len(pages_data) * _LINE_REPEAT_THRESHOLD))
        return {text for text, pages in line_pages.items() if len(pages) >= threshold}

    def _build_units(
        self,
        pages_data: list[tuple[int, list[Line], list[dict]]],
        classifier: HeadingClassifier,
        noisy_table_signatures: set[str],
        noisy_line_signatures: set[str],
    ) -> list[ExtractedUnit]:
        builder = _SectionUnitBuilder(classifier, noisy_table_signatures, noisy_line_signatures)
        for page_index, lines, tables in pages_data:
            builder.process_page(page_index, lines, tables)
        builder.flush()
        return builder.units


class _SectionUnitBuilder:
    """Walks a PDF's (line, table) events in reading order, tracking the
    active section so prose is flushed into per-section units and tables are
    emitted standalone, tagged with the section active when they appear.
    """

    def __init__(
        self,
        classifier: HeadingClassifier,
        noisy_table_signatures: set[str],
        noisy_line_signatures: set[str],
    ):
        self._classifier = classifier
        self._noisy_table_signatures = noisy_table_signatures
        self._noisy_line_signatures = noisy_line_signatures
        self.units: list[ExtractedUnit] = []
        self._stack = SectionStack()
        self._buffer: list[str] = []
        self._buffer_page_start: int | None = None
        self._last_body_line_text = ""
        self._table_counter = 0

    def process_page(self, page_index: int, lines: list[Line], tables: list[dict]) -> None:
        events: list[tuple[float, str, object]] = [(line.top, "line", line) for line in lines]
        events += [(t["bbox"][1], "table", t) for t in tables]
        events.sort(key=lambda e: e[0])

        for _, kind, obj in events:
            if kind == "line":
                self._handle_line(page_index, obj)
            else:
                self._handle_table(page_index, obj)

    def _handle_line(self, page_index: int, line: Line) -> None:
        if _digit_collapsed(line.text.strip()) in self._noisy_line_signatures:
            return
        heading = self._classifier.classify(line)
        if heading:
            self.flush()
            self._stack.push(heading.level, heading.title)
            # Don't let a table with no body text before it in its own
            # section inherit a table_title left over from a prior section.
            self._last_body_line_text = ""
            return
        if self._buffer_page_start is None:
            self._buffer_page_start = page_index
        self._buffer.append(line.text)
        self._last_body_line_text = line.text

    def _handle_table(self, page_index: int, table: dict) -> None:
        if _digit_collapsed(table["text"]) in self._noisy_table_signatures:
            return
        self.flush()
        self._table_counter += 1
        self.units.append(
            ExtractedUnit(
                text=table["text"],
                unit_type="table",
                locator={
                    "page": page_index,
                    "section_path": self._stack.path,
                    "table_title": self._last_body_line_text or self._stack.current_title,
                    "table": self._table_counter,
                },
            )
        )

    def flush(self) -> None:
        if self._buffer:
            text = "\n".join(self._buffer)
            if text.strip():
                self.units.append(
                    ExtractedUnit(
                        text=text,
                        unit_type="section_text",
                        locator={
                            "page": self._buffer_page_start,
                            "section_path": self._stack.path,
                            "section_title": self._stack.current_title,
                        },
                    )
                )
            self._buffer.clear()
        self._buffer_page_start = None
