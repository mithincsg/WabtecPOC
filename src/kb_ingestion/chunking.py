from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extractors import ExtractedUnit

logger = logging.getLogger(__name__)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# --- Normalization -----------------------------------------------------
# Runs right after extraction, before chunking.

_CONTROL_CHARS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0, 32) if c not in (9, 10)) + "]"
)
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_text(text: str) -> str:
    """Unicode-normalize and clean whitespace/artifacts, for prose. Idempotent.

    Collapses runs of spaces, which is right for text reflowed out of a PDF
    and wrong for source code — see normalize_code.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)  # de-hyphenate line-wrapped words
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_code(text: str) -> str:
    """Normalization for source code: same cleanup as normalize_text minus
    anything that alters layout.

    Indentation *is* syntax in Python, and alignment carries meaning in a
    test script's branch structure. Running prose normalization over a
    script collapses every indent to a single space, so a model asked to
    imitate the retrieved example has no correct example to imitate.
    Line-wrap de-hyphenation is dropped too: it would silently join
    `some-\nname` in a string literal. Idempotent.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip("\n")


# --- Tokenization --------------------------------------------------------


# Roughly how a subword tokenizer splits this corpus: letter runs, single
# digits (identifiers like "08880" become several tokens, not one) and each
# punctuation mark on its own.
_PIECE_RE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z0-9]")

# Measured against BAAI/bge-m3 over a sample of real chunks from all three
# sources: the piece count runs 1.03x (track XML) to 1.20x (API stubs) under
# the real count, so one factor covers the corpus within about 15%.
_PIECES_TO_TOKENS = 1.15


class TokenCounter:
    """Approximate token counts for chunk packing, computed locally.

    Deliberately not a real tokenizer. Nothing in this app is embedded any
    more, so `chunk_max_tokens` only bounds how much text a chunk carries
    into a prompt — an estimate within ~15% is as useful as an exact count,
    and it costs no dependency, no model download and no network call at
    startup.

    A plain word count is not good enough to substitute: on the track XML,
    where a line is `BlockId=2001 | Milepost=123.45`, it undercounts by more
    than 3x and would silently triple every chunk.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        return round(len(_PIECE_RE.findall(text)) * _PIECES_TO_TOKENS)


# --- Chunking (token-budget packing within pre-scoped structural units) ---


@dataclass(frozen=True)
class RowChunk:
    text: str
    row_start: int
    row_end: int


def _split_into_paragraphs(text: str) -> list[str]:
    return [p for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]


def _split_into_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(paragraph.strip()) if s]


def _hard_split_by_tokens(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Last-resort split for a single piece that alone exceeds max_tokens
    (e.g. a very long sentence or an oversized table row): greedily pack
    words until the token budget is hit.
    """
    words = text.split()
    if not words:
        return [text] if text else []

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and counter.count(candidate) > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _carry_overlap(
    current: list[str], counter: TokenCounter, overlap_tokens: int
) -> tuple[list[str], int]:
    if overlap_tokens <= 0:
        return [], 0
    kept: list[str] = []
    kept_tokens = 0
    for piece in reversed(current):
        piece_tokens = counter.count(piece)
        if kept and kept_tokens + piece_tokens > overlap_tokens:
            break
        kept.insert(0, piece)
        kept_tokens += piece_tokens
    return kept, kept_tokens


def _pack_pieces(
    pieces: list[str],
    counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    joiner: str,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = counter.count(piece)

        if piece_tokens > max_tokens:
            if current:
                chunks.append(joiner.join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_split_by_tokens(piece, counter, max_tokens))
            continue

        if current and current_tokens + piece_tokens > max_tokens:
            chunks.append(joiner.join(current))
            current, current_tokens = _carry_overlap(current, counter, overlap_tokens)

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append(joiner.join(current))

    return chunks


def chunk_text(
    text: str, counter: TokenCounter, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Paragraph -> sentence recursive splitting, packed into token windows
    with overlap. Used for PDF section text, module docstrings, and
    unparsable ("raw") Python files.
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    pieces: list[str] = []
    for paragraph in paragraphs:
        if counter.count(paragraph) > max_tokens:
            pieces.extend(_split_into_sentences(paragraph))
        else:
            pieces.append(paragraph)

    return _pack_pieces(pieces, counter, max_tokens, overlap_tokens, joiner="\n\n")


def chunk_code(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Keep a function/class/table whole when it fits the token budget (a
    function/method is the natural retrieval unit for reusable APIs, a table
    should keep its header with its rows). Oversized content is split on
    line boundaries, with the first line (signature or table header)
    repeated on every sub-chunk so each stays self-describing.
    """
    if counter.count(text) <= max_tokens:
        return [text] if text.strip() else []

    lines = text.splitlines()
    if not lines:
        return []

    header = lines[0]
    body_lines = lines[1:]

    chunks: list[str] = []
    current: list[str] = []
    for line in body_lines:
        candidate = "\n".join([header, *current, line])
        if current and counter.count(candidate) > max_tokens:
            chunks.append("\n".join([header, *current]))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join([header, *current]))
    if not chunks:
        chunks = [header]

    return chunks


def chunk_rows(
    row_units: list[ExtractedUnit],
    counter: TokenCounter,
    max_tokens: int,
    max_rows_per_chunk: int,
) -> list[RowChunk]:
    """Pack contiguous xlsx rows (already scoped to a single sheet by the
    caller) into groups bounded by both a token budget and a row-count cap.
    Rows are discrete records, so - unlike prose - no overlap is applied.
    """
    chunks: list[RowChunk] = []
    current_texts: list[str] = []
    current_rows: list[int] = []
    current_tokens = 0

    def flush() -> None:
        if current_texts:
            chunks.append(
                RowChunk(
                    text="\n".join(current_texts),
                    row_start=current_rows[0],
                    row_end=current_rows[-1],
                )
            )

    for unit in row_units:
        row_no = unit.locator.get("row", -1)
        row_tokens = counter.count(unit.text)

        if row_tokens > max_tokens:
            flush()
            current_texts, current_rows, current_tokens = [], [], 0
            for sub in _hard_split_by_tokens(unit.text, counter, max_tokens):
                chunks.append(RowChunk(text=sub, row_start=row_no, row_end=row_no))
            continue

        exceeds_tokens = current_texts and (current_tokens + row_tokens > max_tokens)
        exceeds_row_cap = len(current_texts) >= max_rows_per_chunk
        if exceeds_tokens or exceeds_row_cap:
            flush()
            current_texts, current_rows, current_tokens = [], [], 0

        current_texts.append(unit.text)
        current_rows.append(row_no)
        current_tokens += row_tokens

    flush()
    return chunks


# --- Metadata --------------------------------------------------------------
# Built alongside each chunk as units are packed above.


def file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_chunk_id(source_path: str, locator_key: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk id: same source/locator/index/content -> same id,
    so re-ingesting an unchanged file re-upserts the same id (idempotent).
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    raw = f"{source_path}::{locator_key}::{chunk_index}::{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class ChunkMetadata:
    chunk_id: str
    source_path: str
    source_file: str
    document_type: str
    content_type: str
    chunk_index: int
    total_chunks_in_unit: int
    file_hash: str
    indexed_at: str
    section_path: str | None = None
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    requirement_id: str | None = None
    test_case_id: str | None = None
    module_name: str | None = None
    class_name: str | None = None
    method_name: str | None = None
    table_index: int | None = None
    table_title: str | None = None
    # Track-data only: which subdivision export this chunk came from, so
    # retrieval can be restricted to the track a requirement is tested on.
    subdivision: str | None = None
    # The subdivision's display name ("Ginger"). Not searched or filtered on —
    # it is how the UI's subdivision picker labels an otherwise bare five-digit
    # ID, so it has to survive as metadata rather than only inside section_path.
    subdivision_name: str | None = None
    # Track-data only: the subdivision's own real railroad SCAC (`UP`), from
    # xml_extractor.py's `<RailroadSCAC>` -- an environment fact, not a
    # search result, so it must survive onto every chunk from that file the
    # same way subdivision/subdivision_name do.
    railroad_scac: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Drop Nones and the id, which the index stores separately."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and k != "chunk_id"
        }


def build_metadata(
    *,
    source_path: str,
    source_file: str,
    document_type: str,
    unit: ExtractedUnit,
    chunk_index: int,
    total_chunks_in_unit: int,
    file_hash: str,
    text: str,
    module_name: str | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> ChunkMetadata:
    locator = unit.locator
    extra = unit.extra
    locator_key = "|".join(f"{k}={v}" for k, v in sorted(locator.items()))
    chunk_id = make_chunk_id(source_path, locator_key, chunk_index, text)

    class_name: str | None = None
    method_name: str | None = None
    if unit.unit_type == "class":
        class_name = locator.get("name")
    elif unit.unit_type == "function":
        method_name = locator.get("name")
        class_name = locator.get("class_name")

    return ChunkMetadata(
        chunk_id=chunk_id,
        source_path=source_path,
        source_file=source_file,
        document_type=document_type,
        content_type=unit.unit_type,
        chunk_index=chunk_index,
        total_chunks_in_unit=total_chunks_in_unit,
        file_hash=file_hash,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        section_path=locator.get("section_path") or None,
        page=locator.get("page"),
        sheet=locator.get("sheet"),
        row_start=row_start if row_start is not None else locator.get("row"),
        row_end=row_end if row_end is not None else locator.get("row"),
        requirement_id=extra.get("requirement_id"),
        test_case_id=extra.get("test_case_id"),
        module_name=module_name,
        class_name=class_name,
        method_name=method_name,
        table_index=locator.get("table"),
        table_title=locator.get("table_title"),
        subdivision=locator.get("subdivision"),
        subdivision_name=locator.get("subdivision_name"),
        railroad_scac=locator.get("railroad_scac"),
    )
