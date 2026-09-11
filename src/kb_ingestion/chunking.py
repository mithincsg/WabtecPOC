"""Normalization, token-budget packing, and chunk metadata.

Structure decides the chunk boundaries (see extractors.py); the token budget
here only applies when a single structural unit exceeds it.
"""

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
_CONTROL_CHARS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0, 32) if c not in (9, 10)) + "]"
)
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


# --- Normalization ---------------------------------------------------------


def normalize_text(text: str) -> str:
    """Unicode-normalize and clean whitespace for prose. Idempotent.

    Collapses runs of spaces, which is right for text reflowed out of a PDF
    and wrong for source code - see normalize_code.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)  # de-hyphenate wrapped words
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_code(text: str) -> str:
    """Normalization for source code: the same cleanup minus anything that
    alters layout. Indentation is syntax in Python, and a model asked to
    imitate a retrieved script needs the layout intact. Idempotent.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip("\n")


# --- Tokenization ----------------------------------------------------------


class TokenCounter:
    """The bge-m3 tokenizer, for accurate counts during chunk packing. Falls
    back to a whitespace-word approximation if it cannot be loaded (e.g. no
    network), so the pipeline degrades instead of hard-failing.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._tokenizer = None
        self._load_failed = False

    def _tokenizer_instance(self):
        if self._tokenizer is not None or self._load_failed:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
            logger.warning(
                "Could not load tokenizer for %s (%s); falling back to a "
                "whitespace-based token approximation.",
                self.model_name,
                exc,
            )
            self._load_failed = True
        return self._tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self._tokenizer_instance()
        if tokenizer is None:
            return len(text.split())
        return len(tokenizer.encode(text, add_special_tokens=False))


# --- Chunking --------------------------------------------------------------


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
    """Last resort for a single piece that alone exceeds max_tokens: greedily
    pack words until the budget is hit.
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
    """Paragraph -> sentence splitting, packed into token windows with
    overlap. Used for PDF section text, module docstrings and raw Python.
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
    """Keeps a function/class/table whole when it fits the budget. Oversized
    content is split on line boundaries with the first line (signature or
    table header) repeated, so each sub-chunk stays self-describing.
    """
    if counter.count(text) <= max_tokens:
        return [text] if text.strip() else []

    lines = text.splitlines()
    if not lines:
        return []

    header = lines[0]
    chunks: list[str] = []
    current: list[str] = []
    for line in lines[1:]:
        candidate = "\n".join([header, *current, line])
        if current and counter.count(candidate) > max_tokens:
            chunks.append("\n".join([header, *current]))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join([header, *current]))
    return chunks or [header]


def chunk_rows(
    row_units: list[ExtractedUnit],
    counter: TokenCounter,
    max_tokens: int,
    max_rows_per_chunk: int,
) -> list[RowChunk]:
    """Packs contiguous xlsx rows (already scoped to one sheet by the caller)
    into groups bounded by a token budget and a row-count cap. Rows are
    discrete records, so - unlike prose - no overlap is applied.
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


def file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_chunk_id(source_path: str, locator_key: str, chunk_index: int, text: str) -> str:
    """Deterministic id: the same source/locator/index/content gives the same
    id, so re-ingesting an unchanged file re-upserts rather than duplicates.
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
    embedding_model: str
    ingested_at: str
    # Fingerprint of the ingestion code plus the config fields that affect
    # chunk content (see pipeline.compute_pipeline_fingerprint), stamped on
    # after chunking. Half of the unchanged-file skip.
    pipeline_fingerprint: str | None = None
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
    # Track data only: which subdivision report this chunk came from.
    subdivision: str | None = None

    def to_chroma_dict(self) -> dict[str, Any]:
        """Chroma metadata must be str/int/float/bool - drop Nones and the id
        (stored separately as the Chroma document id).
        """
        return {
            k: v for k, v in self.__dict__.items() if v is not None and k != "chunk_id"
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
    embedding_model: str,
    text: str,
    module_name: str | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> ChunkMetadata:
    locator = unit.locator
    extra = unit.extra
    locator_key = "|".join(f"{k}={v}" for k, v in sorted(locator.items()))

    class_name: str | None = None
    method_name: str | None = None
    if unit.unit_type == "class":
        class_name = locator.get("name")
    elif unit.unit_type == "function":
        method_name = locator.get("name")
        class_name = locator.get("class_name")

    return ChunkMetadata(
        chunk_id=make_chunk_id(source_path, locator_key, chunk_index, text),
        source_path=source_path,
        source_file=source_file,
        document_type=document_type,
        content_type=unit.unit_type,
        chunk_index=chunk_index,
        total_chunks_in_unit=total_chunks_in_unit,
        file_hash=file_hash,
        embedding_model=embedding_model,
        ingested_at=datetime.now(timezone.utc).isoformat(),
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
    )
