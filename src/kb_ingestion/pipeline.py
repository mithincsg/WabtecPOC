from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .chunking import (
    ChunkMetadata,
    TokenCounter,
    build_metadata,
    chunk_code,
    chunk_rows,
    chunk_text,
    file_sha256,
    normalize_code,
    normalize_text,
)
from .config import PipelineConfig, SourceRoot
from .extractors import EXTRACTORS_BY_SUFFIX, ExtractedUnit, PDFExtractor

logger = logging.getLogger(__name__)

_PYTHON_SUFFIXES = {".py", ".pyi"}
# Units whose internal layout carries meaning (indentation, a header row),
# so they're split on line boundaries rather than sentence boundaries.
_LINE_STRUCTURED_UNIT_TYPES = ("function", "class", "table")
# Units where whitespace is meaningful, so prose normalization would corrupt
# them. Tables are excluded deliberately: their cells are joined with " | ",
# and collapsing the padding around that separator is desirable.
_LAYOUT_SENSITIVE_UNIT_TYPES = frozenset({"function", "class", "raw_python"})


@dataclass
class PreparedChunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class DiscoveredFile:
    path: Path
    document_type: str
    # Path relative to the repo root, used as the chunk's stable identity
    # across runs and as the label shown in the UI.
    source_path: str


def discover_files(config: PipelineConfig) -> list[DiscoveredFile]:
    """Every supported file under every configured source root, tagged with
    the document_type that root (or its subfolder) assigns.
    """
    found: list[DiscoveredFile] = []
    for source in config.static_sources:
        if not source.path.is_dir():
            logger.warning("Source folder does not exist, skipping: %s", source.path)
            continue
        for path in sorted(source.path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTRACTORS_BY_SUFFIX:
                continue
            if path.name.startswith("~$"):
                # Office lock file for a workbook currently open in Excel.
                continue
            found.append(
                DiscoveredFile(
                    path=path,
                    document_type=source.type_for(path),
                    source_path=_source_path_for(path, source, config),
                )
            )
    return found


def _source_path_for(path: Path, source: SourceRoot, config: PipelineConfig) -> str:
    try:
        return path.relative_to(config.base_dir).as_posix()
    except ValueError:
        # Source root configured outside the repo (an absolute path) — fall
        # back to a path relative to that root so it still reads sensibly.
        return f"{source.document_type}/{path.relative_to(source.path).as_posix()}"


def extract_and_normalize(file_path: Path, config: PipelineConfig) -> list[ExtractedUnit]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        extractor = PDFExtractor(
            heading_size_ratio=config.pdf_heading_size_ratio,
            table_repeat_threshold=config.pdf_table_repeat_threshold,
        )
    else:
        extractor = EXTRACTORS_BY_SUFFIX[suffix]

    units = extractor.extract(file_path)
    for unit in units:
        if unit.unit_type in _LAYOUT_SENSITIVE_UNIT_TYPES:
            unit.text = normalize_code(unit.text)
        else:
            unit.text = normalize_text(unit.text)
    return [u for u in units if u.text.strip()]


def _chunk_row_units(
    row_units: list[ExtractedUnit],
    config: PipelineConfig,
    counter: TokenCounter,
    *,
    source_path: str,
    source_file: str,
    document_type: str,
    file_hash: str,
) -> list[PreparedChunk]:
    """Spreadsheet rows, grouped by sheet so two sheets' rows never land in
    one chunk. With xlsx_max_rows_per_chunk at its default of 1 this is one
    chunk per test case, which is what makes a retrieved reference test case
    a clean, whole example for the LLM to imitate.
    """
    prepared: list[PreparedChunk] = []
    rows_by_sheet: dict[str, list[ExtractedUnit]] = {}
    for unit in row_units:
        rows_by_sheet.setdefault(unit.locator.get("sheet", ""), []).append(unit)

    for sheet, sheet_rows in rows_by_sheet.items():
        row_by_number = {u.locator.get("row"): u for u in sheet_rows}
        row_chunks = chunk_rows(
            sheet_rows, counter, config.chunk_max_tokens, config.xlsx_max_rows_per_chunk
        )
        for idx, row_chunk in enumerate(row_chunks):
            source_unit = row_by_number.get(row_chunk.row_start, sheet_rows[0])
            representative = ExtractedUnit(
                text="",
                unit_type="test_case",
                locator={"sheet": sheet},
                extra=source_unit.extra,
            )
            prepared.append(
                PreparedChunk(
                    text=row_chunk.text,
                    metadata=build_metadata(
                        source_path=source_path,
                        source_file=source_file,
                        document_type=document_type,
                        unit=representative,
                        chunk_index=idx,
                        total_chunks_in_unit=len(row_chunks),
                        file_hash=file_hash,
                        text=row_chunk.text,
                        row_start=row_chunk.row_start,
                        row_end=row_chunk.row_end,
                    ),
                )
            )
    return prepared


def chunk_file(
    units: list[ExtractedUnit],
    config: PipelineConfig,
    counter: TokenCounter,
    *,
    source_path: str,
    source_file: str,
    document_type: str,
    file_hash: str,
    module_name: str | None = None,
) -> list[PreparedChunk]:
    row_units = [u for u in units if u.unit_type == "test_case"]
    other_units = [u for u in units if u.unit_type != "test_case"]

    prepared = _chunk_row_units(
        row_units,
        config,
        counter,
        source_path=source_path,
        source_file=source_file,
        document_type=document_type,
        file_hash=file_hash,
    )

    for unit in other_units:
        if unit.unit_type in _LINE_STRUCTURED_UNIT_TYPES:
            texts = chunk_code(unit.text, counter, config.chunk_max_tokens)
        else:
            texts = chunk_text(
                unit.text, counter, config.chunk_max_tokens, config.chunk_overlap_tokens
            )

        for idx, text in enumerate(texts):
            prepared.append(
                PreparedChunk(
                    text=text,
                    metadata=build_metadata(
                        source_path=source_path,
                        source_file=source_file,
                        document_type=document_type,
                        unit=unit,
                        chunk_index=idx,
                        total_chunks_in_unit=len(texts),
                        file_hash=file_hash,
                        text=text,
                        module_name=module_name,
                    ),
                )
            )

    return prepared


class ChunkingPipeline:
    """Source folders -> chunks, in memory.

    Nothing is embedded or persisted: every source the app reads is held in
    the in-process BM25 index that src/rag/static_context.py builds from
    these chunks on startup.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.counter = TokenCounter()

    def prepare_file(
        self, file: DiscoveredFile, file_hash: str | None = None
    ) -> tuple[str, list[PreparedChunk]]:
        """Extract, normalize, chunk and build metadata for one file. Pass a
        precomputed hash to avoid re-reading a file the caller already hashed.
        """
        file_hash = file_hash if file_hash is not None else file_sha256(file.path)
        module_name = (
            file.path.stem if file.path.suffix.lower() in _PYTHON_SUFFIXES else None
        )

        units = extract_and_normalize(file.path, self.config)
        return file_hash, chunk_file(
            units,
            self.config,
            self.counter,
            source_path=file.source_path,
            source_file=file.path.name,
            document_type=file.document_type,
            file_hash=file_hash,
            module_name=module_name,
        )
