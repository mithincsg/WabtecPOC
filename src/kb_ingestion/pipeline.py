"""Discover -> extract -> normalize -> chunk -> embed -> upsert."""

from __future__ import annotations

import hashlib
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
from .storage import EmbeddingModel, VectorStore

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PYTHON_SUFFIXES = {".py", ".pyi"}
# Units whose internal layout carries meaning (indentation, a header row),
# so they are split on line boundaries rather than sentence boundaries.
_LINE_STRUCTURED_UNIT_TYPES = ("function", "class", "table")
# Units where whitespace is meaningful, so prose normalization would corrupt
# them. Tables are excluded: their cells are joined with " | " and
# collapsing the padding around that separator is desirable.
_LAYOUT_SENSITIVE_UNIT_TYPES = frozenset({"function", "class", "raw_python"})

# Config fields that change what a chunk's text or metadata looks like.
# embedding_device and embedding_batch_size are excluded deliberately: they
# change how fast embedding runs, not what gets embedded.
_CONTENT_AFFECTING_CONFIG_FIELDS = (
    "chunk_max_tokens",
    "chunk_overlap_tokens",
    "xlsx_max_rows_per_chunk",
    "pdf_heading_size_ratio",
    "pdf_table_repeat_threshold",
    "embedding_model",
)


def compute_pipeline_fingerprint(config: PipelineConfig) -> str:
    """Hash of everything that can change the chunks produced for an
    unchanged input file: every .py file in this package, plus the config
    fields above. Stamped on each chunk, so a re-run skips a file only when
    its bytes AND the code that processed it are unchanged - an extractor
    fix invalidates the cache by itself, with no version number to bump.

    Package-wide rather than per-module on purpose: a fix in one extractor
    re-embeds everything, which costs one full pass but makes it impossible
    for a stale chunk to survive a code change.
    """
    hasher = hashlib.sha256()

    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hasher.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode("utf-8"))
        hasher.update(path.read_bytes())

    content_affecting = tuple(
        getattr(config, name) for name in _CONTENT_AFFECTING_CONFIG_FIELDS
    )
    hasher.update(repr(content_affecting).encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class PreparedChunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class DiscoveredFile:
    path: Path
    document_type: str
    # Path relative to the repo root: the chunk's stable identity across
    # runs, and the label shown in the UI.
    source_path: str


@dataclass
class IngestionStats:
    files_seen: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0


def discover_files(config: PipelineConfig) -> list[DiscoveredFile]:
    """Every supported file under every configured source root, tagged with
    the document_type that root (or its subfolder) assigns.
    """
    found: list[DiscoveredFile] = []
    for source in config.sources:
        if not source.path.is_dir():
            logger.warning("Source folder does not exist, skipping: %s", source.path)
            continue
        for path in sorted(source.path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTRACTORS_BY_SUFFIX:
                continue
            if path.name.startswith("~$"):  # Excel lock file
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
        # Source root configured outside the repo (an absolute path).
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
    embedding_model: str,
) -> list[PreparedChunk]:
    """Spreadsheet rows, grouped by sheet so two sheets' rows never land in
    one chunk. At xlsx_max_rows_per_chunk=1 this is one chunk per test case,
    which is what makes a retrieved reference case a clean whole example.
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
                        embedding_model=embedding_model,
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
    embedding_model: str,
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
        embedding_model=embedding_model,
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
                        embedding_model=embedding_model,
                        text=text,
                        module_name=module_name,
                    ),
                )
            )

    return prepared


class IngestionPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        embedder: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config
        self.counter = TokenCounter(config.embedding_model)
        self.embedder = embedder
        self.vector_store = vector_store
        self.fingerprint = compute_pipeline_fingerprint(config)

    def reset(self) -> None:
        if self.vector_store is not None:
            self.vector_store.reset()

    def prepare_file(
        self, file: DiscoveredFile, file_hash: str | None = None
    ) -> tuple[str, list[PreparedChunk]]:
        """Extract, normalize, chunk and build metadata for one file. Touches
        neither the embedder nor the vector store, so it works standalone for
        --dry-run and for the static BM25 index.
        """
        file_hash = file_hash if file_hash is not None else file_sha256(file.path)
        module_name = (
            file.path.stem if file.path.suffix.lower() in _PYTHON_SUFFIXES else None
        )

        units = extract_and_normalize(file.path, self.config)
        chunks = chunk_file(
            units,
            self.config,
            self.counter,
            source_path=file.source_path,
            source_file=file.path.name,
            document_type=file.document_type,
            file_hash=file_hash,
            embedding_model=self.config.embedding_model,
            module_name=module_name,
        )
        for chunk in chunks:
            chunk.metadata.pipeline_fingerprint = self.fingerprint
        return file_hash, chunks

    def run(self, dry_run: bool = False, prune: bool = False) -> IngestionStats:
        if not dry_run and (self.embedder is None or self.vector_store is None):
            raise ValueError("embedder and vector_store are required unless dry_run=True")

        stats = IngestionStats()
        files = discover_files(self.config)
        seen_source_paths: set[str] = set()

        for file in files:
            stats.files_seen += 1
            seen_source_paths.add(file.source_path)

            current_hash = None
            if not dry_run:
                # Hashing raw bytes is cheap next to parsing, so an unchanged
                # file skips extraction and chunking too, not just embedding.
                current_hash = file_sha256(file.path)
                state = self.vector_store.get_file_state(file.source_path)
                if state == (current_hash, self.fingerprint):
                    stats.files_skipped += 1
                    logger.debug("Unchanged, skipping: %s", file.source_path)
                    continue

            try:
                _file_hash, chunks = self.prepare_file(file, file_hash=current_hash)
            except Exception:
                logger.exception("Failed to process %s", file.path)
                stats.files_failed += 1
                continue

            logger.info(
                "%s [%s] -> %d chunks", file.source_path, file.document_type, len(chunks)
            )

            if dry_run:
                stats.files_processed += 1
                continue

            # Drop this file's previous chunks first, so a changed file with
            # different chunk boundaries leaves nothing stale behind.
            stats.chunks_deleted += self.vector_store.delete_by_source_path(
                file.source_path
            )

            if chunks:
                texts = [c.text for c in chunks]
                embeddings = self.embedder.embed(texts)
                self.vector_store.upsert(
                    [c.metadata.chunk_id for c in chunks],
                    embeddings,
                    texts,
                    [c.metadata.to_chroma_dict() for c in chunks],
                )
                stats.chunks_upserted += len(chunks)

            stats.files_processed += 1

        if not dry_run and prune:
            for stale in self.vector_store.get_all_source_paths() - seen_source_paths:
                deleted = self.vector_store.delete_by_source_path(stale)
                stats.chunks_deleted += deleted
                logger.info("Pruned deleted file: %s (%d chunks)", stale, deleted)

        return stats
