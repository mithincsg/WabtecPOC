from __future__ import annotations

import hashlib
from pathlib import Path

from .config import PipelineConfig

_PACKAGE_ROOT = Path(__file__).resolve().parent

# Config fields that change what a chunk's text or metadata looks like.
# embedding_device and embedding_batch_size are deliberately excluded: they
# change how fast embedding runs, not what gets embedded, so moving between
# CPU and GPU shouldn't force a full re-embed.
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
    fields above. Stamped on each chunk, so a re-run can skip a file only
    when its bytes AND the code that processed it are both unchanged — an
    extractor bug fix invalidates the cache by itself, with no version
    number to remember to bump.

    Deliberately package-wide rather than per-module: a fix in one extractor
    re-embeds everything rather than only that format, which costs one full
    pass after a code change but makes it impossible for a stale chunk to
    survive one.
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
