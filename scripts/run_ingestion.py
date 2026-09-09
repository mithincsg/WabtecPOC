"""Build or refresh the knowledge base.

    python scripts/run_ingestion.py --dry-run   # parse only, no model, no DB
    python scripts/run_ingestion.py             # embed + write to ChromaDB
    python scripts/run_ingestion.py --reset     # wipe the collection first
    python scripts/run_ingestion.py --prune     # drop chunks for deleted files
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb_ingestion.config import PipelineConfig  # noqa: E402
from kb_ingestion.embeddings import LocalBGEM3Embedder  # noqa: E402
from kb_ingestion.pipeline import IngestionPipeline, discover_files  # noqa: E402
from kb_ingestion.vector_store import VectorStore  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk only — no embedding model, no database writes.",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete the collection before ingesting."
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove chunks for files no longer present in the source folders.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig.load(args.config)

    if args.dry_run:
        pipeline = IngestionPipeline(config)
    else:
        pipeline = IngestionPipeline(
            config,
            embedder=LocalBGEM3Embedder(
                model_name=config.embedding_model,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
            ),
            vector_store=VectorStore(config.chroma_persist_dir, config.chroma_collection),
        )
        if args.reset:
            pipeline.reset()

    files = discover_files(config)
    if not files:
        roots = ", ".join(str(s.path) for s in config.sources)
        print(f"No supported files found under: {roots}")
        return 1

    stats = pipeline.run(dry_run=args.dry_run, prune=args.prune)

    print(
        f"\nseen {stats.files_seen}  processed {stats.files_processed}  "
        f"skipped {stats.files_skipped}  failed {stats.files_failed}\n"
        f"chunks upserted {stats.chunks_upserted}  deleted {stats.chunks_deleted}"
    )
    if not args.dry_run:
        print(f"collection now holds {pipeline.vector_store.count()} chunks")
    return 1 if stats.files_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
