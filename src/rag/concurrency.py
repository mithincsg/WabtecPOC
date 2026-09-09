from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retrieval is a handful of independent blocking calls — a ChromaDB HNSW
# query, a BM25 pass, a sentence-transformers forward pass — and every one of
# them releases the GIL inside C or torch. Running them on threads therefore
# gives real wall-clock overlap, which matters because retrieval is the part
# of a request that is *not* the model and so is the part worth shrinking.
#
# Deliberately one flat pool used at a single level: nothing submitted to it
# ever waits on another task submitted to it, so it cannot deadlock by
# exhausting its own workers.
_MAX_WORKERS = max(2, int(os.getenv("RAG_RETRIEVAL_WORKERS", "6")))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="rag")


def run_parallel(tasks: list[Callable[[], T]]) -> list[T]:
    """Runs zero-argument callables concurrently, returning results in order.

    A single task runs inline: submitting it would cost a thread hand-off and
    buy nothing. An exception in any task propagates, as it would if the
    calls had been made in sequence.
    """
    if not tasks:
        return []
    if len(tasks) == 1:
        return [tasks[0]()]

    futures = [_EXECUTOR.submit(task) for task in tasks]
    return [future.result() for future in futures]


def shutdown() -> None:
    """Only for tests and clean process exit; the pool is otherwise
    process-lifetime.
    """
    _EXECUTOR.shutdown(wait=False)
