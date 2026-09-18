from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .cache import LruTtlCache

logger = logging.getLogger(__name__)

# Tokens worth matching on in this domain are not all plain words:
# "TBC137", "L2R9479", "08880", "wcr_loco_sim", "IV132.0". Splitting on
# non-alphanumerics but keeping digits attached to their letters preserves
# parameter and requirement identifiers as single tokens, which is exactly
# what a requirement names and what a generated script has to get exactly
# right.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\d+[A-Za-z]*)*|\d+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# How many times an identifier token counts on the query side. BM25 sums a
# contribution per query token, so a requirement that names TBC137 once but
# says "speed", "restricted" and "enforce" throughout scores records matching
# those common words several times above the one record it actually names —
# TBC137's own record came 13th on a one-sentence requirement. Counting an
# identifier as several occurrences of itself restores the balance without
# touching the corpus or discarding the prose terms, which carry the rest of
# the requirement's meaning (deduplicating the query instead makes it worse:
# the repeated prose is also evidence).
_IDENTIFIER_BOOST = 3


def _is_identifier(token: str) -> bool:
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def boost_identifiers(tokens: list[str]) -> list[str]:
    boosted: list[str] = []
    for token in tokens:
        boosted.extend([token] * (_IDENTIFIER_BOOST if _is_identifier(token) else 1))
    return boosted


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class _Corpus:
    """One built index, and everything scored against it.

    A single immutable object rather than five parallel attributes on the
    index, because searches now run on several threads at once: a searcher
    takes one reference to this and reads nothing else, so a rebuild landing
    mid-search swaps the whole corpus instead of leaving that searcher
    reading new ids against old texts.

    The memoised scores live here too, which makes their lifetime correct by
    construction — they are discarded with the corpus they describe and can
    never be served against a different one.
    """

    bm25: BM25Okapi
    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    count: int
    scores: LruTtlCache = field(default_factory=LruTtlCache)


class KeywordIndex:
    """BM25 over every chunk of every source — the app's only retrieval arm.

    Built in process from the store's documents, and rebuilt whenever the
    chunk count changes; on a corpus of this size the build is well under a
    second, so there's no persistence layer to keep in sync.

    Scoring, unlike the build, is *not* cheap: `BM25Okapi.get_scores` walks
    every chunk in Python for every query. One request scores the same query
    against the same corpus several times over — the API pass, the track pass
    and the parameter pass during script generation — differing only in the
    metadata filter applied afterwards. Scores are therefore memoised per
    token set, which reduces the repeat passes to a filtered walk of an array
    that already exists.
    """

    def __init__(self, store, score_cache_entries: int = 64):
        self._store = store
        self._lock = threading.Lock()
        self._score_cache_entries = score_cache_entries
        self._corpus: _Corpus | None = None

    def _ensure_built(self) -> _Corpus | None:
        count = self._store.count()
        corpus = self._corpus
        if corpus is not None and corpus.count == count:
            return corpus

        with self._lock:
            # Re-check inside the lock: a concurrent request may have built it
            # while this one waited.
            corpus = self._corpus
            if corpus is not None and corpus.count == count:
                return corpus

            ids, texts, metadatas = self._store.get_all_documents()
            tokenized = [tokenize(t) for t in texts]
            if not tokenized:
                # BM25Okapi divides by the corpus average document length, so
                # an empty corpus is a ZeroDivisionError, not an empty result.
                self._corpus = None
                return None

            self._corpus = _Corpus(
                bm25=BM25Okapi(tokenized),
                chunk_ids=ids,
                texts=texts,
                metadatas=metadatas,
                count=count,
                scores=LruTtlCache(max_entries=self._score_cache_entries),
            )
            logger.info("Built BM25 keyword index over %d chunks", len(tokenized))
            return self._corpus

    def search(
        self,
        query: str,
        top_k: int,
        predicate=None,
    ) -> list[KeywordHit]:
        """Top BM25 matches for the query.

        `predicate(metadata) -> bool` filters candidates the same way the
        dense arm's `where` clause does. It's applied in Python rather than
        pushed into the index because BM25 scores the whole corpus in one
        pass regardless; filtering first would mean rebuilding the index per
        filter combination — and it would also defeat the score memo, which
        is keyed on the query alone precisely so that differently filtered
        passes over one query share a single scoring run.
        """
        corpus = self._ensure_built()
        if corpus is None:
            return []

        tokens = boost_identifiers(tokenize(query))
        if not tokens:
            return []

        scores = _scores_for(corpus, tokens)

        hits: list[KeywordHit] = []
        for index in _ranked_indices(scores):
            metadata = corpus.metadatas[index] or {}
            if predicate is not None and not predicate(metadata):
                continue
            hits.append(
                KeywordHit(
                    chunk_id=corpus.chunk_ids[index],
                    text=corpus.texts[index],
                    metadata=metadata,
                    score=float(scores[index]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def invalidate(self) -> None:
        self._corpus = None


def _scores_for(corpus: _Corpus, tokens: list[str]) -> np.ndarray:
    key = tuple(tokens)
    scores = corpus.scores.get(key)
    if scores is None:
        scores = np.asarray(corpus.bm25.get_scores(tokens), dtype=np.float64)
        corpus.scores.put(key, scores)
    return scores


def _ranked_indices(scores: np.ndarray) -> list[int]:
    """Corpus indices with a non-zero score, best first.

    Only positive scores are ranked: a zero means the chunk shares no token
    with the query, so it can never be a keyword hit however the caller's
    predicate filters. Ranking that subset in NumPy rather than sorting the
    whole corpus with a Python key function keeps this negligible beside the
    scoring pass itself.
    """
    positive = np.flatnonzero(scores > 0)
    if positive.size == 0:
        return []
    order = np.argsort(scores[positive])[::-1]
    return positive[order].tolist()
