from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from .config import ConfidenceConfig
from .keyword_index import tokenize
from .retriever import RetrievedChunk
from .schema import ConfidenceBreakdown, TestCase

logger = logging.getLogger(__name__)

# Words that appear in every test case regardless of content, so their
# presence says nothing about whether a case is grounded in the retrieved
# material. Excluded from the grounding overlap so a case doesn't score well
# just for being written in English.
_BOILERPLATE = frozenset(
    """
    a an and are as at be been being but by for from has have if in into is it its
    no not of on or that the then there this to was were when which while with
    test scenario verify onboard shall should case step expected result
    """.split()
)

# The identifiers that must be right for a test case to be executable: TBC
# parameters, requirement IDs, subdivision and block numbers, API calls. A
# case that reuses these correctly is grounded in a way that matching prose
# does not demonstrate, so they're scored separately and weighted higher.
_IDENTIFIER_RE = re.compile(r"\b(?:[A-Za-z]{2,6}\d{2,}|\d{3,}|[a-z_]+_[a-z_]+)\b")


@dataclass
class _ExistingCase:
    text: str
    test_case_id: str | None
    source_file: str | None


class ConfidenceScorer:
    """Scores each generated test case on three axes and combines them.

    retrieval  — how strong the supporting context was. A test case written
                 from thin context is a guess however fluent it reads.
    grounding  — how much of the case's substantive vocabulary, and
                 especially its identifiers (TBC137, block 1015,
                 wcr_loco_sim), traces back to the retrieved context rather
                 than to the model. This is the axis that catches a
                 plausible-sounding invented parameter.
    similarity — cosine similarity of the case against the closest existing
                 test case for the same requirement. High means the model
                 reproduced something a human already wrote and signed off,
                 which is the strongest evidence available here. Low is not
                 automatically bad — a genuinely new boundary case will score
                 low — so it's reported with the ID of what it was compared
                 against, not treated as a verdict.
    """

    def __init__(self, embedder, config: ConfidenceConfig):
        self.embedder = embedder
        self.config = config

    def score(
        self,
        test_cases: list[TestCase],
        chunks: list[RetrievedChunk],
        requirement_id: str | None,
    ) -> None:
        """Fills in each test case's `confidence` in place."""
        if not test_cases:
            return

        retrieval = self._retrieval_score(chunks)
        context_tokens, context_identifiers = self._context_vocabulary(chunks)
        existing = self._existing_cases(chunks, requirement_id)
        similarities = self._similarity_to_existing(test_cases, existing)

        for test_case, (similarity, match) in zip(test_cases, similarities):
            grounding = self._grounding_score(
                test_case.description, context_tokens, context_identifiers
            )
            components = {
                "retrieval": retrieval,
                "grounding": grounding,
                "similarity_to_existing": similarity,
            }
            overall = self._combine(components)
            test_case.confidence = ConfidenceBreakdown(
                overall=_round(overall),
                retrieval=_round(retrieval),
                grounding=_round(grounding),
                similarity_to_existing=_round(similarity),
                closest_existing_id=match.test_case_id if match else None,
                closest_existing_source=match.source_file if match else None,
                needs_review=overall < self.config.review_threshold,
            )

    # --- components -------------------------------------------------------

    @staticmethod
    def _retrieval_score(chunks: list[RetrievedChunk]) -> float:
        """Mean cosine similarity of the top few dense hits.

        The top few rather than all of them: a long tail of weak chunks is
        normal and shouldn't drag the score down, but if even the best
        matches are weak then nothing in the knowledge base really covers
        this requirement.
        """
        similarities = [c.similarity for c in chunks if c.similarity is not None]
        if not similarities:
            return 0.0
        best = sorted(similarities, reverse=True)[:3]
        return _clamp(sum(best) / len(best))

    @staticmethod
    def _context_vocabulary(
        chunks: list[RetrievedChunk],
    ) -> tuple[frozenset[str], frozenset[str]]:
        tokens: set[str] = set()
        identifiers: set[str] = set()
        for chunk in chunks:
            tokens.update(tokenize(chunk.text))
            identifiers.update(m.lower() for m in _IDENTIFIER_RE.findall(chunk.text))
        return frozenset(tokens - _BOILERPLATE), frozenset(identifiers)

    @staticmethod
    def _grounding_score(
        description: str,
        context_tokens: frozenset[str],
        context_identifiers: frozenset[str],
    ) -> float:
        case_tokens = set(tokenize(description)) - _BOILERPLATE
        if not case_tokens:
            return 0.0

        token_overlap = len(case_tokens & context_tokens) / len(case_tokens)

        case_identifiers = {m.lower() for m in _IDENTIFIER_RE.findall(description)}
        if case_identifiers:
            identifier_overlap = len(case_identifiers & context_identifiers) / len(
                case_identifiers
            )
            # Identifiers dominate: a case whose prose echoes the context but
            # whose parameter numbers don't appear in it is the failure mode
            # this score exists to surface.
            return _clamp(0.35 * token_overlap + 0.65 * identifier_overlap)

        # No identifiers to check. Cap the score, because a purely
        # prose-level match is weaker evidence than a case that names
        # specific, verifiable values.
        return _clamp(token_overlap * 0.8)

    def _existing_cases(
        self, chunks: list[RetrievedChunk], requirement_id: str | None
    ) -> list[_ExistingCase]:
        """Retrieved chunks that are themselves existing test cases for this
        requirement. Matched on the requirement_id metadata the spreadsheet
        extractor read out of the Requirement column; when the requirement is
        unknown, any retrieved test case is used, which is weaker evidence
        but better than none.
        """
        wanted = requirement_id.strip().upper().split("_")[0] if requirement_id else None
        existing: list[_ExistingCase] = []
        for chunk in chunks:
            if chunk.metadata.get("content_type") != "test_case":
                continue
            chunk_requirement = str(chunk.metadata.get("requirement_id") or "").upper()
            if wanted and chunk_requirement and not chunk_requirement.startswith(wanted):
                continue
            existing.append(
                _ExistingCase(
                    text=chunk.text,
                    test_case_id=chunk.metadata.get("test_case_id"),
                    source_file=chunk.metadata.get("source_file"),
                )
            )
        return existing

    def _similarity_to_existing(
        self, test_cases: list[TestCase], existing: list[_ExistingCase]
    ) -> list[tuple[float, _ExistingCase | None]]:
        if not existing:
            # Nothing to compare against. Returning 0 would penalise a
            # requirement simply for having no prior test cases, so the axis
            # is dropped and its weight redistributed in _combine.
            return [(-1.0, None) for _ in test_cases]

        try:
            vectors = self.embedder.embed(
                [tc.description for tc in test_cases] + [e.text for e in existing]
            )
        except Exception:  # noqa: BLE001 - scoring must not fail generation
            logger.exception("Could not embed for similarity scoring; skipping that axis")
            return [(-1.0, None) for _ in test_cases]

        split = len(test_cases)
        # One matrix product instead of a Python loop per (generated,
        # existing) pair. With a 1024-dimensional model and a dozen retrieved
        # reference cases the loop was doing hundreds of thousands of
        # interpreted float multiplications per request.
        similarities = _cosine_matrix(vectors[:split], vectors[split:])

        results: list[tuple[float, _ExistingCase | None]] = []
        for row in similarities:
            best_index = int(row.argmax())
            results.append((_clamp(float(row[best_index])), existing[best_index]))
        return results

    def _combine(self, components: dict[str, float]) -> float:
        """Weighted mean over the axes that actually applied. An axis marked
        unavailable (-1) has its weight redistributed across the rest rather
        than counting as a zero.
        """
        total_weight = 0.0
        total = 0.0
        for name, value in components.items():
            if value < 0:
                continue
            weight = float(self.config.weights.get(name, 0.0))
            total += weight * value
            total_weight += weight
        return total / total_weight if total_weight else 0.0


def _cosine_matrix(
    generated: list[list[float]], existing: list[list[float]]
) -> np.ndarray:
    """Cosine similarity of every generated vector against every existing
    one, as a (generated x existing) matrix.

    BGE-M3 vectors come back L2-normalised, so this is usually just the dot
    product; the norms are divided out anyway so the function stays correct
    if normalisation is ever turned off. A zero-norm row would divide by
    zero, so those norms are floored to 1 — the resulting row is all zeros,
    which is the right answer for a vector with no direction.
    """
    a = np.asarray(generated, dtype=np.float32)
    b = np.asarray(existing, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)
