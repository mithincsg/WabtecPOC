"""Datasheet rows, parsing of the model's output, and confidence scoring."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .config import ConfidenceConfig
from .retrieval import RetrievedChunk, tokenize

logger = logging.getLogger(__name__)

# The datasheet's columns in delivered order. This tuple is the single source
# of truth: the Excel export, the React table and the model's expected JSON
# keys all derive from it, so adding a column is one edit here plus one line
# in prompts.yaml.
DATASHEET_COLUMNS: tuple[tuple[str, str], ...] = (
    ("s_no", "S_no"),
    ("requirement", "Requirement"),
    ("description", "Description"),
    ("folder", "Folder"),
    ("optimization_technique", "Optimization_Technique"),
    ("test_type", "Test_Type"),
    ("test_technique", "Test_Technique"),
    ("retired", "Retired?"),
    ("scorable", "Scorable"),
    ("comments", "Comments"),
)

# Values the datasheet expects. A model that answers "positive" or "TRUE" is
# normalised rather than rejected - wording is a formatting detail, not a
# reason to throw away a sound test case.
_TEST_TYPES = ("Positive", "Negative")
_YES_NO = ("Yes", "No")
_TRUE_FALSE = ("True", "False")
_TEST_TECHNIQUES = (
    "Equivalence Partitioning",
    "Boundary Value Analysis",
    "Decision Table",
    "State Transition",
    "Error Guessing",
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class TestCaseParseError(ValueError):
    """The model's response could not be parsed into test cases."""


@dataclass
class ConfidenceBreakdown:
    """Why a test case scored what it did.

    Kept as components rather than one number so a reviewer can tell a case
    that is well grounded but unlike anything already written (probably a
    genuine new scenario) from one that is neither (probably invented).
    """

    overall: float = 0.0
    retrieval: float = 0.0
    grounding: float = 0.0
    similarity_to_existing: float = 0.0
    # The closest existing case, so the reviewer can compare directly.
    closest_existing_id: str | None = None
    closest_existing_source: str | None = None
    needs_review: bool = False


@dataclass
class TestCase:
    s_no: int
    requirement: str
    description: str
    folder: str = ""
    optimization_technique: str = "Default"
    test_type: str = "Positive"
    test_technique: str = "Equivalence Partitioning"
    retired: str = "False"
    scorable: str = "Yes"
    comments: str = ""
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)

    def to_row(self) -> list[str]:
        return [str(getattr(self, attr)) for attr, _ in DATASHEET_COLUMNS]


# --- Parsing the model's JSON ----------------------------------------------


def parse_test_cases(
    raw_response: str,
    *,
    requirement_id: str,
    default_folder: str = "",
    start_s_no: int = 1,
) -> list[TestCase]:
    """Turns the model's JSON into datasheet rows.

    Tolerates a code fence or stray prose around the JSON, because instruct
    models don't follow "JSON only" perfectly and re-prompting for a
    formatting slip would double the wait on CPU. S_no is assigned here, not
    taken from the model, so numbering is dense and 1-based.

    The model is asked for `scenario` / `conditions` / `verify` and the house
    layout is assembled here: every token spent on layout is a token not
    spent on a behaviour. A pre-composed `description` is still accepted, so
    an older prompts.yaml keeps working.
    """
    data = _extract_json(raw_response)

    raw_cases = data.get("test_cases") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise TestCaseParseError("The model's JSON had no non-empty 'test_cases' list.")

    test_cases: list[TestCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        description = _clean(item.get("description")) or _compose_description(
            item.get("scenario"), item.get("conditions"), item.get("verify")
        )
        if not description:
            # A row with no description is not a test case; dropping it beats
            # exporting a blank line into the datasheet.
            continue
        test_cases.append(
            TestCase(
                s_no=start_s_no + len(test_cases),
                requirement=requirement_id,
                description=description,
                # folder / optimization_technique / retired / scorable are
                # constants or already known in Python, so they are no longer
                # asked of the model - but are still read when present.
                folder=_clean(item.get("folder")) or default_folder,
                optimization_technique=_clean(item.get("optimization_technique")) or "Default",
                test_type=_choose(item.get("test_type"), _TEST_TYPES, "Positive"),
                test_technique=_choose(
                    item.get("test_technique"), _TEST_TECHNIQUES, "Equivalence Partitioning"
                ),
                retired=_choose(item.get("retired"), _TRUE_FALSE, "False"),
                scorable=_choose(item.get("scorable"), _YES_NO, "Yes"),
                comments=_clean(item.get("comments")),
            )
        )

    if not test_cases:
        raise TestCaseParseError("The model returned no usable test cases.")
    return test_cases


def _compose_description(scenario: object, conditions: object, verify: object) -> str:
    """The house datasheet layout:

        Test Scenario: <scenario>
        -<condition>

        Verify,
        <behaviour>
    """
    scenario_text = _clean(scenario)
    verify_text = _clean(verify)
    if not (scenario_text and verify_text):
        return ""

    if isinstance(conditions, (list, tuple)):
        items = [_clean(c) for c in conditions]
    else:
        # A model that ignored the array and sent one string still has its
        # conditions one per line more often than not.
        items = list(_clean(conditions).splitlines())

    lines = [f"Test Scenario: {scenario_text}"]
    lines += [f"-{item.lstrip('-').strip()}" for item in items if item.strip()]
    return "\n".join(lines) + f"\n\nVerify,\n{verify_text}"


def _extract_json(raw_response: str) -> dict:
    candidate = raw_response.strip()

    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    block = _outermost_object(candidate)
    if block is None:
        raise TestCaseParseError(
            "The model's response contained no JSON object. It may have run out of "
            "output tokens - try raising llm_max_tokens, or narrow the requirement "
            "so fewer behaviours have to be covered in one generation."
        )

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    # The usual cause of a malformed object is num_predict cutting the
    # response off mid-array. That response cost minutes on CPU and the
    # finished test cases are good, so the truncated tail is dropped and the
    # brackets closed rather than throwing all of it away.
    repaired = _close_truncated_json(block)
    if repaired is not None:
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            logger.warning(
                "The model's JSON was truncated (likely at llm_max_tokens); "
                "recovered the %d complete test case(s) it had finished.",
                len(data.get("test_cases") or []),
            )
            return data

    raise TestCaseParseError(
        "The model's JSON was malformed and could not be repaired. It most likely "
        "ran out of output tokens mid-response - raise llm_max_tokens in "
        "config/rag_config.yaml."
    )


def _scan(text: str):
    r"""Yields (index, char, depth_stack, inside_string) over a JSON-ish text.

    Written out rather than done with a regex because both callers have to
    know about string literals: a `{` inside a description is not a nested
    object, and a `\"` inside one does not end the string.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            yield index, char, stack, True
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
        yield index, char, stack, False


def _outermost_object(text: str) -> str | None:
    r"""The first balanced `{...}`, or everything from the first `{` onward if
    it never balances. A greedy `\{.*\}` would run past the object's own
    closing brace to a later one in trailing prose.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    for index, char, stack, in_string in _scan(body):
        if not in_string and char == "}" and not stack:
            return body[: index + 1]
    return body


def _close_truncated_json(text: str) -> str | None:
    """`text` cut back to its last completed value, with its open brackets
    closed. None when nothing was ever completed.
    """
    cut: int | None = None
    closers: list[str] = []
    for index, char, stack, in_string in _scan(text):
        if in_string or char not in "}]":
            continue
        cut = index + 1
        closers = list(reversed(stack))

    if cut is None:
        return None
    return text[:cut].rstrip().rstrip(",") + "".join(closers)


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        # Some models emit the description as a list of lines despite the
        # schema; joining is closer to the intent than str(list).
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _choose(value: object, allowed: tuple[str, ...], default: str) -> str:
    text = _clean(value)
    if not text:
        return default
    for option in allowed:
        if text.lower() == option.lower():
            return option
    # Partial match, so "boundary value" lands on "Boundary Value Analysis".
    for option in allowed:
        if text.lower() in option.lower() or option.lower() in text.lower():
            return option
    return default


# --- Confidence scoring ----------------------------------------------------

# Words that appear in every test case regardless of content, so their
# presence says nothing about whether a case is grounded. Excluded from the
# overlap so a case doesn't score well just for being written in English.
_BOILERPLATE = frozenset(
    """
    a an and are as at be been being but by for from has have if in into is it its
    no not of on or that the then there this to was were when which while with
    test scenario verify onboard shall should case step expected result
    """.split()
)

# The identifiers that must be right for a test case to be executable: TBC
# parameters, requirement IDs, subdivision and block numbers, API calls.
_IDENTIFIER_RE = re.compile(r"\b(?:[A-Za-z]{2,6}\d{2,}|\d{3,}|[a-z_]+_[a-z_]+)\b")


@dataclass
class _ExistingCase:
    text: str
    test_case_id: str | None
    source_file: str | None


class ConfidenceScorer:
    """Scores each generated test case on three axes and combines them.

    retrieval  - how strong the supporting context was. A case written from
                 thin context is a guess however fluent it reads.
    grounding  - how much of the case's substantive vocabulary, especially
                 its identifiers, traces back to the retrieved context
                 rather than to the model. This is the axis that catches a
                 plausible-sounding invented parameter.
    similarity - cosine against the closest existing test case for the same
                 requirement. High means the model reproduced something a
                 human already signed off. Low is not automatically bad - a
                 genuinely new boundary case scores low - so it is reported
                 with the ID it was compared against, not as a verdict.
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
            overall = self._combine(
                {
                    "retrieval": retrieval,
                    "grounding": grounding,
                    "similarity_to_existing": similarity,
                }
            )
            test_case.confidence = ConfidenceBreakdown(
                overall=_round(overall),
                retrieval=_round(retrieval),
                grounding=_round(grounding),
                similarity_to_existing=_round(similarity),
                closest_existing_id=match.test_case_id if match else None,
                closest_existing_source=match.source_file if match else None,
                needs_review=overall < self.config.review_threshold,
            )

    @staticmethod
    def _retrieval_score(chunks: list[RetrievedChunk]) -> float:
        """Mean cosine of the top few dense hits. The top few rather than all
        of them: a long tail of weak chunks is normal, but if even the best
        matches are weak then nothing in the KB covers this requirement.
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

        # No identifiers to check: cap the score, because a prose-level match
        # is weaker evidence than a case naming specific verifiable values.
        return _clamp(token_overlap * 0.8)

    def _existing_cases(
        self, chunks: list[RetrievedChunk], requirement_id: str | None
    ) -> list[_ExistingCase]:
        """Retrieved chunks that are themselves existing test cases for this
        requirement, matched on the requirement_id the spreadsheet extractor
        read out of the Requirement column. When the requirement is unknown,
        any retrieved test case is used - weaker evidence, but better than
        none.
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
            # requirement for having no prior test cases, so the axis is
            # dropped and its weight redistributed in _combine.
            return [(-1.0, None) for _ in test_cases]

        try:
            vectors = self.embedder.embed(
                [tc.description for tc in test_cases] + [e.text for e in existing]
            )
        except Exception:  # noqa: BLE001 - scoring must not fail generation
            logger.exception("Could not embed for similarity scoring; skipping that axis")
            return [(-1.0, None) for _ in test_cases]

        split = len(test_cases)
        similarities = _cosine_matrix(vectors[:split], vectors[split:])

        results: list[tuple[float, _ExistingCase | None]] = []
        for row in similarities:
            best_index = int(row.argmax())
            results.append((_clamp(float(row[best_index])), existing[best_index]))
        return results

    def _combine(self, components: dict[str, float]) -> float:
        """Weighted mean over the axes that applied. An axis marked
        unavailable (-1) has its weight redistributed rather than counting
        as a zero.
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


def _cosine_matrix(generated: list[list[float]], existing: list[list[float]]) -> np.ndarray:
    """Cosine of every generated vector against every existing one, as one
    matrix product rather than a Python loop per pair. Norms are divided out
    anyway (BGE-M3 already normalises) and floored, so a zero vector gives a
    zero row instead of dividing by zero.
    """
    a = np.asarray(generated, dtype=np.float32)
    b = np.asarray(existing, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(_clamp(value), 3)
