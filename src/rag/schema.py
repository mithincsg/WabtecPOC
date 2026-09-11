from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# The datasheet's columns, in the order they appear in the delivered
# workbook. The tuple is the single source of truth: the Excel export, the
# React table and the LLM's expected JSON keys are all derived from it, so
# adding a column is one edit here plus one line in prompts.yaml.
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

# Values the datasheet expects. A model that answers "positive" or "TRUE"
# gets normalised rather than rejected — the wording is a formatting detail,
# not a reason to throw away an otherwise sound test case.
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


class TestCaseParseError(ValueError):
    """Raised when the model's response could not be parsed into test cases."""


class BehaviourPlanParseError(TestCaseParseError):
    """Raised when test_case_plan's response had no usable behaviour lines.

    Subclasses TestCaseParseError so callers that already catch it (the API
    layer maps it to a 502) handle a failed plan step the same way as a
    failed test-case step, with no separate error-handling path needed.
    """


@dataclass
class Behaviour:
    """One line of test_case_plan's output: a single verifiable behaviour
    the requirement specifies, agreed before any test case is written for it.
    """

    index: int
    kind: str  # "P" (positive) | "B" (boundary) | "N" (negative/suppression)
    text: str

    def __str__(self) -> str:
        return f"{self.index}. [{self.kind}] {self.text}"


# "1|P|<sentence>" - see test_case_plan's system prompt in config/prompts.yaml
# for the exact format the model is instructed to produce.
_BEHAVIOUR_LINE_RE = re.compile(r"^\s*\d+\s*\|\s*([PBN])\s*\|\s*(.+?)\s*$", re.IGNORECASE)


def parse_behaviour_plan(raw_response: str) -> list[Behaviour]:
    """Parses test_case_plan's line-per-behaviour output.

    Re-numbers from 1 regardless of what the model wrote as its own leading
    number, the same reasoning as `parse_test_cases` re-numbering `s_no`:
    dense and 1-based even if the model skips or repeats one.
    """
    behaviours: list[Behaviour] = []
    for line in raw_response.splitlines():
        match = _BEHAVIOUR_LINE_RE.match(line)
        if not match:
            continue
        kind, text = match.groups()
        text = text.strip()
        if not text:
            continue
        behaviours.append(Behaviour(index=len(behaviours) + 1, kind=kind.upper(), text=text))

    if not behaviours:
        raise BehaviourPlanParseError(
            "The model's behaviour-plan response had no usable 'N|type|behaviour' "
            "lines. It may have written prose instead of the expected format, or "
            "run out of output tokens."
        )
    return behaviours


@dataclass
class ConfidenceBreakdown:
    """Why a test case scored what it did.

    Kept as components rather than one number so a reviewer can tell a case
    that's well grounded but unlike anything already written (probably a
    genuine new scenario) from one that's neither (probably invented).
    """

    overall: float = 0.0
    retrieval: float = 0.0
    grounding: float = 0.0
    similarity_to_existing: float = 0.0
    # The closest existing test case for this requirement, so the reviewer
    # can compare directly instead of going looking for it.
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

    def to_dict(self) -> dict:
        return asdict(self)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_test_cases(
    raw_response: str, *, requirement_id: str, default_folder: str = ""
) -> list[TestCase]:
    """Turns the model's JSON into datasheet rows.

    Tolerates a code fence or stray prose around the JSON, because instruct
    models don't follow "JSON only" perfectly and re-prompting for a
    formatting slip would double the wait on CPU. S_no is assigned here, not
    taken from the model, so numbering is always dense and 1-based even if
    the model skips or repeats one.
    """
    data = _extract_json(raw_response)

    raw_cases = data.get("test_cases") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise TestCaseParseError("The model's JSON had no non-empty 'test_cases' list.")

    test_cases: list[TestCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        description = _build_description(item)
        if not description:
            # A row with no description is not a test case; dropping it beats
            # exporting a blank line into the datasheet.
            continue
        test_cases.append(
            TestCase(
                s_no=len(test_cases) + 1,
                requirement=requirement_id,
                description=description,
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


def _build_description(item: dict) -> str:
    """The datasheet's "Description" cell, in the house
    "Test Scenario: ...\\n-condition\\n\\nVerify,\\n..." shape.

    The current test_cases prompt returns scenario/conditions/verify as
    separate fields rather than asking the model to assemble that string
    itself (assembly is mechanical and doing it here means one bad line
    break can't cost an otherwise-sound test case). A response still
    carrying the older single "description" string is used as-is, so a
    partial rollback of the prompt keeps working without a code change.
    """
    scenario = _clean(item.get("scenario"))
    if not scenario:
        return _clean(item.get("description"))

    lines = [f"Test Scenario: {scenario}"]
    conditions = item.get("conditions")
    if isinstance(conditions, (list, tuple)):
        lines.extend(f"-{_clean(c)}" for c in conditions if _clean(c))
    elif _clean(conditions):
        lines.append(f"-{_clean(conditions)}")

    verify = _clean(item.get("verify"))
    if verify:
        lines.append("")
        lines.append("Verify,")
        lines.append(verify)

    return "\n".join(lines)


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
            "output tokens — try raising llm_max_tokens, or narrow the requirement "
            "so fewer behaviours have to be covered in one generation."
        )

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    # The usual cause of a malformed object here is `num_predict` cutting the
    # response off mid-array. On CPU that response cost minutes, and the test
    # cases the model *did* finish are perfectly good, so the truncated tail
    # is dropped and the brackets closed rather than throwing all of it away.
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
        "ran out of output tokens mid-response — raise llm_max_tokens in "
        "config/rag_config.yaml."
    )


def _scan(text: str):
    r"""Yields (index, char, depth_stack, inside_string) over a JSON-ish text.

    Written out rather than done with a regex because both callers below have
    to know about string literals: a `{` inside a description is not a nested
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
    r"""The first balanced `{...}` in the text, or everything from the first
    `{` onward if it never balances.

    A greedy `\{.*\}` would run past the object's own closing brace to a
    later one in trailing prose, which turns a recoverable response into a
    parse failure.
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
    head = text[:cut].rstrip().rstrip(",")
    return head + "".join(closers)


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
