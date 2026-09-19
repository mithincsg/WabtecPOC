from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass

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
# Must stay identical to the seven techniques named in the test_cases system
# prompt in config/prompts.yaml. This tuple is also the decoder's enum, so a
# technique missing here is one the model physically cannot emit — it was
# reasoning correctly but forced to mislabel the case under a listed one.
_TEST_TECHNIQUES = (
    "Equivalence Partitioning",
    "Boundary Value Analysis",
    "Decision Table Testing",
    "State Transition Testing",
    "Error Guessing",
    "All Pairs Testing",
    "Cause Effect Testing",
)


class TestCaseParseError(ValueError):
    """Raised when the model's response could not be parsed into test cases."""


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
    # "No" rather than "Yes": scorability is a deliberate reviewer decision
    # about whether a case counts toward the requirement's score, so the
    # safe default is the one that needs an explicit opt-in.
    scorable: str = "No"
    comments: str = ""

    def to_row(self) -> list[str]:
        return [str(getattr(self, attr)) for attr, _ in DATASHEET_COLUMNS]

    def to_dict(self) -> dict:
        return asdict(self)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Handed to the inference server as a decoding constraint, so the response is
# a JSON object of this shape by construction rather than by the model
# choosing to obey the prompt. "JSON only" in prompts.yaml is a request that
# instruct models mostly honour and reasoning models often don't — qwen3 will
# happily spend its whole output budget narrating its plan in prose, which
# arrives here as "no JSON object" after minutes of CPU time. Constraining
# the decoder is what makes the model swappable without re-tuning prompts.
#
# Only `description` is required of a case: s_no and requirement are assigned
# by parse_test_cases, and every other field has a normalised default, so
# demanding them would only give the model more ways to fail. The enums match
# the tuples above, which is why they're built from them.
#
# `coverage` is a planning field, and it is why it comes first. Nothing
# obliges a constrained decoder to write more than one array element — `]` is
# a legal token after the first one — so on the same requirement the same
# model returned 8 cases, then 10, and the datasheet it was measured against
# has 11. prompts.yaml asks the model to enumerate the behaviours before
# writing cases, but with only `test_cases` in the schema it had nowhere to
# put that list, so the instruction could not be followed and the count was
# free. Ollama builds its grammar in property order, so a required `coverage`
# ahead of `test_cases` makes the enumeration the first thing written; the
# cases then follow a list the model has already committed to. It is not
# turned into datasheet rows — parse_test_cases only compares the two lengths
# and warns — because the reviewer's evidence of a short answer is the
# shortfall itself, not another column.
TEST_CASES_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "coverage": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "folder": {"type": "string"},
                    "optimization_technique": {"type": "string"},
                    "test_type": {"type": "string", "enum": list(_TEST_TYPES)},
                    "test_technique": {"type": "string", "enum": list(_TEST_TECHNIQUES)},
                    "retired": {"type": "string", "enum": list(_TRUE_FALSE)},
                    "scorable": {"type": "string", "enum": list(_YES_NO)},
                    "comments": {"type": "string"},
                },
                "required": ["description"],
            },
        },
    },
    "required": ["coverage", "test_cases"],
}


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
        description = _clean(item.get("description"))
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
                scorable=_choose(item.get("scorable"), _YES_NO, "No"),
                comments=_clean(item.get("comments")),
            )
        )

    if not test_cases:
        raise TestCaseParseError("The model returned no usable test cases.")

    coverage = [
        line.strip()
        for line in (data.get("coverage") or [])
        if isinstance(line, str) and line.strip()
    ]
    if len(test_cases) < len(coverage):
        # The model listed what it meant to cover and then wrote fewer cases
        # than that. Not an error — the rows it did write are sound — but the
        # shortfall is invisible in the datasheet, so it is named here.
        logger.warning(
            "The model planned %d test case(s) but wrote %d. Missing: %s",
            len(coverage),
            len(test_cases),
            "; ".join(coverage[len(test_cases):]),
        )
    return test_cases


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
    # Compared on letters and digits only, so "Cause-Effect testing" and
    # "cause effect  testing" both land on the house spelling instead of
    # falling through to the default.
    normalised = _fold(text)
    for option in allowed:
        if normalised == _fold(option):
            return option
    # Partial match, so "boundary value" lands on "Boundary Value Analysis".
    for option in allowed:
        folded = _fold(option)
        if normalised in folded or folded in normalised:
            return option
    return default


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
