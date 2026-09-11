"""Requirement -> datasheet rows, and approved rows -> automation script.

Test-case generation is two LLM calls: a plan call that enumerates the
behaviours, then one writing call per batch of them. Script generation is a
third, separate call, run against the rows the reviewer kept.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from kb_ingestion.chunking import normalize_text

from .config import PromptLibrary
from .llm_client import LLMClient
from .retrieval import HybridRetriever, RetrievedChunk, StaticContextProvider
from .schema import ConfidenceScorer, TestCase, TestCaseParseError, parse_test_cases
from .utils import LruTtlCache, run_parallel

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEST_CASES = 20
# How many behaviours one writing call covers. Small on purpose: the output
# budget of a single call is what used to truncate a long requirement's rows
# mid-array, and a bounded response always parses.
DEFAULT_BATCH_SIZE = 4
# How many writing calls may be in flight at once. On a CPU box the slots
# share cores, so the gain comes from one call's prefill overlapping
# another's decode, and it fades fast past two. Ollama must be serving at
# least this many slots (OLLAMA_NUM_PARALLEL) for it to do anything.
DEFAULT_PARALLELISM = 2

# Sent to Ollama as `format`, which constrains decoding so the model cannot
# emit invalid JSON and stops at the closing brace instead of running to
# num_predict. The constant datasheet fields are absent deliberately -
# Python fills those in (see schema.parse_test_cases).
TEST_CASE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string"},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "verify": {"type": "string"},
                    "test_type": {"type": "string", "enum": ["Positive", "Negative"]},
                    "test_technique": {
                        "type": "string",
                        "enum": [
                            "Equivalence Partitioning",
                            "Boundary Value Analysis",
                            "Decision Table",
                            "State Transition",
                            "Error Guessing",
                        ],
                    },
                    "comments": {"type": "string"},
                },
                "required": [
                    "scenario",
                    "conditions",
                    "verify",
                    "test_type",
                    "test_technique",
                ],
            },
        }
    },
    "required": ["test_cases"],
}


# --- Requirement understanding ---------------------------------------------

# A requirement ID as used across this project's data ("L2R9479",
# "L2R1145424", "L2R9479_A"): a short letter prefix, digits, then optional
# trailing word characters.
_ID_PATTERN = r"([A-Za-z]{1,6}\d[\w.-]*)"
# The common case: the user pastes "<ID> <text>".
_ID_AT_START_RE = re.compile(rf"^{_ID_PATTERN}\b")
# Fallback for a pasted requirement document, which leads with headings
# before a line holding just the ID, sometimes with list numbering. Some
# requirement files carry the ID only in the filename, so finding nothing
# here is a legitimate result rather than a parse failure.
_ID_OWN_LINE_RE = re.compile(rf"(?m)^\s*(?:\d+[.)]\s+)?{_ID_PATTERN}\s*$")
# The functional area, which becomes the datasheet's Folder column.
# Requirement documents name it in a heading ("15 Speed Enforcement"), so it
# is read out rather than configured.
_AREA_HEADING_RE = re.compile(
    r"(?m)^\s*\d+(?:\.\d+)*\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s*$"
)


@dataclass(frozen=True)
class ParsedRequirement:
    requirement_id: str | None
    raw_text: str
    query_text: str
    functional_area: str | None


def parse_requirement(raw_text: str) -> ParsedRequirement:
    """Pulls out the requirement ID and functional area, then normalizes the
    body the way ingested documents were normalized, so the query is
    embedded consistently with the knowledge base.
    """
    cleaned = raw_text.strip()

    start_match = _ID_AT_START_RE.match(cleaned)
    if start_match:
        requirement_id = start_match.group(1)
        body = cleaned[start_match.end() :].strip() or cleaned
    else:
        own_line = _ID_OWN_LINE_RE.search(cleaned)
        requirement_id = own_line.group(1) if own_line else None
        # Leading heading text stays in the query - domain context for the
        # embedding, not noise.
        body = cleaned

    return ParsedRequirement(
        requirement_id=requirement_id,
        raw_text=raw_text,
        query_text=normalize_text(body),
        functional_area=_detect_functional_area(cleaned),
    )


def _detect_functional_area(text: str) -> str | None:
    """The first numbered title-case heading: "15 Speed Enforcement" ->
    "Speed Enforcement". Deeper headings name sub-behaviours, so the
    shallowest match wins.
    """
    best: tuple[int, str] | None = None
    for match in _AREA_HEADING_RE.finditer(text):
        depth = match.group(0).strip().split()[0].count(".")
        if best is None or depth < best[0]:
            best = (depth, match.group(1).strip())
        if depth == 0:
            break
    return best[1] if best else None


# --- Results ---------------------------------------------------------------


@dataclass
class TestCaseResult:
    requirement_id: str | None
    functional_area: str | None
    test_cases: list[TestCase]
    retrieved_chunks: list[RetrievedChunk]
    raw_response: str
    track_subdivisions: list[str] = field(default_factory=list)
    # The plan call's behaviour list, in `[P] description` form. Coverage is
    # judged against behaviours, not row count.
    behaviours: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.test_cases:
            return 0.0
        return round(
            sum(tc.confidence.overall for tc in self.test_cases) / len(self.test_cases), 3
        )


@dataclass
class ScriptResult:
    requirement_id: str | None
    script: str
    elapsed_seconds: float = 0.0


class TestCaseGenerator:
    """requirement -> understanding -> hybrid retrieval -> context -> LLM ->
    datasheet rows -> confidence scores.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        prompts: PromptLibrary,
        scorer: ConfidenceScorer,
        static_context: StaticContextProvider | None = None,
        static_track_top_k: int = 4,
        examples_max_chars: int = 1500,
        *,
        static_track_max_chars: int = 1200,
        batch_size: int = DEFAULT_BATCH_SIZE,
        plan_model: str | None = None,
        plan_max_tokens: int | None = None,
        plan_temperature: float | None = None,
        plan_context_max_chars: int = 1200,
        parallelism: int = DEFAULT_PARALLELISM,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompts = prompts
        self.scorer = scorer
        self.static_context = static_context
        self.static_track_top_k = static_track_top_k
        self.examples_max_chars = examples_max_chars
        self.static_track_max_chars = static_track_max_chars
        self.batch_size = max(1, batch_size)
        self.plan_model = plan_model
        self.plan_max_tokens = plan_max_tokens
        self.plan_temperature = plan_temperature
        self.plan_context_max_chars = plan_context_max_chars
        self.parallelism = max(1, parallelism)
        # A behaviour list depends only on the requirement and the plan
        # prompt, so regenerating rows after a writing-prompt tweak should
        # not re-enumerate: on this CPU that call is minutes.
        self._plan_cache = LruTtlCache(max_entries=16, ttl_seconds=60 * 60)

    def generate(
        self,
        requirement_text: str,
        *,
        max_test_cases: int = DEFAULT_MAX_TEST_CASES,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> TestCaseResult:
        """Two calls, not one.

        Enumerating the behaviours a requirement specifies and writing them
        up in house style are different jobs, and asking a 7B to do both in
        one response made them compete: it pattern-matched the examples,
        stopped early, and whatever it wrote past the output budget was
        truncated mid-array. So stage one asks only for a plain behaviour
        list, and stage two writes rows for slices of that list.

        How many test cases come back is still decided by the requirement -
        one per behaviour found. `max_test_cases` only caps the list.
        """
        started = time.monotonic()
        parsed = parse_requirement(requirement_text)

        retrieval = self.retriever.retrieve(
            parsed.query_text, doc_types=doc_types, top_k=top_k
        )

        # The blocks are labelled because the system prompt's rules refer to
        # them by name: the knowledge-base and track-data blocks are the only
        # places a concrete value may come from. The wording template is not
        # here - it goes at the front of the writing prompt only, so the plan
        # call never sees it and so it forms a cacheable shared prefix.
        context = _labelled_blocks(
            (
                "KNOWLEDGE BASE - prior datasheets, data dictionaries, "
                "parameter guides. A source of values.",
                retrieval.context
                or "(Nothing in the knowledge base matched this requirement.)",
            ),
            (
                "TRACK DATA - the only source of subdivision, block, milepost, "
                "switch and signal values.",
                self._static_track_context(parsed),
            ),
        )

        # The plan call gets its own, much smaller context: enumerating the
        # behaviours needs the requirement's vocabulary, not every value.
        plan_context = self.retriever.build_context(
            retrieval.chunks, max_chars=self.plan_context_max_chars
        )
        behaviours = self._plan(parsed, plan_context, max_test_cases)
        rows, raw_parts = self._write(parsed, context, behaviours)

        if not rows:
            raise TestCaseParseError(
                "None of the writing calls returned a usable test case. The "
                "behaviour list was: "
                + ("; ".join(text for _, text in behaviours) or "(empty)")
            )

        self.scorer.score(rows, retrieval.chunks, parsed.requirement_id)

        return TestCaseResult(
            requirement_id=parsed.requirement_id,
            functional_area=parsed.functional_area,
            test_cases=rows,
            retrieved_chunks=retrieval.chunks,
            raw_response="\n\n".join(raw_parts),
            track_subdivisions=retrieval.track_subdivisions,
            behaviours=[f"[{kind}] {text}" for kind, text in behaviours],
            elapsed_seconds=round(time.monotonic() - started, 1),
        )

    def _plan(
        self, parsed: ParsedRequirement, context: str, max_test_cases: int
    ) -> list[tuple[str, str]]:
        """Stage one: the behaviour list, as (kind, description) pairs.

        Runs hotter than the writing calls (`plan_temperature`): this step
        wants breadth - the boundary nobody thought of - where the writing
        step wants a tightly constrained format.
        """
        cache_key = (parsed.raw_text.strip(), max_test_cases, self.plan_model)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            logger.info("Reusing the cached behaviour plan for this requirement")
            return cached

        prompt = self.prompts.test_case_plan
        raw = self.llm_client.generate(
            prompt.system,
            prompt.render_user(
                context=context,
                requirement_id=parsed.requirement_id or "(not stated)",
                requirement_text=parsed.raw_text.strip(),
                max_test_cases=str(max_test_cases),
            ),
            max_tokens=self.plan_max_tokens,
            temperature=self.plan_temperature,
            model=self.plan_model,
        )

        behaviours = _parse_behaviours(raw)[:max_test_cases]
        if not behaviours:
            # Not fatal: the writing stage can still enumerate for itself,
            # it is just back to doing two jobs at once for this request.
            logger.warning(
                "The plan call returned no parseable behaviour lines; falling "
                "back to a single writing call that enumerates for itself. "
                "Raw plan response: %.300s",
                raw,
            )
            return []

        self._plan_cache.put(cache_key, behaviours)
        logger.info(
            "Planned %d behaviour(s) for %s: %s",
            len(behaviours),
            parsed.requirement_id or "(unidentified requirement)",
            " | ".join(f"[{kind}] {text}" for kind, text in behaviours),
        )
        return behaviours

    def _write(
        self,
        parsed: ParsedRequirement,
        context: str,
        behaviours: list[tuple[str, str]],
    ) -> tuple[list[TestCase], list[str]]:
        """Stage two: the rows, `batch_size` behaviours per call.

        The batches are independent - each needs only the shared context and
        its own slice - so they run through a pool of `parallelism` workers.
        On CPU that is not free: the slots share the machine's threads and
        Ollama splits the shared-prefix KV cache across them, so each call
        gets slower. It still wins overall, because one slot's prefill
        overlaps another's decode. `writing_parallelism: 1` restores strictly
        sequential behaviour if a box measures worse.
        """
        prompt = self.prompts.test_cases
        agreed = _behaviour_list(behaviours, start=1) if behaviours else _NO_PLAN_LIST
        batches = list(_batches(behaviours, self.batch_size))

        def write_batch(offset: int, chunk: list[tuple[str, str]]) -> str:
            return self.llm_client.generate(
                prompt.system,
                prompt.render_user(
                    # The static blocks come first and the per-batch
                    # instruction last, deliberately: Ollama reuses the KV
                    # cache of an identical prompt *prefix*, so batches 2..N
                    # of a requirement cost decode time only.
                    wording_template=self._examples_context(),
                    context=context,
                    requirement_id=parsed.requirement_id or "(not stated)",
                    requirement_text=parsed.raw_text.strip(),
                    behaviours=agreed,
                    batch_list=(
                        _behaviour_list(chunk, start=offset + 1) if chunk else _NO_PLAN_LIST
                    ),
                    folder_hint=_folder_hint(parsed),
                ),
                json_schema=TEST_CASE_JSON_SCHEMA,
            )

        raws = self._run_batches(
            [(offset, partial(write_batch, offset, chunk)) for offset, chunk in batches]
        )

        rows: list[TestCase] = []
        raw_parts: list[str] = []
        for offset, raw in raws:
            if raw is None:
                continue
            raw_parts.append(raw)
            try:
                rows.extend(
                    parse_test_cases(
                        raw,
                        requirement_id=parsed.requirement_id or "",
                        default_folder=parsed.functional_area or "",
                        # Renumbered below once every batch is in: a
                        # concurrent batch cannot know how many rows preceded it.
                        start_s_no=1,
                    )
                )
            except TestCaseParseError as exc:
                # One unusable batch is not worth discarding the batches that
                # did parse - those rows cost minutes on CPU. The gap shows
                # up as behaviours with no row, which is what a reviewer
                # compares against anyway.
                logger.warning(
                    "Batch starting at behaviour %d produced no usable rows "
                    "(%s); keeping the other batches.",
                    offset + 1,
                    exc,
                )
        for number, row in enumerate(rows, start=1):
            row.s_no = number
        return rows, raw_parts

    def _run_batches(
        self, tasks: list[tuple[int, Callable[[], str]]]
    ) -> list[tuple[int, str | None]]:
        """Runs the writing calls, at most `parallelism` at a time, returning
        their responses in batch order (None where the call failed).
        """
        workers = min(self.parallelism, len(tasks))
        if workers <= 1:
            return [(offset, self._safe_call(offset, task)) for offset, task in tasks]

        logger.info("Writing %d batch(es), %d at a time", len(tasks), workers)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gen-batch") as pool:
            futures = [
                (offset, pool.submit(self._safe_call, offset, task))
                for offset, task in tasks
            ]
            return [(offset, future.result()) for offset, future in futures]

    @staticmethod
    def _safe_call(offset: int, task: Callable[[], str]) -> str | None:
        try:
            return task()
        except Exception as exc:  # noqa: BLE001 - one batch, not the request
            logger.warning(
                "The writing call for the batch starting at behaviour %d "
                "failed (%s); keeping the other batches.",
                offset + 1,
                exc,
            )
            return None

    def _static_track_context(self, parsed: ParsedRequirement) -> str:
        if self.static_context is None:
            return ""
        return self.static_context.track_context(
            parsed.query_text, self.static_track_top_k, self.static_track_max_chars
        )

    def _examples_context(self) -> str:
        if self.static_context is None:
            return ""
        return self.static_context.test_case_examples(self.examples_max_chars)


class TestScriptGenerator:
    """Approved test cases -> a single automation script.

    A separate call from test-case generation rather than one long prompt:
    the test cases are what a human reviews and edits, so the script is
    written against whatever they settled on, not the model's first draft.
    It also keeps each CPU generation inside a sensible request timeout.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        prompts: PromptLibrary,
        *,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        static_context: StaticContextProvider | None = None,
        static_api_top_k: int = 4,
        static_track_top_k: int = 4,
        examples_max_chars: int = 8000,
        static_api_max_chars: int = 3500,
        static_track_max_chars: int = 1200,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.static_context = static_context
        self.static_api_top_k = static_api_top_k
        self.static_track_top_k = static_track_top_k
        self.examples_max_chars = examples_max_chars
        self.static_api_max_chars = static_api_max_chars
        self.static_track_max_chars = static_track_max_chars

    def generate(self, requirement_text: str, test_cases: list[TestCase]) -> ScriptResult:
        if not test_cases:
            raise ValueError("Generate test cases before generating a script.")

        started = time.monotonic()
        parsed = parse_requirement(requirement_text)

        # Retrieval for the script is driven by the test cases as well as the
        # requirement: the behaviours being automated determine which API
        # calls and which reference script are relevant.
        query = "\n".join([parsed.query_text, *(tc.description for tc in test_cases)])

        # Four independent lookups, so they run concurrently. The parameter
        # pass is keyed on the requirement query alone, which test-case
        # generation already asked for: a cache hit rather than another
        # BGE-M3 encode plus two ranking passes. The API and track passes use
        # the wider query, because which API calls matter does depend on the
        # approved rows.
        api_context, track_context, reference_scripts, parameter_context = run_parallel(
            [
                lambda: (
                    self.static_context.api_context(
                        query, self.static_api_top_k, self.static_api_max_chars
                    )
                    if self.static_context
                    else ""
                ),
                lambda: (
                    self.static_context.track_context(
                        query, self.static_track_top_k, self.static_track_max_chars
                    )
                    if self.static_context
                    else ""
                ),
                lambda: (
                    self.static_context.script_examples(self.examples_max_chars)
                    if self.static_context
                    else ""
                ),
                lambda: self.retriever.retrieve(parsed.query_text).context,
            ]
        )

        prompt = self.prompts.test_script
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or "(not stated)",
            requirement_text=parsed.raw_text.strip(),
            test_cases=_numbered_test_cases(test_cases),
            api_context=api_context or "(No API definitions were retrieved.)",
            track_context=track_context or "(No track data was retrieved.)",
            parameter_context=(
                parameter_context or "(No parameter/data-dictionary context was retrieved.)"
            ),
            reference_scripts=reference_scripts or "(No reference scripts were retrieved.)",
            branch_range=_branch_range(test_cases),
        )

        raw = self.llm_client.generate(
            prompt.system, user_prompt, max_tokens=self.max_tokens, num_ctx=self.num_ctx
        )
        return ScriptResult(
            requirement_id=parsed.requirement_id,
            script=_extract_python(raw),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )


# --- Behaviour lists -------------------------------------------------------

# Used in place of the agreed list when the plan call came back unusable:
# the writing call then does both jobs for that request - degraded, but not
# a failed generation.
_NO_PLAN_LIST = (
    "(The behaviour list is unavailable for this requirement. Enumerate the "
    "distinct verifiable behaviours the requirement specifies yourself, and "
    "write one row for each.)"
)

_BEHAVIOUR_KINDS = {"P": "P", "B": "B", "N": "N"}
_LEADING_BULLET = re.compile(r"^[\s\-*•\d.)]+")


def _parse_behaviours(raw: str) -> list[tuple[str, str]]:
    """The plan call's `N|type|behaviour` lines as (kind, description) pairs.

    Deliberately forgiving. The plan response is plain text precisely so it
    cannot fail to parse: a line that lost its type field still names a
    behaviour, and losing it to a strict parser would cost exactly the
    coverage this stage exists to protect.
    """
    behaviours: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        parts = [part.strip() for part in line.strip().split("|")]
        if len(parts) >= 3:
            kind = _BEHAVIOUR_KINDS.get(parts[1][:1].upper(), "P")
            text = parts[-1]
        elif len(parts) == 2:
            kind = _BEHAVIOUR_KINDS.get(parts[0][-1:].upper(), "P")
            text = parts[-1]
        else:
            # A bare sentence - a plan line that lost its formatting, or a
            # stray "Here are the behaviours:" preamble. Length plus the
            # trailing colon separate the two well enough.
            kind = "P"
            text = _LEADING_BULLET.sub("", parts[0])
            if len(text) < 20 or text.rstrip().endswith(":"):
                continue
        text = _LEADING_BULLET.sub("", text).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            behaviours.append((kind, text))
    return behaviours


def _behaviour_list(behaviours: list[tuple[str, str]], *, start: int) -> str:
    return "\n".join(
        f"{start + index}. [{kind}] {text}"
        for index, (kind, text) in enumerate(behaviours)
    )


def _batches(
    behaviours: list[tuple[str, str]], size: int
) -> list[tuple[int, list[tuple[str, str]]]]:
    """(offset, slice) pairs. An empty behaviour list still yields one batch,
    so a failed plan call degrades to a single self-enumerating call rather
    than to no generation at all.
    """
    if not behaviours:
        return [(0, [])]
    return [
        (start, behaviours[start : start + size])
        for start in range(0, len(behaviours), size)
    ]


def _branch_range(test_cases: list[TestCase]) -> str:
    numbers = [tc.s_no for tc in test_cases]
    return (
        f"Write exactly {len(numbers)} branches, numbered "
        f"{', '.join(str(n) for n in numbers)}, in that order."
    )


def _labelled_blocks(*blocks: tuple[str, str]) -> str:
    """Joins the context blocks under `=== LABEL ===` banners naming what
    each may be used for. An empty block is stated rather than omitted, so
    the model can tell "no track data matched" from "track data was never
    offered" - and so a thin block does not look like permission to borrow
    values from the wording template.
    """
    return "\n\n".join(
        f"=== {label} ===\n{(body or '(nothing retrieved)').strip()}"
        for label, body in blocks
    )


def _folder_hint(parsed: ParsedRequirement) -> str:
    if not parsed.functional_area:
        return ""
    return (
        f'Use "{parsed.functional_area}" as the folder unless the retrieved '
        "test cases consistently use a different name for this area."
    )


def _numbered_test_cases(test_cases: list[TestCase]) -> str:
    return "\n\n".join(f"S_no {tc.s_no}:\n{tc.description}" for tc in test_cases)


# --- Salvaging Python out of the script response ---------------------------

# A line that plausibly starts a Python module. Anything the model emits
# before the first of these is prose, not source.
_PY_START = re.compile(
    r"""^(\#\s*-\*-\s*coding|\#!|\#\s*coding[:=]|from\s+\w|import\s+\w|"""
    r'''"""|\'\'\'|def\s+\w|class\s+\w|@\w)'''
)
# Trailing commentary is unindented English: several words, none of the
# punctuation that makes a line Python.
_CODE_PUNCT = set("=(){}[]:#\\")
# Reasoning-model scratchpads, emitted despite `think: false` by models that
# ignore the toggle.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*\w*[ \t]*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def _looks_like_prose(line: str) -> bool:
    stripped = line.strip()
    if not stripped or line[:1].isspace():
        return False
    if any(ch in _CODE_PUNCT for ch in stripped):
        return False
    return len(stripped.split()) >= 4


def _extract_python(raw: str) -> str:
    """The Python source out of whatever the model actually returned.

    A small model told "output only Python" still, fairly often, narrates its
    plan first and only then writes the script - or spends the whole output
    budget narrating and never reaches it. Rejecting the response would cost
    another multi-minute generation, so the prose is stripped instead:
    `<think>` blocks go first, a fenced block wins if the model marked one,
    otherwise everything before the first line that looks like the start of a
    module is dropped, as is any trailing run of unindented prose.

    If nothing looks like Python at all, the response is returned unchanged:
    a visibly wrong script a reviewer can read beats an empty file.
    """
    text = _THINK_BLOCK.sub("", raw).strip()

    fenced = _FENCE.search(text)
    if fenced is not None and fenced.group(1).strip():
        return fenced.group(1).strip()

    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _PY_START.match(line)), None)
    if start is None:
        logger.warning(
            "The script call returned no recognisable Python; exporting the "
            "raw response. First 200 chars: %.200s",
            text,
        )
        return text
    if start:
        logger.warning("Dropped %d line(s) of prose before the script started.", start)

    body = lines[start:]
    end = len(body)
    while end > 0 and (not body[end - 1].strip() or _looks_like_prose(body[end - 1])):
        end -= 1
    if end < len(body):
        logger.warning("Dropped %d line(s) of prose after the script ended.", len(body) - end)
    return "\n".join(body[:end]).strip()
