from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .confidence import ConfidenceScorer
from .llm_client import LLMClient
from .prompts import PromptLibrary
from .requirement_parser import ParsedRequirement, parse_requirement
from .retriever import HybridRetriever, RetrievedChunk
from .schema import Behaviour, TestCase, parse_behaviour_plan, parse_test_cases
from .static_context import StaticContextProvider

logger = logging.getLogger(__name__)

# Document types holding material the script generator imitates rather than
# reads for facts. Named as defaults, overridable per call, so renaming a
# folder in config/config.yaml doesn't require a code change.
DEFAULT_API_DOC_TYPE = "python_apis"
DEFAULT_SCRIPT_DOC_TYPE = "reference_test_scripts"

# Ceiling on one generation, overridden from config. Not a target — see
# TestCaseGenerator.generate.
DEFAULT_MAX_TEST_CASES = 20

# Shown in a prompt wherever a parsed requirement has no ID, rather than
# leaving the placeholder empty.
_UNSTATED_REQUIREMENT_ID = "(not stated)"


@dataclass
class TestCaseResult:
    requirement_id: str | None
    functional_area: str | None
    test_cases: list[TestCase]
    retrieved_chunks: list[RetrievedChunk]
    raw_response: str
    track_subdivisions: list[str] = field(default_factory=list)
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
    """requirement -> understanding -> hybrid retrieval -> context ->
    behaviour plan -> batched datasheet-row generation -> confidence scores.

    Generation is two kinds of LLM call, not one: `test_case_plan` first
    enumerates every verifiable behaviour the requirement specifies (a short,
    schema-free response that cannot itself be truncated mid-structure), then
    `test_cases` is called once per `test_case_batch_size` behaviours to
    write their datasheet rows as JSON. Each batch's response is short enough
    to never approach `llm_max_tokens`, so a sprawling requirement can no
    longer lose its last few behaviours to truncation the way one big
    single-call response could.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        prompts: PromptLibrary,
        scorer: ConfidenceScorer,
        static_context: StaticContextProvider | None = None,
        static_track_top_k: int = 4,
        example_max_chars: int | None = None,
        test_case_batch_size: int = 5,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompts = prompts
        self.scorer = scorer
        self.static_context = static_context
        self.static_track_top_k = static_track_top_k
        self.example_max_chars = example_max_chars
        self.test_case_batch_size = max(1, test_case_batch_size)

    def generate(
        self,
        requirement_text: str,
        *,
        max_test_cases: int = DEFAULT_MAX_TEST_CASES,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
        subdivision_id: str | None = None,
    ) -> TestCaseResult:
        """How many test cases come back is decided by the requirement, not
        by the caller: test_case_plan enumerates the verifiable behaviours
        the requirement specifies and one case is written per behaviour.
        `max_test_cases` is only a ceiling, so a sprawling requirement cannot
        ask for a response longer than the model's output budget.
        """
        started = time.monotonic()
        parsed = parse_requirement(requirement_text)

        retrieval = self.retriever.retrieve(
            parsed.query_text,
            requirement_id=parsed.requirement_id,
            doc_types=doc_types,
            top_k=top_k,
            subdivision_id=subdivision_id,
        )

        # Only KNOWLEDGE BASE and TRACK DATA are labelled into $context: they
        # are where a concrete value may come from. The wording template is
        # sent to the model separately as $wording_template, so it is never
        # mistaken for a third source of values the way merging it in here
        # would invite.
        track_context = self._static_track_context(parsed, subdivision_id)
        context = _append_static_context(
            _labelled(
                "KNOWLEDGE BASE — parameters, data dictionaries, test guide",
                retrieval.context
                or "(Nothing in the knowledge base matched this requirement.)",
            ),
            _labelled(
                "TRACK DATA — the only source of subdivision, block, milepost, "
                "switch and signal values for this requirement",
                track_context,
            ),
        )
        wording_template = (
            self.static_context.test_case_examples(self.example_max_chars)
            if self.static_context
            else ""
        ) or "(No reference test cases were available.)"

        plan_prompt = self.prompts.pair("test_case_plan")
        plan_user_prompt = plan_prompt.render_user(
            requirement_id=parsed.requirement_id or _UNSTATED_REQUIREMENT_ID,
            requirement_text=parsed.raw_text.strip(),
            context=context,
            max_test_cases=str(max_test_cases),
        )
        plan_response = self.llm_client.generate(plan_prompt.system, plan_user_prompt)
        behaviours = parse_behaviour_plan(plan_response)[:max_test_cases]
        behaviours_block = "\n".join(str(b) for b in behaviours)

        test_cases_prompt = self.prompts.test_cases
        folder_hint = _folder_hint(parsed)
        raw_responses = [plan_response]
        test_cases: list[TestCase] = []

        for batch in _chunked(behaviours, self.test_case_batch_size):
            batch_user_prompt = test_cases_prompt.render_user(
                requirement_id=parsed.requirement_id or _UNSTATED_REQUIREMENT_ID,
                requirement_text=parsed.raw_text.strip(),
                context=context,
                wording_template=wording_template,
                behaviours=behaviours_block,
                batch_list="\n".join(str(b) for b in batch),
                folder_hint=folder_hint,
            )
            raw_response = self.llm_client.generate(
                test_cases_prompt.system, batch_user_prompt
            )
            raw_responses.append(raw_response)
            test_cases.extend(
                parse_test_cases(
                    raw_response,
                    requirement_id=parsed.requirement_id or "",
                    default_folder=parsed.functional_area or "",
                )
            )

        # Each batch call numbers its own rows from 1 (parse_test_cases has
        # no notion of the batches around it), so they're renumbered once
        # here into one dense, requirement-wide sequence.
        for index, test_case in enumerate(test_cases, start=1):
            test_case.s_no = index

        self.scorer.score(test_cases, retrieval.chunks, parsed.requirement_id)

        return TestCaseResult(
            requirement_id=parsed.requirement_id,
            functional_area=parsed.functional_area,
            test_cases=test_cases,
            retrieved_chunks=retrieval.chunks,
            raw_response="\n\n---\n\n".join(raw_responses),
            track_subdivisions=list(retrieval.track_selection.subdivisions),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )

    def _static_track_context(
        self, parsed: ParsedRequirement, subdivision_id: str | None = None
    ) -> str:
        if self.static_context is None:
            return ""
        return self.static_context.track_context(
            parsed.query_text, parsed.requirement_id, self.static_track_top_k, subdivision_id
        )


class TestScriptGenerator:
    """Approved test cases -> a single automation script.

    Deliberately a separate call from test-case generation rather than one
    long prompt. The test cases are the thing a human reviews and edits, and
    the script has to be written against whatever they settled on — so the
    script is generated from the edited rows, not from the model's first
    draft of them. It also keeps each CPU generation short enough to stay
    inside a sensible request timeout.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        prompts: PromptLibrary,
        *,
        api_doc_type: str = DEFAULT_API_DOC_TYPE,
        script_doc_type: str = DEFAULT_SCRIPT_DOC_TYPE,
        max_tokens: int | None = None,
        static_context: StaticContextProvider | None = None,
        static_api_top_k: int = 4,
        static_track_top_k: int = 4,
        kb_top_k: int | None = None,
        example_top_k: int = 1,
        example_max_chars: int | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompts = prompts
        self.api_doc_type = api_doc_type
        self.script_doc_type = script_doc_type
        self.max_tokens = max_tokens
        self.static_context = static_context
        self.static_api_top_k = static_api_top_k
        self.static_track_top_k = static_track_top_k
        self.kb_top_k = kb_top_k
        self.example_top_k = example_top_k
        self.example_max_chars = example_max_chars

    def generate(
        self,
        requirement_text: str,
        test_cases: list[TestCase],
        subdivision_id: str | None = None,
    ) -> ScriptResult:
        if not test_cases:
            raise ValueError("Generate test cases before generating a script.")

        started = time.monotonic()
        parsed = parse_requirement(requirement_text)

        # Retrieval for the script is driven by the test cases as well as the
        # requirement: the specific behaviours being automated are what
        # determine which API calls, which track features and which
        # parameters are relevant.
        query = "\n".join([parsed.query_text, *(tc.description for tc in test_cases)])

        # Four context blocks, kept separate because they play four different
        # roles and the model must not confuse them. The API stubs say what
        # may be called and with which keyword arguments; the track and
        # parameter blocks are the *only* places a concrete value may come
        # from; the examples supply the shape of the file and nothing else.
        # Before these last two existed, the reference scripts were the sole
        # source of subdivisions, blocks, mileposts and TBC values in the
        # prompt — so the model copied them, which is exactly what it should
        # not do.
        api_context = (
            self.static_context.api_context(query, self.static_api_top_k)
            if self.static_context
            else ""
        )
        track_context = (
            self.static_context.track_context(
                query, parsed.requirement_id, self.static_track_top_k, subdivision_id
            )
            if self.static_context
            else ""
        )
        parameter_context = (
            ""
            if self.kb_top_k == 0
            else self.retriever.retrieve(
                query,
                requirement_id=parsed.requirement_id,
                top_k=self.kb_top_k,
                subdivision_id=subdivision_id,
            ).context
        )
        reference_scripts = (
            self.static_context.script_examples(
                query, self.example_top_k, self.example_max_chars
            )
            if self.static_context
            else ""
        )

        prompt = self.prompts.test_script
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or _UNSTATED_REQUIREMENT_ID,
            requirement_text=parsed.raw_text.strip(),
            test_cases=_numbered_test_cases(test_cases),
            api_context=api_context or "(No API definitions were retrieved.)",
            track_context=track_context
            or "(No track-data records matched — every track value must be a TODO variable.)",
            parameter_context=parameter_context
            or "(No parameter or data-dictionary context matched — every parameter value must be a TODO variable.)",
            reference_scripts=reference_scripts or "(No reference scripts were retrieved.)",
            branch_range=_branch_range(test_cases),
        )

        raw = self.llm_client.generate(
            prompt.system, user_prompt, max_tokens=self.max_tokens
        )
        return ScriptResult(
            requirement_id=parsed.requirement_id,
            script=_strip_code_fence(raw),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )


def _append_static_context(context: str, *extra_blocks: str) -> str:
    blocks = [context, *(b for b in extra_blocks if b)]
    return "\n\n---\n\n".join(blocks)


def _labelled(label: str, body: str) -> str:
    """Headers the model can be pointed at by name. An empty body yields an
    empty string, so a missing block is dropped rather than announced as a
    heading with nothing under it.
    """
    if not body.strip():
        return ""
    return f"=== {label} ===\n{body}"


def _folder_hint(parsed: ParsedRequirement) -> str:
    if not parsed.functional_area:
        return ""
    return (
        f'Use "{parsed.functional_area}" as the folder unless the retrieved '
        "test cases consistently use a different name for this area."
    )


def _numbered_test_cases(test_cases: list[TestCase]) -> str:
    return "\n\n".join(f"S_no {tc.s_no}:\n{tc.description}" for tc in test_cases)


def _branch_range(test_cases: list[TestCase]) -> str:
    if not test_cases:
        return ""
    numbers = ", ".join(str(tc.s_no) for tc in test_cases)
    return f"Write exactly one branch for each of these S_no values, in order, with no gaps: {numbers}."


def _chunked(behaviours: list[Behaviour], size: int) -> list[list[Behaviour]]:
    return [behaviours[i : i + size] for i in range(0, len(behaviours), size)]


def _strip_code_fence(raw: str) -> str:
    """Removes a ```python fence if the model added one despite being told
    not to — cheaper than rejecting an otherwise correct script.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
