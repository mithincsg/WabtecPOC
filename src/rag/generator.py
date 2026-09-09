from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .confidence import ConfidenceScorer
from .llm_client import LLMClient
from .prompts import PromptLibrary
from .requirement_parser import ParsedRequirement, parse_requirement
from .retriever import HybridRetriever, RetrievedChunk
from .schema import TestCase, parse_test_cases
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
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompts = prompts
        self.scorer = scorer
        self.static_context = static_context
        self.static_track_top_k = static_track_top_k

    def generate(
        self,
        requirement_text: str,
        *,
        max_test_cases: int = DEFAULT_MAX_TEST_CASES,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> TestCaseResult:
        """How many test cases come back is decided by the requirement, not
        by the caller: the prompt asks the model to enumerate the verifiable
        behaviours the requirement specifies and write one case for each.
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
        )

        context = retrieval.context or "(Nothing in the knowledge base matched this requirement.)"
        context = _append_static_context(
            context,
            self._static_track_context(parsed),
            self.static_context.examples_context if self.static_context else "",
        )

        prompt = self.prompts.test_cases
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or "(not stated)",
            requirement_text=parsed.raw_text.strip(),
            context=context,
            max_test_cases=str(max_test_cases),
            folder_hint=_folder_hint(parsed),
            columns=", ".join(label for _, label in _column_labels()),
        )

        raw_response = self.llm_client.generate(prompt.system, user_prompt)
        test_cases = parse_test_cases(
            raw_response,
            requirement_id=parsed.requirement_id or "",
            default_folder=parsed.functional_area or "",
        )

        self.scorer.score(test_cases, retrieval.chunks, parsed.requirement_id)

        return TestCaseResult(
            requirement_id=parsed.requirement_id,
            functional_area=parsed.functional_area,
            test_cases=test_cases,
            retrieved_chunks=retrieval.chunks,
            raw_response=raw_response,
            track_subdivisions=list(retrieval.track_selection.subdivisions),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )

    def _static_track_context(self, parsed: ParsedRequirement) -> str:
        if self.static_context is None:
            return ""
        return self.static_context.track_context(
            parsed.query_text, parsed.requirement_id, self.static_track_top_k
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

    def generate(
        self, requirement_text: str, test_cases: list[TestCase]
    ) -> ScriptResult:
        if not test_cases:
            raise ValueError("Generate test cases before generating a script.")

        started = time.monotonic()
        parsed = parse_requirement(requirement_text)

        # Retrieval for the script is driven by the test cases as well as the
        # requirement: the specific behaviours being automated are what
        # determine which API calls and which reference script are relevant.
        query = "\n".join([parsed.query_text, *(tc.description for tc in test_cases)])

        api_context = (
            self.static_context.api_context(query, self.static_api_top_k)
            if self.static_context
            else ""
        )
        track_context = (
            self.static_context.track_context(
                query, parsed.requirement_id, self.static_track_top_k
            )
            if self.static_context
            else ""
        )
        reference_scripts = self.static_context.examples_context if self.static_context else ""
        # Parameter names/values, message/event names and defaults live in
        # the knowledge-base data dictionaries — the same general retrieval
        # pass test-case generation uses, just re-run against the script's
        # query (requirement + the approved test-case descriptions).
        parameter_context = self.retriever.retrieve(
            query, requirement_id=parsed.requirement_id
        ).context

        prompt = self.prompts.test_script
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or "(not stated)",
            requirement_text=parsed.raw_text.strip(),
            test_cases=_numbered_test_cases(test_cases),
            api_context=api_context or "(No API definitions were retrieved.)",
            track_context=track_context or "(No track data was retrieved.)",
            parameter_context=parameter_context or "(No parameter/data-dictionary context was retrieved.)",
            reference_scripts=reference_scripts or "(No reference scripts were retrieved.)",
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


def _folder_hint(parsed: ParsedRequirement) -> str:
    if not parsed.functional_area:
        return ""
    return (
        f'Use "{parsed.functional_area}" as the folder unless the retrieved '
        "test cases consistently use a different name for this area."
    )


def _column_labels():
    from .schema import DATASHEET_COLUMNS

    return DATASHEET_COLUMNS


def _numbered_test_cases(test_cases: list[TestCase]) -> str:
    return "\n\n".join(f"S_no {tc.s_no}:\n{tc.description}" for tc in test_cases)


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
