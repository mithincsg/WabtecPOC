from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .caf_mapping import CafMapping
from .confidence import ConfidenceScorer
from .llm_client import LLMClient
from .prompts import PromptLibrary
from .requirement_parser import ParsedRequirement, parse_requirement
from .retriever import HybridRetriever, RetrievedChunk
from .schema import TEST_CASES_JSON_SCHEMA, TestCase, parse_test_cases
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
    # The datasheet's Folder column for every row of this result, and where
    # it came from ("caf" when the Change Approval Form mapped it, "heading"
    # when it was read out of the requirement text, "" when neither).
    folder: str = ""
    folder_source: str = ""
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
        caf_mapping: CafMapping | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompts = prompts
        self.scorer = scorer
        self.static_context = static_context
        self.static_track_top_k = static_track_top_k
        self.caf_mapping = caf_mapping

    def generate(
        self,
        requirement_text: str,
        *,
        max_test_cases: int = DEFAULT_MAX_TEST_CASES,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
        subdivision: str | None = None,
    ) -> TestCaseResult:
        """How many test cases come back is decided by the requirement, not
        by the caller: the prompt asks the model to enumerate the verifiable
        behaviours the requirement specifies and write one case for each.
        `max_test_cases` is only a ceiling, so a sprawling requirement cannot
        ask for a response longer than the model's output budget.

        `subdivision` is the one the user picked in the UI, and is the only
        thing that decides which track data these cases are written against.
        Without one, no track data reaches the model at all.
        """
        started = time.monotonic()
        parsed = parse_requirement(requirement_text)
        folder, folder_source = self.resolve_folder(parsed.requirement_id, parsed.functional_area)

        retrieval = self.retriever.retrieve(
            parsed.query_text,
            requirement_id=parsed.requirement_id,
            doc_types=doc_types,
            top_k=top_k,
        )

        track_context, searched_subdivisions = self._static_track_context(parsed, subdivision)
        context = retrieval.context or "(Nothing in the knowledge base matched this requirement.)"
        context = _append_static_context(context, track_context)

        prompt = self.prompts.test_cases
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or "(not stated)",
            requirement_text=parsed.raw_text.strip(),
            context=context,
            max_test_cases=str(max_test_cases),
            folder_hint=_folder_hint(folder, folder_source),
            columns=", ".join(label for _, label in _column_labels()),
        )

        raw_response = self.llm_client.generate(
            prompt.system, user_prompt, json_schema=TEST_CASES_JSON_SCHEMA
        )
        test_cases = parse_test_cases(
            raw_response,
            requirement_id=parsed.requirement_id or "",
            default_folder=folder,
        )

        if folder_source == "caf":
            # The Change Approval Form is the authority on which feature a
            # requirement belongs to, so its value replaces whatever the
            # model wrote rather than merely seeding it.
            for test_case in test_cases:
                test_case.folder = folder

        self.scorer.score(test_cases, retrieval.chunks, parsed.requirement_id)

        return TestCaseResult(
            requirement_id=parsed.requirement_id,
            functional_area=parsed.functional_area,
            folder=folder,
            folder_source=folder_source,
            test_cases=test_cases,
            retrieved_chunks=retrieval.chunks,
            raw_response=raw_response,
            # What the track context was actually drawn from: the picked
            # subdivision, or nothing when none was picked.
            track_subdivisions=list(searched_subdivisions),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )

    def resolve_folder(
        self, requirement_id: str | None, functional_area: str | None
    ) -> tuple[str, str]:
        return resolve_folder(self.caf_mapping, requirement_id, functional_area)

    def _static_track_context(
        self, parsed: ParsedRequirement, subdivision: str | None
    ) -> tuple[str, tuple[str, ...]]:
        if self.static_context is None:
            return "", ()
        return self.static_context.track_context_for(
            parsed.query_text,
            self.static_track_top_k,
            subdivision,
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
        self,
        requirement_text: str,
        test_cases: list[TestCase],
        subdivision: str | None = None,
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
        # The same subdivision the datasheet was generated against, so the
        # blocks and mileposts the script hard-codes are the ones its test
        # cases describe.
        track_context = (
            self.static_context.track_context(
                query, self.static_track_top_k, subdivision
            )
            if self.static_context
            else ""
        )
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
        )

        raw = self.llm_client.generate(
            prompt.system, user_prompt, max_tokens=self.max_tokens
        )
        script = _extract_python(_strip_code_fence(raw))
        if script is None:
            # The model narrated its plan instead of writing the script — the
            # failure mode of a reasoning model whose thinking lands in
            # `content`. One retry with the instruction restated as the last
            # thing it reads, which is where it carries most weight.
            logger.warning(
                "Script generation returned prose, not Python; retrying once "
                "with the output-format rule restated."
            )
            raw = self.llm_client.generate(
                prompt.system,
                user_prompt + _PYTHON_ONLY_REMINDER,
                max_tokens=self.max_tokens,
            )
            script = _extract_python(_strip_code_fence(raw))
        elif _unfilled_slots(script):
            # The model copied the house skeleton's `<...>` slots instead of
            # filling them from the context blocks. That is not truncation:
            # the text is complete, it is just a template, and every slot
            # line is a syntax error. One retry saying so explicitly; the
            # first answer is kept if the retry is no better.
            logger.warning(
                "Generated script still contains %d unfilled <...> slots from "
                "the house skeleton; retrying once.",
                len(_unfilled_slots(script)),
            )
            retry = _extract_python(
                _strip_code_fence(
                    self.llm_client.generate(
                        prompt.system,
                        user_prompt + _NO_PLACEHOLDER_REMINDER,
                        max_tokens=self.max_tokens,
                    )
                )
            )
            if retry is not None and not _unfilled_slots(retry):
                script = retry

        return ScriptResult(
            requirement_id=parsed.requirement_id,
            script=(
                _finalize_script(script)
                if script is not None
                else _as_commented_out(raw)
            ),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )


def _append_static_context(context: str, *extra_blocks: str) -> str:
    blocks = [context, *(b for b in extra_blocks if b)]
    return "\n\n---\n\n".join(blocks)


def resolve_folder(
    caf_mapping: CafMapping | None,
    requirement_id: str | None,
    functional_area: str | None,
) -> tuple[str, str]:
    """The datasheet's Folder for a requirement, and where it came from.

    The Change Approval Form wins when it has a row for the requirement: it
    is the maintained list of which feature each requirement belongs to, so
    it keeps the folder stable across runs. The heading read out of the
    pasted requirement text is only the fallback for a requirement the form
    doesn't cover yet.

    A free function, not just a generator method, because the API's folder
    lookup answers on every keystroke and must not pull in the generator —
    and with it the embedding model — to do it. Generation resolves the
    folder through the same call, so what the UI shows before generating is
    what the rows and the exported spreadsheet will carry.
    """
    caf_folder = caf_mapping.folder_for(requirement_id) if caf_mapping else ""
    if caf_folder:
        return caf_folder, "caf"
    if functional_area:
        return functional_area, "heading"
    return "", ""


def _folder_hint(folder: str, folder_source: str) -> str:
    if not folder:
        return ""
    if folder_source == "caf":
        # Mapped in the Change Approval Form, so there is nothing to weigh
        # up: the row is overwritten with this value afterwards anyway, and
        # saying so keeps the model from writing a different one.
        return f'Use exactly "{folder}" as the folder for every row.'
    return (
        f'Use "{folder}" as the folder unless the retrieved '
        "test cases consistently use a different name for this area."
    )


def _column_labels():
    from .schema import DATASHEET_COLUMNS

    return DATASHEET_COLUMNS


def _numbered_test_cases(test_cases: list[TestCase]) -> str:
    return "\n\n".join(f"S_no {tc.s_no}:\n{tc.description}" for tc in test_cases)


# A slot from the house skeleton in prompts.yaml: `<setup: ...>`, `<the
# scenario>`, `<wcr_state.set_onboard_to_cutout_via_1010_msg()>`. The `<` has
# to be followed immediately by a word character, `{` or `.`, so an ordinary
# comparison (`a < b and c > d`) is not mistaken for a slot.
_SLOT = re.compile(r"<[{\w.][^<\n]*>")
# A slot the model wrapped over several lines: opened here, closed on a later
# line. Its continuation runs until a line that *ends* with `>` — the closing
# bracket of the slot — rather than the first `>` seen, because a slot's text
# can itself contain one (`send_<msg id>_msg(...)`).
_SLOT_OPEN = re.compile(r"<[{\w.][^<>\n]*$")


def _unfilled_slots(script: str) -> list[int]:
    """Line numbers still holding a house-skeleton `<...>` slot.

    A small model sometimes reproduces the skeleton instead of filling it in
    — the reply looks like a script and streams like one, but every slot
    line is a syntax error, so the export would otherwise keep only the
    handful of literal lines above the first branch. Detecting that is what
    lets the caller retry, and `_stub_slots` what keeps the rest of the file.
    """
    return [
        i
        for i, line in enumerate(script.splitlines(), 1)
        if _SLOT.search(line) or _SLOT_OPEN.search(line)
    ]


def _stub_slots(script: str) -> str:
    """Turns unfilled slots into `pass  # TODO:` lines, keeping their text.

    Commenting a slot out would empty its `if case == N:` block and break the
    file a second way, so a slot statement becomes a `pass` carrying the slot
    text as a TODO. Nothing the model wrote is lost, the export stays
    runnable Python, and the reviewer sees exactly which lines were never
    filled in.
    """
    out: list[str] = []
    in_slot = False
    for line in script.splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        stripped = line.strip()
        if in_slot:
            out.append(f"{indent}# {stripped}")
            in_slot = not stripped.endswith(">")
            continue
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if _SLOT.search(line):
            # A slot that opens and closes on this line: stub it and carry on.
            out.append(f"{indent}pass  # TODO: unfilled slot — {stripped}")
            continue
        if _SLOT_OPEN.search(line):
            out.append(f"{indent}pass  # TODO: unfilled slot — {stripped}")
            in_slot = True
            continue
        out.append(line)
    return "\n".join(out)


def _finalize_script(script: str) -> str:
    """The export text for a generated script.

    Three shapes reach here: one that already parses (returned untouched),
    one that echoed the skeleton's slots (stubbed, so the whole script
    survives instead of only its parsing head), and one that ran out of
    output tokens (salvaged).
    """
    slots = _unfilled_slots(script)
    if not slots:
        return _salvage_truncated(script)
    stubbed = _stub_slots(script)
    if _parsing_prefix(stubbed) == stubbed:
        logger.warning(
            "Generated script left %d house-skeleton slot(s) unfilled; "
            "exported with each one as a `pass  # TODO:`. The model copied "
            "the template instead of applying it — check llm_model in "
            "config/rag_config.yaml; a very small model tends to do this.",
            len(slots),
        )
        return (
            "# TODO: the model reproduced the house skeleton instead of "
            "filling it in.\n"
            f"# The {len(slots)} line(s) marked `pass  # TODO: unfilled slot` "
            "still have to be written.\n\n"
            + stubbed.rstrip()
            + "\n"
        )
    return _salvage_truncated(stubbed)


def _salvage_truncated(script: str) -> str:
    """Makes a script that stopped mid-statement parse again.

    A generation can end short of the model's own ending — it ran out of
    output tokens, or (if a total budget is configured at all) llm_client hit
    it and kept the text written so far. Either way the tail is
    half a statement, which makes the whole .txt unrunnable over what may be
    a dozen correct test cases. So trailing lines are dropped until what
    remains parses, and a TODO marks where it stopped. The equivalent for
    datasheet rows is the truncation salvage in schema.py.

    A script that already parses is returned untouched, so this costs one
    `compile` on the normal path.
    """
    candidate = _parsing_prefix(script)
    if candidate is None:
        # Nothing parses at all: hand back what the model wrote rather than an
        # empty file, so the reviewer can see what went wrong.
        logger.warning("Generated script does not parse at all; returning it as-is.")
        return script
    total = len(script.splitlines())
    kept = len(candidate.splitlines())
    if kept == total:
        return script
    logger.warning(
        "Generated script did not parse; kept the first %d of %d lines. "
        "The generation most likely ran out of output tokens — raise "
        "script_max_tokens in config/rag_config.yaml.",
        kept,
        total,
    )
    dropped = "\n".join(
        ("# " + line if line.strip() else "#")
        for line in script.splitlines()[kept:]
    )
    # The lines that did not parse are kept as comments rather than
    # deleted: they streamed past in the log as the generation ran, and an
    # export holding less than the terminal showed reads as a lost run.
    return (
        candidate.rstrip()
        + "\n\n# TODO: generation ended here — the lines below did not "
        "parse and are\n# kept as comments for reference.\n"
        + dropped
        + "\n"
    )


_NO_PLACEHOLDER_REMINDER = """

Your previous answer copied the house skeleton's angle-bracket slots
(`<setup: ...>`, `<the scenario>`) into the reply. Those are instructions to
you, not code: a line containing one is a syntax error and the file will not
run. Write the real call in each slot's place, using the API, track-data and
parameter context blocks above. Where a slot's value is genuinely not in any
context block, write a named variable assigned None with a `# TODO:` saying
what is missing — never the slot text itself. Your reply must contain no
`<` placeholder anywhere.
"""

_PYTHON_ONLY_REMINDER = """

Your previous answer to this was prose explaining what the script should do.
That is not usable: the export is executed as Python. Do not explain, do not
reason in the answer, do not restate the test cases. Reply with the script
file itself and nothing else — the first character of your reply must be the
`#` of the encoding comment, and the last line must be `    main()`. Anything
you would have written as explanation belongs in a `#` comment inside the
code, or in a `# TODO:` where a value was not derivable from the context.
"""

# A generated script always opens with the house skeleton's first lines, so
# these are where its Python starts inside a response that drifted into prose
# first. Deliberately narrow: a looser anchor would latch onto a sentence.
_SCRIPT_ANCHOR = re.compile(
    r"^\s*(?:#\s*This Python file uses"
    r"|from\s+Common\."
    r"|from\s+[\w.]+\s+import\s"
    r"|import\s+\w"
    r"|def\s+main\s*\()",
)


def _extract_python(text: str) -> str | None:
    """The Python source inside a response, or None if there is none.

    A reasoning model whose chain of thought lands in `message.content`
    (rather than the separate `thinking` field `strip_reasoning` handles)
    answers with paragraphs of English, sometimes with the real script buried
    after them. Prose is not Python, so this finds where the code starts and
    drops everything before it; `_salvage_truncated` then trims any prose that
    followed the code.

    Returns None rather than a best effort when nothing in the response is a
    script, so the caller can retry instead of exporting narration.
    """
    lines = text.splitlines()
    starts = [0] + [i for i, line in enumerate(lines) if i and _SCRIPT_ANCHOR.match(line)]
    for start in starts:
        candidate = "\n".join(lines[start:]).strip()
        if _is_script(candidate):
            return candidate
        # A generation that ran out of tokens ends mid-statement, so the whole
        # candidate does not compile even though it is a script. Judge it by
        # the part that does; the caller's `_salvage_truncated` then trims the
        # broken tail and marks it.
        prefix = _parsing_prefix(candidate)
        if prefix is not None and _is_script(prefix):
            return candidate
    return None


def _is_script(text: str) -> bool:
    """Whether `text` is the automation script rather than English about it.

    Parsing alone is not enough: a line of prose such as `Requirement:
    L2R433018` is a valid annotated-name statement, so a paragraph can compile
    by accident. A real script imports the WCR helpers and defines functions,
    so both of those are required as well.
    """
    if "def " not in text or not re.search(r"^\s*(?:import|from)\s", text, re.M):
        return False
    try:
        compile(text, "<generated>", "exec")
    except SyntaxError:
        return False
    return True


def _parsing_prefix(text: str) -> str | None:
    """The longest run of leading lines that compiles, or None if none does."""
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            compile(candidate, "<generated>", "exec")
        except SyntaxError:
            continue
        return candidate
    return None


def _as_commented_out(raw: str) -> str:
    """Last resort: the model's prose, as comments, so the .txt still parses.

    Reached only when the retry also came back without a script. Handing the
    reviewer an empty file hides what happened, and handing them raw English
    in a .py-shaped export hides that it cannot run — so the text is preserved
    verbatim, commented, under a TODO that says what went wrong.
    """
    logger.warning(
        "Script generation produced no Python after a retry; exporting the "
        "model's response as comments. Check llm_think in "
        "config/rag_config.yaml — a reasoning model is answering with its "
        "reasoning."
    )
    body = "\n".join(
        ("# " + line if line.strip() else "#") for line in raw.strip().splitlines()
    )
    return (
        "# This Python file uses the following encoding: utf-8\n"
        "# TODO: no script was generated — the model replied with prose, not\n"
        "# Python. Its reply is preserved below for reference; the script has\n"
        "# to be written by hand or the generation re-run.\n\n" + body + "\n"
    )


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
