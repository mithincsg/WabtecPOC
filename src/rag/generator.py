from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .caf_mapping import CafMapping
from .llm_client import LLMClient
from .prompts import PromptLibrary
from .requirement_parser import parse_requirement
from .keyword_index import KeywordHit
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


@dataclass
class ScriptResult:
    requirement_id: str | None
    script: str
    elapsed_seconds: float = 0.0
    # Method names the script calls (as `wcr_<something>.<name>(`) that do
    # not exist anywhere in data/python_apis -- an invented call like
    # `send_acknowledge_key`, which no amount of better retrieval can
    # prevent, only catch after the fact. Empty when nothing to flag, or
    # when there is no StaticContextProvider to validate against.
    unknown_api_calls: list[str] = field(default_factory=list)
    # `track_group`/`sub_folder` values the script writes that are not a
    # real track name in this subdivision's own TrackNameFeature legend --
    # catches the model writing the subdivision's display name (e.g.
    # "Ginger") where a track name ("Main1", "Siding", ...) belongs. Empty
    # when nothing to flag, when no subdivision was picked, or when there is
    # no StaticContextProvider to validate against.
    unknown_track_groups: list[str] = field(default_factory=list)


class TestCaseGenerator:
    """requirement -> understanding -> parameter-config context -> LLM ->
    datasheet rows.

    Deliberately takes no `HybridRetriever`: the embedded knowledge base is
    not retrieved for this call at all, only the parameter configuration
    guide's records (see `generate`).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptLibrary,
        static_context: StaticContextProvider | None = None,
        static_parameter_top_k: int = 6,
        caf_mapping: CafMapping | None = None,
    ):
        self.llm_client = llm_client
        self.prompts = prompts
        self.static_context = static_context
        self.static_parameter_top_k = static_parameter_top_k
        self.caf_mapping = caf_mapping

    def generate(
        self,
        requirement_text: str,
        *,
        max_test_cases: int = DEFAULT_MAX_TEST_CASES,
        subdivision: str | None = None,
    ) -> TestCaseResult:
        """How many test cases come back is decided by the requirement, not
        by the caller: the prompt asks the model to enumerate the verifiable
        behaviours the requirement specifies and write one case for each.
        `max_test_cases` is only a ceiling, so a sprawling requirement cannot
        ask for a response longer than the model's output budget.

        `subdivision` is accepted so the caller can send the same request
        shape as the script call, but no track data reaches this generation:
        a datasheet row describes a behaviour to verify, not the blocks it
        runs on.

        Context comes from the parameter configuration guide's records
        (`data/parameter_config/`) only — the embedded knowledge base is not
        retrieved for this call at all. The parameter a requirement names is
        the thing its test cases assert against, and its valid range is what
        a boundary case is written from, so the records are what a row needs.
        """
        started = time.monotonic()
        parsed = parse_requirement(requirement_text)
        logger.info(
            "Test-case generation started: requirement=%s, functional_area=%s",
            parsed.requirement_id or "(not stated)",
            parsed.functional_area or "(none)",
        )
        folder, folder_source = self.resolve_folder(parsed.requirement_id)
        logger.info("Folder resolved: %r (source=%s)", folder, folder_source or "none")

        parameter_hits = (
            self.static_context.parameter_hits(parsed.query_text, self.static_parameter_top_k)
            if self.static_context
            else []
        )
        parameter_chunks = [_chunk_from_hit(hit) for hit in parameter_hits]
        # Same query and predicate as parameter_hits above, so this is a cache
        # hit against keyword_index's per-query memoisation, not a second scan.
        parameter_context = (
            self.static_context.parameter_context(parsed.query_text, self.static_parameter_top_k)
            if self.static_context
            else ""
        )
        # No track data here: a datasheet description states the behaviour to
        # verify, not the blocks it runs on. Track values are fetched for the
        # script call, which is what hard-codes them.
        context = parameter_context or "(No parameter records matched this requirement.)"

        prompt = self.prompts.test_cases
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or "(not stated)",
            requirement_text=parsed.raw_text.strip(),
            context=context,
            max_test_cases=str(max_test_cases),
            folder_hint=_folder_hint(folder),
            columns=", ".join(label for _, label in _column_labels()),
        )
        logger.info(
            "Context assembled: %d chars (parameter records only); "
            "sending %d-char prompt to the datasheet LLM call",
            len(context),
            len(user_prompt),
        )
        _log_context_block("parameter records", parameter_context)

        raw_response = self.llm_client.generate(
            prompt.system, user_prompt, json_schema=TEST_CASES_JSON_SCHEMA
        )
        test_cases = parse_test_cases(
            raw_response,
            requirement_id=parsed.requirement_id or "",
            default_folder=folder,
        )
        logger.info("Parsed %d test case(s) from the LLM response", len(test_cases))

        if folder_source == "caf":
            # The Change Approval Form is the authority on which feature a
            # requirement belongs to, so its value replaces whatever the
            # model wrote rather than merely seeding it.
            for test_case in test_cases:
                test_case.folder = folder

        elapsed = round(time.monotonic() - started, 1)
        logger.info(
            "Test-case generation finished in %.1fs: %d case(s)", elapsed, len(test_cases)
        )

        return TestCaseResult(
            requirement_id=parsed.requirement_id,
            functional_area=parsed.functional_area,
            folder=folder,
            folder_source=folder_source,
            test_cases=test_cases,
            retrieved_chunks=parameter_chunks,
            raw_response=raw_response,
            # Always empty: no track data is searched for a datasheet.
            track_subdivisions=[],
            elapsed_seconds=elapsed,
        )

    def resolve_folder(self, requirement_id: str | None) -> tuple[str, str]:
        return resolve_folder(self.caf_mapping, requirement_id)


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
        static_parameter_top_k: int = 6,
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
        self.static_parameter_top_k = static_parameter_top_k

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
        logger.info(
            "Script generation started: requirement=%s, %d test case(s), subdivision=%s",
            parsed.requirement_id or "(not stated)",
            len(test_cases),
            subdivision or "(none)",
        )

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
        # Parameter names/values, message/event names and defaults come from
        # two places: the TBC/CFG/THE records converted out of the parameter
        # configuration guide, and the knowledge-base data dictionaries — the
        # same general retrieval pass test-case generation uses, re-run
        # against the script's query (requirement + approved descriptions).
        # The records go first: a script asserting on TBC137 needs that
        # parameter's own row, not the paragraph nearest to it.
        parameter_records = (
            self.static_context.parameter_context(query, self.static_parameter_top_k)
            if self.static_context
            else ""
        )
        parameter_context = _append_static_context(
            parameter_records,
            self.retriever.retrieve(query, requirement_id=parsed.requirement_id).context,
        )

        prompt = self.prompts.test_script
        user_prompt = prompt.render_user(
            requirement_id=parsed.requirement_id or "(not stated)",
            requirement_text=parsed.raw_text.strip(),
            test_cases=_numbered_test_cases(test_cases),
            api_context=api_context or "(No API definitions were retrieved.)",
            track_context=track_context or "(No track data was retrieved.)",
            parameter_context=parameter_context or "(No parameter/data-dictionary context was retrieved.)",
        )
        logger.info(
            "Context assembled: api=%d chars, track=%d chars, parameter/kb=%d chars; "
            "sending %d-char prompt to the script LLM call",
            len(api_context),
            len(track_context),
            len(parameter_context),
            len(user_prompt),
        )

        raw = self.llm_client.generate(
            prompt.system,
            user_prompt,
            max_tokens=self.max_tokens,
            prefill=_SCRIPT_PREFILL,
            stop_when=_script_looks_complete,
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
                prefill=_SCRIPT_PREFILL,
                stop_when=_script_looks_complete,
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
                        prefill=_SCRIPT_PREFILL,
                        stop_when=_script_looks_complete,
                    )
                )
            )
            if retry is not None and not _unfilled_slots(retry):
                script = retry

        final_script = _finalize_script(script) if script is not None else _as_commented_out(raw)
        final_script = _fix_test_type_table(final_script, test_cases)
        final_script = _ensure_subdiv_id(final_script, subdivision)
        final_script = _add_position_provenance_comments(final_script, self.static_context, subdivision)
        elapsed = round(time.monotonic() - started, 1)
        unknown_calls = self._unknown_api_calls(final_script)
        _log_unknown_api_calls(unknown_calls)
        unknown_groups = self._unknown_track_groups(final_script, subdivision)
        _log_unknown_track_groups(unknown_groups)
        logger.info(
            "Script generation finished in %.1fs: %d chars", elapsed, len(final_script)
        )
        return ScriptResult(
            requirement_id=parsed.requirement_id,
            script=final_script,
            elapsed_seconds=elapsed,
            unknown_api_calls=unknown_calls,
            unknown_track_groups=unknown_groups,
        )

    def _unknown_api_calls(self, script: str) -> list[str]:
        """Method names `script` calls that data/python_apis never defines.

        Retrieval (core methods, triggers, keyword search) can only improve
        the odds the model calls a real method correctly -- it cannot rule
        out the model inventing a plausible-sounding name it never saw
        documentation for, the way `send_acknowledge_key` was invented from
        a requirement's wording with no such method anywhere in the API.
        This is the check that catches that case instead of a reviewer.
        """
        if self.static_context is None:
            return []
        known = self.static_context.known_api_methods()
        called = self.static_context.api_call_names(script)
        return sorted(name for name in called if name not in known)

    def _unknown_track_groups(self, script: str, subdivision: str | None) -> list[str]:
        """`track_group`/`sub_folder` values `script` writes that are not a
        real track name for `subdivision`.

        Same shape as `_unknown_api_calls`: retrieval and the prompt rules
        can only improve the odds the model picks a real track name, not
        rule out it reaching instead for a similarly-worded field it also
        saw in the same context block (the subdivision's own display name,
        e.g. "Ginger" for subdivision 8101, sitting beside the real track
        names in the track-data context). This is the check that catches
        that mix-up after the fact.
        """
        if self.static_context is None or not subdivision:
            return []
        known = self.static_context.known_track_groups(subdivision)
        if not known:
            # No TrackNameFeature legend indexed for this subdivision --
            # nothing to validate against, so say nothing rather than flag
            # every value as unknown.
            return []
        used = _track_group_values(script)
        return sorted(name for name in used if name not in known)


def _log_unknown_api_calls(unknown_calls: list[str]) -> None:
    if unknown_calls:
        logger.warning(
            "Generated script calls %d method(s) not found anywhere in "
            "data/python_apis -- likely invented, not just mis-called: %s",
            len(unknown_calls),
            ", ".join(unknown_calls),
        )


# A `"track_group": "Main1"` or `"sub_folder": "Main1"` keyword argument, as
# written by `wcr_loco_sim.set_position(...)` / `wcr_track.set_track_to_use(...)`
# per the house skeleton -- either single- or double-quoted, since the model
# is not required to pick one style.
_TRACK_GROUP_KWARG_RE = re.compile(
    r"""["'](?:track_group|sub_folder)["']\s*:\s*["']([^"']+)["']"""
)


def _track_group_values(script: str) -> set[str]:
    return set(_TRACK_GROUP_KWARG_RE.findall(script))


def _log_unknown_track_groups(unknown_groups: list[str]) -> None:
    if unknown_groups:
        logger.warning(
            "Generated script uses %d track_group/sub_folder value(s) not "
            "found in this subdivision's own TrackNameFeature legend -- "
            "likely the subdivision's display name written where a track "
            "name belongs: %s",
            len(unknown_groups),
            ", ".join(unknown_groups),
        )


def _chunk_from_hit(hit: KeywordHit) -> RetrievedChunk:
    """A parameter-config keyword hit, in the shape the API's
    retrieved-context response already knows how to read.

    `similarity` stays None — these are BM25-only hits, never embedded.
    """
    return RetrievedChunk(
        chunk_id=hit.chunk_id,
        text=hit.text,
        metadata=hit.metadata,
        bm25_score=hit.score,
        score=hit.score,
        matched_by=["keyword"],
    )


def _log_context_block(label: str, text: str) -> None:
    logger.info("--- %s context (%d chars) ---\n%s", label, len(text), text or "(empty)")


def _append_static_context(context: str, *extra_blocks: str) -> str:
    blocks = [b for b in (context, *extra_blocks) if b]
    return "\n\n---\n\n".join(blocks)


def resolve_folder(
    caf_mapping: CafMapping | None,
    requirement_id: str | None,
) -> tuple[str, str]:
    """The datasheet's Folder for a requirement, and where it came from.

    The Change Approval Form is the only source: it is the maintained list
    of which feature each requirement belongs to. A requirement the form
    doesn't cover yet gets an empty folder rather than a guess — the
    heading read out of the pasted requirement text names a functional
    area, not necessarily the CAF feature, so filling the column with it
    would put an unmapped requirement in a folder that looks authoritative
    but isn't.

    A free function, not just a generator method, because the API's folder
    lookup answers on every keystroke and must not pull in the generator —
    and with it the embedding model — to do it. Generation resolves the
    folder through the same call, so what the UI shows before generating is
    what the rows and the exported spreadsheet will carry.
    """
    caf_folder = caf_mapping.folder_for(requirement_id) if caf_mapping else ""
    if caf_folder:
        return caf_folder, "caf"
    return "", ""


def _folder_hint(folder: str) -> str:
    if not folder:
        return ""
    # Mapped in the Change Approval Form, so there is nothing to weigh up:
    # the row is overwritten with this value afterwards anyway, and saying
    # so keeps the model from writing a different one.
    return f'Use exactly "{folder}" as the folder for every row.'


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


# The `run_test.set_test_case_function_table({...})` call from the house
# skeleton -- captures the dict literal on its own so it can be read and
# replaced without disturbing the surrounding call syntax. Assumes a flat
# `{"Type": "handler", ...}` literal (what the skeleton asks for); a dict
# containing nested braces would not match, and is left alone rather than
# mishandled.
_FUNCTION_TABLE_CALL_RE = re.compile(
    r"(run_test\.set_test_case_function_table\(\s*)(\{[^{}]*\})(\s*\))"
)
_TABLE_ENTRY_RE = re.compile(r"""["']([^"']+)["']\s*:\s*["']([^"']+)["']""")
# The house skeleton's handler signature -- `def handle_test_case(data_sheet=None):`
# -- matched generically on the parameter name rather than the literal
# function name, since a script is free to call it anything.
_HANDLER_DEF_RE = re.compile(r"^def\s+(\w+)\s*\(\s*data_sheet\b", re.M)


def _fix_test_type_table(script: str, test_cases: list[TestCase]) -> str:
    """Repairs `run_test.set_test_case_function_table(...)` so every test
    type actually present in `test_cases` has an entry, deterministically.

    Which test types are present is a mechanical fact already sitting on
    `test_cases` (`TestCase.test_type`) -- not something that needs a model
    to enumerate correctly on every generation the way the house skeleton's
    `<{"Positive": "handle_test_case", ...} for the test types actually
    present>` slot currently asks it to. Observed gap: a datasheet with both
    Positive and Negative rows produced a script whose table named only
    "Positive", so every Negative-typed row's branch existed but
    `run_test.run()` never dispatched to it.

    Existing entries are kept exactly as written -- a script may
    legitimately route different test types to different handlers, and this
    is a repair, not a rewrite. Only a test type with no entry at all is
    added, pointed at whichever handler an existing entry already uses, or
    failing that, the script's own `def <handler>(data_sheet...)`. Leaves
    the script untouched if there's no table call to fix, every type is
    already covered, or there is no handler to route a new entry to.
    """
    match = _FUNCTION_TABLE_CALL_RE.search(script)
    if match is None:
        return script

    entries = dict(_TABLE_ENTRY_RE.findall(match.group(2)))
    needed = sorted({tc.test_type for tc in test_cases if tc.test_type})
    missing = [test_type for test_type in needed if test_type not in entries]
    if not missing:
        return script

    handler = next(iter(entries.values()), None)
    if handler is None:
        handler_match = _HANDLER_DEF_RE.search(script)
        handler = handler_match.group(1) if handler_match else None
    if handler is None:
        return script

    for test_type in missing:
        entries[test_type] = handler
    logger.warning(
        "Generated script's set_test_case_function_table was missing %d "
        "test type(s) present in the approved test cases; added routed to "
        "%r: %s",
        len(missing),
        handler,
        ", ".join(missing),
    )

    fixed_dict = "{" + ", ".join(f'"{k}": "{v}"' for k, v in entries.items()) + "}"
    return script[: match.start(2)] + fixed_dict + script[match.end(2) :]


# A `wcr_track.set_track_to_use({...})` call's dict literal, captured the
# same flat-dict way `_FUNCTION_TABLE_CALL_RE` captures the dispatch table.
_SET_TRACK_CALL_RE = re.compile(
    r"(wcr_track\.set_track_to_use\(\s*)(\{[^{}]*\})(\s*\))"
)
_SUBDIV_ID_KEY_RE = re.compile(r"""["']subdivID["']""")


def _ensure_subdiv_id(script: str, subdivision: str | None) -> str:
    """Adds a missing `subdivID` key to every `set_track_to_use(...)` call,
    set to `subdivision` -- the same value the UI's picker sent into this
    generation, not a guess.

    Every real reference script includes `subdivID` in this call regardless
    of `TrackType` (`data/Examples/reference_test_scripts`), even though the
    API docstring only calls it required for `'AUX_POOL'` -- the house
    convention is stricter than the bare API minimum, and the model has
    been observed to drop it for `'AUX'`. A call that already has the key
    is left exactly as written.
    """
    if not subdivision:
        return script

    def _replace(match: "re.Match[str]") -> str:
        dict_text = match.group(2)
        if _SUBDIV_ID_KEY_RE.search(dict_text):
            return match.group(0)
        inner = dict_text[1:-1].rstrip()
        sep = ", " if inner.strip() else ""
        value = subdivision if subdivision.isdigit() else f'"{subdivision}"'
        return f"{match.group(1)}{{{inner}{sep}\"subdivID\": {value}}}{match.group(3)}"

    fixed = _SET_TRACK_CALL_RE.sub(_replace, script)
    if fixed != script:
        logger.warning(
            "Generated script had a set_track_to_use(...) call missing "
            "subdivID; added subdivID=%s per the house convention.",
            subdivision,
        )
    return fixed


# One `wcr_loco_sim.set_position({...})` call and whatever follows it on the
# same physical line -- captured separately so a comment already present
# after the call (`rest` containing `#`) is left alone rather than doubled.
_SET_POSITION_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<call>wcr_loco_sim\.set_position\(\s*(?P<dict>\{[^{}]*\})\s*\))(?P<rest>[^\n]*)$",
    re.MULTILINE,
)
_POSITION_FIELD_RE = re.compile(
    r"""["'](track_group|point)["']\s*:\s*["']?([^,"'}]+?)["']?\s*[,}]"""
)


def _add_position_provenance_comments(
    script: str, static_context: "StaticContextProvider | None", subdivision: str | None
) -> str:
    """Adds the block-id/milepost-range comment the system prompt asks for
    (`# 385000(point) is within block 13022 (380000-390000)`) to every
    `set_position(...)` call that doesn't already carry a trailing comment,
    resolved deterministically against the real track data instead of left
    to the model to remember on every generation -- observed gap: none of
    11 branches in one real generation carried it despite the prompt rule.

    A no-op wherever the real block can't be resolved (`find_block`
    returning None: unknown subdivision, a `track_group` not in this
    subdivision's own track names, or a `point` outside every block on
    it) -- silence in that case, never a guessed value appended. Also a
    no-op on a line that already has a `#` after the call, so a comment the
    model did write is never duplicated.
    """
    if static_context is None or not subdivision:
        return script

    def _replace(match: "re.Match[str]") -> str:
        if "#" in match.group("rest"):
            return match.group(0)
        fields = dict(_POSITION_FIELD_RE.findall(match.group("dict")))
        track_group = fields.get("track_group")
        point_text = fields.get("point")
        if not track_group or point_text is None:
            return match.group(0)
        try:
            point = float(point_text)
        except ValueError:
            return match.group(0)
        block = static_context.find_block(subdivision, track_group, point)
        if block is None:
            return match.group(0)
        block_id, start, end = block
        point_disp = int(point) if point.is_integer() else point
        start_disp = int(start) if start.is_integer() else start
        end_disp = int(end) if end.is_integer() else end
        comment = f"  # {point_disp}(point) is within block {block_id} ({start_disp}-{end_disp})"
        return match.group(0) + comment

    fixed = _SET_POSITION_LINE_RE.sub(_replace, script)
    if fixed != script:
        logger.warning(
            "Generated script had set_position(...) call(s) with no "
            "block/milepost provenance comment; added one from the real "
            "track data where the block could be resolved."
        )
    return fixed


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


# The script call is the one generation a JSON schema cannot constrain, so
# the constraint is applied to the answer's opening instead: the model is
# handed the house skeleton's first line as the start of its own turn and
# continues from there. A reply that begins inside a comment line of a Python
# file does not begin "We are given one test case..." — which is what a
# chattier or reasoning-tuned model otherwise spends the whole output budget
# doing, reaching the exporter as prose and leaving `_as_commented_out` to
# salvage it. Only the encoding comment is prefilled: enough to fix the
# answer's form, not enough to put a value in the model's mouth.
_SCRIPT_PREFILL = "# This Python file uses the following encoding: utf-8\n"

# The house skeleton's required closing lines, per the system prompt's own
# OUTPUT FORMAT rule: "the last line is `main()`. Nothing before, nothing
# after." A bare `main()` call (as opposed to the `def main():` earlier in
# the same skeleton) only ever legitimately appears once, here, so seeing it
# is an unambiguous signal the script is complete -- not a guess from a
# character or token count. Tolerates a trailing comment and either quote
# style, since neither changes whether the script is actually finished.
_SCRIPT_END_RE = re.compile(
    r"if\s+__name__\s*==\s*[\"']__main__[\"']\s*:\s*\n\s*main\(\)\s*(#.*)?\s*$"
)


def _script_looks_complete(text: str) -> bool:
    """True once `text` ends with the script's required closing lines.

    Passed as `stop_when` to the script-generation LLM calls, so the stream
    is closed the moment a correct script is finished rather than waiting on
    a token budget -- which is what let a reasoning model that finishes a
    correct script and then keeps going (opening `<think>` and rewriting the
    whole thing again) run for tens of minutes before the budget finally cut
    it off. Checking the real ending scales to any script length, unlike a
    fixed token limit low enough to catch a runaway repeat but high enough
    to never truncate a genuinely long, correct script for a large datasheet.
    """
    return bool(_SCRIPT_END_RE.search(text.rstrip()))

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


_FENCED_BLOCK = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", re.DOTALL | re.M)


def _strip_code_fence(raw: str) -> str:
    """The script inside a markdown fence, if the model wrapped it in one.

    Cheaper than rejecting an otherwise correct script. The fence is not
    always the first thing in the reply — a chat-tuned model introduces it
    ("Here is the script:") and explains it afterwards — so every fenced
    block is considered, not only one at position 0. The largest block that
    is a script wins, and a reply that already reads as a script untouched is
    returned untouched, so a fence marker inside a comment cannot trigger
    this.
    """
    text = raw.strip()
    if "```" not in text or _is_script(text):
        return text
    blocks = [m.group(1).strip() for m in _FENCED_BLOCK.finditer(text)]
    scripts = [block for block in blocks if _is_script(block)]
    if scripts:
        return max(scripts, key=len)
    if blocks:
        # Nothing fenced parses — most likely the generation was cut off
        # mid-block. Hand the longest one on anyway, for the truncation
        # salvage to trim and mark.
        return max(blocks, key=len)
    # An opening fence with no closing one: the budget ran out inside it.
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        return "\n".join(lines[1:]).strip()
    return text
