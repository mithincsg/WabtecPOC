from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Callable

from kb_ingestion.config import PipelineConfig
from kb_ingestion.extractors.xml_extractor import normalize_subdivision
from kb_ingestion.pipeline import ChunkingPipeline, discover_files

from .keyword_index import KeywordHit, KeywordIndex

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class Subdivision:
    """One subdivision folder under data/track_data, as the UI lists it.

    Identified by its number and nothing else: the display names the old HTML
    reports carried ("Ginger") are not in the XML export, and a picker that
    shows a name for some subdivisions and a bare number for others reads as
    a gap in the data rather than a choice.
    """

    id: str
    chunks: int


class _StaticDocumentStore:
    """The corpus KeywordIndex builds its BM25 index over: every chunk of
    every configured source, read straight from disk at startup.
    """

    def __init__(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]):
        self._chunk_ids = chunk_ids
        self._texts = texts
        self._metadatas = metadatas

    def count(self) -> int:
        return len(self._chunk_ids)

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        return self._chunk_ids, self._texts, self._metadatas


def _build_static_chunks(config: PipelineConfig) -> _StaticDocumentStore:
    pipeline = ChunkingPipeline(config)

    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for file in discover_files(config):
        try:
            _hash, chunks = pipeline.prepare_file(file)
        except Exception:
            logger.exception("Failed to parse static-context file %s", file.path)
            continue
        for chunk in chunks:
            chunk_ids.append(chunk.metadata.chunk_id)
            texts.append(chunk.text)
            metadatas.append(chunk.metadata.to_dict())

    logger.info("Built static context over %d chunks", len(chunk_ids))
    return _StaticDocumentStore(chunk_ids, texts, metadatas)


def _list_subdivisions(store: _StaticDocumentStore) -> list[Subdivision]:
    _ids, _texts, metadatas = store.get_all_documents()
    counts: dict[str, int] = {}
    for metadata in metadatas:
        if metadata.get("document_type") != "track_data":
            continue
        subdivision = str(metadata.get("subdivision") or "").strip()
        if not subdivision:
            continue
        counts[subdivision] = counts.get(subdivision, 0) + 1

    found = [
        Subdivision(id=subdivision, chunks=count)
        for subdivision, count in sorted(counts.items())
    ]
    logger.info(
        "Track data covers %d subdivision(s): %s",
        len(found),
        ", ".join(s.id for s in found) or "none",
    )
    return found


def _railroad_scac_by_subdivision(store: _StaticDocumentStore) -> dict[str, str]:
    """`{subdivision: SCAC}`, read off the `railroad_scac` metadata
    `xml_extractor.py` stamps from each file's own `<RailroadSCAC>` field --
    not assumed to be `UP`, or any other single value, for every
    subdivision: a script for a different subdivision could genuinely need
    a different railroad, and this reads whichever one that subdivision's
    own real track database actually states.
    """
    _ids, _texts, metadatas = store.get_all_documents()
    scacs: dict[str, str] = {}
    for metadata in metadatas:
        if metadata.get("document_type") != "track_data":
            continue
        subdivision = str(metadata.get("subdivision") or "").strip()
        scac = str(metadata.get("railroad_scac") or "").strip()
        if subdivision and scac and subdivision not in scacs:
            scacs[subdivision] = scac
    return scacs


# A `TrackName=<value>` field, as `xml_extractor._collect` renders every
# TrackNameFeature record's own fields (`TrackName=Main1 | TrackValue=2 |
# NumberBlocks=89`) -- the field name is `|`/whitespace-delimited from its
# value the same way every other record line is, so this is the same shape
# `_parameter_ids`-style pattern-matching relies on elsewhere in this module.
_TRACK_NAME_FIELD_RE = re.compile(r"\bTrackName=(\S+)")


def _track_groups_by_subdivision(store: _StaticDocumentStore) -> dict[str, frozenset[str]]:
    """`{subdivision: {real track names}}`, read off each subdivision's own
    TrackNameFeature legend -- the only source a script's `track_group` (or
    `set_track_to_use`'s `sub_folder`) may come from.

    Exists to validate against, not to retrieve from: a subdivision's
    TrackNameFeature chunk sits in the same BM25 pool as its
    `SubdivisionName` field (e.g. "Ginger" for subdivision 8101), which
    names the subdivision itself, not any track inside it, and reads
    similarly enough in context that a generation has written the wrong one
    as `track_group`. `known_track_groups` is what a generated script gets
    checked against afterward, the same way `known_api_methods` catches an
    invented API call.
    """
    _ids, texts, metadatas = store.get_all_documents()
    groups: dict[str, set[str]] = {}
    for text, metadata in zip(texts, metadatas):
        if (metadata or {}).get("document_type") != "track_data":
            continue
        if (metadata or {}).get("table_title") != "TrackNameFeature":
            continue
        subdivision = str((metadata or {}).get("subdivision") or "").strip()
        if not subdivision:
            continue
        names = groups.setdefault(subdivision, set())
        names.update(_TRACK_NAME_FIELD_RE.findall(text))
    return {subdivision: frozenset(names) for subdivision, names in groups.items()}


def _block_ranges_by_subdivision(
    store: _StaticDocumentStore,
) -> dict[str, list[tuple[str, float, float, str]]]:
    """`{subdivision: [(track name, start milepost, end milepost, block id)]}`,
    read off every `BlockFeature` record.

    Each `BlockFeature` record is one line of `field=value` pairs (see
    `xml_extractor._collect`), including its own resolved `TrackName`
    alongside `BlockId`/`StartMilepost`/`EndMilepost` — no second record to
    join against. Built from the same chunks `track_context` searches, not
    a separate parse of the XML, so it stays in sync with whatever
    `static_track_top_k`/chunking currently indexes.

    For resolving a script's own `(track_group, point)` back to the real
    block it falls in, e.g. for the trailing provenance comment on
    `set_position` — see `find_block`.
    """
    _ids, texts, metadatas = store.get_all_documents()
    ranges: dict[str, list[tuple[str, float, float, str]]] = {}
    for text, metadata in zip(texts, metadatas):
        if (metadata or {}).get("document_type") != "track_data":
            continue
        if (metadata or {}).get("table_title") != "BlockFeature":
            continue
        subdivision = str((metadata or {}).get("subdivision") or "").strip()
        if not subdivision:
            continue
        for line in text.splitlines():
            fields = dict(part.split("=", 1) for part in line.split(" | ") if "=" in part)
            block_id = fields.get("BlockId")
            track_name = fields.get("TrackName")
            start = fields.get("StartMilepost")
            end = fields.get("EndMilepost")
            if not (block_id and track_name and start and end):
                continue
            try:
                start_f, end_f = float(start), float(end)
            except ValueError:
                continue
            ranges.setdefault(subdivision, []).append((track_name, start_f, end_f, block_id))
    return ranges


# Identifiers as a requirement writes them: "TBC137", "CFG 65", "THE12".
_PARAMETER_ID_RE = re.compile(r"\b(TBC|CFG|THE)\s*(\d+)\b", re.IGNORECASE)


def _parameter_ids(text: str) -> list[str]:
    ids: list[str] = []
    for prefix, number in _PARAMETER_ID_RE.findall(text or ""):
        identifier = f"{prefix.upper()}{number}"
        if identifier not in ids:
            ids.append(identifier)
    return ids


def _index_parameter_records(store: _StaticDocumentStore) -> dict[str, KeywordHit]:
    """Parameter records by their identifier, for exact lookup.

    BM25 alone does not guarantee the parameter a requirement names comes
    back: "TBC137" is one rare token against a requirement's worth of common
    ones ("speed", "restricted", "enforce"), and a handful of unrelated
    records that match several of those outscore the one record that matters.
    The identifiers stated in the requirement are not a ranking question, so
    they are looked up directly instead.
    """
    ids, texts, metadatas = store.get_all_documents()
    records: dict[str, KeywordHit] = {}
    for chunk_id, text, metadata in zip(ids, texts, metadatas):
        if (metadata or {}).get("document_type") != "parameter_config":
            continue
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        found = _parameter_ids(first_line.split("|")[0])
        if len(found) == 1 and found[0] not in records:
            records[found[0]] = KeywordHit(
                chunk_id=chunk_id, text=text, metadata=metadata or {}, score=0.0
            )
    return records


def _index_api_records(store: _StaticDocumentStore) -> dict[tuple[str, str], KeywordHit]:
    """`python_apis` chunks by `(class name, method name)`, for the core and
    triggered methods to fetch directly instead of via keyword search.

    Keyed by the pair, not by bare method name: a method name alone is not
    unique across classes (`set_bos` alone is defined on 50 different
    office-message classes), so a bare-name index could only ever return one
    arbitrary class's version under that name, regardless of which class a
    script's call actually resolves to.
    """
    ids, texts, metadatas = store.get_all_documents()
    records: dict[tuple[str, str], KeywordHit] = {}
    for chunk_id, text, metadata in zip(ids, texts, metadatas):
        if (metadata or {}).get("document_type") != "python_apis":
            continue
        class_name = (metadata or {}).get("class_name")
        method_name = (metadata or {}).get("method_name")
        if class_name and method_name:
            records.setdefault((class_name, method_name), KeywordHit(
                chunk_id=chunk_id, text=text, metadata=metadata or {}, score=0.0
            ))
    return records


# A top-level `name: ClassName` instance declaration, exactly the shape
# every API object is declared in -- `wcr_track: PublicTrackFile`,
# `run_test: RunTest`, `wabtest: PublicWabtest`. Read from the real source
# rather than assumed from a naming convention: most instances happen to be
# prefixed `wcr_`, but `run_test` and `wabtest` are not, and both are called
# in every reference script's `main()` -- a regex hardcoded to a `wcr_`
# prefix would silently never see either one. Captures both the object's
# name and its class: a method name alone is not unique (`set_bos` alone is
# defined on 50 different office-message classes), so which class an object
# belongs to is needed to tell those apart later.
_INSTANCE_DECL_RE = re.compile(r"^([a-zA-Z_]\w*):\s*([A-Za-z_]\w*)\s*$", re.MULTILINE)


def _python_apis_source_paths(config: PipelineConfig) -> list[Path]:
    roots = [root.path for root in config.static_sources if root.document_type == "python_apis"]
    return [path for root in roots for path in Path(root).glob("**/*.py*")]


def _declared_instances(paths: list[Path]) -> dict[str, str]:
    """`{object name: class name}` for every instance the real API source
    declares at module level, for building a call-matching regex that
    covers the whole file rather than just whichever objects happen to
    follow a `wcr_` naming convention, and for resolving which class a
    called method actually belongs to.
    """
    instances: dict[str, str] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, class_name in _INSTANCE_DECL_RE.findall(text):
            instances.setdefault(name, class_name)
    return instances


def _build_api_call_regex(instance_names) -> re.Pattern[str]:
    """A `<declared instance>.<method>(` call, capturing *both* the object
    name and the method name -- for exactly the instances `_declared_instances`
    found, never a guessed prefix. Both are captured, not just the method
    name, because the method name alone is not unique across classes (see
    `_declared_instances`); resolving which object it was called on is what
    lets a caller resolve which class's version of that method this is.
    """
    if not instance_names:
        return re.compile(r"[^\s\S]")  # an impossible character class: never matches
    alternation = "|".join(re.escape(name) for name in sorted(instance_names))
    return re.compile(rf"\b({alternation})\.([a-zA-Z_]\w*)\s*\(")


def resolved_api_calls(
    script_text: str, call_re: re.Pattern[str], instance_to_class: dict[str, str]
) -> set[tuple[str, str]]:
    """`{(class name, method name)}` for every `<api object>.<method>(...)`
    call in a script, matched against `call_re` (see `_build_api_call_regex`)
    and resolved through `instance_to_class` (see `_declared_instances`).

    Shared by `_discover_core_api_methods` (over the reference scripts) and
    by a generated script's post-generation validation (over the model's
    output) -- the same shape of call, read the same way, for two different
    purposes.
    """
    calls: set[tuple[str, str]] = set()
    for instance, method in call_re.findall(script_text or ""):
        class_name = instance_to_class.get(instance)
        if class_name:
            calls.add((class_name, method))
    return calls


def api_call_names(
    script_text: str, call_re: re.Pattern[str], instance_to_class: dict[str, str]
) -> set[str]:
    """Bare method names called in a script -- for checking a name exists
    *anywhere* in the API surface (see `StaticContextProvider.known_api_methods`),
    where which exact class it belongs to does not matter.
    """
    return {method for _, method in resolved_api_calls(script_text, call_re, instance_to_class)}

# A method used in most approved reference scripts is structural (setup,
# cleanup, verification plumbing every script needs), not situational to one
# requirement -- so it should never have to win a keyword-popularity contest
# against unrelated methods to get documented for the model. "Most" rather
# than "all": data/Examples has only 3 reference scripts today, too few for
# a stricter threshold to be meaningful, and the methods just under 100% here
# (e.g. set_track_to_use) are ones no wording-based rule could gate anyway --
# which physical track a script uses is an environment-setup detail, never
# part of a requirement or test case's behaviour (see _names_parameter_identifier
# for the contrasting case where the wording genuinely does carry the signal).
_CORE_METHOD_MIN_COVERAGE = 0.5

# A class name ending in a digit -- every message/component-specific class
# in data/python_apis is named this way (`PublicOffice1051`, `PublicEMC600`,
# `PublicIvocOffice1821`, ...), one class per specific message or component
# number, as opposed to a shared/general class (`PublicOfficeBase`,
# `PublicTrackFile`, `RunTest`) that has no such suffix. A numbered class's
# methods are tied to whatever specific message content that one number
# means, not to "every script" the way _CORE_METHOD_MIN_COVERAGE otherwise
# assumes -- confirmed the hard way: `PublicOffice1051`'s 9-call sequence
# crossed the coverage threshold from appearing in 2 of 3 reference scripts,
# got guaranteed into every prompt regardless of whether a given requirement
# needs an "authority" message at all, and a real generation run got stuck
# in an unbounded repetition loop reproducing that one sequence. Frequency
# alone cannot tell a genuinely universal call apart from a
# message-specific one that just happens to appear in most of a 3-script
# sample; this excludes the latter category structurally instead.
_NUMBERED_CLASS_RE = re.compile(r"\d$")


def _discover_core_api_methods(
    examples_dir: Path | None,
    call_re: re.Pattern[str],
    instance_to_class: dict[str, str],
) -> tuple[tuple[str, str], ...]:
    """Which `(class, method)` pairs are structural rather than situational,
    read off the real reference scripts in
    `<examples_dir>/reference_test_scripts` instead of a hand-maintained
    list -- so this adapts automatically as reference scripts are added or
    changed, rather than needing someone to remember to update a constant.

    Identified by `(class, method)`, not by bare method name: a method name
    alone is not unique (`set_bos` alone is defined on 50 different
    office-message classes), so counting bare names would risk promoting a
    name to "guaranteed" while pointing at an arbitrary one of many
    same-named methods instead of the specific class a script actually uses.

    `call_re` and `instance_to_class` must be built from the real declared
    API objects (see `_build_api_call_regex`/`_declared_instances`), not a
    guessed prefix -- built the wrong way, this silently undercounts every
    method called through an object outside that guess (`run_test.run()`,
    `wabtest...`), even though those calls are exactly the kind of
    universal, structural call this function exists to find.
    """
    if not examples_dir:
        logger.warning(
            "Static context (python_apis): no examples_dir configured -- no "
            "core API methods discovered; API docs will rely on keyword "
            "search alone"
        )
        return ()

    scripts_dir = Path(examples_dir) / "reference_test_scripts"
    scripts = sorted(scripts_dir.glob("*.txt")) if scripts_dir.is_dir() else []
    if not scripts:
        logger.warning(
            "Static context (python_apis): no reference scripts found under "
            "%s -- no core API methods discovered; API docs will rely on "
            "keyword search alone",
            scripts_dir,
        )
        return ()

    counts: dict[tuple[str, str], int] = {}
    for script in scripts:
        text = script.read_text(encoding="utf-8", errors="replace")
        for class_name, method in resolved_api_calls(text, call_re, instance_to_class):
            if _NUMBERED_CLASS_RE.search(class_name):
                continue
            counts[(class_name, method)] = counts.get((class_name, method), 0) + 1

    threshold = len(scripts) * _CORE_METHOD_MIN_COVERAGE
    core = tuple(sorted(pair for pair, count in counts.items() if count > threshold))
    logger.info(
        "Static context (python_apis): %d core (class, method) pair(s) "
        "discovered from %d reference script(s): %s",
        len(core),
        len(scripts),
        ", ".join(f"{c}.{m}" for c, m in core) or "(none)",
    )
    return core


def _names_parameter_identifier(query: str) -> bool:
    """Reuses `_parameter_ids` -- a requirement naming TBC137 has already
    stated it will set and restore that parameter, so `set_default_tbc`'s
    docs are as certain to be needed as `set_tbc`'s.
    """
    return bool(_parameter_ids(query))


# Methods that are common but not universal, so they are guaranteed a slot
# only when the query gives a concrete, checkable reason to expect them --
# a rule instead of a keyword-popularity contest against unrelated methods.
# Each predicate takes the script-generation query and decides whether that
# method's real docs should be force-included alongside the core list.
#
# Unlike a physical track (an environment-setup detail, never part of a
# requirement's wording -- see _discover_core_api_methods), a TBC/CFG value genuinely
# is part of the behaviour a requirement describes, so it is normal for a
# requirement naming TBC137 to also need set_default_tbc to restore it.
#
# Identified by (class, method): `set_default_tbc` happens to be unique to
# `PublicDebugClient` (verified against data/python_apis), but every lookup
# in this module is keyed this way regardless, so a future method that is
# NOT unique can never be silently resolved to the wrong class the way a
# bare-name lookup would risk (see `_index_api_records`).
_TRIGGERED_API_METHODS: tuple[tuple[Callable[[str], bool], tuple[str, str]], ...] = (
    (_names_parameter_identifier, ("PublicDebugClient", "set_default_tbc")),
)


def _format_hits(hits: list[KeywordHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source_file") or hit.metadata.get("source_path") or "unknown"
        locator = hit.metadata.get("method_name") or hit.metadata.get("section_path") or ""
        label = f"{source} > {locator}" if locator else source
        blocks.append(f"[{index}] {hit.metadata.get('document_type', 'static')} | {label}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(blocks)


class StaticContextProvider:
    """python_apis, track_data and parameter_config, keyword-searched from an
    in-process BM25 index built straight off disk. This is the whole of the
    app's retrieval — there is no dense arm and no vector store.

    parameter_config is the parameter configuration guide after
    `scripts/convert_parameter_guide.py` has turned its tables into one JSON
    record per parameter — near-identical rows keyed by exact identifiers,
    which is the same shape as track data and belongs here for the same
    reasons.

    track_data is one folder per subdivision, each holding that subdivision's
    `-subdiv.xml` export. Chunks are stamped with the folder's subdivision,
    which is what the UI's picker filters on.

    `data/Examples/` is deliberately *not* loaded here. Sending those files
    at request time cost ~8-10k characters of prompt on every generation —
    all of it prefill, all of it identical from one request to the next —
    to teach the model a shape that does not change. The shape has instead
    been distilled by hand into the house wording pattern and house
    skeleton in config/prompts.yaml. The folder stays in the repo as the
    source those templates were derived from and the material to re-derive
    them from when the house style changes; it is design-time input now,
    not runtime input.

    Built once, at construction — same lifetime as the running backend
    process. Restart the server to pick up edits to any source folder.
    """

    def __init__(
        self,
        ingestion_config: PipelineConfig,
    ):
        self._config = ingestion_config
        store = _build_static_chunks(ingestion_config)
        self._chunk_count = store.count()
        self._keyword_index = KeywordIndex(store)
        self._subdivisions = _list_subdivisions(store)
        self._railroad_scac = _railroad_scac_by_subdivision(store)
        self._track_groups = _track_groups_by_subdivision(store)
        self._block_ranges = _block_ranges_by_subdivision(store)
        self._parameter_records = _index_parameter_records(store)
        self._api_records = _index_api_records(store)
        self._instance_to_class = _declared_instances(_python_apis_source_paths(ingestion_config))
        self._api_call_re = _build_api_call_regex(self._instance_to_class.keys())
        self._core_api_methods = _discover_core_api_methods(
            ingestion_config.examples_dir, self._api_call_re, self._instance_to_class
        )

    def api_call_names(self, script_text: str) -> set[str]:
        """Every real API method called in `script_text`, by bare name --
        matched against every object this static index's own
        `data/python_apis` source actually declares (see
        `_declared_instances`) rather than a guessed naming convention.
        For checking a name exists at all (`known_api_methods`); see
        `_resolved_api_calls` where which exact class matters.
        """
        return api_call_names(script_text, self._api_call_re, self._instance_to_class)

    def _resolved_api_calls(self, script_text: str) -> set[tuple[str, str]]:
        """Every real API call in `script_text`, as `(class, method)` pairs."""
        return resolved_api_calls(script_text, self._api_call_re, self._instance_to_class)

    def api_context(self, query: str, top_k: int) -> str:
        """WCR test-automation API documentation for the script call, formatted
        for a prompt. See `api_hits` for the records themselves.
        """
        return _format_hits(self.api_hits(query, top_k))

    def api_hits(self, query: str, top_k: int) -> list[KeywordHit]:
        """WCR test-automation API documentation for the script call.

        `self._core_api_methods` (discovered from real reference scripts by
        `_discover_core_api_methods`, not a hand-maintained list) are
        fetched directly and always included: a keyword search over the
        full API surface has been observed to rank them far outside a small
        top_k on a real requirement (their docstrings share little
        vocabulary with typical requirement wording, while unrelated
        methods sharing surface words like "train"/"speed" outrank them),
        and since most approved scripts need a real signature for them
        regardless of what the requirement is about, there is nothing for a
        search to usefully decide here.

        `_TRIGGERED_API_METHODS` are fetched directly when their predicate
        matches the query -- common but not universal, so they are
        guaranteed only when the query gives a concrete, checkable reason to
        expect them, rather than left to the same keyword-popularity contest
        that misses the core methods.

        Whatever slots remain go to the ordinary keyword search, for
        genuinely variable calls (driving commands, wayside message
        helpers) that cannot be enumerated in advance.
        """
        if top_k <= 0:
            return []

        included = self._guaranteed_api_hits(query)
        guaranteed = len(included)
        remaining = top_k - guaranteed
        if remaining > 0:
            hits = self._keyword_index.search(
                query,
                remaining,
                predicate=lambda m: (
                    m.get("document_type") == "python_apis"
                    and (m.get("class_name"), m.get("method_name")) not in included
                ),
            )
            for hit in hits:
                key = (hit.metadata.get("class_name"), hit.metadata.get("method_name"))
                if key[0] and key[1]:
                    included.setdefault(key, hit)

        logger.info(
            "Static context (python_apis): %d guaranteed (core+triggered) + "
            "%d searched = %d hit(s) (top_k=%d)",
            guaranteed,
            len(included) - guaranteed,
            len(included),
            top_k,
        )
        return list(included.values())

    def _guaranteed_api_hits(self, query: str) -> dict[tuple[str, str], KeywordHit]:
        """`self._core_api_methods`, plus whichever `_TRIGGERED_API_METHODS`
        predicates match `query` -- the `(class, method)` pairs `api_hits`
        fetches directly rather than via keyword search. A missing pair logs
        a warning instead of raising, since a rename in `data/python_apis`
        should degrade to "one less guaranteed hit" for a reviewer to notice
        in the generated script, not fail the whole request.
        """
        included: dict[tuple[str, str], KeywordHit] = {}
        for pair in self._core_api_methods:
            self._add_guaranteed_hit(included, pair, "core")
        for predicate, pair in _TRIGGERED_API_METHODS:
            if predicate(query):
                self._add_guaranteed_hit(included, pair, "triggered")
        return included

    def _add_guaranteed_hit(
        self,
        included: dict[tuple[str, str], KeywordHit],
        pair: tuple[str, str],
        reason: str,
    ) -> None:
        if pair in included:
            return
        hit = self._api_records.get(pair)
        if hit is None:
            logger.warning(
                "Static context (python_apis): %s method %s.%s not found in "
                "the indexed API surface -- check data/python_apis for a "
                "rename",
                reason,
                *pair,
            )
            return
        included[pair] = hit

    def parameter_context(self, query: str, top_k: int) -> str:
        """TBC/CFG/THE records from the parameter configuration guide, formatted
        for a prompt. See `parameter_hits` for the records themselves.
        """
        return _format_hits(self.parameter_hits(query, top_k))

    def parameter_hits(self, query: str, top_k: int) -> list[KeywordHit]:
        """TBC/CFG/THE records from the parameter configuration guide.

        Exact lookup first, keyword search only as a fallback. A requirement
        that names its parameters (TBC137, CFG16) has stated its own scope,
        so those records are the whole answer and `top_k` does not apply to
        them. Only a requirement that names none falls back to BM25, which
        still ranks an exact identifier well.

        Returned as raw hits, not just the formatted block, so a caller can
        report them as the retrieved context — these are the only chunks
        test-case generation has to show.
        """
        if top_k <= 0:
            return []
        named = [
            self._parameter_records[identifier]
            for identifier in _parameter_ids(query)
            if identifier in self._parameter_records
        ]
        if named:
            # A requirement that names its parameters has already said which
            # ones it is about, so nothing is padded in beside them. BM25
            # ranks the rest on shared prose, and the guide is 900 records of
            # the same prose: on a work-zone requirement naming TBC290 and
            # CFG22, TBC412 ("...calculated position uncertainty of the
            # leading edge of the train...") outscored TBC290's own record
            # and reached the prompt, where every record reads as in-scope.
            # A parameter the requirement never mentions is not a ranking
            # question to be answered less confidently — it is the wrong
            # answer, and one the reviewer cannot spot in a finished case.
            logger.info(
                "Static context (parameter_config): %d record(s) named in the requirement (%s); "
                "no keyword padding",
                len(named),
                ", ".join(_parameter_ids(query)),
            )
            return named

        hits = self._keyword_index.search(
            query, top_k, predicate=lambda m: m.get("document_type") == "parameter_config"
        )
        logger.info(
            "Static context (parameter_config): %d/%d hit(s) by keyword "
            "(the requirement names no TBC/CFG/THE identifier)",
            len(hits),
            top_k,
        )
        return hits

    def track_context(
        self,
        query: str,
        top_k: int,
        subdivision: str | None,
    ) -> str:
        return self.track_context_for(query, top_k, subdivision)[0]

    def track_context_for(
        self,
        query: str,
        top_k: int,
        subdivision: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        """Track context for the subdivision picked in the UI, plus which
        subdivisions it was actually drawn from.

        The picked subdivision is the only thing that decides this. There is
        no requirement -> subdivision map any more: a stored map and a UI
        picker are two answers to one question, and the one the user just
        chose in front of the result has to win, so keeping both only created
        a way for the control to appear to do nothing.

        No subdivision means no track data, deliberately. Searching every
        subdivision instead would let BM25 return blocks and mileposts
        belonging to track this requirement is not tested on — a wrong value
        that looks right, which is worse than the `# TODO:` the generator
        writes when the track block is empty.

        The caller gets the subdivisions back rather than re-deriving them,
        because what was searched is reported in the UI and must be the same
        set that produced these hits.
        """
        if top_k <= 0 or not subdivision:
            logger.info("Static context (track_data): skipped, no subdivision picked")
            return "", ()

        chosen = normalize_subdivision(subdivision)
        if not chosen:
            logger.info("Static context (track_data): subdivision %r did not normalize", subdivision)
            return "", ()

        def predicate(metadata: dict[str, Any]) -> bool:
            return (
                metadata.get("document_type") == "track_data"
                and metadata.get("subdivision") == chosen
            )

        hits = self._keyword_index.search(query, top_k, predicate=predicate)
        logger.info(
            "Static context (track_data): %d/%d hit(s) in subdivision %s",
            len(hits),
            top_k,
            chosen,
        )
        return self._with_railroad_scac(_format_hits(hits), chosen), (chosen,)

    def _with_railroad_scac(self, context: str, subdivision: str) -> str:
        """Prepends this subdivision's real `RailroadSCAC`, guaranteed, not
        left to the keyword search above to happen to retrieve it.

        Which railroad a subdivision belongs to is an environment fact, like
        which physical track a block is on -- never part of a requirement or
        test case's wording, so no search query could ever be built to find
        it (see the same reasoning for `set_track_to_use` in
        `_discover_core_api_methods`). A script that guesses `scac` from an
        API docstring's own example list (`'COMMON'`, first in the list, was
        one seen this way) instead of this subdivision's real value sends
        cleanup and setup messages the onboard silently ignores, since it
        only accepts them from the railroad it is actually configured for.
        """
        scac = self._railroad_scac.get(subdivision)
        if not scac:
            return context
        fact = (
            f"Railroad SCAC for subdivision {subdivision} (from the real track "
            f"database -- use this for every 'scac' argument in this script, "
            f"not an example value from an API docstring): {scac}"
        )
        return f"{fact}\n\n{context}" if context else fact

    @property
    def chunk_count(self) -> int:
        """How much the index holds, for the health endpoint."""
        return self._chunk_count

    @property
    def subdivisions(self) -> list[Subdivision]:
        """Every subdivision the static index actually holds track data for —
        read off the indexed chunks rather than by listing the folder, so the
        dropdown can only ever offer track the search can really return.
        """
        return self._subdivisions

    def known_track_groups(self, subdivision: str | None) -> frozenset[str]:
        """Real track names (`track_group`/`sub_folder` values) for
        `subdivision`, from its own TrackNameFeature legend.

        For validating a generated script's `track_group` argument against,
        not for retrieval -- see `_track_groups_by_subdivision`. Empty
        (rather than raising) for a subdivision with no indexed
        TrackNameFeature record, so a caller degrades to "nothing to check
        against" instead of failing the whole request.
        """
        chosen = normalize_subdivision(subdivision) if subdivision else None
        if not chosen:
            return frozenset()
        return self._track_groups.get(chosen, frozenset())

    def find_block(
        self, subdivision: str | None, track_group: str, point: float
    ) -> tuple[str, float, float] | None:
        """The real block on `track_group` in `subdivision` covering
        `point`, as `(block_id, start_milepost, end_milepost)`, or None.

        None covers every case where the real block can't be resolved --
        unknown subdivision, a `track_group` this subdivision's own
        TrackNameFeature legend doesn't name, or a point outside every one
        of that track's blocks. Never a nearest-match guess: a caller gets
        a real answer or nothing to build a comment from.

        `track_group` is matched the same way `set_position` itself
        matches it (case-insensitive; see `data/python_apis`), so a script
        that already wrote a validly-cased `Main1` or `main1` resolves to
        the same block either way.
        """
        chosen = normalize_subdivision(subdivision) if subdivision else None
        if not chosen:
            return None
        wanted = (track_group or "").strip().lower()
        if not wanted:
            return None
        for track_name, start, end, block_id in self._block_ranges.get(chosen, []):
            if track_name.strip().lower() != wanted:
                continue
            if start <= point <= end:
                return block_id, start, end
        return None

    def known_api_methods(self) -> frozenset[str]:
        """Every method name the real `data/python_apis` surface defines.

        For validating a generated script's calls against, not for
        retrieval: a call to a name outside this set does not exist in the
        API at all (e.g. an invented `send_acknowledge_key`), which a
        keyword search or a guaranteed-include list can only reduce the odds
        of, never rule out.
        """
        return frozenset(method for _class, method in self._api_records)
