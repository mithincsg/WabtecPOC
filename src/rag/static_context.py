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
from .requirement_parser import requirement_chapter

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


def _track_legends_by_subdivision(store: _StaticDocumentStore) -> dict[str, list[dict[str, str]]]:
    """`{subdivision: [TrackNameFeature record fields]}`, in the legend's own
    order -- sent with every track context, since which track names exist is
    never something a requirement's wording could make a keyword search find.
    """
    _ids, texts, metadatas = store.get_all_documents()
    legends: dict[str, list[dict[str, str]]] = {}
    for text, metadata in zip(texts, metadatas):
        if (metadata or {}).get("document_type") != "track_data":
            continue
        if (metadata or {}).get("table_title") != "TrackNameFeature":
            continue
        subdivision = str((metadata or {}).get("subdivision") or "").strip()
        if not subdivision:
            continue
        for line in text.splitlines():
            fields = dict(part.split("=", 1) for part in line.split(" | ") if "=" in part)
            if fields.get("TrackName"):
                legends.setdefault(subdivision, []).append(fields)
    return legends


# The prefix `xml_extractor` puts on a nested record's line to say which
# block it belongs to (`BlockFeature 2015 > SignalId=...`).
_PARENT_BLOCK_RE = re.compile(r"\bBlockFeature (\d+) >")

# Track tables every track context already carries in summary form (the
# legend, the block table, the guaranteed SCAC line), so keyword search never
# spends a slot on them.
_SUMMARISED_TRACK_TABLES = frozenset({"TrackNameFeature", "BlockFeature", "Subdivision"})


def _miles(raw: float) -> str:
    return f"{raw / 10000:.4f}".rstrip("0").rstrip(".")


def _block_line(block_id: str, start: float, end: float) -> str:
    """`block: start-end [start-end in miles]` -- both forms, because the
    XML's mileposts are 1/10000 mile, set_position takes miles and 01041
    set_segment takes the raw value, and a model left to convert one into
    the other gets it wrong.
    """
    return f"{block_id}: {start:.0f}-{end:.0f} [{_miles(start)}-{_miles(end)}]"


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


def _index_icd_fields(store: _StaticDocumentStore) -> list[tuple[re.Pattern[str], list[KeywordHit]]]:
    """ICD field chunks grouped by field name, each with a pattern matching
    that name as a requirement or test case writes it: any case, words
    joined by space, hyphen, underscore or nothing (`train type`,
    `TrainType`, `head-end only`). Longest names first, so `Head End Only
    Speed Restriction` is listed ahead of anything it contains.
    """
    ids, texts, metadatas = store.get_all_documents()
    by_name: dict[str, list[KeywordHit]] = {}
    for chunk_id, text, metadata in zip(ids, texts, metadatas):
        if (metadata or {}).get("document_type") != "icd":
            continue
        name = str(metadata.get("section_path") or "").split(" > ")[-1]
        words = re.findall(r"[A-Za-z0-9]+", name)
        if len(words) < 2:
            continue
        by_name.setdefault(" ".join(words).lower(), []).append(
            KeywordHit(chunk_id=chunk_id, text=text, metadata=metadata, score=0.0)
        )
    return [
        (re.compile(r"\b" + r"[\s_-]*".join(map(re.escape, name.split())) + r"\b", re.IGNORECASE), hits)
        for name, hits in sorted(by_name.items(), key=lambda item: -len(item[0]))
    ]


# The first line every icd_field chunk is rendered with (see
# `icd_extractor.icd_units`): the field name, then every message it decodes
# identically in, as `"<msg_id> <message name>"` entries joined by `"; "`.
_ICD_HEADER_RE = re.compile(r"^ICD field (.+?) \(message (.+)\)$")


def _icd_msg_ids(field_chunk: str) -> set[str]:
    first_line = field_chunk.strip().splitlines()[0] if field_chunk.strip() else ""
    header = _ICD_HEADER_RE.match(first_line)
    if not header:
        return set()
    return {entry.partition(" ")[0] for entry in header.group(2).split("; ")}


def _message_instances_by_id(instance_to_class: dict[str, str]) -> dict[str, str]:
    """`{msg id: declared instance}` for every `wcr_office_<id>` /
    `wcr_ivoc_office_<id>` object `data/python_apis` declares -- the office
    messages a script can actually send, as opposed to every message the ICD
    happens to define (`icd_messages.json` also carries locomotive-side and
    file-transfer messages no office object sends).
    """
    return {
        instance.rsplit("_", 1)[-1]: instance
        for instance in instance_to_class
        if instance.startswith(("wcr_office_", "wcr_ivoc_office_"))
    }


def _index_message_terms(
    store: _StaticDocumentStore, sendable_msg_ids: set[str]
) -> tuple[dict[str, list[tuple[re.Pattern[str], str]]], dict[str, str]]:
    """`{msg id: [(pattern, matched term), ...]}` for every office message a
    script can send -- built off the same icd_field chunks `_index_icd_fields`
    groups by field name, so no source is read twice.

    A term is the message's own name, one of its multi-word field names, or
    one of its enumeration meanings (e.g. "Work zone", the Bulletin Type
    value that names the 01041 message a work-zone requirement needs) --
    exactly the vocabulary a requirement uses to talk about a message
    without ever writing its ICD number. Single-word names are left out for
    the same reason `_index_icd_fields` leaves them out: too common to be a
    reliable signal. A term shared by more than a handful of messages (a
    generic enumeration value like "Not used" or "Reserved") is dropped too
    -- it would select every message that has it, which is no selection at
    all.

    Alongside the term index, also returns `{term: "<field>=<code> (<term>)"}`
    for every term that came from an enumeration value -- e.g. "work zone"
    -> "Bulletin Type=6 (Work zone)". A term alone tells the model which
    object matched, not which value to pass its setter: a real generation
    matched "work zone" to 01041, correctly called `set_type`, and then
    still invented a wrong method name (`set_bulletin_type`, which belongs
    to a different class) and a wrong-cased value (`"WORK_ZONE"`) for it --
    the exact ICD field/code/meaning that matched was known at retrieval
    time but never reached the object card. `message_object_hits` looks
    this hint up so the card can state the real value instead.
    """
    ids, texts, metadatas = store.get_all_documents()
    term_msg_ids: dict[str, set[str]] = {}
    term_hint: dict[str, str] = {}
    msg_names: dict[str, str] = {}
    for text, metadata in zip(texts, metadatas):
        if (metadata or {}).get("document_type") != "icd":
            continue
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        header = _ICD_HEADER_RE.match(first_line)
        if not header:
            continue
        field_name, message_list = header.groups()
        msg_ids_here = []
        for entry in message_list.split("; "):
            msg_id, _, name = entry.partition(" ")
            if msg_id not in sendable_msg_ids:
                continue
            msg_ids_here.append(msg_id)
            msg_names.setdefault(msg_id, name)
        if not msg_ids_here:
            continue

        terms = {field_name} if len(field_name.split()) >= 2 else set()
        for line in text.splitlines()[1:]:
            enum_match = re.match(rf"^{re.escape(field_name)}=(\S+) \((.+)\)$", line)
            if enum_match:
                code, meaning = enum_match.groups()
                if len(meaning.split()) >= 2:
                    terms.add(meaning)
                    term_hint.setdefault(meaning.lower(), f"{field_name}={code} ({meaning})")
        for term in terms:
            term_msg_ids.setdefault(term.lower(), set()).update(msg_ids_here)

    for msg_id, name in msg_names.items():
        if len(name.split()) >= 2:
            term_msg_ids.setdefault(name.lower(), set()).add(msg_id)

    by_msg: dict[str, list[tuple[re.Pattern[str], str]]] = {}
    for term_lower, msg_ids in term_msg_ids.items():
        if len(msg_ids) > 3:
            continue
        pattern = re.compile(
            r"\b" + r"[\s_-]*".join(map(re.escape, term_lower.split())) + r"\b", re.IGNORECASE
        )
        for msg_id in msg_ids:
            by_msg.setdefault(msg_id, []).append((pattern, term_lower))
    return by_msg, term_hint


def _method_signature(method_source: str, instance: str) -> tuple[str, str, list[str]]:
    """`(<instance>.<method>(...) signature, docstring's first line, dict
    keys the docstring documents)` off one method chunk's real source --
    for rendering a compact, object-qualified line instead of sending the
    whole method body.
    """
    lines = method_source.splitlines()
    def_line = next((line.strip() for line in lines if line.strip().startswith("def ")), "")
    def_line = def_line.rstrip(":")
    signature = re.sub(rf"^def\s+(\w+)\(self,?\s*", rf"{instance}.\1(", def_line, count=1)

    doc_lines: list[str] = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith(('"""', "'''")):
                in_docstring = True
                stripped = stripped.strip("\"'")
                if stripped:
                    doc_lines.append(stripped)
            continue
        if stripped.endswith(('"""', "'''")):
            break
        if stripped:
            doc_lines.append(stripped)
    summary = doc_lines[0] if doc_lines else ""
    keys = sorted(set(re.findall(r"'(\w+)'\s*:", method_source)))
    return signature, summary, keys


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
# than "all": data/Examples has only a handful of reference scripts, too few
# for a stricter threshold to be meaningful, and the methods just under 100% here
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


@dataclasses.dataclass(frozen=True)
class ExampleFamily:
    """What the reference script(s) for one requirement chapter call beyond
    the core methods: the calls a requirement from the same chapter is
    likely to need too."""

    reference_ids: tuple[str, ...]
    api_pairs: tuple[tuple[str, str], ...]
    message_ids: tuple[str, ...]


def _discover_example_families(
    examples_dir: Path | None,
    call_re: re.Pattern[str],
    instance_to_class: dict[str, str],
    core: tuple[tuple[str, str], ...],
    message_instance: dict[str, str],
) -> dict[str, ExampleFamily]:
    """`{chapter: ExampleFamily}`, read off the reference scripts.

    A method only one reference script uses never becomes core, and its
    docstring rarely shares words with a requirement: on switch
    requirements `wcr_user_act.user_action` ranked 486th and
    `wcr_start_up.automation_setup` 1061st, far outside the keyword slots,
    so a switch script never saw the calls L2R7983 is built from. The
    chapter (`13` for `13.1.1 Facing Approach`) is what ties a new
    requirement to the reference script written for the same feature. It is
    read from that script's requirement text heading, so a reference script
    whose text has no numbered heading joins no family.

    Only (class, method) names and office message ids are taken -- API
    vocabulary, like the core methods -- never a value.
    """
    if not examples_dir:
        return {}
    scripts_dir = Path(examples_dir) / "reference_test_scripts"
    class_to_msg_id = {
        instance_to_class[instance]: msg_id
        for msg_id, instance in message_instance.items()
        if instance in instance_to_class
    }
    core_pairs = set(core)

    grouped: dict[str, dict[str, set]] = {}
    for script in sorted(scripts_dir.glob("*.txt")) if scripts_dir.is_dir() else []:
        requirement_id = script.stem.split("_", 1)[0]
        texts = sorted(Path(examples_dir).glob(f"{requirement_id}_*Text*.txt"))
        chapter = (
            requirement_chapter(texts[0].read_text(encoding="utf-8", errors="replace"))
            if texts
            else None
        )
        if not chapter:
            continue
        family = grouped.setdefault(chapter, {"ids": set(), "pairs": set(), "msgs": set()})
        family["ids"].add(requirement_id)
        for class_name, method in resolved_api_calls(
            script.read_text(encoding="utf-8", errors="replace"), call_re, instance_to_class
        ):
            if class_name in class_to_msg_id:
                family["msgs"].add(class_to_msg_id[class_name])
            elif not _NUMBERED_CLASS_RE.search(class_name) and (class_name, method) not in core_pairs:
                family["pairs"].add((class_name, method))

    families = {
        chapter: ExampleFamily(
            reference_ids=tuple(sorted(found["ids"])),
            api_pairs=tuple(sorted(found["pairs"])),
            message_ids=tuple(sorted(found["msgs"])),
        )
        for chapter, found in grouped.items()
    }
    for chapter, family in sorted(families.items()):
        logger.info(
            "Static context (python_apis): chapter %s family from %s: %d method(s), office message(s) %s",
            chapter,
            ", ".join(family.reference_ids),
            len(family.api_pairs),
            ", ".join(family.message_ids) or "none",
        )
    return families


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


# Docstring lines that tell the model nothing a call needs: the types are
# already in the signature, most methods return None, and every stub body is
# a bare `...`.
_API_DOC_NOISE_RE = re.compile(r'^(?::(?:type|rtype)\b.*|:return:\s*None|\.\.\.|""")$')


def _compact_api_source(text: str) -> str:
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _API_DOC_NOISE_RE.match(line):
            continue
        kept.append(line if line.startswith(("def ", "@")) else "  " + line)
    return "\n".join(kept).replace("<br>", "").replace("&lt;", "<").replace("&gt;", ">")


def _format_api_hits(hits: list[KeywordHit], class_to_instance: dict[str, str]) -> str:
    """API method chunks for the prompt, each labelled with the object it is
    called through (`wcr_display.press_key`) -- a bare method name is not
    unique across classes -- and compacted with `_compact_api_source`.
    Formatting only: the indexed text BM25 ranks is untouched.
    """
    blocks = []
    for index, hit in enumerate(hits, start=1):
        class_name = hit.metadata.get("class_name")
        method = hit.metadata.get("method_name")
        owner = class_to_instance.get(class_name) or class_name
        label = f"{owner}.{method}" if owner and method else (method or class_name or "unknown")
        blocks.append(f"[{index}] {label}\n{_compact_api_source(hit.text)}")
    return "\n\n".join(blocks)


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
        track_block_window: int = 8,
    ):
        self._config = ingestion_config
        self._track_block_window = track_block_window
        store = _build_static_chunks(ingestion_config)
        self._chunk_count = store.count()
        self._keyword_index = KeywordIndex(store)
        self._subdivisions = _list_subdivisions(store)
        self._railroad_scac = _railroad_scac_by_subdivision(store)
        self._track_groups = _track_groups_by_subdivision(store)
        self._track_legends = _track_legends_by_subdivision(store)
        self._block_ranges = _block_ranges_by_subdivision(store)
        self._parameter_records = _index_parameter_records(store)
        self._icd_fields = _index_icd_fields(store)
        self._api_records = _index_api_records(store)
        self._instance_to_class = _declared_instances(_python_apis_source_paths(ingestion_config))
        self._class_to_instance: dict[str, str] = {}
        for instance, class_name in self._instance_to_class.items():
            self._class_to_instance.setdefault(class_name, instance)
        self._api_call_re = _build_api_call_regex(self._instance_to_class.keys())
        self._core_api_methods = _discover_core_api_methods(
            ingestion_config.examples_dir, self._api_call_re, self._instance_to_class
        )
        self._message_instance = _message_instances_by_id(self._instance_to_class)
        self._message_terms, self._message_term_hints = _index_message_terms(
            store, set(self._message_instance)
        )
        self._example_families = _discover_example_families(
            ingestion_config.examples_dir,
            self._api_call_re,
            self._instance_to_class,
            self._core_api_methods,
            self._message_instance,
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

    def api_context(
        self,
        query: str,
        top_k: int,
        exclude_classes: frozenset[str] = frozenset(),
        chapter: str | None = None,
    ) -> str:
        """WCR test-automation API documentation for the script call, formatted
        for a prompt. See `api_hits` for the records themselves.
        """
        return _format_api_hits(
            self.api_hits(query, top_k, exclude_classes, chapter), self._class_to_instance
        )

    def api_hits(
        self,
        query: str,
        top_k: int,
        exclude_classes: frozenset[str] = frozenset(),
        chapter: str | None = None,
    ) -> list[KeywordHit]:
        """WCR test-automation API documentation for the script call.

        `exclude_classes` leaves out a class already covered by a
        `message_object_hits` card -- that card already lists every one of
        its methods, so a keyword-search slot spent repeating one of them
        here would be a wasted slot, not a second copy worth having.

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

        With `chapter`, the methods the same chapter's reference script
        calls beyond the core ones are fetched directly too (see
        `_discover_example_families`).

        Whatever slots remain go to the ordinary keyword search, for
        genuinely variable calls (driving commands, wayside message
        helpers) that cannot be enumerated in advance.
        """
        if top_k <= 0:
            return []

        included = self._guaranteed_api_hits(query, chapter)
        guaranteed = len(included)
        remaining = top_k - guaranteed
        if remaining > 0:
            hits = self._keyword_index.search(
                query,
                remaining,
                predicate=lambda m: (
                    m.get("document_type") == "python_apis"
                    and (m.get("class_name"), m.get("method_name")) not in included
                    and m.get("class_name") not in exclude_classes
                ),
            )
            for hit in hits:
                key = (hit.metadata.get("class_name"), hit.metadata.get("method_name"))
                if key[0] and key[1]:
                    included.setdefault(key, hit)

        logger.info(
            "Static context (python_apis): %d guaranteed (core+triggered+family) + "
            "%d searched = %d hit(s) (top_k=%d)",
            guaranteed,
            len(included) - guaranteed,
            len(included),
            top_k,
        )
        return list(included.values())

    def message_object_hits(
        self, query: str, top_k: int, chapter: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        """`[(declared instance, class, matched terms)]` for the office
        message(s) this requirement's own wording identifies -- see
        `_index_message_terms` for what counts as a term -- ranked by how
        many distinct terms matched, most first.

        Exact term matching, not keyword ranking: aggregating BM25 scores
        per class was tried and is unreliable here (on one reference
        requirement it ranked an unrelated office class first), the same
        way plain keyword search cannot be trusted to surface a named
        TBC/CFG/THE identifier (see `parameter_hits`).

        A single matched term is not kept, even though `_index_message_terms`
        already requires it to be multi-word and rare: verified against
        `data/Examples`, a signal-target requirement having nothing to do
        with any office message matched 01041 on "restricted speed" alone --
        that phrase names a signal-target category there, not the Bulletin
        Dataset field it names in ICD terms. Every genuine match found in
        that same verification cleared two or more independent terms, so
        requiring corroboration from a second term is a real bar, not an
        arbitrary one.

        With `chapter`, slots the wording left open go to the office messages
        the same chapter's reference script sends: a switch requirement
        worded without any ICD term still needs the authority and bulletin
        its reference script sets up (see `_discover_example_families`).
        """
        if top_k <= 0:
            return []
        scored: list[tuple[int, str, list[str]]] = []
        for msg_id, terms in self._message_terms.items():
            matched_terms = {term for pattern, term in terms if pattern.search(query or "")}
            if len(matched_terms) < 2:
                continue
            # An enumeration term is rendered as its real "<field>=<code>
            # (<meaning>)" line where one is known, instead of the bare
            # meaning -- see _index_message_terms for why: the meaning alone
            # told a real generation which object to call, not which exact
            # value to pass its setter.
            matched = sorted({self._message_term_hints.get(term, term) for term in matched_terms})
            scored.append((len(matched), msg_id, matched))
        scored.sort(key=lambda item: (-item[0], item[1]))

        # A message whose matched terms are all already matched by a
        # higher-ranked one adds no evidence of its own: 01043 (Bulletin
        # Cancellation) matches "work zone" only because it shares 01041's
        # Bulletin Type table, and its card would cost ~1.9k tokens of prompt.
        hits: list[tuple[str, str, list[str]]] = []
        kept_terms: list[set[str]] = []
        for _count, msg_id, matched in scored:
            if len(hits) >= top_k:
                break
            if any(set(matched) <= earlier for earlier in kept_terms):
                continue
            instance = self._message_instance.get(msg_id)
            class_name = self._instance_to_class.get(instance) if instance else None
            if instance and class_name:
                hits.append((instance, class_name, matched))
                kept_terms.append(set(matched))

        family = self._example_families.get(chapter or "")
        for msg_id in family.message_ids if family else ():
            instance = self._message_instance.get(msg_id)
            class_name = self._instance_to_class.get(instance) if instance else None
            if len(hits) >= top_k or not class_name or any(hit[0] == instance for hit in hits):
                continue
            hits.append(
                (instance, class_name, [f"sent by reference {', '.join(family.reference_ids)} (chapter {chapter})"])
            )

        logger.info(
            "Static context (message objects): %d office-message object(s) "
            "matched: %s",
            len(hits),
            ", ".join(f"{instance} ({', '.join(matched)})" for instance, _, matched in hits)
            or "none",
        )
        return hits

    def message_object_context(self, hits: list[tuple[str, str, list[str]]]) -> str:
        """`message_object_hits`' result, rendered as one card per object:
        every method of its class, one line each -- `<instance>.<method>(...)`,
        its docstring's first line, and any dict keys its params document.

        This is what tells the model which object a stimulus call belongs
        to: an API-surface method chunk on its own carries a bare method
        name (`set_type`), and that name alone is not unique -- ~50 classes
        define `set_bos`. A card names the real declared instance
        (`wcr_office_01041`) up front and lists every one of its methods
        under it, so the model has a real object to call instead of picking
        one of many classes' same-named methods at random.
        """
        return "\n\n".join(
            self._object_card(instance, class_name, matched) for instance, class_name, matched in hits
        )

    def _object_card(self, instance: str, class_name: str, matched_terms: list[str]) -> str:
        lines = [f"{instance}: {class_name}  (matched: {', '.join(matched_terms)})"]
        for (cls, _method), hit in self._api_records.items():
            if cls != class_name:
                continue
            signature, summary, keys = _method_signature(hit.text, instance)
            summary = re.sub(r"^This function (?:will )?", "", summary)
            line = signature.removesuffix(" -> None")
            if summary:
                line += f"  # {summary}"
            if keys:
                line += f" keys: {', '.join(keys)}"
            lines.append(line)
        return "\n".join(lines)

    def _guaranteed_api_hits(
        self, query: str, chapter: str | None = None
    ) -> dict[tuple[str, str], KeywordHit]:
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
        family = self._example_families.get(chapter or "")
        for pair in family.api_pairs if family else ():
            self._add_guaranteed_hit(included, pair, "example family")
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
        """TBC/CFG/THE records from the parameter configuration guide, looked
        up by the identifiers the requirement states.

        Exact lookup is the whole of it: a requirement that names TBC137 or
        CFG16 has stated its own scope, so those records are the answer and
        `top_k` does not apply. A requirement that names none gets no
        records — see below for why there is no keyword fallback.

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

        # No identifier named, no records. BM25 always returns its top_k, so a
        # fallback here cannot answer "no parameter applies" — and the guide is
        # 900 records of one vocabulary ("on-board segment", "train", "track",
        # "message"), so the ranking is decided by shared boilerplate. On
        # L2R8289 (bulletin-dataset validation, which turns on no parameter at
        # all) the top four scored 324/323/319/318 — a 2% spread across four
        # unrelated records. The prompt then presents that block as the only
        # source of TBC/CFG/THE values, so the model writes a case around
        # whichever record the noise returned.
        logger.info(
            "Static context (parameter_config): no records "
            "(the requirement names no TBC/CFG/THE identifier)"
        )
        return []

    def icd_context(
        self,
        query: str,
        top_k: int,
        message_objects: list[tuple[str, str, list[str]]] | tuple = (),
    ) -> str:
        """Decoded ICD message fields the requirement names -- what
        `Train Type=1` or `Restricted Speed=1` means -- formatted for a
        prompt.

        Looked up by field name, not ranked. The field names are a closed
        vocabulary, and BM25 over requirement prose loses them the same way
        it lost TBC identifiers: a work-zone requirement repeating "speed" and
        "restriction" ranked Head End PTC Subdivision ID above Train Type,
        which it named. Only multi-word names are matched, so a bare "Speed"
        or "Direction" field never rides in on ordinary wording. `top_k` is a
        ceiling.

        With `message_objects` (see `message_object_hits`), only fields of
        those messages are kept, and the fields behind their enumeration
        matches are added: a work-zone requirement matched 01041 through
        "Bulletin Type=6 (Work zone)" but named only "calculated position
        uncertainty", which pulled that field from two locomotive reports
        (01071, 02054) the test never sends -- and never the Bulletin Type
        table the test does set. Without message objects, name matching
        alone decides, as before.
        """
        if top_k <= 0:
            return ""
        candidates = [
            hit
            for pattern, field_hits in self._icd_fields
            if pattern.search(query or "")
            for hit in field_hits
        ]
        if message_objects:
            msg_ids = {instance.rsplit("_", 1)[-1] for instance, _, _ in message_objects}
            enum_fields = {
                term.split("=", 1)[0]
                for _, _, matched in message_objects
                for term in matched
                if "=" in term
            }
            candidates += [
                hit
                for pattern, field_hits in self._icd_fields
                if any(pattern.fullmatch(field) for field in enum_fields)
                for hit in field_hits
            ]
            candidates = [hit for hit in candidates if msg_ids & _icd_msg_ids(hit.text)]
        hits = list({hit.chunk_id: hit for hit in candidates}.values())[:top_k]
        logger.info(
            "Static context (icd): %d field record(s)%s (%s)",
            len(hits),
            " of the matched message object(s)" if message_objects else " named in the requirement",
            ", ".join(sorted({h.metadata.get("section_path", "").split(" > ")[-1] for h in hits})) or "none",
        )
        return _format_hits(hits)

    def track_context(
        self,
        query: str,
        top_k: int,
        subdivision: str | None,
        include_blocks: bool = False,
    ) -> str:
        return self.track_context_for(query, top_k, subdivision, include_blocks)[0]

    def track_context_for(
        self,
        query: str,
        top_k: int,
        subdivision: str | None,
        include_blocks: bool = False,
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

        The track legend is always sent, and with `include_blocks` (the
        script call) so is a block table: the first `track_block_window`
        blocks of each track plus every block a retrieved feature row
        belongs to. Keyword search over requirement prose returned one
        arbitrary BlockFeature chunk and no legend, so the track names and
        milepost ranges every script needs were left to luck. The search
        therefore skips the tables these now cover and spends its slots on
        situational feature rows (signals, crossings, speed restrictions).
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
                and metadata.get("table_title") not in _SUMMARISED_TRACK_TABLES
            )

        hits = self._keyword_index.search(query, top_k, predicate=predicate)
        sections = [self._track_legend(chosen)]
        if include_blocks:
            referenced = {block for hit in hits for block in _PARENT_BLOCK_RE.findall(hit.text)}
            sections.append(self._block_table(chosen, referenced))
        sections.append(_format_hits(hits))
        logger.info(
            "Static context (track_data): legend%s + %d/%d feature hit(s) in subdivision %s",
            " + block table" if include_blocks else "",
            len(hits),
            top_k,
            chosen,
        )
        context = "\n\n".join(section for section in sections if section)
        return self._with_railroad_scac(context, chosen), (chosen,)

    def _track_legend(self, subdivision: str) -> str:
        legend = self._track_legends.get(subdivision)
        if not legend:
            return ""
        tracks = "; ".join(
            f"{fields['TrackName']} (value {fields.get('TrackValue', '?')}, "
            f"{fields.get('NumberBlocks', '?')} blocks)"
            for fields in legend
        )
        return f"Tracks in subdivision {subdivision} (TrackNameFeature): {tracks}"

    def _block_table(self, subdivision: str, referenced: set[str]) -> str:
        by_track: dict[str, list[tuple[float, float, str]]] = {}
        for track, start, end, block in self._block_ranges.get(subdivision, []):
            by_track.setdefault(track, []).append((start, end, block))
        if not by_track:
            return ""

        legend_order = [fields["TrackName"] for fields in self._track_legends.get(subdivision, [])]
        tracks = [t for t in legend_order if t in by_track] + [t for t in by_track if t not in legend_order]
        window = self._track_block_window
        lines = [
            f"Blocks (first {window} per track): block: start-end raw XML milepost "
            f"[miles]. set_position \"point\" takes miles; 01041 set_segment "
            f"\"start\"/\"end\" take the raw value."
        ]
        shown: set[str] = set()
        for track in tracks:
            blocks = sorted(by_track[track])[:window]
            shown.update(block for _, _, block in blocks)
            lines.append(f"{track}: " + "; ".join(_block_line(b, s, e) for s, e, b in blocks))

        extra = [
            f"{track} {_block_line(block, start, end)}"
            for track, start, end, block in self._block_ranges.get(subdivision, [])
            if block in referenced and block not in shown
        ]
        if extra:
            lines.append("Blocks the feature rows below belong to: " + "; ".join(extra))
        return "\n".join(lines)

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
