# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Turns a requirement into a review-ready test-case datasheet (`.xlsx`) and an
automation script draft (`.txt`), grounded in existing documents:

```
data/python_apis/       .py/.pyi   ──(ast)──┐
data/track_data/<subdiv>/  -subdiv.xml  ──(ElementTree)──┤
data/parameter_config/  .json   ──(one record per TBC/CFG/THE)──┤
                                                                │
                            chunking (structure-aware) → one in-process
                                                          BM25 index
                                                                │
data/Examples/     req → ref test cases → ref script triples    (design-time only:
                   distilled by hand into the templates in prompts.yaml, never sent)
                                                                │
requirement ──→ requirement understanding ──→ BM25 retrieval ──┤
                                                                │
                context + prompts.yaml ──→ Qwen (Ollama)
                                                                │
                datasheet rows ──→ .xlsx
                                                                │
                approved rows ──→ second generation ──→ test script ──→ .txt
```

**There is no vector store and no embedding model.** Every source folder is
parsed from disk at server start into a single in-process BM25 index
(`src/rag/static_context.py`). This is deliberate: all three sources are
either one giant stub file or thousands of near-identical rows (per
subdivision, per parameter), not prose an embedding model gains from, and
all three are full of exact identifiers (API names, block numbers,
`TBC137`) BM25 already handles better than dense search.

`data/knowledge_base/` is **not read at request time**. It holds the source
PDFs that `scripts/convert_parameter_guide.py` converts into
`data/parameter_config/`, and nothing else.

`data/Examples/` is separate again, and is not sent to the model at all: its
patterns are distilled by hand into the house wording pattern and house
skeleton in `config/prompts.yaml`, so the shape costs a few hundred tokens of
cacheable system prompt instead of ~10k characters of prefill on every
request.

## Setup

```bash
python -m venv venv && venv\Scriptsctivate      # Windows
pip install -r requirements.txt
# create .env yourself (gitignored, no committed template) — see the
# "--- ... ---" section comments in .env for what each key does
ollama pull qwen2.5:7b
ollama serve
```

`.env` holds per-machine values: backend host/port, CORS origins, log level,
`MAX_CONCURRENT_GENERATIONS`, Ollama host override, frontend dev-server
port/proxy target. Everything else — chunking, retrieval depth, prompts —
lives in `config/*.yaml`.

**The backend makes no network calls at startup** beyond talking to Ollama.
Nothing is downloaded: no model, no tokenizer. Keep it that way — a
dependency that reaches the network at import or startup does not belong
here. `TokenCounter` in `chunking.py` estimates token counts locally with a
regex piece count scaled by a factor measured against the real bge-m3
tokenizer (1.03x–1.20x across the three sources, so one factor covers the
corpus within ~15%). Since nothing is embedded, `chunk_max_tokens` only
bounds how much text a chunk carries into a prompt, and an estimate is as
useful there as an exact count. A plain word count is *not* an acceptable
substitute: on the track XML it undercounts by more than 3x and would
silently triple every chunk.

## Common commands

There is no ingestion step. Every source folder under `static_sources:` in
`config/config.yaml` is read fresh from disk on every server start, so
**editing any of them takes effect on the next restart** — no build, no
index to rebuild, no state file.

Convert the Parameter Configuration Guide after dropping a new revision of it
into `data/knowledge_base/`:

```bash
python scripts/convert_parameter_guide.py   # the guide PDF -> data/parameter_config/
```

The running app reads the JSON, not the PDF, so a new revision changes nothing
until this is re-run and the server restarted.

Update the requirement -> feature (Folder) mapping after editing `data/CAF.xlsx`:

```bash
python scripts/convert_caf_mapping.py       # data/CAF.xlsx -> config/caf_mapping.json
```

The app reads `config/caf_mapping.json`, not the workbook, so this must be
re-run for a workbook edit to take effect; the app then picks up the
regenerated JSON on its next request, no restart needed.

Run the backend:

```bash
python src/api/main.py
```

Reads `BACKEND_HOST`/`BACKEND_PORT` from `.env`, checks the port is actually
free before binding (steps to the next one if not — avoids opaque
`WinError 10013` from a leftover process or a Windows-reserved port range).
For auto-reload during dev, use uvicorn directly instead (no port fallback,
so pick a free port yourself):

```bash
uvicorn api.main:app --app-dir src --reload --port 8000
```

Run the frontend:

```bash
cd frontend && npm install && npm run dev      # http://localhost:5173
npm run build     # production build
npm run preview   # preview the build
```

There is no automated test suite in this repo currently (no `tests/`
directory, no pytest config) — verify changes by starting the server and
exercising the API/frontend directly.

## Architecture

### Parsing and chunking (`src/kb_ingestion/`)

Despite the package name, nothing is ingested anywhere — this is the parse
and chunk stage, and its output goes straight into the in-process BM25 index.
`config/config.yaml`'s `static_sources:` lists every folder read. Every chunk
is stamped with a `document_type` from its root; `subfolders_as_type: true`
lets an immediate subfolder override it, no code change.

Chunking preserves structure — nothing unrelated ever shares a chunk:
- **PY/PYI** (`extractors/python_extractor.py`): module -> class -> method; a
  method is only split if it alone busts the token budget.
- **XML** (`extractors/xml_extractor.py`): the `<subdiv>-subdiv.xml` track
  export. Subdivision -> record type -> one `field=value` line per record,
  nested records prefixed with their parent's ID (`BlockFeature 2001 > ...`)
  so a flattened line still says which block it belongs to. Each unit starts
  with a label row naming the record type and its fields in **split** form
  (`Block Signal Feature | Signal Id | ...`), because BM25 tokenizes camel
  case whole — without it a requirement asking about a "signal" matches none
  of these rows. Chunking repeats that first line on every split chunk. The
  exports pad fixed-width fields with literal NUL character references, which
  is not well-formed XML at any version, so illegal character references are
  stripped before parsing rather than losing a 2 MB file to one padded field.
- **JSON** (`extractors/json_extractor.py`): the converted Parameter
  Configuration Guide. One parameter record = one chunk, led by its
  identifier and title (`TBC137 | Restricted Speed`) so BM25 can return the
  parameter a requirement names rather than its neighbours.

`extractors/` also still holds PDF, XLSX and TXT parsers. No configured
source uses them today; they are kept because they are generic format
parsers, and adding a folder of PDFs to `static_sources:` should not need
new code.

Key files: `chunking.py` (normalisation, token budgets, metadata),
`pipeline.py` (`ChunkingPipeline`: discover -> extract -> normalise -> chunk,
in memory, nothing persisted).

### The static index: python_apis, track_data and parameter_config (`src/rag/static_context.py`)

`python_apis/`, `track_data/` and `parameter_config/` are declared under
`static_sources:` in `config/config.yaml` and held in one in-process BM25
index. This is the whole of the app's retrieval.
`static_api_top_k`/`static_track_top_k`/`static_parameter_top_k` in
`config/rag_config.yaml` control how many chunks of each are pulled per
request.

`track_data/` is one folder per subdivision (`data/track_data/08101/`), each
holding that subdivision's `<subdiv>-subdiv.xml` export: the full serialized
track database, carrying WIU addresses and security keys, per-block
element/heading/elevation series, device status indices and acquisitions.

The HTML report that used to sit beside each XML has been removed, along with
`html_extractor.py`. It was the human view of a strict subset of the same
data, and indexing both meant two chunks competing to answer the same
question. **Only XML is read now** — `.html`/`.htm` are no longer registered
suffixes, so dropping a report back into a folder does nothing.

One consequence is visible in the UI: the display name ("Ginger") lived only
in the HTML report, so the subdivision picker lists bare numbers. That is the
intended state, not a gap — a picker showing a name for some subdivisions and
a number for others would read as missing data. `Subdivision` in
`static_context.py` therefore carries `id` and `chunks` and nothing else.

Track data reaches **only** the script call. A datasheet description states
the behaviour to verify, not the blocks it runs on, so test-case generation
sends nothing from `track_data/` (its `track_subdivisions` in the response
is always empty).

Which subdivision the script call searches is decided by **one** thing: the
subdivision picked in the UI (`subdivision` on the generate requests,
`GET /api/track-subdivisions` for the list).

There is deliberately no requirement -> subdivision map any more. A stored
map and a picker are two answers to the same question, and the one the user
just chose in front of the result has to win — so keeping both only created
a way for the control to appear to do nothing. The picker is **required**
in the UI (`canGenerate` in `frontend/src/App.jsx`), and the backend treats
a missing subdivision as "send no track data" rather than "search
everything": searching every subdivision would let BM25 return blocks and
mileposts from track the requirement is not tested on, and a wrong value
that looks right is worse than the `# TODO:` the model writes when the
track block is empty.

`data/Examples/` (config `examples_dir`) holds requirement -> reference test
cases -> reference script triples for a handful of other requirements. It is
**not read at request time.** Nothing in it reaches the model directly.

Its patterns are distilled by hand into two templates in
`config/prompts.yaml` — the **house wording pattern** in the `test_cases`
system prompt (how a datasheet `description` is phrased: condition bullets,
a bare `Verify` line, one observable behaviour) and the **house skeleton**
in the `test_script` system prompt (encoding comment -> docstring -> import
-> `main()` -> `handle_test_case` branch table -> `# User Defined Functions`
-> `__main__` guard, plus the two branch styles and the fixed conventions).

Sending the files instead cost ~8-10k characters on *every* generation, all
of it prefill, and all of it byte-identical from one request to the next —
repeated work to teach the model a shape that never changes. The distilled
templates live in the **system** prompt, which is both where "how to write"
belongs and the prefix an inference server can cache across requests.

Both templates are written with angle-bracket slots (`<subdivision>`, `<the
observable behaviour>`) and contain **no** real subdivision, block,
milepost, signal or TBC value. That is deliberate and must stay that way: a
template holding a real value would reintroduce the copying defect that
"Examples are templates, never a source of values" (below) exists to
prevent. A template you cannot copy a value out of enforces the rule
structurally rather than by instruction.

`data/Examples/` stays in the repo as **design-time** input: the source
those templates were derived from, and the material to re-derive them from
when the house style changes. `examples_dir` in `config/config.yaml` still
points at it. Editing the folder now has no effect on generation until
someone updates `config/prompts.yaml` to match — that is the intended
trade.

`parameter_config/` holds the Parameter Configuration Guide's tables as one
JSON record per parameter, produced from the PDF by
`scripts/convert_parameter_guide.py` (`extractors/json_extractor.py` renders
each record back into a chunk that leads with the identifier and title). The
guide is the authority for TBC/CFG/THE values, their units, valid ranges,
North American defaults and owning railroad — exactly the values a test case
asserts against and a boundary case is written from. Left as a PDF it chunks
into prose, so asking about `TBC137` returns whatever share of a guide page
the chunk boundary happened to catch, with its neighbours' ranges alongside
it; one record per parameter returns that parameter.

Retrieval of those records is exact lookup first, BM25 only as a fallback.
Any TBC/CFG/THE identifier the requirement states is looked up by exact id in
`static_context.py`, and when the requirement names any, **those records are
the whole parameter block** — nothing is padded in beside them and
`static_parameter_top_k` does not apply. BM25 ranks the remaining ~900
records on shared prose, and the guide is 900 records of the same prose: on
a work-zone requirement naming TBC290 and CFG22, `TBC412` ("...calculated
position uncertainty of the leading edge of the train...") scored *above*
TBC290's own record and filled a padding slot, where `_format_hits` presents
every record identically and the prompt calls the block the only place a
value may come from. A parameter the requirement never mentions is not a
ranking question to be answered less confidently — it is the wrong answer,
and one the reviewer cannot spot once it is written into a finished case.
`static_parameter_top_k` still governs the fallback, for a requirement that
names no identifier at all; the identifier boost below is what makes that
ranking sound.

The conversion is a build step, not a request-time parse: nothing reads the
guide PDF while the app runs. Re-run the script for a new revision, then
restart the server.

### Requirement -> folder mapping (`src/rag/caf_mapping.py`)

`data/CAF.xlsx` (the Change Approval Form export: `Sr No`, `Section`,
`Feature`, `Requirement No`) is the source of the datasheet's `Folder`
column, but the running app never reads that workbook directly. Instead:

```bash
python scripts/convert_caf_mapping.py   # data/CAF.xlsx -> config/caf_mapping.json
```

converts it once into `config/caf_mapping.json` (path in
`config/config.yaml`'s `caf_mapping_file`) — `{"requirements": {"<ID>":
{"feature": ..., "section": ...}}}` — and `CafMapping` only ever loads that
JSON file. A feature is written once against the first requirement it
covers and left blank below in the workbook, so the conversion carries the
value down; lookups are case- and whitespace-insensitive, and `L2R8279_A`
falls back to `L2R8279`. Re-run the conversion script whenever `data/CAF.xlsx`
changes — the app is not watching the workbook, only the JSON file the
script produces, which it re-reads whenever that file's mtime changes (no
restart needed).

When the mapping covers a requirement, that feature is what every generated
row's `Folder` carries — the value overwrites whatever the model wrote
rather than just seeding it, so the same requirement always lands in the
same folder. The functional area read out of the requirement heading
(`requirement_parser.py`) is only the fallback for a requirement the mapping
doesn't cover yet. `POST /api/requirements/folder` resolves it through the
same call without touching the model, so the UI shows the folder under the
requirement box as soon as the requirement number is typed.

### Keyword retrieval (`src/rag/keyword_index.py`)

One ranking, BM25 over every chunk of the static index. There is no dense
arm, no fusion and no vector store: the sources are identifier-dense records,
which is what BM25 is good at, and the embedding stack that used to sit
alongside it has been removed entirely.

Query tokens that mix letters and digits (`tbc137`, `l2r7983`, `iv132`) are
counted `_IDENTIFIER_BOOST` times on the **query** side only
(`boost_identifiers` in `keyword_index.py`; the corpus is tokenized plainly).
BM25 sums a contribution per query token, so without this a requirement that
names `TBC137` once but repeats "speed", "restricted" and "enforce" throughout
ranks records matching those common words above the one record it names —
measured on a one-sentence requirement, TBC137 came 13th of the parameter
records and never reached the prompt; boosted, it is 1st. Deduplicating the
query instead was tried and is worse (18th): the repeated prose is evidence
too. Bare numbers are deliberately not boosted — the corpus is full of them
and IDF already separates a rare one.

Scoring is memoised per query token set: `BM25Okapi.get_scores` walks every
chunk in Python, and one script request scores the same query three times
(API pass, track pass, parameter pass), differing only in the metadata filter
applied afterwards. Don't reintroduce repeated unmemoised scans.

### Output formats (`src/rag/exporters.py`)

`.xlsx`: sheet `Datasheet`, columns A–J exactly matching existing workbooks
(`S_no`, `Requirement`, `Description`, `Folder`, `Optimization_Technique`,
`Test_Type`, `Test_Technique`, `Retired?`, `Scorable`, `Comments`) and
nothing else — the export is exactly the delivered format.

`.txt` script: `exporters.py` itself only prepends a provenance header
(requirement ID, generation time) — the encoding comment, proprietary
docstring, `from Common.WCR_public import *`, `main()` with the test-case
function table, and `handle_test_case`'s one-`if`-per-row structure are all
instructions to the LLM in `config/prompts.yaml`'s script prompt, not code
here. Where a value isn't derivable from context, the model is instructed to
leave a named variable and `# TODO:` rather than guess. Script generation is
a **second** LLM call, run against the rows the reviewer kept (including
edits) — not the model's first draft.

The script call is the one generation that cannot be constrained by a JSON
schema (it must return Python), so "Output ONLY the Python source" in the
prompt is a request, and a reasoning model whose chain of thought lands in
`message.content` answers it with paragraphs of English ("We are given one
test case… From the context…").

What replaces the schema is **prefill**: `LLMClient.generate` takes a
`prefill=` argument, and the script call passes the house skeleton's first
line (`_SCRIPT_PREFILL` in `generator.py`, the encoding comment and nothing
more). `OllamaClient` sends it as a trailing **assistant** message, which
every Ollama chat template in use renders without its end-of-turn token (the
standard `{{ if not $last }}<|im_end|>` idiom), so the model continues that
text instead of opening a fresh turn. Its first token is therefore already
inside a Python comment line — it cannot begin a paragraph — and
`_rejoin_prefill` stitches the prefill back onto the answer so callers see one
continuous script whether the server continued the turn or ignored it and
started over. Only the encoding comment is prefilled: enough to fix the
answer's *form*, not enough to put a *value* in the model's mouth (see
"Examples are templates, never a source of values").

Prefill also sidesteps a trap specific to hybrid reasoning models. qwen3's
Ollama template appends `<|im_start|>assistant\n<think>` after a trailing
**user** message unconditionally, so the model always starts inside a chain of
thought regardless of `llm_think: off`, and on a verbose model the whole
`script_max_tokens` budget can be spent before any code is written — which
reaches the exporter as prose. A trailing assistant message takes that branch
of the template out of play. If a model still answers with prose, the budget
is the next thing to check: raise `script_max_tokens`.

The rest is enforced in code, in `generator.py`:

- `_strip_code_fence` pulls the script out of a markdown fence wherever the
  fence sits — a chat-tuned model introduces it ("Here is the script:") and
  explains it afterwards, so a fence is not only found at position 0. A reply
  that already reads as a script is returned untouched.
- `_extract_python` decides whether the response *is* a script and, if the
  model narrated first and then wrote one, returns the code from the first
  house-skeleton anchor onward. Parsing alone is not the test — a prose line
  like `Requirement: L2R433018` is a valid annotated-name statement, so a
  paragraph can compile by accident; `_is_script` additionally requires an
  import and a `def`. A truncated script is still recognised, via its
  longest parsing prefix, and left for `_salvage_truncated` to trim.
- No script in the response means one retry with the format rule restated at
  the **end** of the user prompt, which is where an instruction carries most
  weight.
- A small model sometimes reproduces the house skeleton instead of applying
  it, `<setup: ...>` slots and all. That streams past in the log looking like
  a script, but every slot line is a syntax error, so trimming to the parsing
  prefix would export only the handful of literal lines above the first
  branch — a file that looks empty next to what the terminal showed.
  `_unfilled_slots` detects it and retries once with
  `_NO_PLACEHOLDER_REMINDER`; if the retry is no better, `_stub_slots`
  rewrites each slot as `pass  # TODO: unfilled slot — <slot text>`, which
  keeps the whole script, keeps the file parsing, and names every line still
  to be written. Reaching that path means the model is too small for the
  job — check `llm_model` in `config/rag_config.yaml`.
- Whatever `_salvage_truncated` drops off the end is kept in the export as
  comments, not deleted. The user watched those lines stream past; an export
  holding less than the terminal showed reads as a lost generation.
- If the retry also returns prose, `_as_commented_out` exports it commented
  under a `# TODO:` saying no script was generated. The export stays runnable
  Python in every path — never English in a `.py`-shaped file, and never an
  empty file that hides what happened.

Reaching that last path means the model, not the pipeline, is the problem —
check `llm_model`/`llm_think` in `config/rag_config.yaml`.

### Examples are templates, never a source of values

Every concrete value in a generated artefact — subdivision, block, milepost,
switch, signal, TBC parameter, API name — must come from a real source for
*that* requirement: track values from `data/track_data/` (via
`static_context.py`), API names/kwargs from `data/python_apis/`, and
TBC/CFG/THE values, units and valid ranges from `data/parameter_config/`.
`data/Examples/` fixes shape only; copying one of its concrete values into a
new artefact is a defect. A prompt can only obey "don't guess" if it actually
contains a real source for the values it asks for — this is why script
generation runs its own track-data and static-API passes rather than relying
on the reference scripts alone.

### Prompts (`config/prompts.yaml`)

Every system/user prompt lives here. Edits take effect on the next request —
no restart, no code change. Placeholders are `$name`/`${name}`
(`string.Template`), not `{name}`, so JSON braces inside prompts need no
escaping.

### No "how many test cases" control — deliberate

How many cases a requirement needs is a property of the requirement (one per
verifiable behaviour), not something a user can guess beforehand. The model
enumerates the behaviours stated and writes one case per behaviour.
`max_test_cases` in `config/rag_config.yaml` is a ceiling to keep a sprawling
requirement inside the model's output budget, not a target — don't turn it
into a user-facing "number of cases" input.

### Performance design (CPU-bound 7B model)

The generation call is the floor on request latency; everything else is
arranged to not add to it. When touching `src/rag/` or `src/api/`, preserve these:

- BM25 scoring is memoised per query (`keyword_index.py`) — don't reintroduce
  repeated scans of the same query with only the metadata filter differing.
  One script request runs three passes over the same query, so this is what
  keeps retrieval off the critical path.
- The BM25 index is built once per process, at first use, under a lock
  (`services.py`). It takes about two seconds over ~7,000 chunks, and
  `warm_up()` pays even that at startup so no request does.
- Generation endpoints in `src/api/main.py` are `async`; a semaphore
  (`MAX_CONCURRENT_GENERATIONS`, default 1) serializes actual generations
  since a CPU-bound model does not get faster with concurrent requests — it
  slows both down. Raise this default only where the model has headroom
  (GPU, remote Ollama).
- A truncated LLM response (ran out of output tokens mid-array) is salvaged
  in `src/rag/schema.py`: completed test cases are recovered, the incomplete
  tail dropped. The script equivalent is `_salvage_truncated` in
  `generator.py`, which drops trailing lines until the script parses and
  marks the cut with a `# TODO:` — the export has to stay runnable Python.
- Generations are streamed (`src/rag/llm_client.py`) and, by default, are
  **not** time-limited: `llm_request_timeout_seconds: 0` and
  `llm_stall_timeout_seconds: 0` mean a generation runs until the model is
  done. On CPU no wall-clock number separates a slow generation from a stuck
  one — a script prompt can sit in prefill for minutes before its first token
  — and cutting one off loses the whole run. A dead Ollama still fails fast,
  because a dropped connection is a `ConnectionError`, not a timeout.
  Streaming is what makes the unlimited wait observable: progress is logged
  as tokens arrive. Set either knob non-zero only if you want a generation
  to give up — the total budget keeps the partial answer and salvages it,
  the stall timeout fails the request. Do not "fix" a slow generation by
  shrinking the prompt; that trades away the grounding the output needs.
- `llm_keep_alive: 30m` in `config/rag_config.yaml` keeps model weights
  resident between requests.

### Swapping `llm_model` is a config-only change

Changing `llm_model` in `config/rag_config.yaml` must never need a code edit,
so anything that differs between model families is detected at runtime.

The case that matters today is reasoning models (`qwen3`, `deepseek-r1`,
`gpt-oss`). Ollama streams their chain of thought in a **separate**
`message.thinking` field and only then starts filling `message.content`, so a
naive reader collects an empty answer while the whole `num_predict` budget is
spent reasoning — which surfaced as `qwen3:4b produced no output in 723s`.
`llm_think` (`auto` | `off` | `on`, default `auto`) handles it:
`OllamaClient._think_flag` asks `/api/show` for the model's `capabilities`
(cached per model) and sends `think: false` only to models that advertise
`thinking`, sending nothing for models that don't — Ollama rejects the flag
on those. `strip_reasoning` additionally removes inline `<think>` fences for
models that embed reasoning in the content instead. With `llm_think: on` the
reasoning is kept, and `llm_max_tokens`/`script_max_tokens` must then be large
enough to cover the thinking *and* the answer; a run that never gets past
thinking now fails with that advice rather than "produced no output".

The second half of the same problem is output format. "Return ONLY a JSON
object" in `prompts.yaml` is a request that instruct models mostly honour and
reasoning models often don't — qwen3 will spend a whole output budget
narrating its plan, which reaches the parser as *"The model's response
contained no JSON object"* after minutes of CPU time. So the datasheet call
passes `TEST_CASES_JSON_SCHEMA` (`schema.py`, built from the same tuples that
normalise the columns) as `json_schema=` on `LLMClient.generate`, and
`OllamaClient` sends it as Ollama's `format`: the decoder is constrained, so
the response is a JSON object of the right shape whatever the model would
otherwise have written. Only `description` is required of a case — `s_no`
and `requirement` are assigned by `parse_test_cases` and everything else has a
normalised default, so requiring them would only add ways to fail.

The schema's other job is the **count**. Nothing obliges a constrained
decoder to write more than one array element — `]` is a legal token after
the first — and the prompt's only number was a ceiling, so on one
requirement the same model returned 8 cases, then 10, against a delivered
datasheet of 11; a run returning 1 is the same coin. `prompts.yaml` asked
the model to enumerate the behaviours before writing cases, but with only
`test_cases` in the schema there was nowhere to put that list. So the schema
now requires a `coverage` array of short strings *before* `test_cases`, and
Ollama builds its grammar in property order: the enumeration is the first
thing written, and the cases follow a list the model has already committed
to. `coverage` never becomes a datasheet column — `parse_test_cases` only
compares the two lengths and logs a warning naming the lines that got no
case, because the reviewer's evidence of a short answer is the shortfall
itself. The prompt
still describes the shape (a constrained model writes better JSON when it also
knows what the fields mean), the truncation salvage still applies, and an
inference server too old for structured output is retried once without the
constraint. **Script generation deliberately passes no schema** — it must
return Python.

Follow the same rule for any future model-dependent behaviour: probe the
server for the capability, default to `auto`, don't hard-code model names.

If generations are too slow: lower `llm_max_tokens`/`script_max_tokens`,
lower the `static_*_top_k` values, or switch `llm_model` to a smaller variant
in config — no code change needed for any of these. Script generation is the
slowest call in the app (it sends API stubs, track data and parameter
records, then writes the longest output), so it is the one that runs into the
budget first.

Prefer reducing the *unchanging* parts of the prompt over the retrieved
parts. On CPU, prefill is a real fraction of wall clock — not the rounding
error it is on a GPU — and text that is byte-identical across requests is
the cheapest thing to cut, since it costs prefill every time while teaching
the model nothing new. Distilling `data/Examples/` into the prompt
templates was exactly this trade. Trimming retrieved context is the last
resort, not the first: that is the grounding the output depends on.

### Configuration surface

| File | Controls |
|---|---|
| `config/config.yaml` | `static_sources` (every folder read), `examples_dir` (design-time only), chunk sizes, PDF heuristics, chunk tokenizer |
| `config/rag_config.yaml` | Static-context top-k, Ollama host/model/limits/timeouts, `max_test_cases` ceiling |
| `config/prompts.yaml` | Every system and user prompt |
| `data/parameter_config/*.json` | TBC/CFG/THE parameter records — generated from the guide PDF by `scripts/convert_parameter_guide.py` |
| `config/caf_mapping.json` | Requirement → feature, i.e. the datasheet's `Folder` column — generated from `data/CAF.xlsx` by `scripts/convert_caf_mapping.py` |
| `.env` | Backend host/port, CORS origins, log level, `MAX_CONCURRENT_GENERATIONS`, Ollama host override, frontend dev-server port/proxy target |

Prefer adding new behavior-affecting knobs to `config/*.yaml`, and new
per-machine values to `.env` — this split is intentional throughout the repo.

### Abstraction points

`LLMClient` (in `src/rag/`) is a protocol specifically so a different
inference server (vLLM, llama.cpp, any OpenAI-compatible API) can replace
Ollama without touching the retrieval or generation pipeline.

### API routes (`src/api/main.py`)

`GET /api/health`, `POST /api/requirements/upload`,
`POST /api/requirements/folder` (CAF folder lookup, no model),
`GET /api/track-subdivisions` (the subdivision picker's list, read off the
built static index so it can only offer track that actually parsed),
`POST /api/test-cases`
(datasheet generation), `POST /api/test-script` (script generation, second
LLM call), `POST /api/test-cases/export` and `POST /api/test-script/export`
(`.xlsx`/`.txt` downloads). `services.py` wires the shared static-context and
generator instances; `models.py` holds the request/response schemas.

`/api/health` reports `indexed_chunks` (how much the BM25 index holds) and
whether the configured model is reachable.

### Layout

```
config/                     config.yaml, rag_config.yaml, prompts.yaml, caf_mapping.json
data/knowledge_base/         source PDFs for the converter scripts; NOT read at request time
data/python_apis/            PY · PYI (indexed)
data/track_data/<subdiv>/    <subdiv>-subdiv.xml (indexed)
data/parameter_config/       parameter records (indexed); generated, not hand-edited
data/Examples/               reference triples; design-time source for the prompts.yaml templates, not read at request time
data/CAF.xlsx                requirement -> feature (Folder) source; converted, not read at request time
scripts/convert_caf_mapping.py   data/CAF.xlsx -> config/caf_mapping.json
scripts/convert_parameter_guide.py  guide PDF -> data/parameter_config/
src/kb_ingestion/
  extractors/                 one parser per format (python, xml, json; pdf/xlsx/text kept, unused)
  chunking.py                 normalisation, token budgets, chunk metadata
  pipeline.py                   discover -> extract -> chunk, in memory
src/rag/
  requirement_parser.py        requirement ID + functional area
  caf_mapping.py                requirement -> feature (the Folder column), from config/caf_mapping.json
  static_context.py             the BM25 index over every source folder, and the searches over it
  keyword_index.py               BM25 scoring, tokenization, identifier boost
  cache.py                       the generation result cache
  prompts.py                     prompts.yaml loader
  schema.py                      datasheet rows, model-output parsing, truncation salvage
  llm_client.py                  Ollama behind a protocol
  generator.py                   test-case and test-script generators
  exporters.py                   .xlsx and .txt
src/api/                     FastAPI app (main.py), request/response models, service container
frontend/                    React + Vite
```
