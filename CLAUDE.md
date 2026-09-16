# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Turns a requirement into a review-ready test-case datasheet (`.xlsx`) and an automation script draft (`.txt`), grounded in existing documents via RAG:

```
data/knowledge_base/  PDF · XLSX · TXT   ──(PyMuPDF, openpyxl)──┐
                                                                  │
                            chunking (structure-aware) → BGE-M3 → ChromaDB
                                                                  │
data/python_apis/  .py/.pyi   ──(ast)──┐                         │
data/track_data/<subdiv>/  .html + .xml  ──(html.parser, ElementTree)──┤
                                                   BM25-only, never embedded
data/Examples/     req → ref test cases → ref script triples      (design-time only:
                   distilled by hand into the templates in prompts.yaml, never sent)
                                                                  │
requirement ──→ requirement understanding ──→ hybrid retrieval ──┤
                                            (BGE-M3 dense + BM25, RRF)
                                                                  │
                context + prompts.yaml ──→ Qwen2.5-7B (Ollama)
                                                                  │
                datasheet rows + confidence ──→ .xlsx
                                                                  │
                approved rows ──→ second generation ──→ test script ──→ .txt
```

Only `data/knowledge_base/` is embedded into ChromaDB. `data/python_apis/`
and `data/track_data/` are parsed with the same extractors but kept in an
in-process, BM25-only index (`src/rag/static_context.py`) instead — they're
either one giant stub file or thousands of near-identical per-subdivision
rows, not prose an embedding model gains from, and both are full of exact
identifiers (API names, block numbers) BM25 already handles better than
dense search. `data/Examples/` is separate again, and is not sent to the
model at all: its patterns are distilled by hand into the house wording
pattern and house skeleton in `config/prompts.yaml`, so the shape costs a
few hundred tokens of cacheable system prompt instead of ~10k characters of
prefill on every request.

## Setup

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
# create .env yourself (gitignored, no committed template) — see the
# "--- ... ---" section comments in .env for what each key does
ollama pull qwen2.5:7b
ollama serve
```

`.env` holds per-machine values: backend host/port, CORS origins, log level,
`MAX_CONCURRENT_GENERATIONS`, retrieval worker count, Ollama host override,
frontend dev-server port/proxy target. Everything else — chunking, retrieval
weights, prompts, confidence thresholds — lives in `config/*.yaml`.

First ingestion downloads `BAAI/bge-m3` from Hugging Face (~2 GB); needs
internet access at least once.

## Common commands

Build/update the knowledge base:

```bash
python scripts/run_ingestion.py --dry-run   # parse only: no model, no writes
python scripts/run_ingestion.py             # embed and write to ChromaDB
python scripts/run_ingestion.py --prune     # forget files you deleted
python scripts/run_ingestion.py --reset     # start over
```

Re-running skips files whose bytes **and** processing code (`pipeline_fingerprint`)
are both unchanged since last embedded (stamped on the chunks themselves in
ChromaDB — no separate state file). Changing an extractor bumps the
fingerprint and re-embeds the whole KB once. Only `sources:` in
`config/config.yaml` is affected — `static_sources:` (python_apis, track_data)
is read fresh from disk on every server start, not ingested.

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
directory, no pytest config) — verify changes by running ingestion and the
API/frontend directly.

## Architecture

### Ingestion (`src/kb_ingestion/`)

`config/config.yaml`'s `sources:` list currently has only `data/knowledge_base`
active (`python_apis`/`track_data` are commented out there — they're ingested
differently, see below). Every chunk is stamped with a `document_type` from
its root; `subfolders_as_type: true` lets an immediate subfolder override it
(e.g. `knowledge_base/reference_test_cases/` becomes its own filterable type,
no code change).

Chunking preserves structure — nothing unrelated ever shares a chunk:
- **PDF** (`extractors/pdf_extractor.py`): heading hierarchy built while
  reading (numbering like `3.1.1`, font-size/bold fallback). A chunk never
  spans two sections; every prose chunk carries `section_path`. Tables become
  their own chunks. Running headers/footers and per-page admin tables are
  dropped as boilerplate.
- **XLSX** (`extractors/xlsx_extractor.py`): one row = one logical test case
  = one chunk (`xlsx_max_rows_per_chunk: 1`). `Requirement`/`S_no` columns
  become `requirement_id`/`test_case_id` metadata (what confidence scoring
  matches on).
- **PY/PYI** (`extractors/python_extractor.py`): module → class → method; a
  method is only split if it alone busts the token budget.
- **HTML** (`extractors/html_extractor.py`): subdivision → named section
  (Blocks, Switches, Signals, Speed Restrictions) → track-feature records,
  tagged with the numeric subdivision ID (read from the containing folder
  first, then the document).
- **XML** (`extractors/xml_extractor.py`): the `-subdiv.xml` sibling.
  Subdivision → record type → one `field=value` line per record, nested
  records prefixed with their parent's ID (`BlockFeature 2001 > …`) so a
  flattened line still says which block it belongs to. Each unit starts with
  a label row naming the record type and its fields in **split** form
  (`Block Signal Feature | Signal Id | …`), because BM25 tokenizes camel case
  whole — without it a requirement asking about a "signal" matches none of
  these rows. Chunking repeats that first line on every split chunk. The
  exports pad fixed-width fields with literal `&#x0;`, which is not
  well-formed XML at any version, so illegal character references are
  stripped before parsing rather than losing a 2 MB file to one padded field.
- **TXT** (`extractors/text_extractor.py`): plain-text documents in
  `knowledge_base/`.

Key files: `chunking.py` (normalisation, token budgets, metadata),
`incremental.py` (fingerprint behind the unchanged-file skip),
`embeddings.py` (BGE-M3 wrapper), `vector_store.py` (ChromaDB wrapper),
`pipeline.py` (orchestration).

### Static context: python_apis and track_data (`src/rag/static_context.py`)

`python_apis/` and `track_data/` are declared under `static_sources:` in
`config/config.yaml`, parsed with the same extractors as above, but held in
an in-process BM25-only index instead of ChromaDB — see the file's header
comment for the reasoning. `static_api_top_k`/`static_track_top_k` in
`config/rag_config.yaml` control how many chunks of each are pulled per
request.

`track_data/` is one folder per subdivision (`data/track_data/08101/`), each
holding **both** halves of that subdivision's export, and both are indexed:

- `<subdiv>.<rev>.html` — the readable report: a handful of grouped, labelled
  tables (Blocks, Switches, Signals, Speed Restrictions).
- `<subdiv>-subdiv.xml` — the full serialized track database, carrying fields
  the report never prints (WIU addresses and security keys, per-block
  element/heading/elevation series, device status indices, acquisitions).

The folder name is what stamps a chunk's `subdivision`, so one filter covers
the pair regardless of what either file states internally. Together they are
~5,000 BM25 chunks and about 11s of the backend's startup; the XML is roughly
four fifths of that.

Which subdivision a request searches is decided by **one** thing: the
subdivision picked in the UI (`subdivision` on the generate requests,
`GET /api/track-subdivisions` for the list). The same value is sent again
with the script call, so the script's hard-coded blocks match its
datasheet's.

There is deliberately no requirement → subdivision map any more. A stored
map and a picker are two answers to the same question, and the one the user
just chose in front of the result has to win — so keeping both only created
a way for the control to appear to do nothing. The picker is **required**
in the UI (`canGenerate` in `frontend/src/App.jsx`), and the backend treats
a missing subdivision as "send no track data" rather than "search
everything": searching every subdivision would let BM25 return blocks and
mileposts from track the requirement is not tested on, and a wrong value
that looks right is worse than the `# TODO:` the model writes when the
track block is empty.

One consequence worth knowing: `HybridRetriever` no longer runs a track
pass at all. It only ever searches the embedded `knowledge_base`
collection; track data reaches the prompt solely through
`static_context.py`. (That pass had in fact been dead since track data
moved out of ingestion — it filtered the embedded collection for a
`document_type` that is no longer ingested.)

`data/Examples/` (config `examples_dir`) holds requirement → reference test
cases → reference script triples for a handful of other requirements. It is
**not read at request time.** Nothing in it reaches the model directly.

Its patterns are distilled by hand into two templates in
`config/prompts.yaml` — the **house wording pattern** in the `test_cases`
system prompt (how a datasheet `description` is phrased: condition bullets,
a bare `Verify` line, one observable behaviour) and the **house skeleton**
in the `test_script` system prompt (encoding comment → docstring → import →
`main()` → `handle_test_case` branch table → `# User Defined Functions` →
`__main__` guard, plus the two branch styles and the fixed conventions).

Sending the files instead cost ~8–10k characters on *every* generation, all
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
restart, no ingestion).

When the mapping covers a requirement, that feature is what every generated
row's `Folder` carries — the value overwrites whatever the model wrote
rather than just seeding it, so the same requirement always lands in the
same folder. The functional area read out of the requirement heading
(`requirement_parser.py`) is only the fallback for a requirement the mapping
doesn't cover yet. `POST /api/requirements/folder` resolves it through the
same call without touching the model, so the UI shows the folder under the
requirement box as soon as the requirement number is typed.

### Hybrid retrieval (`src/rag/retriever.py`, `keyword_index.py`)

Two independent rankings over the embedded `knowledge_base` collection,
fused with reciprocal rank fusion (RRF, combines by rank not score):
- **Dense** (BGE-M3 cosine over ChromaDB) — catches paraphrase.
- **Keyword** (BM25) — catches exact identifiers dense search blurs (`TBC137`
  vs `TBC139`, block `1015` vs `1025`, API names).

`dense_weight`/`keyword_weight` in `config/rag_config.yaml` tune the balance;
0 disables an arm. BM25 index is built in-process from the collection and
rebuilt whenever chunk count changes.

### Confidence scoring (`src/rag/confidence.py`)

Three component scores + weighted overall (`confidence_weights` in
`config/rag_config.yaml`), because they fail differently:

| Component | Measures | Low means |
|---|---|---|
| `retrieval` | Mean cosine of strongest retrieved chunks | KB doesn't really cover this requirement |
| `grounding` | How much row vocabulary/identifiers trace to retrieved context | Model may have invented a param/block/API name |
| `similarity_to_existing` | Cosine vs. closest existing test case for same requirement | Genuinely new boundary case, or fabrication |

`similarity_to_existing` matches are reported by ID (e.g. `L2R7983_1`) for
direct reviewer comparison. When a requirement has no existing test cases,
that axis is dropped and its weight redistributed (not counted as zero).
Rows below `confidence_review_threshold` are flagged in the UI and the
`Needs_Review` export column.

### Output formats (`src/rag/exporters.py`)

`.xlsx`: sheet `Datasheet`, columns A–J exactly matching existing workbooks
(`S_no`, `Requirement`, `Description`, `Folder`, `Optimization_Technique`,
`Test_Type`, `Test_Technique`, `Retired?`, `Scorable`, `Comments`). Confidence
columns (`Confidence`, `Confidence_Retrieval`, `Confidence_Grounding`,
`Similarity_To_Existing`, `Closest_Existing_Case`, `Needs_Review`) are
**appended after J**, never inserted among them.

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
`static_context.py`), parameters/message names from `data/knowledge_base/`,
API names/kwargs from `data/python_apis/`. `data/Examples/` fixes shape only;
copying one of its concrete values into a new artefact is a defect. A
prompt can only obey "don't guess" if it actually contains a real source for
the values it asks for — this is why script generation runs its own
track-data and static-API passes rather than relying on the reference
scripts alone.

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

- Embedding cache keyed on exact text (`cache.py`) — nothing is embedded twice.
- BM25 scoring is memoised per query (`keyword_index.py`) — don't reintroduce
  repeated scans of the same query with only the metadata filter differing.
- Both arms of a retrieval pass (ChromaDB HNSW query, BM25 scan) run
  concurrently via the pool in `concurrency.py` (`RAG_RETRIEVAL_WORKERS`).
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
otherwise have written. Only `description` is required in the schema — `s_no`
and `requirement` are assigned by `parse_test_cases` and everything else has a
normalised default, so requiring them would only add ways to fail. The prompt
still describes the shape (a constrained model writes better JSON when it also
knows what the fields mean), the truncation salvage still applies, and an
inference server too old for structured output is retried once without the
constraint. **Script generation deliberately passes no schema** — it must
return Python.

Follow the same rule for any future model-dependent behaviour: probe the
server for the capability, default to `auto`, don't hard-code model names.

If generations are too slow: lower `llm_max_tokens`/`script_max_tokens`,
lower `retrieval_top_k`/`max_context_chars`, or switch `llm_model` to a
smaller variant in config — no code change needed for any of these. Script
generation is the slowest call in the app (it sends API stubs, track data
and data-dictionary context, then writes the longest output), so it is the
one that runs into the budget first.

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
| `config/config.yaml` | `sources` (embedded), `static_sources` (BM25-only), `examples_dir` (design-time only), chunk sizes, PDF heuristics, embedding model, ChromaDB location |
| `config/rag_config.yaml` | Hybrid-search weights/depth, context budget, static-context top-k, Ollama host/model/limits/timeouts, `max_test_cases` ceiling, confidence weights/threshold |
| `config/prompts.yaml` | Every system and user prompt |
| `config/caf_mapping.json` | Requirement → feature, i.e. the datasheet's `Folder` column — generated from `data/CAF.xlsx` by `scripts/convert_caf_mapping.py` |
| `.env` | Backend host/port, CORS origins, log level, `MAX_CONCURRENT_GENERATIONS`, `RAG_RETRIEVAL_WORKERS`, Ollama host override, frontend dev-server port/proxy target |

Prefer adding new behavior-affecting knobs to `config/*.yaml`, and new
per-machine values to `.env` — this split is intentional throughout the repo.

### Abstraction points

`EmbeddingModel` and `LLMClient` (in `src/rag/`) are protocols specifically so
a served embedding endpoint or a different inference server (vLLM, llama.cpp,
any OpenAI-compatible API) can replace either without touching the retrieval
or generation pipeline.

### API routes (`src/api/main.py`)

`GET /api/health`, `POST /api/requirements/upload`,
`POST /api/requirements/folder` (CAF folder lookup, no model),
`GET /api/track-subdivisions` (the subdivision picker's list, read off the
built static index so it can only offer track that actually parsed),
`POST /api/test-cases`
(datasheet generation), `POST /api/test-script` (script generation, second
LLM call), `POST /api/test-cases/export` and `POST /api/test-script/export`
(`.xlsx`/`.txt` downloads). `services.py` wires the shared retriever/generator
instances; `models.py` holds the request/response schemas.

### Layout

```
config/                     config.yaml, rag_config.yaml, prompts.yaml, caf_mapping.json
data/knowledge_base/         PDF · XLSX · TXT (embedded)
data/python_apis/            PY · PYI (static, BM25-only)
data/track_data/<subdiv>/    HTML report + -subdiv.xml (static, BM25-only)
data/Examples/               reference triples; design-time source for the prompts.yaml templates, not read at request time
data/CAF.xlsx                requirement -> feature (Folder) source; converted, not read at request time
scripts/convert_caf_mapping.py   data/CAF.xlsx -> config/caf_mapping.json
src/kb_ingestion/
  extractors/                 one parser per format (pdf, xlsx, python, html, xml, text)
  chunking.py                 normalisation, token budgets, chunk metadata
  incremental.py               fingerprint behind the unchanged-file skip
  embeddings.py                BGE-M3 wrapper
  vector_store.py              ChromaDB wrapper
  pipeline.py                   orchestration
src/rag/
  requirement_parser.py        requirement ID + functional area
  caf_mapping.py                requirement -> feature (the Folder column), from config/caf_mapping.json
  static_context.py             python_apis/track_data, BM25-only, never embedded
  keyword_index.py               BM25 arm over the embedded collection
  retriever.py                   hybrid search + RRF + context assembly
  cache.py                       embedding / result caches
  concurrency.py                 the retrieval thread pool
  prompts.py                     prompts.yaml loader
  schema.py                      datasheet rows, model-output parsing, truncation salvage
  confidence.py                  the three scores
  llm_client.py                  Ollama behind a protocol
  generator.py                   test-case and test-script generators
  exporters.py                   .xlsx and .txt
src/api/                     FastAPI app (main.py), request/response models, service container
frontend/                    React + Vite
scripts/run_ingestion.py     ingestion CLI
```
