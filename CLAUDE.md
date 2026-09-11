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
data/track_data/   .html      ──(html.parser)──┤   BM25-only, never embedded
data/Examples/     req → ref test cases → ref script triples ──┘ (labelled, budgeted)
                                                                  │
requirement ──→ requirement understanding ──→ hybrid retrieval ──┤
                                            (BGE-M3 dense + BM25, RRF)
                                                                  │
        context + prompts.yaml ──→ Qwen2.5-7B plan call (behaviour list)
                                    └──→ batched writing calls (4 each)
                                                                  │
                datasheet rows + confidence ──→ .xlsx
                                                                  │
                approved rows ──→ second generation ──→ test script ──→ .txt
```

Only `data/knowledge_base/` is embedded into ChromaDB. `data/python_apis/`
and `data/track_data/` are parsed with the same extractors but kept in an
in-process, BM25-only index (`src/rag/retrieval.py`) instead — they're
either one giant stub file or thousands of near-identical per-subdivision
rows, not prose an embedding model gains from, and both are full of exact
identifiers (API names, block numbers) BM25 already handles better than
dense search. `data/Examples/` is separate again: a handful of reference
triples sent to the LLM under a character budget, never searched or
ranked.

## Setup

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
# create .env yourself (gitignored, no committed template) — see the
# "--- ... ---" section comments in .env for what each key does
ollama pull qwen2.5:7b
set OLLAMA_NUM_PARALLEL=2    # so the writing batches can overlap
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
- **PDF** (`extractors.py`, PDF): heading hierarchy built while
  reading (numbering like `3.1.1`, font-size/bold fallback). A chunk never
  spans two sections; every prose chunk carries `section_path`. Tables become
  their own chunks. Running headers/footers and per-page admin tables are
  dropped as boilerplate.
- **XLSX** (`extractors.py`, XLSX): one row = one logical test case
  = one chunk (`xlsx_max_rows_per_chunk: 1`). `Requirement`/`S_no` columns
  become `requirement_id`/`test_case_id` metadata (what confidence scoring
  matches on).
- **PY/PYI** (`extractors.py`, PY/PYI): module → class → method; a
  method is only split if it alone busts the token budget.
- **HTML** (`extractors.py`, HTML): subdivision → named section
  (Blocks, Switches, Signals, Speed Restrictions) → track-feature records,
  tagged with the numeric subdivision ID.
- **TXT** (`extractors.py`, TXT): plain-text documents in
  `knowledge_base/`.

Key files: `chunking.py` (normalisation, token budgets, metadata),
`storage.py` (BGE-M3 wrapper + ChromaDB wrapper), `pipeline.py`
(orchestration, and the fingerprint behind the unchanged-file skip).

### Static context: python_apis, track_data, and Examples (`src/rag/retrieval.py`)

`python_apis/` and `track_data/` are declared under `static_sources:` in
`config/config.yaml`, parsed with the same extractors as above, but held in
an in-process BM25-only index instead of ChromaDB — see the file's header
comment for the reasoning. `static_api_top_k`/`static_track_top_k` in
`config/rag_config.yaml` control how many chunks of each are pulled per
request.

Track data is searched across every ingested subdivision for every
requirement — there is no requirement → subdivision mapping. BM25 ranking
over the query surfaces whichever subdivision's rows actually match; the
matched chunk's `subdivision` metadata is still kept for citation and is
reported back as `track_subdivisions` so a reviewer can see which
subdivision(s) a generation actually drew values from.

`data/Examples/` (config `examples_dir`) holds requirement → reference test
cases → reference script triples for a handful of other requirements,
included in every generation request — not searched, not ranked. They exist
to fix the *shape* of the output (datasheet phrasing, script layout, naming
style), never to supply values: see "Examples are templates, never a source
of values" below.

They are **split by call and budgeted**, and both halves of that matter.
The datasheet prompt gets only the reference *test cases*
(`examples_test_case_max_chars`); the script prompt gets only the reference
*scripts* (`examples_script_max_chars`), each example taking an equal share
truncated on a line boundary. Unbudgeted, the folder is ~157,000 characters
(~39k tokens) against `llm_num_ctx`, and Ollama resolves an over-long prompt
by dropping the *start* of it — the requirement, the retrieved context, and
the rules forbidding value reuse. The visible symptom is every requirement
producing the same unrelated test cases, copied from whichever example
survived at the tail. `OllamaClient` now logs a loud warning whenever an
assembled prompt is estimated to exceed the window, since Ollama itself
reports nothing. The script call has its own wider window
(`script_num_ctx`), because it also carries reference scripts and the API
surface.

The three context blocks in each prompt are wrapped in `=== LABEL ===`
banners (`_labelled_blocks` in `generator.py`) naming what each may be used
for — the system prompts' rules refer to them by name, and unlabelled, the
most output-shaped text in the prompt (the examples) simply wins.

### Hybrid retrieval (`src/rag/retrieval.py`)

Two independent rankings over the embedded `knowledge_base` collection,
fused with reciprocal rank fusion (RRF, combines by rank not score):
- **Dense** (BGE-M3 cosine over ChromaDB) — catches paraphrase.
- **Keyword** (BM25) — catches exact identifiers dense search blurs (`TBC137`
  vs `TBC139`, block `1015` vs `1025`, API names).

`dense_weight`/`keyword_weight` in `config/rag_config.yaml` tune the balance;
0 disables an arm. BM25 index is built in-process from the collection and
rebuilt whenever chunk count changes.

### Confidence scoring (`src/rag/schema.py`)

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

The model's answer is passed through `_extract_python` in `generator.py`
before it is exported. A small instruct model told "output only Python"
still, fairly often, narrates a plan first ("We are given a requirement…",
a numbered list of the steps it intends, a discussion of which values it
could not find) and only then writes the script — or spends the whole
output budget narrating and never reaches it. `_extract_python` drops
`<think>` blocks, prefers a fenced block when the model marked one, and
otherwise cuts everything before the first line that looks like the start
of a Python module and any trailing run of unindented prose. It salvages
rather than rejects, because rejecting costs another multi-minute
generation; a response with no Python in it at all is exported unchanged
(with a warning) so the reviewer can see what happened. The prompt fights
the same problem from the other side: the system prompt names the first
line the reply must have, and the user prompt repeats it last.

### Examples are templates, never a source of values

Every concrete value in a generated artefact — subdivision, block, milepost,
switch, signal, TBC parameter, API name — must come from a real source for
*that* requirement: track values from `data/track_data/` (via
`retrieval.py`), parameters/message names from `data/knowledge_base/`,
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

### Test-case generation is two calls, batched

`TestCaseGenerator.generate` runs a **plan** call then one **writing** call per
`test_case_batch_size` behaviours. The plan call (`test_case_plan` in
prompts.yaml) sees no examples and emits no JSON — just `N|type|behaviour`
lines against a six-point coverage checklist, at `plan_temperature` (hotter:
this step wants breadth). Each writing call (`test_cases`) writes rows for one
slice of that list under `llm_max_tokens`, which is therefore a *per-batch*
budget, not a per-requirement one.

This exists because coverage was being lost to truncation, not to the model
running out of ideas: one call carrying ~20 negative constraints, a 6000-char
example block and a whole requirement's worth of JSON ran past `num_predict`
and had its tail dropped by the salvage path in `schema.py`. Things to
preserve when touching this:

- **`format` (Ollama structured output).** `TEST_CASE_JSON_SCHEMA` in
  `generator.py` is sent as `format` on the writing calls, so decoding cannot
  produce invalid JSON. Never set it on the script call — a grammar over
  free-form Python does nothing useful.
- **Constant fields are Python's job.** The model returns `scenario` /
  `conditions` / `verify` / `test_type` / `test_technique` / `comments` only;
  `schema.py` assembles the house description layout and fills in `Folder`,
  `Optimization_Technique`, `Retired?` and `Scorable`. Adding a field back to
  the prompt costs output budget that would otherwise be a test case.
  `parse_test_cases` still accepts a pre-composed `description`, so an older
  prompts.yaml keeps working.
- **Prefix order.** In the writing prompt the static blocks (wording template,
  then retrieved context) come first and the per-batch instruction last, so
  Ollama reuses the KV cache across batches. Moving the requirement back to
  the top would undo it.
- **Degrading, not failing.** An unparseable plan falls back to one
  self-enumerating writing call; one unparseable batch is logged and the rest
  are kept. `_parse_behaviours` is deliberately forgiving for the same reason.
- **Batches run concurrently.** `writing_parallelism` (default 2) bounds how
  many writing calls are in flight; the batches are independent, so this
  shortens a request without sending fewer tokens. One call's prefill
  (compute-bound) overlaps another's decode (memory-bound), which is where
  the gain comes from — not from real parallelism, since the slots share the
  same cores and Ollama splits the shared-prefix KV cache across them. It
  needs `OLLAMA_NUM_PARALLEL` set to at least the same number in the
  environment `ollama serve` runs in; below that the requests queue
  server-side and nothing changes. `s_no` is assigned after every batch is
  in, since a concurrent batch cannot know how many rows preceded it, and a
  call that fails outright is now logged and skipped like an unparseable
  one.

### No "how many test cases" control — deliberate

How many cases a requirement needs is a property of the requirement (one per
verifiable behaviour), not something a user can guess beforehand. The model
enumerates the behaviours stated and writes one case per behaviour.
`max_test_cases` in `config/rag_config.yaml` caps how many behaviours the plan
call may enumerate — a ceiling, not a target. Don't turn it into a
user-facing "number of cases" input.

### Prefill is the wall — measure before tuning

On a CPU-only box this pipeline is **prefill-bound, not decode-bound**.
Measured on the 6-core/12-thread machine it was developed on
(`scripts/bench_generation.py`, 1,977-token prompt, qwen2.5:7b q4_K_M):

| Setting | Prefill | Decode |
|---|---|---|
| `num_thread=4` | 14.8 tok/s | 7.5 tok/s |
| `num_thread=6` (physical cores) | 15.9 tok/s | 6.1 tok/s |
| Ollama's own default | 16.6 tok/s | 8.5 tok/s |
| **`num_thread=12` (logical)** | **20.4 tok/s** | 7.6 tok/s |
| qwen2.5:3b, `num_thread=6` | 37.6 tok/s | 14.2 tok/s |

Two things follow, and both contradict advice that sounds right:

- **One thread per logical processor beat one per physical core by 28%.**
  The "SMT over-subscribes the cores" rule of thumb is about memory-bound
  decode; prefill is compute-bound matrix work that uses both siblings.
  `llm_num_threads: auto` therefore resolves to `os.cpu_count()`. Re-measure
  on different hardware rather than reasoning about it.
- **A prompt token costs ~3× what an output token costs**, at ~50s per
  1,000 prompt tokens. So the cheapest available speedup is always to send
  fewer prompt tokens, and every block in a prompt needs a character
  budget — including the ones that look small.

That last point is why `static_track_max_chars` and `static_api_max_chars`
exist. Track-data chunks are dense rows of identifiers that tokenize at
~2.3 characters per token where prose runs at ~4: four unbudgeted track
chunks measured 4,796 characters but **2,086 tokens**, larger than the
retrieved knowledge base, the requirement and the system prompt combined,
and ~110s of every call. `top_k` cannot bound that, because chunk size is a
property of the subdivision. When adding a new context block, budget it in
characters and check what it costs in *tokens* — the two are not
proportional across this corpus.

The plan call gets its own, much smaller slice (`plan_context_max_chars`,
knowledge base only). Enumerating which behaviours exist needs the
requirement's vocabulary, not its values, so track data is left out of that
call entirely.

`plan_model` can point the enumeration call at a smaller model
(`qwen2.5:3b` measured 2.4× faster). It is `null` by default: coverage is
the thing the plan call exists to protect, and trading it for ~50s is not
obviously right. Enable it if wall-clock matters more, and check the
behaviour lists afterwards.

### Performance design (CPU-bound 7B model)

The generation call is the floor on request latency; everything else is
arranged to not add to it. When touching `src/rag/` or `src/api/`, preserve these:

- Embedding cache keyed on exact text (`utils.py`) — nothing is embedded twice.
- BM25 scoring is memoised per query (`retrieval.py`) — don't reintroduce
  repeated scans of the same query with only the metadata filter differing.
- Both arms of a retrieval pass (ChromaDB HNSW query, BM25 scan) run
  concurrently via the pool in `utils.py` (`RAG_RETRIEVAL_WORKERS`).
- Generation endpoints in `src/api/main.py` are `async`; a semaphore
  (`MAX_CONCURRENT_GENERATIONS`, default 1) serializes actual generations
  since a CPU-bound model does not get faster with concurrent requests — it
  slows both down. Raise this default only where the model has headroom
  (GPU, remote Ollama).
- A truncated LLM response (ran out of output tokens mid-array) is salvaged
  in `src/rag/schema.py`: completed test cases are recovered, the incomplete
  tail dropped. Batched writing plus `format` should keep this from firing;
  it stays as the safety net.
- The script call reuses the datasheet call's retrieval for its parameter
  context — same requirement query, so it is a cache hit in `retrieval.py`
  rather than another encode plus two ranking passes. Its API and track
  passes still use the wider query (requirement + approved rows).
- `OllamaClient` streams (`stream: true`) and logs `prompt_eval_count` /
  `prompt_eval_duration` / `eval_count` / `eval_duration` from the final
  chunk. Prefill-bound and decode-bound slowness have opposite fixes and
  those counters are the only way to tell them apart.
- `llm_num_threads: auto` resolves to one thread per physical core
  (`GenerationConfig.resolved_num_threads`); Ollama's own default counts
  logical processors and over-subscribes SMT machines.
- `llm_keep_alive: 30m` in `config/rag_config.yaml` keeps model weights
  resident between requests.

If generations are too slow: lower `llm_max_tokens` (per batch) or
`test_case_batch_size`, lower `retrieval_top_k`/`max_context_chars`, or switch
`llm_model` to a smaller Qwen2.5 variant in config — no code change needed for
any of these. Not yet done, in rough order of payoff if it is still too slow:
run the plan call on `qwen2.5:3b` while the writing calls stay on 7B, cache
the plan per requirement ID (done), and fire the writing batches
concurrently (done — `writing_parallelism`; raising it past 2 wants
measurement, since it splits the shared-prefix cache across slots).
`examples_script_max_chars: 8000` is the largest single block in the script
prompt and the next thing to trim if that call in particular is slow.

### Configuration surface

| File | Controls |
|---|---|
| `config/config.yaml` | `sources` (embedded), `static_sources` (BM25-only), `examples_dir`, chunk sizes, PDF heuristics, embedding model, ChromaDB location |
| `config/rag_config.yaml` | Hybrid-search weights/depth, context budget, static-context top-k **and character budgets**, Ollama host/model/limits, plan-call model/budget/temperature/context, `test_case_batch_size`, `writing_parallelism`, `max_test_cases` ceiling, confidence weights/threshold |
| `config/prompts.yaml` | Every system and user prompt |
| `.env` | Backend host/port, CORS origins, log level, `MAX_CONCURRENT_GENERATIONS`, `RAG_RETRIEVAL_WORKERS`, Ollama host override, frontend dev-server port/proxy target |

Prefer adding new behavior-affecting knobs to `config/*.yaml`, and new
per-machine values to `.env` — this split is intentional throughout the repo.

### Abstraction points

`EmbeddingModel` and `LLMClient` (in `src/rag/`) are protocols specifically so
a served embedding endpoint or a different inference server (vLLM, llama.cpp,
any OpenAI-compatible API) can replace either without touching the retrieval
or generation pipeline.

### API routes (`src/api/main.py`)

`GET /api/health`, `POST /api/requirements/upload`, `POST /api/test-cases`
(datasheet generation: the plan call plus one writing call per batch, behind
one request), `POST /api/test-script` (script generation, second
LLM call), `POST /api/test-cases/export` and `POST /api/test-script/export`
(`.xlsx`/`.txt` downloads). `services.py` wires the shared retriever/generator
instances and holds the request/response schemas.

### Layout

```
config/                     config.yaml, rag_config.yaml, prompts.yaml
data/knowledge_base/         PDF · XLSX · TXT (embedded)
data/python_apis/            PY · PYI (static, BM25-only)
data/track_data/              HTML (static, BM25-only)
data/Examples/               requirement -> reference test cases -> reference script triples
src/kb_ingestion/
  extractors.py               one parser per format (pdf, xlsx, python, html, text)
  chunking.py                 normalisation, token budgets, chunk metadata
  storage.py                  BGE-M3 wrapper + ChromaDB wrapper
  pipeline.py                 orchestration + the unchanged-file fingerprint
src/rag/
  config.py                   rag_config.yaml settings + prompts.yaml loader
  utils.py                    embedding / result caches, the retrieval thread pool
  retrieval.py                BM25 arm, hybrid search + RRF, static context (python_apis,
                              track_data, Examples - BM25-only, never embedded)
  schema.py                   datasheet rows, model-output parsing + salvage, confidence
  llm_client.py               Ollama behind a protocol
  generator.py                requirement parsing, test-case and test-script generators
  exporters.py                .xlsx and .txt
src/api/
  main.py                     FastAPI app: routes and the entry point
  services.py                 request/response schemas + the service container
frontend/                    React + Vite
scripts/run_ingestion.py     ingestion CLI
scripts/bench_generation.py  prefill/decode rates per thread count and model
```
