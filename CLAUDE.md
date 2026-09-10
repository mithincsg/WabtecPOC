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
data/Examples/     req → ref test cases → ref script triples ──┘ (always sent in full)
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
dense search. `data/Examples/` is separate again: a handful of reference
triples always sent to the LLM in full, never searched or ranked.

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
  tagged with the numeric subdivision ID.
- **TXT** (`extractors/text_extractor.py`): plain-text documents in
  `knowledge_base/`.

Key files: `chunking.py` (normalisation, token budgets, metadata),
`incremental.py` (fingerprint behind the unchanged-file skip),
`embeddings.py` (BGE-M3 wrapper), `vector_store.py` (ChromaDB wrapper),
`pipeline.py` (orchestration).

### Static context: python_apis, track_data, and Examples (`src/rag/static_context.py`)

`python_apis/` and `track_data/` are declared under `static_sources:` in
`config/config.yaml`, parsed with the same extractors as above, but held in
an in-process BM25-only index instead of ChromaDB — see the file's header
comment for the reasoning. `static_api_top_k`/`static_track_top_k` in
`config/rag_config.yaml` control how many chunks of each are pulled per
request.

Track data is additionally mapped, not searched blindly:
`config/track_mapping.yaml` says which subdivisions a requirement applies to
(e.g. `L2R9479: ["08880"]`); `unmapped: exclude` (or `all`) controls the
fallback. `8880`/`08880` both work, and `L2R9479_A` falls back to `L2R9479`.

`data/Examples/` (config `examples_dir`) holds requirement → reference test
cases → reference script triples for a handful of other requirements, always
included in full for every generation request — not searched, not budgeted,
small enough that ranking would add nothing. They exist to fix the *shape*
of the output (datasheet phrasing, script layout, naming style), never to
supply values: see "Examples are templates, never a source of values" below.

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
  tail dropped.
- `llm_keep_alive: 30m` in `config/rag_config.yaml` keeps model weights
  resident between requests.

If generations are too slow: lower `llm_max_tokens`, lower `retrieval_top_k`/
`max_context_chars`, or switch `llm_model` to a smaller Qwen2.5 variant in
config — no code change needed for any of these.

### Configuration surface

| File | Controls |
|---|---|
| `config/config.yaml` | `sources` (embedded), `static_sources` (BM25-only), `examples_dir`, chunk sizes, PDF heuristics, embedding model, ChromaDB location |
| `config/rag_config.yaml` | Hybrid-search weights/depth, context budget, static-context top-k, Ollama host/model/limits, `max_test_cases` ceiling, confidence weights/threshold |
| `config/prompts.yaml` | Every system and user prompt |
| `config/track_mapping.yaml` | Requirement → subdivision mapping |
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
(datasheet generation), `POST /api/test-script` (script generation, second
LLM call), `POST /api/test-cases/export` and `POST /api/test-script/export`
(`.xlsx`/`.txt` downloads). `services.py` wires the shared retriever/generator
instances; `models.py` holds the request/response schemas.

### Layout

```
config/                     config.yaml, rag_config.yaml, prompts.yaml, track_mapping.yaml
data/knowledge_base/         PDF · XLSX · TXT (embedded)
data/python_apis/            PY · PYI (static, BM25-only)
data/track_data/              HTML (static, BM25-only)
data/Examples/               requirement -> reference test cases -> reference script triples
src/kb_ingestion/
  extractors/                 one parser per format (pdf, xlsx, python, html, text)
  chunking.py                 normalisation, token budgets, chunk metadata
  incremental.py               fingerprint behind the unchanged-file skip
  embeddings.py                BGE-M3 wrapper
  vector_store.py              ChromaDB wrapper
  pipeline.py                   orchestration
src/rag/
  requirement_parser.py        requirement ID + functional area
  track_mapping.py              requirement -> subdivision
  static_context.py             python_apis/track_data/Examples, BM25-only, never embedded
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
