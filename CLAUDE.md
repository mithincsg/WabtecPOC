# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Turns a requirement into a review-ready test-case datasheet (`.xlsx`) and an automation script draft (`.txt`), grounded in existing documents via RAG:

```
                  ┌── data/knowledge_base/  PDF · XLSX · TXT   (PyMuPDF, openpyxl)
ingestion ────────┼── data/python_apis/     PYI · PY           (ast)
                  └── data/track_data/      HTML               (html.parser)
                            │
              structure-aware + hierarchy chunking → BGE-M3 → ChromaDB
                            │
requirement ──→ requirement understanding ──→ hybrid retrieval ──→ context
                                              (BGE-M3 dense + BM25, RRF)
                            │
              context + prompts.yaml ──→ Qwen2.5-7B-Instruct (Ollama)
                            │
              datasheet rows + confidence ──→ .xlsx
                            │
              approved rows ──→ second generation ──→ test script ──→ .txt
```

## Setup

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                            # then adjust for your machine

ollama pull qwen2.5:7b-instruct
ollama serve
```

`.env` (gitignored) holds per-machine values (host/port, CORS, log level, Ollama
host, frontend dev-server port/proxy). Everything else — chunking, retrieval
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
are both unchanged since last embedded — no separate state file. Changing an
extractor bumps the fingerprint and re-embeds the whole KB once.

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

Run tests:

```bash
pytest -q
```

53 tests, no model download and no live Ollama needed — embedding model and
LLM are replaced with deterministic fakes; retrieval runs against a real
temporary ChromaDB. Extractor tests run against the actual documents in
`data/`, so a parsing regression on real PDFs/workbooks fails the suite.
`pythonpath = src` is set in `pytest.ini`, so imports in tests are like
`from rag.retriever import ...` / `from api.main import ...`, not `from src...`.
Run a single test file/case the normal pytest way, e.g. `pytest tests/test_retrieval.py -k some_case`.
The `slow` marker denotes tests needing the real bge-m3 model or a live Ollama server.

## Architecture

### Ingestion (`src/kb_ingestion/`)

Three source roots (declared in `config/config.yaml`, not code), each parsed
by the extractor suited to its format:

| Folder | Holds | Parsed with |
|---|---|---|
| `data/knowledge_base/` | Requirement specs, reference datasheets, reference scripts, data dictionaries, config guides | PyMuPDF, openpyxl, plain text |
| `data/python_apis/` | WCR test-automation API stubs | stdlib `ast` |
| `data/track_data/` | One HTML subdivision report per subdivision | stdlib `html.parser` |

Every chunk is stamped with a `document_type` from its root; `subfolders_as_type: true`
lets an immediate subfolder override it (e.g. `knowledge_base/reference_test_cases/`
becomes its own filterable type, no code change).

Chunking preserves structure — nothing unrelated ever shares a chunk:
- **PDF**: heading hierarchy built while reading (numbering like `3.1.1`, font-size/bold
  fallback). A chunk never spans two sections; every prose chunk carries `section_path`.
  Tables become their own chunks. Running headers/footers and per-page admin tables are
  dropped as boilerplate.
- **XLSX**: one row = one logical test case = one chunk (`xlsx_max_rows_per_chunk: 1`).
  `Requirement`/`S_no` columns become `requirement_id`/`test_case_id` metadata (what
  confidence scoring matches on).
- **PY/PYI**: module → class → method; a method is only split if it alone busts the
  token budget.
- **HTML**: subdivision → named section (Blocks, Switches, Signals, Speed Restrictions)
  → track-feature records, tagged with the numeric subdivision ID.

Key files: `chunking.py` (normalisation, token budgets, metadata), `incremental.py`
(fingerprint behind the unchanged-file skip), `embeddings.py` (BGE-M3 wrapper),
`vector_store.py` (ChromaDB wrapper), `pipeline.py` (orchestration), `extractors/`
(one parser per format).

### Track data is mapped, not searched blindly

`config/track_mapping.yaml` says which subdivisions a requirement applies to
(e.g. `L2R9479: ["08880"]`); `unmapped: exclude` (or `all`) controls the
fallback. `8880`/`08880` both work, and `L2R9479_A` falls back to `L2R9479`.
Track data is retrieved in its own filtered pass, appended after general
results — it never crowds out requirement prose on score alone.

### Hybrid retrieval (`src/rag/retriever.py`, `keyword_index.py`)

Two independent rankings fused with reciprocal rank fusion (RRF, combines by
rank not score):
- **Dense** (BGE-M3 cosine over ChromaDB) — catches paraphrase.
- **Keyword** (BM25) — catches exact identifiers dense search blurs (`TBC137`
  vs `TBC139`, block `1015` vs `1025`, API names). Not optional — a wrong
  parameter number makes a test case useless.

`dense_weight`/`keyword_weight` in `config/rag_config.yaml` tune the balance;
0 disables an arm. BM25 index is built in-process from the collection and
rebuilt whenever chunk count changes.

### Confidence scoring (`src/rag/confidence.py`)

Three component scores + weighted overall, because they fail differently:

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
columns are **appended after J**, never inserted among them (`include_confidence: false`
to omit). On screen, Confidence is reordered to sit second (after `S_no`) since
it's what reviewers triage on — only the view is reordered, not the export.

`.txt` script follows the existing script pattern (encoding comment,
proprietary docstring, `from Common.WCR_public import *`, `main()` with the
test-case function table, `handle_test_case` with one `if case == N` branch
per row), with a header naming the requirement and generation time. Where a
value isn't derivable from context, the model is instructed to leave a named
variable and `# TODO:` rather than guess. Script generation is a **second**
LLM call, run against the rows the reviewer kept (including edits) — not the
model's first draft.

### Examples are templates, never a source of values

`data/Examples/` (config `examples_dir`) holds requirement → reference
test cases → reference script triples for a handful of other requirements.
They exist to fix the *shape* of the output: datasheet phrasing, script
layout, naming style. Every concrete value in them — subdivision, block,
milepost, switch, signal, TBC parameter — belongs to that example's
requirement, and copying one into a new artefact is a defect, not a
shortcut.

Both prompts therefore label their context blocks (`=== TRACK DATA ===`,
`=== PARAMETERS AND DATA DICTIONARY ===`, `=== SHAPE TEMPLATE ===`) and
state which block each kind of token may come from. Keep that separation
when editing `config/prompts.yaml`: an unlabelled concatenation reads to the
model as one undifferentiated pile of context, and the examples — being the
most fluent, most script-shaped text in the prompt — win.

Two structural rules hold this up, and both matter:

- **Every prompt must contain a real source for the values it asks for.**
  Script generation runs its own track-data pass and its own knowledge-base
  pass (`script_kb_top_k`) for exactly this reason. Before those existed,
  the reference scripts were the *only* place a subdivision or TBC value
  appeared anywhere in the script prompt, so "don't guess" was unsatisfiable
  and the model copied.
- **Examples are budgeted, not sent whole** (`example_max_chars`, and
  `script_example_top_k` for how many reference scripts). The test-case
  prompt gets the example requirements and their reference *test cases*; the
  script prompt gets the reference *script* whose subject matter best matches
  the requirement (ranked by a small BM25 index over the examples). Sending
  all four full scripts to both prompts is ~105 KB and overran `llm_num_ctx`,
  and Ollama trims an over-long prompt from the **start** — silently dropping
  the system rules that forbid copying from those examples.

When track values come out as `# TODO:` variables, check
`config/track_mapping.yaml` against `data/track_data/` before touching the
prompt: a requirement mapped to a subdivision with no report on disk has no
track context to ground anything in, which is the correct outcome for a
missing input, not a prompt bug.

### Prompts (`config/prompts.yaml`)

Every system/user prompt lives here. Edits take effect on the next request —
no restart, no code change. Placeholders are `$name` (`string.Template`), not
`{name}`, so JSON braces inside prompts need no escaping.

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
  Reference test cases used in confidence scoring never change, so they must
  not be re-embedded per request.
- BM25 `get_scores` is memoised per query — don't reintroduce repeated scans
  of the same query with only the metadata filter differing.
- Both arms of a retrieval pass (ChromaDB HNSW query, BM25 scan) run
  concurrently via the pool in `concurrency.py`.
- Identical requirements are served from a one-hour cache (`cached: true` in
  the response); cache key includes model + mtime of `config/prompts.yaml`.
- Generation endpoints are `async`; a semaphore (`MAX_CONCURRENT_GENERATIONS`,
  default 1) serializes actual generations since a CPU-bound model does not
  get faster with concurrent requests — it slows both down. Raise this default
  only where the model has headroom (GPU, remote Ollama).
- A truncated LLM response (ran out of output tokens mid-array) is salvaged:
  completed test cases are recovered, the incomplete tail dropped.
- `llm_keep_alive: 30m` in `config/rag_config.yaml` keeps model weights
  resident between requests. The server also warms up the embedding model and
  LLM on a background thread at startup.

If generations are too slow: lower `llm_max_tokens`, lower `retrieval_top_k`/
`max_context_chars`, or switch `llm_model` to `qwen2.5:3b-instruct` in config
— no code change needed for any of these.

### Configuration surface

| File | Controls |
|---|---|
| `config/config.yaml` | Source folders, chunk sizes, PDF heuristics, embedding model, ChromaDB location |
| `config/rag_config.yaml` | Hybrid-search weights/depth, context budget, static/script retrieval depths (`script_kb_top_k`, `script_example_top_k`, `example_max_chars`), Ollama host/model/limits, `max_test_cases` ceiling, confidence weights/threshold |
| `config/prompts.yaml` | Every system and user prompt |
| `config/track_mapping.yaml` | Requirement → subdivision mapping |
| `.env` | Backend host/port, CORS origins, log level, Ollama host override, `MAX_CONCURRENT_GENERATIONS`, frontend dev-server port/proxy target |

Prefer adding new behavior-affecting knobs to `config/*.yaml`, and new
per-machine values to `.env` — this split is intentional throughout the repo.

### Abstraction points

`EmbeddingModel` and `LLMClient` (in `src/rag/`) are protocols specifically so
a served embedding endpoint or a different inference server (vLLM, llama.cpp,
any OpenAI-compatible API) can replace either without touching the retrieval
or generation pipeline.

### Layout

```
config/                     config.yaml, rag_config.yaml, prompts.yaml, track_mapping.yaml
data/knowledge_base/        PDF · XLSX · TXT
data/python_apis/           PYI · PY
data/track_data/            HTML
src/kb_ingestion/
  extractors/               one parser per format
  chunking.py               normalisation, token budgets, chunk metadata
  incremental.py            fingerprint behind the unchanged-file skip
  embeddings.py             BGE-M3 wrapper
  vector_store.py           ChromaDB wrapper
  pipeline.py               orchestration
src/rag/
  requirement_parser.py     requirement ID + functional area
  track_mapping.py          requirement → subdivision
  keyword_index.py          BM25 arm
  retriever.py              hybrid search + RRF + context assembly
  cache.py                  embedding / result caches
  concurrency.py            the retrieval thread pool
  prompts.py                prompts.yaml loader
  schema.py                 datasheet rows, model-output parsing
  confidence.py             the three scores
  llm_client.py             Ollama behind a protocol
  generator.py              test-case and test-script generators
  exporters.py              .xlsx and .txt
src/api/                    FastAPI app, models, service container
frontend/                   React + Vite
scripts/run_ingestion.py    ingestion CLI
tests/                      pytest suite
```
