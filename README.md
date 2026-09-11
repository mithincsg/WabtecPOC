# PTC Test Case Generator

Turns a requirement into a review-ready test-case datasheet and an automation
script draft, grounded in your existing documents.

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

`.env` (gitignored; `.env.example` is the checked-in template) holds the
values that differ per machine rather than per deployment behaviour: the
backend host/port, allowed CORS origins, log level, the Ollama host, and the
frontend dev-server port and proxy target. Everything else — chunking,
retrieval weights, prompts, confidence thresholds — stays in `config/*.yaml`.

The first ingestion run downloads `BAAI/bge-m3` from Hugging Face (~2 GB), so
the machine needs internet access at least once.

## Build the knowledge base

Drop documents into the three folders, then:

```bash
python scripts/run_ingestion.py --dry-run   # parse only: no model, no writes
python scripts/run_ingestion.py             # embed and write to ChromaDB
python scripts/run_ingestion.py --prune     # forget files you deleted
python scripts/run_ingestion.py --reset     # start over
```

Re-running skips any file whose bytes **and** whose processing code are both
unchanged since it was last embedded. Both are checked against the
`file_hash`/`pipeline_fingerprint` stamped on that file's chunks in ChromaDB,
so there is no separate state file to fall out of sync. Fixing an extractor
changes the fingerprint for every file, so the next run re-embeds the whole
knowledge base once; runs after that go back to touching only what changed

## Run it

Two processes:

```bash
python src/api/main.py
```

This reads `BACKEND_HOST`/`BACKEND_PORT` from `.env` and checks the port is
actually free before handing it to uvicorn, stepping to the next one if not
— a leftover process on the port (or a Windows-reserved range from
Hyper-V/WSL2/Docker) otherwise fails with an opaque `WinError 10013`. For
auto-reload during development, use the uvicorn CLI instead (it does not get
this fallback, so pick a free port yourself if it fails to bind):

```bash
uvicorn api.main:app --app-dir src --reload --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173. Paste a requirement (or load a `.txt`),
generate, review the rows, download the `.xlsx`, then generate the script and
download the `.txt`.

There is no "how many test cases" control, deliberately. How many cases a
requirement needs is a property of the requirement — one per verifiable
behaviour it states — and it is not something a user can sensibly guess at
before seeing the result. The model is asked to enumerate the behaviours the
requirement specifies and write one case for each: a requirement naming a
single condition yields a single case, one naming six target types with a
boundary each yields twelve. `max_test_cases` in `config/rag_config.yaml` is a
ceiling on that enumeration, not a target.

Datasheet generation is itself **two calls** (see "Two-stage test-case
generation" below): one that enumerates the behaviours, then one per batch of
four that writes their rows.

## The three source folders

| Folder | Holds | Parsed with |
|---|---|---|
| `data/knowledge_base/` | Requirement specs, reference datasheets, reference scripts, data dictionaries, configuration guides | PyMuPDF, openpyxl, plain text |
| `data/python_apis/` | The WCR test-automation API stubs | stdlib `ast` |
| `data/track_data/` | One HTML subdivision report per subdivision | stdlib `html.parser` |

Roots are declared in `config/config.yaml`, not in code. Every chunk is
stamped with a `document_type` taken from its root; set
`subfolders_as_type: true` on a root and an immediate subfolder name
overrides it, which is how `knowledge_base/reference_test_cases/` becomes its
own filterable type without a code change.

### Track data is searched across every subdivision

Each subdivision report lists thousands of track features. There is no
requirement → subdivision mapping — every requirement searches every
ingested subdivision, and BM25 ranking over the query surfaces whichever
subdivision's rows actually match. Track data is retrieved in its own
filtered pass (`document_type: track_data`) and appended after the general
results, so it can never crowd out requirement prose on score alone. The
matched chunk's `subdivision` metadata is kept for citation and reported
back as `track_subdivisions`, so a reviewer can see which subdivision(s) a
generation actually drew values from.

## How chunking preserves structure

Nothing unrelated ever shares a chunk.

- **PDF** — a heading hierarchy is built while reading (numbering like `3.1.1`,
  with font-size and bold tiers as the fallback for unnumbered titles). A
  chunk never spans two sections, and every prose chunk carries its
  `section_path` (`"3 Data Records > 3.1 Record Header"`). Tables become their
  own chunks, tagged with their section and detected title. Running
  headers/footers and per-page admin tables are dropped as boilerplate, and a
  ruled box that isn't really tabular (a framed "intentionally left blank"
  page) is rejected rather than emitted as a table.
- **XLSX** — one row is one logical test case is one chunk
  (`xlsx_max_rows_per_chunk: 1`), so a retrieved reference test case is always
  whole. The `Requirement` and `S_no` columns become `requirement_id` and
  `test_case_id` metadata, which is what confidence scoring matches on.
- **PY/PYI** — module → class → method. A class with methods becomes a compact
  header chunk (signature and docstring) plus one chunk per method; a method
  is never split unless it alone busts the token budget.
- **HTML** — subdivision → named section (Blocks, Switches, Signals, Speed
  Restrictions) → track-feature records, one chunk per section, tagged with
  the numeric subdivision ID.

## Hybrid retrieval

Two independent rankings, fused with reciprocal rank fusion:

- **Dense** (BGE-M3 cosine over ChromaDB) catches paraphrase — a requirement
  saying "most restrictive of" finds a test case written as "lowest applicable
  speed".
- **Keyword** (BM25) catches the exact identifiers dense search blurs
  together: `TBC137` vs `TBC139`, block `1015` vs `1025`, an API name like
  `wcr_loco_sim.set_position`. A wrong parameter number makes a test case
  useless, so this arm is not optional.

RRF combines the two by rank rather than score, so a BM25 score of 14.2 and a
cosine similarity of 0.71 can be merged without pretending they are
comparable numbers. `dense_weight` and `keyword_weight` in
`config/rag_config.yaml` tune the balance; set either to 0 to disable that arm.

The BM25 index is built in process from the collection and rebuilt whenever
the chunk count changes, so it follows ingestion automatically.

## Confidence scoring

Every generated row gets three component scores and a weighted overall score.
Components rather than one number, because they fail differently:

| Component | What it measures | Low means |
|---|---|---|
| `retrieval` | Mean cosine of the strongest retrieved chunks | Nothing in the knowledge base really covers this requirement |
| `grounding` | How much of the row's vocabulary — and especially its identifiers — traces back to the retrieved context | The model may have invented a parameter, block or API name |
| `similarity_to_existing` | Cosine of the row against the closest **existing** test case for the same requirement | Either a genuinely new boundary case, or fabrication |

`similarity_to_existing` is the strongest signal available, because a high
score means the model reproduced something a human already wrote and signed
off; the ID it matched (`L2R7983_1`) is reported so a reviewer can compare
directly. It is not a verdict on its own — a real new boundary case scores low
— which is why `grounding` sits beside it. When a requirement has no existing
test cases, that axis is dropped and its weight redistributed rather than
counted as zero.

Rows below `confidence_review_threshold` are flagged in the UI and in the
`Needs_Review` export column.

## Output formats

The exported `.xlsx` is sheet `Datasheet` with columns A–J exactly as in your
existing workbooks — `S_no`, `Requirement`, `Description`, `Folder`,
`Optimization_Technique`, `Test_Type`, `Test_Technique`, `Retired?`,
`Scorable`, `Comments` — so it drops into the existing execution flow
unchanged. Confidence columns are **appended** after J, never inserted among
them; pass `include_confidence: false` to omit them.

On screen, Confidence sits second, right after `S_no`. With eleven columns the
near-constant ones push it behind a horizontal scroll, and it is the signal
reviewers triage on. Only the view is reordered; the export is not.

The script `.txt` follows the pattern of your existing scripts (encoding
comment, proprietary docstring, `from Common.WCR_public import *`, `main()`
with the test-case function table, `handle_test_case` with one `if case == N`
branch per row) and carries a header naming the requirement and generation
time so a draft can't be mistaken for hand-written code. Where a value isn't
derivable from the context the model is instructed to leave a named variable
and a `# TODO:` rather than guess a plausible-looking number.

Script generation is a **second** call, run against the rows the reviewer
kept — including their edits — not against the model's first draft of them.

## Two-stage test-case generation

Enumerating what a requirement needs tested and writing those rows in house
style are different jobs, and a 7B asked to do both in one response did
neither well. It pattern-matched the reference examples and stopped where they
stopped, and whatever it did write past `num_predict` was truncated mid-array
and silently dropped — the coverage was not missing because the model could
not think of the cases, it was missing because the response was cut off.

So the datasheet is generated in two stages:

1. **Plan** (`test_case_plan` in `config/prompts.yaml`). A small prompt with no
   examples and no JSON. It emits one `N|type|behaviour` line per verifiable
   behaviour, worked through an explicit six-point coverage checklist
   (positive path per named condition, each "most restrictive of" comparison,
   each threshold inside and outside, each suppression condition, each state
   transition, each error path). ~200 output tokens, and it runs at
   `plan_temperature` — hotter than the writing calls, because this step wants
   breadth. The list is logged, and returned to the UI as `behaviours`, so
   coverage can be judged before any prose is paid for.
2. **Write** (`test_cases`). One call per `test_case_batch_size` behaviours,
   each with a bounded output budget that always fits. Ollama's structured
   output (`format`) constrains decoding to the datasheet schema, so the
   response cannot be invalid JSON and stops cleanly at the closing brace.

Two things follow from that split and are worth preserving:

- **The model no longer emits what Python already knows.** `Folder`,
  `Optimization_Technique`, `Retired?` and `Scorable` are filled in by
  `schema.py`, and the house description layout (`Test Scenario:`, `-` bullets,
  `Verify,`) is assembled there from the `scenario` / `conditions` / `verify`
  the model returns. That is roughly a quarter of the old output budget
  returned to actual test-case content, and it lets the wording examples
  shrink to `examples_test_case_max_chars: 1500` — they only have to
  demonstrate voice now, not layout.
- **Prompt order is prefix-cacheable.** Ollama reuses the KV cache of an
  identical prompt *prefix*, so the static blocks (wording template, retrieved
  context) come first in the writing prompt and the per-batch instruction
  last. Batches 2..N of a requirement cost decode time only. Reordering those
  blocks would quietly undo it.

A plan call that comes back unparseable is not fatal: the writing stage falls
back to enumerating for itself in a single call, and says so in the log. A
single writing batch that fails to parse is dropped and the other batches are
kept.

## Prompts

Every prompt is in `config/prompts.yaml`. Edits take effect on the next
request; no restart, no code change. Placeholders are `$name`
(`string.Template`), not `{name}`, so the JSON braces inside the prompts need
no escaping.

## Speed on CPU

A 7B model on CPU is slow, so two settings in `config/rag_config.yaml` matter:

- `llm_keep_alive: 30m` keeps the weights resident between requests. Without
  it, every request after an idle gap pays the load cost again.
- `llm_num_threads` — leave at 0 to let Ollama choose, or set it to your
  physical core count if generation is slower than expected.

The server also warms up the embedding model and the LLM on a background
thread at startup, so the first real request doesn't pay for either.

### What the server does to stay out of the model's way

The generation itself is the floor on how fast a request can be. Everything
around it is arranged so that it is the *only* thing you wait for:

- **Nothing is embedded or scored twice.** The embedding model sits behind a
  cache keyed on exact text. The requirement query was previously embedded
  once per retrieval pass — general, track data, then again per doc type
  during script generation — and confidence scoring re-embedded the retrieved
  reference test cases on every request even though they never change.
- **BM25 scores are memoised per query.** `get_scores` walks the whole corpus
  in Python, and one request ran it several times over the same query with
  only the metadata filter differing. It now runs once and the filtered passes
  share the result.
- **Both arms of every retrieval pass run concurrently.** A ChromaDB HNSW
  query and a BM25 scan overlap almost perfectly, since one waits inside C and
  the other inside NumPy.
- **Repeat requests are replayed, not regenerated.** An identical requirement
  is answered from a one-hour cache; the response carries `cached: true` and
  the UI says so and offers a Regenerate button, so a cached answer is never
  the only answer available. The cache key includes the model and the mtime of
  `config/prompts.yaml`, so editing a prompt still takes effect immediately.
- **Generations queue rather than compete.** The model already uses every
  core, so two at once do not finish sooner — they each take about twice as
  long and start tripping the request timeout. The endpoints are `async` (so
  health, uploads and exports stay responsive during a multi-minute
  generation) and a semaphore admits `MAX_CONCURRENT_GENERATIONS` — default 1
  — at a time. Raise it only where the model has headroom: a GPU, or a remote
  Ollama.
- **A truncated response is salvaged, not discarded.** If the model runs out
  of output tokens mid-array, the test cases it finished are recovered and the
  incomplete tail dropped, instead of failing a request that cost minutes.
  With batched writing and Ollama's structured output this should no longer
  fire; it stays as the safety net.
- **The script call reuses the datasheet call's retrieval.** Parameter and
  data-dictionary context is keyed on the requirement query, which has not
  changed since the rows were generated, so the second call gets a cache hit
  instead of another BGE-M3 encode plus two ranking passes. The API and
  track-data passes still use the wider query (requirement + approved rows),
  because which API calls matter genuinely does depend on the rows.
- **Ollama is called with `stream: true`.** The generators still want the
  whole string, but a multi-minute non-streaming POST is one long silence that
  a proxy or client timeout can end for reasons no log explains — and the
  final streamed chunk carries `prompt_eval_count`, `prompt_eval_duration`,
  `eval_count` and `eval_duration`, which are logged. Those four are the only
  way to tell "prefill is too big" from "we are decoding too many tokens",
  and the two have opposite fixes.
- **`llm_num_threads: auto` uses one thread per logical processor.** This is
  measured, not assumed — see below.

### Prefill is the wall

Measured with `python scripts/bench_generation.py` on the 6-core/12-thread
box this was built on (1,977-token prompt, qwen2.5:7b q4_K_M):

| Setting | Prefill | Decode |
|---|---|---|
| `num_thread=4` | 14.8 tok/s | 7.5 tok/s |
| `num_thread=6` (physical cores) | 15.9 tok/s | 6.1 tok/s |
| Ollama's own default | 16.6 tok/s | 8.5 tok/s |
| **`num_thread=12` (logical)** | **20.4 tok/s** | 7.6 tok/s |
| qwen2.5:3b, `num_thread=6` | 37.6 tok/s | 14.2 tok/s |

A prompt token costs about three times what an output token costs, and
~50 seconds per thousand. So **the reliable way to make this faster is to
send fewer prompt tokens**, and the surprising part is which blocks are
big: track-data chunks tokenize at ~2.3 characters per token against
prose's ~4, so four of them were 4,796 characters but 2,086 tokens — more
than the knowledge base, requirement and system prompt combined. Hence
`static_track_max_chars` / `static_api_max_chars`; `top_k` alone cannot
bound a block whose chunk size depends on which subdivision it came from.

Run the benchmark on your own hardware before changing thread counts — the
one-thread-per-physical-core rule of thumb measured 28% *slower* here.

If generations are still too slow, in order of impact: lower
`static_track_max_chars` and `max_context_chars` (prompt tokens are the
expensive ones), set `plan_model: qwen2.5:3b` so the enumeration call runs
on a smaller model, lower `llm_max_tokens` (the per-batch budget), or change
`llm_model` itself. Each is a config edit and needs no code change.

## Configuration

| File | Controls |
|---|---|
| `config/config.yaml` | Source folders, chunk sizes, PDF heuristics, embedding model, ChromaDB location |
| `config/rag_config.yaml` | Hybrid-search weights and depth, context budget, Ollama host/model/limits, plan-call and batch settings, the `max_test_cases` ceiling, confidence weights and threshold |
| `config/prompts.yaml` | Every system and user prompt |
| `.env` | Backend host/port, CORS origins, log level, Ollama host override, `MAX_CONCURRENT_GENERATIONS`, frontend dev-server port/proxy target |

## Layout

```
config/                     the four files above
data/knowledge_base/        PDF · XLSX · TXT
data/python_apis/           PYI · PY
data/track_data/            HTML
src/kb_ingestion/
  extractors.py             one parser per format
  chunking.py               normalisation, token budgets, chunk metadata
  storage.py                BGE-M3 wrapper + ChromaDB wrapper
  pipeline.py               orchestration
src/rag/
  config.py                 settings + prompts.yaml loader
  utils.py                  caches + the retrieval thread pool
  retrieval.py              BM25 arm, hybrid search + RRF, static context
  schema.py                 datasheet rows, model-output parsing, confidence
  llm_client.py             Ollama behind a protocol
  generator.py              test-case and test-script generators
  exporters.py              .xlsx and .txt
src/api/                  main.py (routes, entry point), services.py (schemas + container)
frontend/                   React + Vite
scripts/run_ingestion.py    ingestion CLI
tests/                      pytest suite
```

Both `EmbeddingModel` and `LLMClient` are protocols, so a served embedding
endpoint or a different inference server (vLLM, llama.cpp, anything
OpenAI-compatible) can replace either without touching the pipeline.

## Tests

```bash
pytest -q
```

53 tests, no model download and no live Ollama needed — the embedding model
and the LLM are replaced with deterministic fakes, while retrieval runs
against a real temporary ChromaDB. The extractor tests run against the actual
documents in `data/`, so a parsing regression on your real PDFs and workbooks
fails the suite.
