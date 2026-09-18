# PTC Test Case Generator

Turns a requirement into a review-ready test-case datasheet and an automation
script draft, grounded in your existing documents.

```
data/python_apis/          PYI · PY            (stdlib ast)
data/track_data/<subdiv>/  <subdiv>-subdiv.xml (stdlib ElementTree)
data/parameter_config/     JSON records        (one per TBC/CFG/THE)
                            │
              structure-aware chunking → one in-process BM25 index
                            │
requirement ──→ requirement understanding ──→ keyword retrieval ──→ context
                            │
              context + prompts.yaml ──→ Qwen (Ollama)
                            │
              datasheet rows ──→ .xlsx
                            │
              approved rows ──→ second generation ──→ test script ──→ .txt
```

There is no vector store and no embedding model. Every source folder is read
from disk when the server starts and held in a single in-process BM25 index.
This suits the material: it is API stubs, track records and parameter rows —
identifier-dense tables rather than prose — and the values a generated script
must get exactly right (`TBC137`, block `1015`, `wcr_loco_sim.set_position`)
are precisely the ones keyword search is good at and dense search smears
across near-neighbours.

## Setup

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama serve
```

Create `.env` yourself — it is gitignored and there is no committed template;
the `--- ... ---` section comments in an existing `.env` say what each key
does. It holds the values that differ per machine rather than per deployment
behaviour: backend host/port, allowed CORS origins, log level,
`MAX_CONCURRENT_GENERATIONS`, the Ollama host, and the frontend dev-server
port and proxy target. Everything else — chunking, retrieval depth, prompts —
stays in `config/*.yaml`.

Nothing is downloaded and the backend makes no network calls at startup —
its only outbound traffic is to Ollama. Chunk token budgets are estimated
locally; there is no tokenizer or model to fetch, so the app runs on a
disconnected machine.

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

Then open http://localhost:5173. Paste a requirement (or load a `.txt`), pick
a subdivision, generate, review the rows, download the `.xlsx`, then generate
the script and download the `.txt`.

There is no build or ingestion step. Every source folder is re-read on server
start, so **editing one takes effect on the next restart** — no index to
rebuild, no state file to fall out of sync. Building the index takes about
two seconds over ~7,000 chunks, paid at startup so no request does.

There is no "how many test cases" control, deliberately. How many cases a
requirement needs is a property of the requirement — one per verifiable
behaviour it states — and it is not something a user can sensibly guess at
before seeing the result. The model is asked to enumerate the behaviours the
requirement specifies and write one case for each: a requirement naming a
single condition yields a single case, one naming six target types with a
boundary each yields twelve. `max_test_cases` in `config/rag_config.yaml` is a
ceiling that keeps a sprawling requirement inside the model's output budget,
not a target.

## The three source folders

| Folder | Holds | Parsed with |
|---|---|---|
| `data/python_apis/` | The WCR test-automation API stubs | stdlib `ast` |
| `data/track_data/<subdiv>/` | One `<subdiv>-subdiv.xml` track export per subdivision | stdlib `ElementTree` |
| `data/parameter_config/` | One JSON record per TBC/CFG/THE parameter | generated, see below |

Roots are declared under `static_sources:` in `config/config.yaml`, not in
code. Every chunk is stamped with a `document_type` taken from its root.

Two other folders exist but are **not read at request time**:

- `data/knowledge_base/` holds the source PDFs the converter scripts read.
- `data/Examples/` holds reference requirement/datasheet/script triples. Its
  patterns are distilled by hand into the templates in `config/prompts.yaml`;
  editing the folder changes nothing until those templates are updated.

Two build steps turn source documents into what the app reads:

```bash
python scripts/convert_parameter_guide.py   # guide PDF -> data/parameter_config/
python scripts/convert_caf_mapping.py       # data/CAF.xlsx -> config/caf_mapping.json
```

Re-run the first after dropping in a new revision of the Parameter
Configuration Guide, then restart the server. Re-run the second after editing
the CAF workbook; the app re-reads the JSON whenever its mtime changes, so no
restart is needed for that one.

### Track data is XML only, and picked rather than searched blindly

Each subdivision folder holds the full serialized track database as XML,
carrying fields no report prints: WIU addresses and security keys, per-block
element/heading/elevation series, device status indices, acquisitions. The
human-readable HTML report that used to sit beside it has been removed — it
was the same data with less in it, and indexing both meant two chunks
competing to answer the same question. `.html` is no longer a recognised
suffix, so putting a report back does nothing.

One visible consequence: the subdivision display name ("Ginger") lived only in
that report, so the picker lists bare numbers. That is the intended state, not
missing data.

Each export lists thousands of track features. Searching all of them for every
requirement drowns retrieval in track the requirement isn't tested on, so the
subdivision is chosen explicitly in the UI (its list comes from
`GET /api/track-subdivisions`, read off the built index so it can only offer
track that actually parsed). That choice is the only thing that selects track
data, and it is required before generating; the same value is sent again with
the script call so the script's hard-coded blocks match its datasheet's.

A request that somehow arrives without one gets no track data rather than a
search across every subdivision — a block or milepost from the wrong track
looks right and is wrong, which is worse than the `# TODO:` placeholder the
model writes when the track block is empty.

## How chunking preserves structure

Nothing unrelated ever shares a chunk.

- **PY/PYI** — module → class → method. A class with methods becomes a compact
  header chunk (signature and docstring) plus one chunk per method; a method
  is never split unless it alone busts the token budget.
- **XML** — subdivision → record type → one `field=value` line per record.
  Nested records are prefixed with their parent's identifier
  (`BlockFeature 2001 > ...`), so a flattened heading line still says which
  block it belongs to. Each record type opens with a label row naming its
  fields in split form (`Block Signal Feature | Signal Id | ...`), because
  BM25 tokenizes camel case whole — without it, a requirement asking about a
  "signal" matches none of these rows. The exports pad fixed-width fields with
  literal NUL character references, which is not well-formed XML at any
  version, so those are stripped before parsing rather than losing a 2 MB file
  to one padded field.
- **JSON** — one parameter record per chunk, led by its identifier and title
  (`TBC137 | Restricted Speed`), so asking about a parameter returns that
  parameter rather than whatever share of a guide page a chunk boundary caught.

## Retrieval

One ranking: BM25 over every chunk of the index, filtered per pass by
`document_type` (and, for track data, by the chosen subdivision).

Query tokens mixing letters and digits (`tbc137`, `l2r7983`, `iv132`) are
counted several times on the query side only. BM25 sums a contribution per
query token, so without this a requirement naming `TBC137` once but repeating
"speed", "restricted" and "enforce" throughout ranks records matching those
common words above the one record it names — measured on a one-sentence
requirement, `TBC137` came 13th and never reached the prompt; boosted, it is
1st. Bare numbers are deliberately not boosted: the corpus is full of them and
IDF already separates a rare one.

Parameter records go further. Any TBC/CFG/THE identifier the requirement
states is looked up by exact id, and when the requirement names any, those
records are the whole parameter block — nothing is padded in beside them.
BM25 is the fallback for a requirement that names none.

`static_api_top_k`, `static_track_top_k` and `static_parameter_top_k` in
`config/rag_config.yaml` set how many chunks each pass returns.

## Output formats

The exported `.xlsx` is sheet `Datasheet` with columns A–J exactly as in your
existing workbooks — `S_no`, `Requirement`, `Description`, `Folder`,
`Optimization_Technique`, `Test_Type`, `Test_Technique`, `Retired?`,
`Scorable`, `Comments` — so it drops into the existing execution flow
unchanged, with no extra columns. The on-screen table shows the same columns
in the same order.

The script `.txt` follows the pattern of your existing scripts (encoding
comment, proprietary docstring, `from Common.WCR_public import *`, `main()`
with the test-case function table, `handle_test_case` with one `if case == N`
branch per row) and carries a header naming the requirement and generation
time so a draft can't be mistaken for hand-written code. Where a value isn't
derivable from the context the model is instructed to leave a named variable
and a `# TODO:` rather than guess a plausible-looking number.

Script generation is a **second** call, run against the rows the reviewer
kept — including their edits — not against the model's first draft of them.

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

The server warms up the index and the LLM on a background thread at startup,
so the first real request doesn't pay for either.

### What the server does to stay out of the model's way

The generation itself is the floor on how fast a request can be. Everything
around it is arranged so that it is the *only* thing you wait for:

- **BM25 scores are memoised per query.** `get_scores` walks the whole corpus
  in Python, and one script request scores the same query three times — API,
  track and parameter passes — differing only in the filter applied
  afterwards. It runs once and the filtered passes share the result.
- **Repeat requests are replayed, not regenerated.** An identical requirement
  is answered from a one-hour cache; the response carries `cached: true` and
  the UI says so and offers a Regenerate button, so a cached answer is never
  the only answer available. The cache key includes the model and the mtime of
  `config/prompts.yaml`, so editing a prompt still takes effect immediately.
- **Generations queue rather than compete.** The model already uses every
  core, so two at once do not finish sooner — they each take about twice as
  long. The endpoints are `async` (so health, uploads and exports stay
  responsive during a multi-minute generation) and a semaphore admits
  `MAX_CONCURRENT_GENERATIONS` — default 1 — at a time. Raise it only where
  the model has headroom: a GPU, or a remote Ollama.
- **A truncated response is salvaged, not discarded.** If the model runs out
  of output tokens mid-array, the test cases it finished are recovered and the
  incomplete tail dropped, instead of failing a request that cost minutes. The
  script equivalent trims back to the last line that parses and marks the cut
  with a `# TODO:`, so the export stays runnable Python.

If generations are still too slow, in order of impact: lower `llm_max_tokens`
and `script_max_tokens`, lower the `static_*_top_k` values, or change
`llm_model` to a smaller variant — all one-line config changes needing no code
edit.

## Configuration

| File | Controls |
|---|---|
| `config/config.yaml` | Source folders, chunk sizes, PDF heuristics, chunk tokenizer |
| `config/rag_config.yaml` | Retrieval depth per source, Ollama host/model/limits, the `max_test_cases` ceiling |
| `config/prompts.yaml` | Every system and user prompt |
| `config/caf_mapping.json` | Requirement → feature, the datasheet's `Folder` column; generated |
| `.env` | Backend host/port, CORS origins, log level, Ollama host override, `MAX_CONCURRENT_GENERATIONS`, frontend dev-server port/proxy target |

## Layout

```
config/                     config.yaml, rag_config.yaml, prompts.yaml, caf_mapping.json
data/python_apis/           PYI · PY (indexed)
data/track_data/<subdiv>/   <subdiv>-subdiv.xml (indexed)
data/parameter_config/      parameter records (indexed); generated, not hand-edited
data/knowledge_base/        source PDFs for the converters; not read at request time
data/Examples/              reference triples; design-time source for the prompt templates
data/CAF.xlsx               requirement -> feature source; converted, not read at request time
scripts/convert_parameter_guide.py  guide PDF -> data/parameter_config/
scripts/convert_caf_mapping.py      data/CAF.xlsx -> config/caf_mapping.json
src/kb_ingestion/
  extractors/               one parser per format (python, xml, json; pdf/xlsx/text kept, unused)
  chunking.py               normalisation, token budgets, chunk metadata
  pipeline.py               discover -> extract -> chunk, in memory
src/rag/
  requirement_parser.py     requirement ID + functional area
  caf_mapping.py            requirement -> feature (the Folder column)
  static_context.py         the BM25 index over every source folder, and the searches over it
  keyword_index.py          BM25 scoring, tokenization, identifier boost
  cache.py                  the generation result cache
  prompts.py                prompts.yaml loader
  schema.py                 datasheet rows, model-output parsing
  llm_client.py             Ollama behind a protocol
  generator.py              test-case and test-script generators
  exporters.py              .xlsx and .txt
src/api/                    FastAPI app, models, service container
frontend/                   React + Vite
```

`LLMClient` is a protocol, so a different inference server (vLLM, llama.cpp,
anything OpenAI-compatible) can replace Ollama without touching the pipeline.

## Tests

There is no automated test suite in this repo currently. Verify changes by
starting the server and exercising the API and frontend directly.
