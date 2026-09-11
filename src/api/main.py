from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import anyio.to_thread  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from api.models import (  # noqa: E402
    ExportScriptRequest,
    ExportTestCasesRequest,
    GenerateScriptRequest,
    GenerateScriptResponse,
    GenerateTestCasesRequest,
    GenerateTestCasesResponse,
    HealthResponse,
    RequirementUploadResponse,
    RetrievedChunkOut,
    TestCaseOut,
)
from api.services import Services  # noqa: E402
from rag.cache import stable_key  # noqa: E402
from rag.exporters import datasheet_to_xlsx, download_name, script_to_text  # noqa: E402
from rag.llm_client import LLMConnectionError, LLMResponseError  # noqa: E402
from rag.prompts import PromptError  # noqa: E402
from rag.requirement_parser import parse_requirement  # noqa: E402
from rag.schema import TestCaseParseError  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
]

services = Services(REPO_ROOT)

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# Requirements are pasted or uploaded as text; anything larger than this is
# not a requirement.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
UPLOAD_SUFFIXES = {".txt", ".md"}

# How many generations may run at once. On a CPU-only box the model already
# uses every core, so a second concurrent generation does not finish sooner —
# it makes both take roughly twice as long, and the fixed request timeout then
# starts catching requests that would have succeeded. Queueing instead keeps
# each one at its normal speed. Raise it only where the model has headroom
# (a GPU, or a remote Ollama).
MAX_CONCURRENT_GENERATIONS = max(1, int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1")))

# Created in `lifespan` rather than at import: an asyncio.Semaphore binds
# itself to the first event loop that contends on it, and a module-level one
# would outlive the loop it was bound to.
_generation_gate: asyncio.Semaphore | None = None


async def _generate(work):
    """Runs one blocking generation on a worker thread, admitting at most
    MAX_CONCURRENT_GENERATIONS of them at a time.

    The endpoints are `async def` so the event loop keeps serving
    `/api/health`, uploads and exports while a generation that can take
    minutes is in flight. The semaphore is the other half: it stops several
    of those generations from fighting over the same cores, which on CPU
    makes all of them slower rather than any of them sooner.
    """
    assert _generation_gate is not None, "lifespan did not run"
    async with _generation_gate:
        return await anyio.to_thread.run_sync(work)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generation_gate
    _generation_gate = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    # Warming up on a background thread keeps the port open immediately —
    # the frontend can render and call /api/health while the model loads.
    threading.Thread(target=services.warm_up, daemon=True).start()
    yield


app = FastAPI(title="PTC Test Case Generator", version="1.0", lifespan=lifespan)

# The React dev server runs on a different port, so the browser treats API
# calls as cross-origin. Localhost only by default; override CORS_ORIGINS in
# .env deliberately if the app is ever served from elsewhere.
_cors_env = os.getenv("CORS_ORIGINS")
_cors_origins = (
    [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    if _cors_env
    else _DEFAULT_CORS_ORIGINS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # Without this, a cross-origin fetch (anything not going through the Vite
    # dev proxy) can read the response body but not this header, so the
    # browser falls back to a generic "download" filename with no extension
    # — easy to mistake for the download being broken.
    expose_headers=["Content-Disposition"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return await anyio.to_thread.run_sync(_health)


def _health() -> HealthResponse:
    """What the UI shows on load: whether the knowledge base has anything in
    it and whether the model is reachable. Neither is fatal, but both are
    worth knowing before waiting on a generation.
    """
    try:
        chunk_count = services.vector_store.count()
    except Exception:  # noqa: BLE001
        logger.exception("Could not read the knowledge base")
        chunk_count = 0

    models = services.llm_client.available_models()
    configured = services.settings.generation.model
    return HealthResponse(
        knowledge_base_chunks=chunk_count,
        embedding_model=services.settings.retrieval.embedding_model,
        llm_model=configured,
        # Ollama reports tags as "qwen2.5:7b-instruct"; a config value
        # without the tag still refers to the same model.
        llm_available=any(m == configured or m.startswith(configured) for m in models),
        mapped_requirements=services.track_mapping.known_requirement_ids,
    )


@app.post("/api/requirements/upload", response_model=RequirementUploadResponse)
async def upload_requirement(file: UploadFile = File(...)) -> RequirementUploadResponse:
    """Reads a requirement out of an uploaded text file, so a user can drop
    in a requirement document instead of pasting it. The text is returned to
    the browser rather than held server-side — the user can then edit it
    before generating, and the server stays stateless.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(
            400,
            f"Upload a plain-text requirement ({', '.join(sorted(UPLOAD_SUFFIXES))}), "
            f"or paste the text directly. Got {suffix or 'no extension'}.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "That file is larger than 2 MB — paste the requirement instead.")

    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise HTTPException(400, "That file is empty.")

    return RequirementUploadResponse(
        filename=file.filename or "requirement.txt",
        requirement_text=text,
        requirement_id=parse_requirement(text).requirement_id,
    )


@app.post("/api/test-cases", response_model=GenerateTestCasesResponse)
async def generate_test_cases(
    request: GenerateTestCasesRequest,
) -> GenerateTestCasesResponse:
    generation = services.settings.generation

    # Keyed on everything that can change the answer, the model and the
    # prompt file included, so editing config/prompts.yaml is still picked up
    # on the next request exactly as PromptLibrary promises.
    key = stable_key(
        request.requirement_text,
        request.top_k,
        request.doc_types,
        request.subdivision_id,
        generation.model,
        generation.max_test_cases,
        services.prompt_revision(),
    )
    if not request.refresh:
        cached = services.result_cache.get(key)
        if cached is not None:
            logger.info("Serving test cases from cache")
            return cached.model_copy(update={"cached": True})

    def work():
        return services.test_case_generator.generate(
            request.requirement_text,
            max_test_cases=generation.max_test_cases,
            doc_types=request.doc_types,
            top_k=request.top_k,
            subdivision_id=request.subdivision_id,
        )

    try:
        result = await _generate(work)
    except (LLMConnectionError, LLMResponseError) as exc:
        # 503: the pipeline is fine, the model endpoint isn't. The message
        # carries the actual remedy (start Ollama, pull the model).
        raise HTTPException(503, str(exc)) from exc
    except TestCaseParseError as exc:
        raise HTTPException(502, str(exc)) from exc
    except PromptError as exc:
        raise HTTPException(500, str(exc)) from exc

    response = GenerateTestCasesResponse(
        requirement_id=result.requirement_id,
        functional_area=result.functional_area,
        test_cases=[TestCaseOut.from_domain(tc) for tc in result.test_cases],
        retrieved=[_chunk_out(c) for c in result.retrieved_chunks],
        track_subdivisions=result.track_subdivisions,
        mean_confidence=result.mean_confidence,
        review_threshold=services.settings.confidence.review_threshold,
        elapsed_seconds=result.elapsed_seconds,
    )
    services.result_cache.put(key, response)
    return response


@app.post("/api/test-script", response_model=GenerateScriptResponse)
async def generate_test_script(request: GenerateScriptRequest) -> GenerateScriptResponse:
    test_cases = [tc.to_domain() for tc in request.test_cases]

    try:
        result = await _generate(
            lambda: services.script_generator.generate(
                request.requirement_text, test_cases, request.subdivision_id
            )
        )
    except (LLMConnectionError, LLMResponseError) as exc:
        raise HTTPException(503, str(exc)) from exc
    except PromptError as exc:
        raise HTTPException(500, str(exc)) from exc

    return GenerateScriptResponse(
        requirement_id=result.requirement_id,
        script=result.script,
        elapsed_seconds=result.elapsed_seconds,
    )


@app.post("/api/test-cases/export")
def export_test_cases(request: ExportTestCasesRequest) -> Response:
    """The datasheet as .xlsx. The rows come back from the browser rather
    than from server memory, so what gets exported is exactly what the user
    sees — including any edits they made in the table.
    """
    content = datasheet_to_xlsx(
        [tc.to_domain() for tc in request.test_cases],
        include_confidence=request.include_confidence,
    )
    filename = download_name("test_cases", request.requirement_id, "xlsx")
    return Response(
        content=content,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/test-script/export")
def export_test_script(request: ExportScriptRequest) -> Response:
    content = script_to_text(request.script, request.requirement_id)
    filename = download_name("test_script", request.requirement_id, "txt")
    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _chunk_out(chunk) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        source=chunk.source_label,
        document_type=str(chunk.metadata.get("document_type") or ""),
        similarity=round(chunk.similarity, 3) if chunk.similarity is not None else None,
        bm25_score=round(chunk.bm25_score, 2) if chunk.bm25_score is not None else None,
        matched_by=chunk.matched_by,
        excerpt=chunk.text[:600],
    )


def _resolve_port(host: str, port: int, max_attempts: int = 10) -> int:
    """Binds a throwaway socket to confirm `port` is actually usable before
    handing it to uvicorn, and steps to the next port if not.

    A stray dev-server process left listening on the configured port, or a
    Windows-reserved range (Hyper-V/WSL2/Docker exclude blocks), makes
    uvicorn fail with an opaque `WinError 10013`/`10048` after the reload
    watcher has already printed its banner. Failing here instead gives a
    clear reason and a working port on the first try.
    """
    import socket

    for candidate in range(port, port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, candidate))
            except OSError as exc:
                if candidate == port:
                    logger.warning(
                        "Port %d unavailable (%s). This is usually a leftover "
                        "process still bound to it (`netstat -ano | findstr :%d` "
                        "then `taskkill /PID <pid> /F`) or a Windows-reserved "
                        "range (`netsh interface ipv4 show excludedportrange "
                        "protocol=tcp`). Trying the next port instead.",
                        port,
                        exc,
                        port,
                    )
                continue
            return candidate

    raise RuntimeError(
        f"No free port found in {port}-{port + max_attempts - 1} on {host}. "
        "Set BACKEND_PORT in .env to a known-free port."
    )


if __name__ == "__main__":
    # `uvicorn api.main:app --app-dir src --port 8000` (see README) reads its
    # own --host/--port flags and never imports this block. This entry point
    # is for `python src/api/main.py`, which honours BACKEND_HOST/PORT from
    # .env instead, and additionally checks the port is actually free first.
    import uvicorn

    _host = os.getenv("BACKEND_HOST", "127.0.0.1")
    _port = _resolve_port(_host, int(os.getenv("BACKEND_PORT", "8000")))
    if str(_port) != os.getenv("BACKEND_PORT", "8000"):
        logger.info("Starting on port %d instead", _port)

    uvicorn.run(app, host=_host, port=_port)
