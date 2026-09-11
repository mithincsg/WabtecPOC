"""Ollama behind the LLMClient protocol."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Protocol

import requests

from .config import GenerationConfig
from .utils import LruTtlCache

logger = logging.getLogger(__name__)

# /api/health is polled by the UI and answers with the model list, which can
# only change when someone runs `ollama pull`.
_TAGS_TTL_SECONDS = 15.0


class LLMConnectionError(RuntimeError):
    """The model endpoint could not be reached, or didn't answer in time."""


class LLMResponseError(RuntimeError):
    """The endpoint answered, but with an error or an unusable body."""


class LLMClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        *,
        temperature: float | None = None,
        json_schema: dict | None = None,
        model: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        ...


class OllamaClient:
    """Calls a local Ollama server, behind the LLMClient protocol so a
    differently served endpoint (vLLM, llama.cpp, an OpenAI-compatible API)
    can replace it without touching the generators.

    Two settings matter on a CPU-only box, which is why they are config
    rather than constants: `keep_alive` holds the model in RAM between
    requests (loading 7B weights costs tens of seconds), and `num_threads`
    sets the thread count, which Ollama's own default gets wrong on SMT
    machines - see GenerationConfig.resolved_num_threads.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        # One pooled session for the process: otherwise every generation is
        # a fresh TCP connect and handshake.
        self._session = requests.Session()
        self._tags = LruTtlCache(max_entries=1, ttl_seconds=_TAGS_TTL_SECONDS)
        self._num_threads = config.resolved_num_threads()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        *,
        temperature: float | None = None,
        json_schema: dict | None = None,
        model: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        # num_ctx is per call because the two generations are different
        # shapes: the datasheet call is mostly retrieved prose, the script
        # call additionally carries reference scripts and the API surface.
        window = num_ctx or self.config.num_ctx
        options = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p,
            "num_predict": max_tokens or self.config.max_tokens,
            "num_ctx": window,
        }
        if self._num_threads > 0:
            options["num_thread"] = self._num_threads

        # `model` is per call so the cheap enumeration stage can run on a
        # smaller model. Both stay resident as long as
        # OLLAMA_MAX_LOADED_MODELS allows; if it does not, the swap costs a
        # load on every call and is a net loss.
        model_name = model or self.config.model
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Streamed even though every caller wants the whole string: a
            # multi-minute non-streaming POST is one long silence a proxy or
            # client timeout can end for reasons no log explains.
            "stream": True,
            "keep_alive": self.config.keep_alive,
            "think": self.config.think,
            "options": options,
        }
        if json_schema is not None:
            # Ollama constrains decoding to the schema, so the model cannot
            # emit invalid JSON. The repair path in schema.py stays as the
            # fallback for servers too old to support this.
            payload["format"] = json_schema

        self._warn_if_prompt_overruns_context(
            system_prompt, user_prompt, options["num_predict"], window
        )

        started = time.monotonic()
        content, metrics = self._stream("/api/chat", payload, on_token)
        if not content:
            raise LLMResponseError("Ollama returned no message content.")

        self._log_metrics(model_name, content, metrics, time.monotonic() - started)
        return content

    def warm_up(self) -> bool:
        """Loads the model with an empty generation so the first real request
        doesn't pay the load cost. Returns False rather than raising: a cold
        model is a slow start, not a failure.
        """
        try:
            self._post(
                "/api/chat",
                {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False,
                    "keep_alive": self.config.keep_alive,
                    "think": self.config.think,
                    "options": {"num_predict": 1},
                },
            )
            return True
        except (LLMConnectionError, LLMResponseError) as exc:
            logger.warning("Could not warm up %s: %s", self.config.model, exc)
            return False

    def available_models(self) -> list[str]:
        cached = self._tags.get("tags")
        if cached is not None:
            return cached
        try:
            response = self._session.get(
                f"{self.config.ollama_host.rstrip('/')}/api/tags", timeout=5
            )
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError):
            # Not cached: an unreachable Ollama is the state most likely to
            # change in the next few seconds.
            return []
        self._tags.put("tags", models)
        return models

    def _stream(
        self, path: str, payload: dict, on_token: Callable[[str], None] | None
    ) -> tuple[str, dict[str, Any]]:
        """Accumulates a streamed /api/chat response into one string.

        The final chunk carries Ollama's timing counters, returned alongside
        the text: they are the only way to tell "prefill is too big" from
        "we are decoding too many tokens", and those have opposite fixes.
        """
        parts: list[str] = []
        metrics: dict[str, Any] = {}
        started = time.monotonic()
        first_token_at: float | None = None
        for line in self._post_stream(path, payload):
            try:
                chunk = json.loads(line)
            except ValueError:
                # Should be impossible with iter_lines, but one unparseable
                # chunk is not worth losing the response already streamed in.
                logger.debug("Skipping unparseable stream line: %.120s", line)
                continue
            if chunk.get("error"):
                raise LLMResponseError(f"Ollama reported: {chunk['error']}")
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                if first_token_at is None:
                    first_token_at = time.monotonic() - started
                parts.append(piece)
                if on_token is not None:
                    on_token(piece)
            if chunk.get("done"):
                metrics = chunk
        metrics = dict(metrics)
        metrics["_ttft_seconds"] = first_token_at
        return "".join(parts), metrics

    def _log_metrics(
        self, model: str, content: str, metrics: dict, wall_seconds: float
    ) -> None:
        prompt_tokens = metrics.get("prompt_eval_count")
        prompt_ns = metrics.get("prompt_eval_duration") or 0
        out_tokens = metrics.get("eval_count")
        out_ns = metrics.get("eval_duration") or 0
        if prompt_tokens is None and out_tokens is None:
            logger.info("%s produced %d chars in %.1fs", model, len(content), wall_seconds)
            return
        # load, ttft and unaccounted are logged because they catch problems
        # the other numbers hide: a keep_alive that did not hold shows as
        # seconds of load on a warm server, and wall time that is neither
        # prefill, decode nor load is the machine paging the weights back in
        # (measured at 100-265s per call on a box with 7 GB free RAM).
        load_ns = metrics.get("load_duration") or 0
        counted = (prompt_ns + out_ns + load_ns) / 1e9
        logger.info(
            "%s: prefill %s tok in %.1fs (%.0f tok/s), decode %s tok in %.1fs "
            "(%.1f tok/s), load %.1fs, ttft %.1fs, %d chars, %.1fs wall "
            "(%.1fs unaccounted)",
            model,
            prompt_tokens,
            prompt_ns / 1e9,
            (prompt_tokens or 0) / (prompt_ns / 1e9) if prompt_ns else 0.0,
            out_tokens,
            out_ns / 1e9,
            (out_tokens or 0) / (out_ns / 1e9) if out_ns else 0.0,
            load_ns / 1e9,
            metrics.get("_ttft_seconds") or 0.0,
            len(content),
            wall_seconds,
            max(0.0, wall_seconds - counted),
        )
        if metrics.get("done_reason") == "length":
            logger.warning(
                "The response stopped at num_predict (%s tokens), not at the "
                "model's own stopping point - it was cut off mid-answer. "
                "Raise llm_max_tokens, or lower test_case_batch_size so each "
                "call writes fewer rows.",
                out_tokens,
            )

    def _warn_if_prompt_overruns_context(
        self, system_prompt: str, user_prompt: str, num_predict: int, num_ctx: int
    ) -> None:
        """Ollama does not report an over-long prompt: it silently trims from
        the start of it, which is exactly where the requirement text and the
        rules live. That failure looks like "the model ignored my
        requirement", so it is worth a loud line in the log.

        Characters / 3.5 is a deliberately pessimistic token estimate for
        this material, which is dense with badly-tokenizing identifiers.
        """
        estimated = int((len(system_prompt) + len(user_prompt)) / 3.5)
        available = num_ctx - num_predict
        if estimated > available:
            logger.warning(
                "Prompt is roughly %d tokens but only %d are left in the "
                "%d-token context after reserving %d for the response - "
                "Ollama will trim the START of the prompt (the requirement "
                "and the retrieved context). Lower "
                "examples_test_case_max_chars / examples_script_max_chars / "
                "max_context_chars, or raise llm_num_ctx, in "
                "config/rag_config.yaml.",
                estimated,
                available,
                num_ctx,
                num_predict,
            )

    def _post(self, path: str, payload: dict) -> dict:
        """A non-streaming request. Only warm_up uses this."""
        response = self._request(path, payload, stream=False)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"Ollama returned a non-JSON body: {response.text[:500]}"
            ) from exc

    def _post_stream(self, path: str, payload: dict):
        """Yields the non-empty lines of a streamed response.

        A transport failure before the first line is retried on a fresh
        connection; one part-way through is not, since those tokens have
        already been handed to the caller.
        """
        response = self._request(path, payload, stream=True)
        try:
            with response:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        yield line
        except requests.RequestException as exc:
            raise LLMConnectionError(
                f"The connection to Ollama at {self.config.ollama_host} dropped "
                f"part-way through the response: {exc}"
            ) from exc

    def _request(
        self, path: str, payload: dict, *, stream: bool, _attempt: int = 1
    ) -> requests.Response:
        url = f"{self.config.ollama_host.rstrip('/')}{path}"
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.config.request_timeout_seconds,
                stream=stream,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"{self.config.model} did not answer within "
                f"{self.config.request_timeout_seconds}s. On CPU, lower "
                f"llm_max_tokens, or switch llm_model to a smaller model "
                f"in config/rag_config.yaml."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self.config.ollama_host}. Start it with "
                f"`ollama serve`, and make sure `ollama pull {self.config.model}` has run."
            ) from exc
        except requests.RequestException as exc:
            # Anything else the transport can raise, most often a stale
            # pooled connection the server closed while reloading the model
            # for a different num_ctx. One retry, because the generation is
            # idempotent and a second attempt on a fresh connection turns a
            # spurious 503 into a served request; only once, because a real
            # outage should fail fast.
            if _attempt == 1:
                logger.warning(
                    "Request to Ollama failed at the transport level (%s); "
                    "retrying once on a fresh connection.",
                    exc,
                )
                self._session.close()
                self._session = requests.Session()
                return self._request(path, payload, stream=stream, _attempt=2)
            raise LLMConnectionError(
                f"The request to Ollama at {self.config.ollama_host} failed: {exc}"
            ) from exc

        if response.status_code == 404:
            raise LLMResponseError(
                f"Ollama does not have the model {self.config.model!r}. "
                f"Run `ollama pull {self.config.model}`."
            )
        if response.status_code != 200:
            raise LLMResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
            )
        return response
