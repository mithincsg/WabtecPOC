from __future__ import annotations

import logging
import time
from typing import Protocol

import requests

from .cache import LruTtlCache
from .config import GenerationConfig

logger = logging.getLogger(__name__)

# `/api/health` is polled by the UI and answers with the model list. Asking
# Ollama for its tags on every poll costs a round trip that can only change
# when someone runs `ollama pull`, so the answer is held briefly.
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
    ) -> str:
        ...


class OllamaClient:
    """Calls a local Ollama server. Behind the LLMClient protocol so a
    differently served endpoint (vLLM, llama.cpp, an OpenAI-compatible API)
    can replace it without touching the generators.

    Two settings matter on a CPU-only box, which is why they're config rather
    than constants:

    `keep_alive` holds the model in RAM between requests. Loading 7B weights
    from disk takes tens of seconds; without this, every request after an
    idle gap pays that again and the UI looks broken.

    `num_threads` sets the thread count. Ollama's default sometimes
    over-subscribes on machines with SMT, which is slower than using one
    thread per physical core.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        # One pooled session for the process. Each generation is a fresh TCP
        # connect and handshake otherwise, and on a long-lived server that is
        # pure overhead paid on every request.
        self._session = requests.Session()
        self._tags = LruTtlCache(max_entries=1, ttl_seconds=_TAGS_TTL_SECONDS)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        # `num_ctx` is per call because the two generations are different
        # shapes: the datasheet call is mostly retrieved prose, the script
        # call additionally carries reference scripts and the API surface and
        # needs a wider window to hold them without the prompt being trimmed.
        window = num_ctx or self.config.num_ctx
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_predict": max_tokens or self.config.max_tokens,
            "num_ctx": window,
        }
        if self.config.num_threads > 0:
            options["num_thread"] = self.config.num_threads

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": options,
        }

        self._warn_if_prompt_overruns_context(
            system_prompt, user_prompt, options["num_predict"], window
        )

        started = time.monotonic()
        data = self._post("/api/chat", payload)
        content = (data.get("message") or {}).get("content")
        if not content:
            raise LLMResponseError(f"Ollama returned no message content: {data!r}")

        logger.info(
            "%s produced %d chars in %.1fs",
            self.config.model,
            len(content),
            time.monotonic() - started,
        )
        return content

    def _warn_if_prompt_overruns_context(
        self, system_prompt: str, user_prompt: str, num_predict: int, num_ctx: int
    ) -> None:
        """Ollama does not report an over-long prompt: it silently trims from
        the start of it, which is exactly where the requirement text and the
        rules live. That failure looks like "the model ignored my
        requirement and returned the same cases as yesterday", so it is worth
        a loud line in the log.

        Characters / 3.5 is a deliberately pessimistic token estimate for
        this material — dense with identifiers, which tokenize badly.
        """
        estimated = int((len(system_prompt) + len(user_prompt)) / 3.5)
        available = num_ctx - num_predict
        if estimated > available:
            logger.warning(
                "Prompt is roughly %d tokens but only %d are left in the "
                "%d-token context after reserving %d for the response — "
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

    def warm_up(self) -> bool:
        """Loads the model with an empty generation so the first real request
        doesn't pay the load cost. Called on server start; returns False
        rather than raising, since a cold model is a slow start, not a
        failure.
        """
        try:
            self._post(
                "/api/chat",
                {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False,
                    "keep_alive": self.config.keep_alive,
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
            # change in the next few seconds, and reporting it stale would
            # keep the UI showing the model as down after it came back.
            return []
        self._tags.put("tags", models)
        return models

    def _post(self, path: str, payload: dict, _attempt: int = 1) -> dict:
        url = f"{self.config.ollama_host.rstrip('/')}{path}"
        try:
            response = self._session.post(
                url, json=payload, timeout=self.config.request_timeout_seconds
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
            # Anything else the transport can raise mid-stream — a dropped
            # connection, a chunked-encoding error when Ollama is killed
            # part-way through a long CPU generation, or a pooled connection
            # the server closed while it was reloading the model for a
            # different num_ctx. Without this the exception escapes as an
            # opaque 500 instead of the 503 the UI knows how to explain.
            #
            # One retry, because the commonest cause is a stale pooled
            # connection rather than a server that is actually down: the
            # generation itself is idempotent, so a second attempt on a fresh
            # connection costs a reload at worst and turns a spurious 503
            # into a served request. Only once — a real outage should fail
            # fast, not after several multi-minute attempts.
            if _attempt == 1:
                logger.warning(
                    "Request to Ollama failed at the transport level (%s); "
                    "retrying once on a fresh connection.",
                    exc,
                )
                self._session.close()
                self._session = requests.Session()
                return self._post(path, payload, _attempt=2)
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
        try:
            return response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"Ollama returned a non-JSON body: {response.text[:500]}"
            ) from exc
