from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator, Protocol

import requests

from .cache import LruTtlCache
from .config import GenerationConfig

logger = logging.getLogger(__name__)

# `/api/health` is polled by the UI and answers with the model list. Asking
# Ollama for its tags on every poll costs a round trip that can only change
# when someone runs `ollama pull`, so the answer is held briefly.
_TAGS_TTL_SECONDS = 15.0

# How often a long generation reports progress to the log. A CPU-bound 7B
# model writing a full test script runs for many minutes; without this the
# server looks hung to anyone watching the log.
_PROGRESS_LOG_SECONDS = 30.0

# Opening the TCP connection to a local Ollama is either instant or broken;
# the long waits all happen after the connection is up.
_CONNECT_TIMEOUT_SECONDS = 10.0

# What a model can do only changes when someone re-pulls it, so the
# capability lookup behind the thinking flag is held for the process.
_CAPS_TTL_SECONDS = 600.0

# Models that reason but predate Ollama's separate `thinking` stream field
# emit the chain of thought inline, fenced like this, in the middle of what
# is supposed to be strict JSON or runnable Python.
_INLINE_THINK_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)


def strip_reasoning(text: str) -> str:
    """Removes inline chain-of-thought fences from a model response.

    Costs nothing on a model that never emits them, and is the difference
    between parseable and unparseable output on one that does - which is why
    it runs for every model rather than for a known list.
    """
    cleaned = _INLINE_THINK_RE.sub("", text)
    # An unclosed fence means the budget ran out mid-thought: only what came
    # before it is usable.
    opened = re.search(r"<(think|thinking|reasoning)>", cleaned, re.IGNORECASE)
    if opened:
        cleaned = cleaned[: opened.start()]
    return cleaned.strip()


def _messages(system_prompt: str, user_prompt: str, prefill: str | None) -> list[dict]:
    """The chat turns for one generation, optionally opening the answer.

    A trailing assistant message is how a chat endpoint is told "the reply has
    already started, continue it". Every Ollama chat template in use here
    renders the last message without its end-of-turn token (the standard
    `{{ if not $last }}<|im_end|>` idiom), so generation resumes inside that
    text instead of at a fresh, empty assistant turn.

    Two things follow, and both matter for the script call. The model cannot
    open with a paragraph of English, because its first token continues a line
    of Python. And on a hybrid reasoning model the template's unconditional
    `<think>` opener is attached to a trailing *user* message only, so
    prefilling skips it — the answer budget stops being spent on a chain of
    thought that the caller throws away.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def _rejoin_prefill(prefill: str | None, content: str) -> str:
    """The whole answer, prefill included, however the server handled it.

    The caller asked for text that starts with `prefill`, so that is what it
    gets back whether the server continued the prefilled turn (content is the
    continuation) or ignored the trailing assistant message and started over
    (content already repeats it). Getting this wrong either way is visible in
    the export: a dropped prefill loses the script's first line, a doubled one
    writes it twice.
    """
    if not prefill:
        return content
    if content.lstrip().startswith(prefill.strip()):
        return content.lstrip()
    return prefill + content


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
        json_schema: dict | None = None,
        prefill: str | None = None,
    ) -> str:
        """Runs one generation. `json_schema`, when given, constrains
        decoding to a JSON value of that shape — the datasheet call uses it,
        the script call (which must return Python) does not.

        `prefill` is text the answer must continue from: it is sent as the
        opening of the model's own turn, so the first token it chooses is
        already inside that text rather than free to start a paragraph. It is
        the script call's equivalent of the schema — the one constraint
        available when the answer has to be Python, not JSON. An endpoint
        that cannot prefill must still return the answer with `prefill`
        prepended, so callers see one continuous text either way.
        """
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

    Generations are streamed. Not for a token-by-token UI — the API answers
    in one piece either way — but because a non-streamed POST gives only a
    single wall-clock deadline for the whole request, and there is no way to
    set that deadline correctly: too short and a script generation that is
    working fine, just slowly, is killed after minutes of CPU time; too long
    and a genuinely wedged server holds the request open for just as long.
    Streaming separates the two questions. `stall_timeout_seconds` bounds the
    gap between tokens (is the model still producing?) and
    `request_timeout_seconds` bounds the total (has this run long enough that
    we should take what we have?) — and when the total is reached, the text
    written so far is returned instead of thrown away, which the truncation
    salvage in schema.py can still turn into usable test cases.

    Either limit set to 0 is removed entirely: the request waits as long as
    the model takes. That is the sane default on a CPU box, where a large
    script prompt can spend minutes in prefill before the first token and no
    fixed number distinguishes that from a hang. What still protects the
    request is that a dead Ollama drops the connection rather than going
    quiet, and that arrives as a ConnectionError, not a timeout.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        # One pooled session for the process. Each generation is a fresh TCP
        # connect and handshake otherwise, and on a long-lived server that is
        # pure overhead paid on every request.
        self._session = requests.Session()
        self._tags = LruTtlCache(max_entries=1, ttl_seconds=_TAGS_TTL_SECONDS)
        self._caps = LruTtlCache(max_entries=8, ttl_seconds=_CAPS_TTL_SECONDS)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        json_schema: dict | None = None,
        prefill: str | None = None,
    ) -> str:
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_predict": max_tokens or self.config.max_tokens,
            "num_ctx": self.config.num_ctx,
        }
        if self.config.num_threads > 0:
            options["num_thread"] = self.config.num_threads

        payload = {
            "model": self.config.model,
            "messages": _messages(system_prompt, user_prompt, prefill),
            "stream": True,
            "keep_alive": self.config.keep_alive,
            "options": options,
        }
        think = self._think_flag()
        if think is not None:
            payload["think"] = think
        if json_schema is not None:
            # Structured output: the server constrains the decoder to this
            # schema, so the response is a JSON object of the right shape
            # whatever the model would otherwise have written. Without it a
            # model that narrates instead of answering costs a full
            # generation and fails in the parser.
            payload["format"] = json_schema

        logger.info(
            "Calling %s at %s (system=%d chars, user=%d chars%s; prompt token count "
            "is reported by Ollama once the call finishes)",
            self.config.model,
            self.config.ollama_host,
            len(system_prompt),
            len(user_prompt),
            ", schema-constrained" if json_schema is not None else "",
        )
        started = time.monotonic()
        try:
            raw, thought_chars, stopped_early, prompt_tokens, completion_tokens = (
                self._stream_chat(payload, started)
            )
        except LLMResponseError as exc:
            # An Ollama too old for structured output rejects `format` as a
            # bad request. Losing the constraint is far better than losing
            # the feature, so drop it and let the prompt do the asking.
            if "format" not in payload or "format" not in str(exc).lower():
                raise
            logger.warning(
                "%s rejected the JSON schema (%s); retrying unconstrained.",
                self.config.ollama_host,
                exc,
            )
            payload.pop("format")
            raw, thought_chars, stopped_early, prompt_tokens, completion_tokens = (
                self._stream_chat(payload, started)
            )
        content = strip_reasoning(raw)
        elapsed = time.monotonic() - started

        if not content:
            if thought_chars:
                # The model worked; it just never got past reasoning, because
                # the answer budget was spent thinking.
                raise LLMResponseError(
                    f"{self.config.model} spent all {elapsed:.0f}s and "
                    f"{thought_chars} chars reasoning without writing an answer. "
                    f"It is a thinking model: set llm_think: off in "
                    f"config/rag_config.yaml, or raise llm_max_tokens/"
                    f"script_max_tokens enough to cover the reasoning as well."
                )
            raise LLMResponseError(
                f"{self.config.model} produced no output in {elapsed:.0f}s. "
                f"Check that Ollama is healthy and that the prompt fits "
                f"llm_num_ctx in config/rag_config.yaml."
            )

        logger.info(
            "%s produced %d chars (%s completion tokens, %s prompt tokens) in %.1fs%s",
            self.config.model,
            len(content),
            completion_tokens if completion_tokens is not None else "?",
            prompt_tokens if prompt_tokens is not None else "?",
            elapsed,
            " (cut off at the total budget)" if stopped_early else "",
        )
        return _rejoin_prefill(prefill, content)

    def _stream_chat(
        self, payload: dict, started: float
    ) -> tuple[str, int, bool, int | None, int | None]:
        """Reads a streamed /api/chat response.

        Returns the answer text, how many characters of separate `thinking`
        the model streamed alongside it, whether it was cut short by the
        total budget, and the prompt/completion token counts Ollama reports
        on its final event (`prompt_eval_count`/`eval_count`) — the actual
        count for whatever model is configured, not a guess from character
        length, so this keeps working across a model swap with no code
        change. Either is None when the stream ended before that event (a
        stall, a dropped connection, the total budget).
        A reasoning model can emit `message.thinking` for minutes before its
        first `message.content` token, so counting that is what tells a
        thinking model apart from a silent one - both look like an empty
        answer otherwise.
        A stall — no token for `stall_timeout_seconds` — is the transport's
        read timeout, so it arrives here as an exception rather than a short
        read.
        """
        stall = self.config.stall_timeout_seconds
        budget = self.config.request_timeout_seconds
        chunks: list[str] = []
        thought_chars = 0
        last_log = started
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            with self._open_stream("/api/chat", payload) as response:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        # Ollama emits one JSON object per line; anything else
                        # is a proxy or an error page, not a token.
                        raise LLMResponseError(
                            f"Ollama returned a non-JSON stream line: {line[:500]}"
                        ) from None
                    if event.get("error"):
                        raise LLMResponseError(f"Ollama reported: {event['error']}")

                    message = event.get("message") or {}
                    piece = message.get("content") or ""
                    if piece:
                        chunks.append(piece)
                        print(piece, end="", flush=True)
                    thought_chars += len(message.get("thinking") or "")
                    if event.get("done"):
                        prompt_tokens = event.get("prompt_eval_count")
                        completion_tokens = event.get("eval_count")
                        print()
                        break

                    now = time.monotonic()
                    if budget > 0 and now - started > budget:
                        logger.warning(
                            "%s hit the %ds total budget with %d chars written; "
                            "keeping the partial response. Raise "
                            "llm_request_timeout_seconds, lower llm_max_tokens/"
                            "script_max_tokens, or set a smaller llm_model in "
                            "config/rag_config.yaml.",
                            self.config.model,
                            budget,
                            sum(len(c) for c in chunks),
                        )
                        return "".join(chunks), thought_chars, True, None, None
                    if now - last_log >= _PROGRESS_LOG_SECONDS:
                        last_log = now
                        logger.info(
                            "%s still generating: %d chars after %.0fs%s",
                            self.config.model,
                            sum(len(c) for c in chunks),
                            now - started,
                            f" (plus {thought_chars} chars of reasoning)"
                            if thought_chars
                            else "",
                        )
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"{self.config.model} produced nothing for {stall}s (after "
                f"{time.monotonic() - started:.0f}s and "
                f"{sum(len(c) for c in chunks)} chars). That is a stalled "
                f"model rather than a slow one: check the Ollama server, or "
                f"raise llm_stall_timeout_seconds in config/rag_config.yaml."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            # A connection dropped mid-stream after real output means a
            # crashed or restarted Ollama, not an unreachable one — and
            # whatever it managed to write is still worth salvaging.
            if chunks:
                logger.warning("Ollama stream ended early: %s", exc)
                return "".join(chunks), thought_chars, True, None, None
            raise LLMConnectionError(
                f"Could not reach Ollama at {self.config.ollama_host}. Start it with "
                f"`ollama serve`, and make sure `ollama pull {self.config.model}` has run."
            ) from exc
        except requests.RequestException as exc:
            if chunks:
                logger.warning("Ollama stream failed part-way: %s", exc)
                return "".join(chunks), thought_chars, True, None, None
            raise LLMConnectionError(
                f"The request to Ollama at {self.config.ollama_host} failed: {exc}"
            ) from exc

        return "".join(chunks), thought_chars, False, prompt_tokens, completion_tokens

    def _think_flag(self) -> bool | None:
        """What to send as `think` for the configured model, or None to omit.

        Ollama rejects the field outright on a model that cannot think, so it
        can only be sent once the model has said it can - which is what makes
        swapping llm_model between qwen2.5 and qwen3 a config edit and
        nothing more.
        """
        mode = str(self.config.think or "auto").strip().lower()
        if mode in ("on", "true", "yes", "1"):
            wanted = True
        elif mode in ("off", "false", "no", "0"):
            wanted = False
        elif mode == "auto":
            wanted = False
        else:
            raise LLMResponseError(
                f"llm_think must be auto, on or off in config/rag_config.yaml, "
                f"not {self.config.think!r}."
            )
        if not self._supports_thinking():
            return None
        if mode == "auto":
            logger.info(
                "%s is a thinking model; reasoning disabled (llm_think: auto). "
                "Set llm_think: on to keep it, and raise llm_max_tokens to "
                "cover it.",
                self.config.model,
            )
        return wanted

    def _supports_thinking(self) -> bool:
        cached = self._caps.get(self.config.model)
        if cached is not None:
            return cached
        try:
            response = self._session.post(
                f"{self.config.ollama_host.rstrip('/')}/api/show",
                json={"model": self.config.model},
                timeout=10,
            )
            response.raise_for_status()
            capabilities = response.json().get("capabilities") or []
        except (requests.RequestException, ValueError) as exc:
            # Not knowing is not fatal: omitting `think` just leaves the model
            # on its own default, which is how it behaved before this existed.
            logger.warning("Could not read capabilities for %s: %s", self.config.model, exc)
            return False
        supported = "thinking" in capabilities
        self._caps.put(self.config.model, supported)
        return supported

    def _open_stream(self, path: str, payload: dict) -> requests.Response:
        url = f"{self.config.ollama_host.rstrip('/')}{path}"
        try:
            response = self._session.post(
                url,
                json=payload,
                stream=True,
                # (connect, read). On a streamed response the read half
                # applies per chunk, which is exactly the stall timeout; the
                # total budget is enforced in the read loop instead. None is
                # requests' "wait forever", which is what a stall timeout of 0
                # asks for.
                timeout=(_CONNECT_TIMEOUT_SECONDS, self.config.stall_timeout_seconds or None),
            )
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"{self.config.model} produced no first token within "
                f"{self.config.stall_timeout_seconds}s. A cold model loading "
                f"from disk can exceed that — raise llm_stall_timeout_seconds "
                f"in config/rag_config.yaml."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self.config.ollama_host}. Start it with "
                f"`ollama serve`, and make sure `ollama pull {self.config.model}` has run."
            ) from exc
        self._raise_for_status(response)
        return response

    def warm_up(self) -> bool:
        """Loads the model with an empty generation so the first real request
        doesn't pay the load cost. Called on server start; returns False
        rather than raising, since a cold model is a slow start, not a
        failure.
        """
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {"num_predict": 1},
        }
        try:
            # Warming up with thinking left on would make a reasoning model
            # start a full chain of thought for a one-token request.
            think = self._think_flag()
            if think is not None:
                payload["think"] = think
            self._post(
                "/api/chat",
                payload,
                timeout=self.config.stall_timeout_seconds or None,
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

    def _post(self, path: str, payload: dict, timeout: float | None) -> dict:
        """One non-streamed request. Only the warm-up uses this — real
        generations stream, see `_stream_chat`.
        """
        url = f"{self.config.ollama_host.rstrip('/')}{path}"
        try:
            response = self._session.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"{self.config.model} did not answer within {timeout}s."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self.config.ollama_host}. Start it with "
                f"`ollama serve`, and make sure `ollama pull {self.config.model}` has run."
            ) from exc
        except requests.RequestException as exc:
            # Anything else the transport can raise — a dropped connection, a
            # chunked-encoding error when Ollama is killed part-way. Without
            # this the exception escapes as an opaque 500 instead of the 503
            # the UI knows how to explain.
            raise LLMConnectionError(
                f"The request to Ollama at {self.config.ollama_host} failed: {exc}"
            ) from exc

        self._raise_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"Ollama returned a non-JSON body: {response.text[:500]}"
            ) from exc

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.status_code == 404:
            raise LLMResponseError(
                f"Ollama does not have the model {self.config.model!r}. "
                f"Run `ollama pull {self.config.model}`."
            )
        if response.status_code != 200:
            raise LLMResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
            )
