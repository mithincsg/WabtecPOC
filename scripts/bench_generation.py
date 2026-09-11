"""Measure what a generation call actually costs on this machine.

Prefill, not decode, is the wall on a CPU-only box: a datasheet writing call
carries a few thousand prompt tokens, and at ~20 tokens/s that is minutes
before the first output token appears. Every tuning decision in
`config/rag_config.yaml` — context budgets, thread count, which model runs
the plan call — is a bet about prefill and decode rates, and those rates are
hardware-specific enough that the defaults in this repo were wrong for the
first box they were measured on.

So: measure, then set the config. Ollama reports `prompt_eval_count`,
`prompt_eval_duration`, `eval_count` and `eval_duration` on every response;
this prints them as rates, plus the wall time they do not account for
(model loads, queueing, prompt-cache copying — none of which show up in the
other four numbers).

    python scripts/bench_generation.py                 # threads sweep, default model
    python scripts/bench_generation.py --models qwen2.5:7b,qwen2.5:3b
    python scripts/bench_generation.py --threads 0,6,12 --prompt-tokens 4000

Note that switching `num_thread` or model between runs makes Ollama restart
the runner, which shows up as `load`. The rates themselves are measured
inside one call and are unaffected.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.config import RAGSettings  # noqa: E402

# Deliberately I-ETMS-flavoured: identifier-dense text tokenizes very
# differently from prose (~2.3 characters per token against ~4), and a
# benchmark on lorem ipsum would flatter the prefill rate.
_PARAGRAPH = (
    "The on-board segment shall generate a target for a track-based "
    "subdivision or district level speed restriction of type generic, "
    "tonnage, TPOB or axle count. When the track database indicates the "
    "speed is restricted, the target speed shall be set to the most "
    "restrictive speed defined in the track database or TBC137. "
)


def build_prompt(target_tokens: int) -> str:
    # ~3.2 characters per token for this material; close enough to land
    # within a few percent of the requested size.
    repeats = max(1, int(target_tokens * 3.2 / len(_PARAGRAPH)))
    return _PARAGRAPH * repeats


def run(host: str, model: str, threads: int | None, prompt: str, timeout: int) -> None:
    options = {"num_predict": 16, "num_ctx": 8192, "temperature": 0.1}
    if threads:
        options["num_thread"] = threads
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a verification engineer."},
            {"role": "user", "content": prompt + "\nSummarise in one word."},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": options,
    }

    started = time.monotonic()
    response = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    wall = time.monotonic() - started

    prompt_tokens = data.get("prompt_eval_count") or 0
    prompt_s = (data.get("prompt_eval_duration") or 1) / 1e9
    out_tokens = data.get("eval_count") or 0
    out_s = (data.get("eval_duration") or 1) / 1e9
    load_s = (data.get("load_duration") or 0) / 1e9

    print(
        f"{model:14s} threads={str(threads or 'ollama-default'):15s} "
        f"prefill {prompt_tokens:5d} tok {prompt_s:6.1f}s ({prompt_tokens / prompt_s:5.1f} tok/s)  "
        f"decode {out_tokens:3d} tok {out_s:5.1f}s ({out_tokens / out_s:4.1f} tok/s)  "
        f"load {load_s:5.1f}s  wall {wall:6.1f}s  "
        f"unaccounted {max(0.0, wall - prompt_s - out_s - load_s):5.1f}s",
        flush=True,
    )


def main() -> None:
    settings = RAGSettings.load(REPO_ROOT / "config" / "rag_config.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.generation.ollama_host)
    parser.add_argument(
        "--models",
        default=settings.generation.model,
        help="comma-separated Ollama tags to compare",
    )
    parser.add_argument(
        "--threads",
        default="0,%d" % (max(1, (__import__("os").cpu_count() or 2) // 2)),
        help="comma-separated num_thread values; 0 means Ollama's own choice",
    )
    parser.add_argument("--prompt-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_tokens)
    print(
        f"~{args.prompt_tokens} prompt tokens, 16 output tokens, "
        f"host {args.host}\n"
    )
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for raw in [t.strip() for t in args.threads.split(",") if t.strip()]:
            threads = int(raw)
            try:
                run(args.host, model, threads or None, prompt, args.timeout)
            except Exception as exc:  # noqa: BLE001 - a failed row is not fatal
                print(f"{model} threads={raw}: FAILED {exc}", flush=True)


if __name__ == "__main__":
    main()
