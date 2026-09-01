"""Prompt-processing throughput at depth, and the memory it costs.

The agentic and MTP benches both use short prompts, so neither measures prefill
in any meaningful way — which makes them blind to exactly the parameter that
governs it. `--ubatch-size` sets the physical batch llama.cpp processes at once:
larger means better GPU utilisation on long prompts and larger compute buffers,
smaller means the reverse. On a box where the weights leave ~22 GiB of headroom,
that tradeoff is worth measuring rather than assuming.

Every prompt carries a fresh nonce so no depth reuses another's cached prefix.
`cache_n` is reported per request; if it is ever non-zero the number is
contaminated and the run says so rather than quietly reporting a fast lie.

Usage:
    uv run python benches/bench_prefill.py --base-url http://127.0.0.1:8090/v1 \
        --label ub512 --depths 4096,16384,65536
"""

import argparse
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_result, wait_ready  # noqa: E402

# A prompt is only contaminated when a meaningful share of it came from
# cache. Below this it is just the shared template prefix.
CACHE_CONTAMINATION_FRAC = 0.01  # 1%

CLAUSES = [
    "The service layer validates each request against the schema before dispatch.",
    "Every handler logs its latency to the metrics endpoint for later aggregation.",
    "Migrations run inside a transaction so a partial failure rolls the batch back.",
    "The scheduler retries failed jobs with exponential backoff and a jitter window.",
    "Session tokens rotate on privilege change and are revoked on explicit logout.",
    "Cache entries carry a soft expiry so a stale read can serve while refreshing.",
    "The ingest worker batches rows until the buffer fills or the flush timer fires.",
    "Read replicas lag the primary, so writes are routed to the leader only.",
    "Configuration resolves from environment variables and then the config file.",
    "The audit trail records actor, resource, and the previous value of each field.",
]


def make_prompt(approx_tokens, nonce):
    """~1 token per word, unique per (depth, nonce) so nothing shares a prefix."""
    rng = random.Random(f"{nonce}:{approx_tokens}")
    words = f"Reference set {nonce} depth {approx_tokens}.".split()
    while len(words) < approx_tokens:
        words.extend(rng.choice(CLAUSES).split())
    return " ".join(words[:approx_tokens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--depths", default="4096,16384,65536")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    wait_ready(args.base_url)
    max_cached_frac = 0.0
    depths = [int(d) for d in args.depths.split(",")]
    out, contaminated = {}, False

    with httpx.Client() as client:
        for depth in depths:
            runs = []
            for _ in range(args.reps):
                nonce = uuid.uuid4().hex[:8]
                body = {
                    "model": args.model,
                    "messages": [{"role": "user",
                                  "content": make_prompt(depth, nonce) + "\n\nReply with OK."}],
                    # 1 token: we are timing prompt processing, not generation.
                    "max_tokens": 1,
                    "temperature": 1.0,
                }
                t0 = time.monotonic()
                r = client.post(f"{args.base_url}/chat/completions", json=body, timeout=args.timeout)
                wall = time.monotonic() - t0
                r.raise_for_status()
                t = r.json().get("timings") or {}
                # cache_n > 0 is NOT contamination on its own: llama.cpp reuses
                # the chat-template prefix, which is a constant ~9 tokens on
                # every request regardless of depth. Measured across all four
                # ubatch arms it was exactly 9 every time — 0.01% of a 73k
                # prompt, and identical across arms, so it biases nothing.
                # Flag only when a MATERIAL share of the prompt was served from
                # cache. A guard that fires on every run is one nobody heeds.
                pn, cn = t.get("prompt_n") or 0, t.get("cache_n") or 0
                frac = (cn / pn) if pn else 0.0
                max_cached_frac = max(max_cached_frac, frac)
                if frac > CACHE_CONTAMINATION_FRAC:
                    contaminated = True
                runs.append({
                    "prompt_n": t.get("prompt_n"),
                    "prefill_tok_s": t.get("prompt_per_second"),
                    "prompt_ms": t.get("prompt_ms"),
                    "cache_n": t.get("cache_n"),
                    "wall_s": round(wall, 2),
                })
            rates = [x["prefill_tok_s"] for x in runs if x["prefill_tok_s"]]
            out[depth] = {
                "prefill_tok_s_median": round(statistics.median(rates), 1) if rates else None,
                "prompt_n": runs[0]["prompt_n"],
                "runs": runs,
            }
            print(f"depth {depth:>7}  prefill={out[depth]['prefill_tok_s_median']:>8} tok/s  "
                  f"(prompt_n={runs[0]['prompt_n']})", flush=True)

    payload = {"label": args.label, "depths": depths, "reps": args.reps,
               "prefix_cache_contaminated": contaminated,
               "max_cached_fraction": round(max_cached_frac, 5),
               "results": out}
    if contaminated:
        print(f"WARNING: up to {max_cached_frac:.1%} of a prompt was served from cache "
              f"(threshold {CACHE_CONTAMINATION_FRAC:.0%}) — these rates are optimistic "
              "and must not be compared across arms.")
    if not args.no_save:
        print(f"saved: {save_result(f'prefill-{args.label}', payload)}")


if __name__ == "__main__":
    raise SystemExit(main())
