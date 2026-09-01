"""Decode throughput and MTP acceptance, measured from the server's own timings.

llama.cpp returns a `timings` object on every chat completion:

    predicted_per_second   decode tok/s        prompt_per_second  prefill tok/s
    draft_n                tokens drafted      draft_n_accepted   tokens accepted

so acceptance is read directly rather than scraped out of the log. draft_n == 0
means MTP contributed nothing to that request — which is how this bench tells
"MTP is off" apart from "MTP is on but never accepted", a distinction the tok/s
number alone cannot make.

Prompts are agentic in shape (plan, then emit code or a diff) because MTP
acceptance is strongly content-dependent: it runs high on boilerplate and low on
dense prose, and averaging over the wrong mix produces a number that does not
predict real agent throughput.

Usage:
    uv run python benches/bench_mtp.py --base-url http://127.0.0.1:8090/v1 \
        --label mtp-n3 --reps 3
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_result, wait_ready  # noqa: E402

# (name, prompt, max_tokens). Chosen to span the acceptance range an agent sees.
WORKLOADS = [
    ("boilerplate_code",
     "Write a Python dataclass `Order` with fields id:int, customer:str, "
     "total:float, status:str, plus `to_dict` and `from_dict` methods and full "
     "type hints. Output only the code.", 700),
    ("unified_diff",
     "Produce a unified diff that adds retry-with-backoff to this function:\n"
     "```python\nasync def fetch(url):\n    async with httpx.AsyncClient() as c:\n"
     "        r = await c.get(url)\n        return r.json()\n```\n"
     "Output only the diff.", 700),
    ("prose_reasoning",
     "Explain the trade-offs between optimistic and pessimistic locking for a "
     "high-write order table, and say which you would pick and why.", 700),
    ("shell_and_json",
     "Emit a JSON object with keys `commands` (array of 6 shell commands that "
     "set up a Python project with a venv, ruff and pytest) and `notes` "
     "(a string). Output only JSON.", 700),
]


def run_once(client, base_url, model, prompt, max_tokens, timeout, temperature, top_p):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    t0 = time.monotonic()
    r = client.post(f"{base_url}/chat/completions", json=body, timeout=timeout)
    wall = time.monotonic() - t0
    r.raise_for_status()
    d = r.json()
    t = d.get("timings") or {}
    draft_n = t.get("draft_n") or 0
    accepted = t.get("draft_n_accepted") or 0
    return {
        "decode_tok_s": t.get("predicted_per_second"),
        "prefill_tok_s": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"),
        "prompt_n": t.get("prompt_n"),
        "draft_n": draft_n,
        "draft_n_accepted": accepted,
        # Acceptance is undefined, not zero, when nothing was drafted.
        "acceptance": round(accepted / draft_n, 4) if draft_n else None,
        "wall_s": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    wait_ready(args.base_url)

    # Ask the server what speculation it is actually running, instead of
    # inferring it from --label. A mislabelled arm is how a sweep silently
    # compares a config against itself.
    spec_type = "unknown"
    try:
        import httpx as _h
        pr = _h.get(args.base_url.rsplit("/v1", 1)[0] + "/props", timeout=10.0).json()
        spec_type = (pr.get("default_generation_settings", {})
                       .get("params", {}).get("speculative.types") or "unknown")
    except Exception:
        pass

    per_workload = {}
    all_decode, all_draft, all_acc = [], 0, 0
    with httpx.Client() as client:
        for name, prompt, mx in WORKLOADS:
            runs = []
            for _ in range(args.reps):
                runs.append(run_once(client, args.base_url, args.model, prompt, mx,
                                     args.timeout, args.temperature, args.top_p))
            dec = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
            drafted = sum(r["draft_n"] for r in runs)
            acc = sum(r["draft_n_accepted"] for r in runs)
            per_workload[name] = {
                "decode_tok_s_median": round(statistics.median(dec), 2) if dec else None,
                "prefill_tok_s_median": round(
                    statistics.median([r["prefill_tok_s"] for r in runs if r["prefill_tok_s"]]), 1)
                    if any(r["prefill_tok_s"] for r in runs) else None,
                "draft_n": drafted,
                "draft_n_accepted": acc,
                "acceptance": round(acc / drafted, 4) if drafted else None,
                "runs": runs,
            }
            all_decode.extend(dec)
            all_draft += drafted
            all_acc += acc
            a = per_workload[name]["acceptance"]
            print(f"{name:20} decode={per_workload[name]['decode_tok_s_median']:>7} tok/s  "
                  f"acceptance={'n/a (MTP off)' if a is None else f'{a:.3f}'}  "
                  f"drafted={drafted}", flush=True)

    payload = {
        "label": args.label,
        "base_url": args.base_url,
        "reps": args.reps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "decode_tok_s_median": round(statistics.median(all_decode), 2) if all_decode else None,
        "decode_tok_s_mean": round(statistics.mean(all_decode), 2) if all_decode else None,
        "total_drafted": all_draft,
        "total_accepted": all_acc,
        "overall_acceptance": round(all_acc / all_draft, 4) if all_draft else None,
        # "did speculation draft anything", NOT "is MTP specifically running".
        # --spec-type takes a list and the ngram-* strategies draft without MTP,
        # so an ngram-only run legitimately has drafts and no MTP. Naming this
        # mtp_active printed "MTP active: True" for an arm with no NextN head
        # loaded, which is exactly the confusion this bench exists to prevent.
        "speculation_active": all_draft > 0,
        "spec_type": spec_type,
        "workloads": per_workload,
    }
    print()
    print(f"decode median  : {payload['decode_tok_s_median']} tok/s")
    print(f"spec active    : {payload['speculation_active']}  ({spec_type})")
    print(f"acceptance     : {payload['overall_acceptance']}")
    if not args.no_save:
        print(f"saved: {save_result(f'mtp-{args.label}', payload)}")


if __name__ == "__main__":
    raise SystemExit(main())
