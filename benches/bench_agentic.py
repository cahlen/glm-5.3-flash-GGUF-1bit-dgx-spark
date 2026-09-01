"""Agentic reliability bench: the 10 scenarios an autonomous coding agent needs.

Complements bench_toolcalls.py. That bench asks "how often does the model pick the
right tool across many similar tasks" (breadth, Wilson intervals). This one asks
"does the serving stack honour the agentic contract at all" across 10 *distinct*
shapes of request — plain text, code, JSON, required tool calls, multi-turn tool
results, a large tool set, post-tool reasoning, a long system prompt, diffs, and
shell commands.

The load-bearing check is scenario 04/08/09/10: those are sent with
tool_choice="required". A 200 OK carrying zero tool calls is recorded as a
SERVING failure, not a model miss — it means the template or the tool-call parser
is broken, and no amount of prompt tuning fixes it. Reported separately.

Usage:
    uv run python benches/bench_agentic.py --base-url http://localhost:8090/v1 \
        --label glm-5.3-flash-iq1s-49k --trials 3
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_spec import EXTRA_CALLS, REQUIRED_SCENARIOS, SCENARIOS  # noqa: E402
from common import save_result, wait_ready  # noqa: E402


def _post(client, base_url, body, timeout):
    t0 = time.monotonic()
    r = client.post(f"{base_url}/chat/completions", json=body, timeout=timeout)
    dt = time.monotonic() - t0
    r.raise_for_status()
    return r.json(), dt


def _extract(resp):
    """(text, tool_calls) from an OpenAI-shaped response.

    llama.cpp puts reasoning in `reasoning_content` when --reasoning-format
    deepseek is active; we deliberately do NOT concatenate it into `text`,
    because a check that passes only because the answer appeared inside the
    think block is not a passing answer.
    """
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    return text, calls


def run_scenario(client, base_url, scenario, *, model, trials, timeout,
                 reasoning_effort=None, temperature=None, top_p=None,
                 min_p=None, top_k=None, max_tokens=8192):
    results = []
    for i in range(trials):
        body = {
            "model": model,
            "messages": scenario["messages"],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if scenario["tools"]:
            body["tools"] = scenario["tools"]
            if scenario["tool_choice"]:
                body["tool_choice"] = scenario["tool_choice"]
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        # llama.cpp defaults min_p=0.05 and top_k=40 and applies them ON TOP of
        # top_p. Zhipu/Unsloth's llama.cpp guidance is --min-p 0.01, so the
        # default truncates 5x harder than the model's authors recommend.
        if min_p is not None:
            body["min_p"] = min_p
        if top_k is not None:
            body["top_k"] = top_k
        if reasoning_effort:
            # GLM-5.3 reads reasoning_effort out of the chat template kwargs.
            body["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}

        trial = {"trial": i}
        try:
            resp, dt = _post(client, base_url, body, timeout)
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:400]
            trial.update(ok=False, reason=f"HTTP {e.response.status_code}: {detail}",
                         transport_error=True, latency_s=None)
            results.append(trial)
            continue
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            trial.update(ok=False, reason=f"transport: {type(e).__name__}: {e}",
                         transport_error=True, latency_s=None)
            results.append(trial)
            continue

        text, calls = _extract(resp)
        ok, reason = scenario["check"](text, calls)
        usage = resp.get("usage") or {}
        trial.update(
            ok=ok,
            reason=reason,
            transport_error=False,
            latency_s=round(dt, 2),
            n_tool_calls=len(calls),
            tool_names=[c.get("function", c).get("name") for c in calls],
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            # kept for post-hoc diagnosis; truncated so results files stay small
            text_head=text[:300],
            finish_reason=resp["choices"][0].get("finish_reason"),
        )
        results.append(trial)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8090/v1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--reasoning-effort", default=None,
                    help="passed as chat_template_kwargs.reasoning_effort")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--min-p", type=float, default=None,
                    help="llama.cpp default is 0.05; zai-org/Unsloth recommend 0.01")
    ap.add_argument("--top-k", type=int, default=None,
                    help="llama.cpp default is 40; not specified by the model authors")
    # 8192, not 2048: GLM-5.3 at the default Max reasoning effort routinely
    # spends >2000 tokens in <think> before emitting any content. At 2048 a
    # trivial "write binary_search" returned finish_reason=length with an empty
    # message — the bench was measuring the budget, not the model.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    wait_ready(args.base_url)

    wanted = set(args.only.split(",")) if args.only else None
    scenarios = [s for s in SCENARIOS if not wanted or s["id"] in wanted]

    per_scenario = {}
    with httpx.Client() as client:
        for s in scenarios:
            trials = run_scenario(
                client, args.base_url, s,
                model=args.model, trials=args.trials, timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                temperature=args.temperature, top_p=args.top_p,
                min_p=args.min_p, top_k=args.top_k,
                max_tokens=args.max_tokens,
            )
            passed = sum(1 for t in trials if t["ok"])
            # Lenient: the expected tool was called correctly, but extra calls
            # rode along. Distinguishes waste from a wrong action.
            lenient = sum(1 for t in trials
                          if t["ok"] or str(t.get("reason", "")).startswith(EXTRA_CALLS))
            lat = [t["latency_s"] for t in trials if t.get("latency_s") is not None]
            # The hard contract: required tool_choice must never yield zero calls.
            zero_call_violations = sum(
                1 for t in trials
                if s["id"] in REQUIRED_SCENARIOS
                and not t.get("transport_error")
                and t.get("n_tool_calls") == 0
            )
            per_scenario[s["id"]] = {
                "kind": s["kind"],
                "tool_choice": s["tool_choice"],
                "expect_tool": s["expect_tool"],
                "passed": passed,
                "trials": len(trials),
                "pass_rate": round(passed / len(trials), 3) if trials else None,
                "passed_lenient": lenient,
                "pass_rate_lenient": round(lenient / len(trials), 3) if trials else None,
                "median_latency_s": round(statistics.median(lat), 2) if lat else None,
                "required_zero_call_violations": zero_call_violations,
                "detail": trials,
            }
            if passed == len(trials):
                mark = "PASS"
            elif lenient == len(trials):
                mark = "EXTRA"      # right call every time, but with extras
            elif passed == 0:
                mark = "FAIL"
            else:
                mark = "FLAKY"
            extra = ""
            if zero_call_violations:
                extra = f"  <-- {zero_call_violations} REQUIRED-but-zero-tool-calls"
            reasons = {t["reason"] for t in trials if not t["ok"]}
            why = ("  " + "; ".join(sorted(reasons))[:160]) if reasons else ""
            print(f"{mark:5} {s['id']:28} {passed}/{len(trials)}{extra}{why}", flush=True)

    total_pass = sum(v["passed"] for v in per_scenario.values())
    total_lenient = sum(v["passed_lenient"] for v in per_scenario.values())
    total_trials = sum(v["trials"] for v in per_scenario.values())
    violations = sum(v["required_zero_call_violations"] for v in per_scenario.values())
    scen_pass = sum(1 for v in per_scenario.values() if v["passed"] == v["trials"])

    payload = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "trials_per_scenario": args.trials,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "top_k": args.top_k,
        "scenarios_fully_passed": scen_pass,
        "scenarios_total": len(scenarios),
        "trial_pass_rate": round(total_pass / total_trials, 3) if total_trials else None,
        "trial_pass_rate_lenient": round(total_lenient / total_trials, 3) if total_trials else None,
        "required_zero_call_violations": violations,
        "results": per_scenario,
    }

    print()
    print(f"scenarios fully passed : {scen_pass}/{len(scenarios)}")
    print(f"trial pass rate        : {total_pass}/{total_trials} strict, "
          f"{total_lenient}/{total_trials} lenient (extra calls forgiven)")
    print(f"REQUIRED violations    : {violations}   (must be 0)")

    if not args.no_save:
        path = save_result(f"agentic-{args.label}", payload)
        print(f"saved: {path}")
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
