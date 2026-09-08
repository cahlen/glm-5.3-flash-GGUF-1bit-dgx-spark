"""Replay a CAPTURED OpenCode request N times and measure the spread.

Five attempts to reproduce the production runaway from a reconstructed prompt all
failed, and each failure produced a wrong theory. This replays the real thing:
the exact body OpenCode sent, 53 tool definitions and ~19.4k prompt tokens,
straight off the capture proxy.

The question is no longer "what causes it" but "how often does it happen".
The first captured turn completed normally in ~80s, while the same prompt in an
earlier session burned 14,949 tokens — so the behaviour is intermittent and only
a distribution can describe it.

    python3 replay_real.py <capture.jsonl> <seq> [reps]

stream is forced to False so usage.completion_tokens is available directly.
That is a deviation from the captured body and is called out in the output;
sampling should not depend on transport, but it is an assumption, not a fact.
"""
import json
import statistics
import sys
import time
import urllib.request

CAPTURE = sys.argv[1]
SEQ = int(sys.argv[2])
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
URL = "http://127.0.0.1:8090/v1/chat/completions"

body = None
for line in open(CAPTURE):
    rec = json.loads(line)
    if rec["seq"] == SEQ:
        body = json.loads(rec["body"])
        break
if body is None:
    sys.exit(f"seq {SEQ} not found in {CAPTURE}")

streamed = body.get("stream")
body["stream"] = False

print(f"replaying capture seq={SEQ}: {len(body.get('messages', []))} messages, "
      f"{len(body.get('tools') or [])} tools, max_tokens={body.get('max_tokens')}, "
      f"model={body.get('model')}")
print(f"(captured stream={streamed}, forced to False so usage is reported)")
print()
print(f"{'rep':<4} {'prompt_tok':>11} {'completion':>11} {'reason_ch':>10} "
      f"{'finish':<12} {'wall':>8}  calls", flush=True)

toks, walls, fins, exhausted = [], [], [], 0
for rep in range(REPS):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
    except Exception as e:
        print(f"{rep:<4} ERROR {type(e).__name__}: {e}", flush=True)
        continue
    dt = time.monotonic() - t0
    ch = d["choices"][0]
    m = ch["message"]
    u = d.get("usage", {})
    calls = [c["function"]["name"] for c in (m.get("tool_calls") or [])]
    ct = u.get("completion_tokens") or 0
    fr = ch.get("finish_reason")
    toks.append(ct)
    walls.append(dt)
    fins.append(fr)
    if fr == "length" and not calls and len((m.get("content") or "").strip()) < 40:
        exhausted += 1
    print(f"{rep:<4} {u.get('prompt_tokens', 0):>11} {ct:>11} "
          f"{len(m.get('reasoning_content') or ''):>10} {str(fr):<12} {dt:>7.1f}s  {calls}",
          flush=True)

if toks:
    print()
    print(f"n={len(toks)}  median={statistics.median(toks):.0f} tok  "
          f"min={min(toks)}  max={max(toks)}  "
          f"mean={statistics.mean(toks):.0f}")
    print(f"wall: median={statistics.median(walls):.1f}s  max={max(walls):.1f}s")
    print(f"finish_reasons: {dict((f, fins.count(f)) for f in set(fins))}")
    print(f"budget-exhausted (length + no call + no content): {exhausted}/{len(toks)}")
    over8k = sum(1 for t in toks if t > 8000)
    print(f"runs over 8000 tokens: {over8k}/{len(toks)}")
