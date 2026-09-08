# Harness

The scenario suite in [`../benches/`](../benches/) measures tool-call
correctness on small prompts. It cannot measure reasoning length, latency, or
anything that only appears at production prompt size — its largest completion is
288 tokens, so a 1024-token reasoning budget can never bind on it.

Every claim in the README about the reasoning budget comes from this directory
instead: replaying **one real captured OpenCode request** (17,908 prompt tokens,
53 tool definitions) against the live server, and running a real multi-turn
agent loop whose tool calls actually execute.

## The captured request is deliberately not shipped

`opencode_capture.jsonl` holds real requests recorded off a working session:
source files, paths, and task context from actual work. It is not in this repo
and should not be. **Record your own.**

Any transparent proxy in front of the server will do. The format is one JSON
object per line:

```json
{"seq": 3, "body": "{\"model\": \"glm-5.3-flash\", \"messages\": [...], \"tools\": [...]}"}
```

`body` is the verbatim request body as a JSON **string**. `seq` is any integer
you pass to the scripts to select which captured turn to replay. That is the
entire contract.

Point the scripts at your capture with `GLM53_CAPTURE=/path/to/capture.jsonl`,
and at a non-local server with `GLM53_URL=http://host:port`.

**Replay a captured request, not a reconstructed one.** Five attempts to
reproduce the runaway from a hand-written prompt all failed and each produced a
different wrong theory. The real body reproduced it immediately.

## Scripts

| script | what it does | produced |
|---|---|---|
| `replay_real.py` | replays one captured request N times, reports completion length, finish reason, tool calls, cap-outs | the headline table |
| `agent_loop.py` | real multi-turn loop; executes the model's tool calls in a disposable sandbox, then scores whether the software it built passes its own tests | every agent-loop result |
| `budget_sweep.sh` | replay sweep across budget values (`off` = unrestricted) | `results/20260904-budget-sweep-*`, `results/20260908-budget-headline-rerun.txt` |
| `budget_floor.sh` | replay + agent loops at 128/256/512, looking for the floor | `results/20260908-budget-floor-*` |
| `budget_quality.sh` | scenario suite + agent loop per budget arm | `results/20260904-budget-quality-*` |
| `budget_agentloop_ab.sh` | n=5 agent loops per arm, 1024 vs 2048 | `results/20260904-agentloop-*` |

Reproduce the headline comparison with:

```bash
REPS=20 ./harness/budget_sweep.sh off 2048
```

The sweep scripts install a temporary systemd drop-in to change
`--reasoning-budget`, restart the server between arms, and remove the drop-in on
exit — including on failure. They will interrupt anything else using the server.

## `agent_loop.py` executes model-generated shell

That is the point of it, and the sandbox is not a formality. Commands run with a
minimal `PATH`, no inherited environment, `HOME` and `TMPDIR` pointed at a
disposable working directory that is deleted and recreated per run, a 60s
timeout, and a denylist rejecting privilege escalation, network access, system
paths, and absolute paths outside the sandbox. Read it before you run it.

The venv is provisioned up front because network access is refused inside the
sandbox. The first version of this harness omitted that, and the agent spent ten
turns hunting for an interpreter — which measured the denylist, not the model.
