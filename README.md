# GLM-5.3-Flash on the NVIDIA DGX Spark

The measured-optimal configuration for serving **GLM-5.3-Flash GGUF (`UD-IQ1_S`,
1-bit)** on a single **DGX Spark / GB10** (119 GiB unified memory) for
**autonomous agentic coding** — tool calls, code edits, shell commands, repo
analysis.

Every number below was measured on that hardware. The scripts that produced them
are in this repo and the raw results are in [`results/`](results/).

---

## The configuration

```bash
llama-server \
  --model .../GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf \
  --alias glm-5.3-flash --host 0.0.0.0 --port 8090 \
  --n-gpu-layers 999 --ctx-size 131072 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 --jinja \
  --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 \
  --reasoning-effort high --reasoning-budget 2048 \\
  --reasoning-format deepseek --reasoning-preserve
```

That is the command actually running, copied from `ps`, not a reconstruction.

| | |
|---|---|
| Model | `unsloth/GLM-5.3-Flash-GGUF` · **`UD-IQ1_S`** · 86.7 GiB, 3 shards |
| Arch | `glm5next` — 46 blocks, 288 experts (8 active), `nextn_predict_layers=1` |
| llama.cpp | [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) · `glm5next/upstream` · **`d07e71e`** (pinned deliberately — see section 7) |
| Resident | **90.9 GiB**, leaving ~22 GiB |
| Throughput | **27.5–29 tok/s** decode · ~283 tok/s prefill @4k |
| Context | 131072, Q8 KV |

**Do not pass `--spec-draft-model` / `-md`.** GLM-5.3-Flash ships its MTP head
*inside the same GGUF*. `--spec-type draft-mtp` alone loads it (+2.79 GiB).
Pointing `-md` at the GGUF loads a **second full copy of the weights** and
hard-locks the machine.

## Quick start

```bash
hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-IQ1_S/*" \
  --local-dir ~/models/GLM-5.3-Flash-UD-IQ1_S

./serving/glm53-build.sh                 # pinned llama.cpp build
sudo bash deploy/install-gpu-mem-guard.sh   # REQUIRED — see finding 1
cp deploy/glm53.service ~/.config/systemd/user/   # edit ExecStart path
systemctl --user daemon-reload && systemctl --user enable --now glm53
loginctl enable-linger "$USER"

./serving/glm53-health.sh
```

Edit [`serving/glm53.env`](serving/glm53.env) and `systemctl --user restart
glm53` to change anything. The unit contains no flags.

Paths are portable: `glm53.env` uses `${HOME}`, and `deploy/glm53.service` uses
systemd's `%h` specifier, so nothing is pinned to one account. Both assume the
repo is at `~/dev/glm-5.3-flash-GGUF-1bit-dgx-spark` and the weights at `~/models/...`; change
`GLM53_MODEL` / `GLM53_BIN` and the unit's `ExecStart` if you put them elsewhere.

Everything here targets **one** machine shape: a single GB10 with ~128 GB of
unified memory. The context and quant conclusions are specific to that; the
watchdog, MTP-depth and `reasoning_effort` findings generalise further.

---

## The single most important setting: `--reasoning-budget`

Everything else in this document is worth single-digit percentages. This one
decides whether the setup works at all.

Left unrestricted (llama.cpp's default), GLM-5.3 on a realistic agentic prompt
frequently reasons until it hits the output ceiling and returns **nothing the
client can act on** — no tool call, no content, after eleven minutes. Measured by
replaying a **real captured OpenCode request** (17,908 prompt tokens, 53 tool
definitions), n=20 per arm, identical input:

| | unrestricted | **`--reasoning-budget 2048`** |
|---|---|---|
| returned a tool call | 11/20 | **20/20** |
| hit the cap with nothing actionable | **9/20 (45%)** | **0/20** |
| over 8,000 completion tokens | 13/20 (65%) | 0/20 |
| median completion | 15,867 tok | **2,145 tok** |
| median turn latency | **11.2 min** | **1.6 min** |
| spread across runs | 166–16,384 (**144×**) | 2,092–2,297 (**1.1×**) |

Fisher exact on the cap-outs: **p ≈ 0.0008**.

Three things worth drawing out.

**The capped reasoning was not buying anything.** The tool calls returned under
the budget are the same ones the fast unrestricted runs produced. The extra
14,000 tokens of thinking did not lead somewhere better; it led nowhere.

**`reason_chars` sits at ~8,000 on every single budgeted run** — the budget binds
every time. On a prompt this size the model always wants to think longer than is
useful, so this is not clipping an occasional outlier; it is correcting a
systematic bias.

**Variance, not just latency, is the win.** 144× down to 1.1×. An agent loop with
a coin-flip chance of an eleven-minute stall is unusable regardless of its median.

### Verified across a real multi-turn conversation

The budget was first validated on single turns (n=20 replays of one captured
request). A conversation is where accumulated history could plausibly bring the
failure back, so it was tested end-to-end: a live agentic loop with tool calls
actually executed in a sandbox and results fed back.

**16 turns, 0 budget-exhausted, 5.1 minutes total.** The prompt grew 593 →
11,403 tokens with no degradation; completions stayed between 13 and 2,337
tokens. The agent built a working Python CLI with **10 passing tests**.

`reason_chars` per turn ran 0, 22, 81, 199, 249, 383, 605, 693, 711, 1476, 2470,
4194 — mostly **well under** the 2048-token cap. That matters: the budget is not
clipping normal operation, it only binds on the runaway turns. If 2048 were too
tight, every turn would sit pinned at the ceiling with truncated thinking.

Harness: `~/glm53-capture/agent_loop.py` (sandboxed; it executes model-generated
shell, which is why the sandbox is not a formality).

### The MCP tool surface: ~30s once per session, not per turn

A real OpenCode request carried **53 tools totalling 63,710 characters** — 82% of
a 17,908-token prompt. Disabling three servers (`chrome-devtools` 29 tools/25,429
chars, `shadcn` 7/4,940, `sequential-thinking` 1/4,259) leaves 16 tools and 29,008
chars, cutting the prompt to 9,689 tokens.

Replaying the same captured 16-turn session with both tool sets:

| seq | full: prompt / processed / prefill | trimmed: prompt / processed / prefill |
|---|---|---|
| 3 (first) | 17,908 / **17,908** / **64.8s** | 9,689 / — / ~34s cold |
| 4 | 18,822 / 914 / 4.3s | 10,603 / **914** / 3.9s |
| 5 | 19,049 / 227 / 1.5s | 10,830 / **227** / 1.4s |
| 6 | 26,877 / **7,828** / **36.7s** | 18,658 / **7,828** / 33.2s |
| 7–18 | 150–1,682 / 1.3–6.8s | identical `processed` |

**`processed` is identical on every turn.** The tool block sits at the front of
the prompt, so it is cached after turn 1 and never reprocessed. Trimming buys:

- **~30s once per session** (turn 1 prefill, 64.8s → ~34s)
- **~8% faster prefill thereafter** — attention over a shorter cached context
- **8,219 tokens of context permanently freed** (~8% more headroom before
  compaction, and compaction is expensive)

It does **not** save time per turn. An earlier draft of this document claimed it
did; that was wrong and only caught by measuring a full session.

**The bigger lever is visible in the same table.** Turn 6 costs 33–37s in *both*
arms because the conversation history jumped (18,358 → 46,453 chars). A single
history-growth turn costs as much as the entire tool block. If sessions feel
slow, shorter conversations beat a shorter tool list.

### Priority order, corrected

This document originally led with decode throughput. That was the wrong
emphasis, and the ordering below is what the measurements actually support:

1. **`--reasoning-budget 2048`** — 45% total-failure → 0%. Nothing else compares.
2. **Session hygiene** — start fresh conversations; a history-growth turn costs
   30s+ of prefill regardless of configuration.
3. **Trim the MCP tool surface** — ~30s off session start, ~8% more context.
4. **Quant choice** — decides whether the model runs at all.
5. **MTP depth, ubatch, samplers** — real, measured, single-digit percentages.

Items 4 and 5 occupied most of the tuning effort. Item 1 was a setting nobody
had touched.

### Why this was missed for so long

The bench suite measured **correctness** on ten single-step scenarios with a
small tool surface, and **decode throughput** on short prompts. It could not see
this: the failure needs a large tool surface and a multi-step task, and it
presents as latency rather than a wrong answer. Throughput was tuned to a ±3%
noise floor while a 45% total-failure rate sat unmeasured in a dimension nobody
sampled.

Five plausible causes were proposed and refuted before the real one was found —
`reasoning_effort`, tool-surface size, prompt wording, `--reasoning-preserve`,
and the budget itself on an earlier single-sample test. Every refutation that
rested on one sample was worthless against a distribution this wide. See
`~/glm53-capture/` for the capture proxy and the replay harness; **reproduce with
that, not with a reconstructed prompt** — five reconstructions all failed to
trigger it.

## Findings

### 1. The watchdog was killing the server — this is the big one

If you run a memory watchdog on a unified-memory box, **it will shoot your model
server**, because a ~91 GiB resident model is always the largest GPU process and
legitimately parks `MemAvailable` near any sane floor.

On 2026-08-31 `gpu-mem-guard.service` SIGKILLed this server **11 times in under
three hours** — `code=killed, status=9/KILL`, no core dump, no llama.cpp fault.
It presents exactly like a mysterious llama.cpp crash, and a supervisor script
restarting it makes it look like a flaky server rather than an external kill.

```
gpu-mem-guard: LOW MEM MemAvailable=20391MB ... < 20G floor
gpu-mem-guard: -> KILL biggest GPU pid=1584552 (91328MiB GPU) cmd=[...llama-server...]
```

The fix ([`deploy/install-gpu-mem-guard.sh`](deploy/install-gpu-mem-guard.sh))
exempts the managed unit and kills the next-biggest GPU process instead. Proven
under the identical condition that used to kill it:

```
LOW MEM MemAvailable=19012MB < 20G floor
all GPU processes are exempt: 1615937(glm53.service) — NOT killing.
```

Restarts after the fix, across every benchmark below: **0**.

### 2. Context is nearly free — the weights are the whole budget

Only **12 of the 46 blocks carry attention KV**. `head_count_kv` is a per-layer
array that is `0` for the other 34 — those are linear-attention layers whose
recurrent state does not grow with context.

Resident memory, measured with `nvidia-smi --query-compute-apps` after load:

| config | resident | delta |
|---|---|---|
| 49K, f16 | 88.94 GiB | — |
| 128K, f16 | 91.28 GiB | **+2.34** — 2.7× the context |
| 128K, q8_0 | 90.97 GiB | −0.32 — quantizing the KV |
| 256K, q8_0 | 94.25 GiB | +3.28 |

**2.7× the context costs 2.34 GiB and 0% throughput.** If you are running this
model at a small context to save memory, you are saving almost nothing — the
86.7 GiB of weights is the entire budget.

### 2b. q8_0 vs f16 KV barely matters here — pick either

A corollary of the above: if the KV cache is small, quantizing it saves little.
Measured at matched settings (128K, MTP depth 2, 3 trials per scenario):

| | q8_0 | f16 |
|---|---|---|
| resident | **90.97 GiB** | 91.28 GiB |
| decode | 27.75 tok/s | 27.86 tok/s |
| MTP acceptance | 0.640 | 0.633 |
| agentic strict | 22/30 | 22/30 |
| agentic lenient | 25/30 | 26/30 |

**No measurable difference in speed or quality**; q8_0 saves 0.32 GiB (0.35%).
This repo defaults to q8_0 because the saving is real and free, but **f16 is an
equally valid choice and is the safer one if your llama.cpp was built without
the q8_0 flash-attention kernels.**

`GGML_CUDA_FA_ALL_QUANTS=OFF` (the default, and what this repo builds) compiles
exactly four KV kernel variants — `f16/f16`, `bf16/bf16`, `q8_0/q8_0`,
`q4_0/q4_0` — and additionally **requires K and V to be the same type**
(`fattn.cu`: the `K->type != V->type` rejection exists only when the flag is
off). `q4_1`, `q5_0` and `q5_1` are refused as KV types outright. An unsupported
pairing does not error: it returns `BEST_FATTN_KERNEL_NONE` and llama.cpp
**silently falls back to non-flash attention**, so you get a working server that
is quietly slower.

Turning it `=ON` unlocks mixed K/V types and the remaining quants at roughly
double the compile time. Not worth it here: the only thing it buys is a smaller
KV cache, and the table above shows the KV cache is already so small that
quantizing it at all is worth 0.32 GiB against 86.7 GiB of weights.

Do not generalise this to models with conventional attention — there, KV
quantization is a large win. It is small *here* because 34 of 46 layers keep no
KV at all.

### 3. MTP draft depth 2 beats llama.cpp's default of 3

Swept with [`serving/mtp-sweep.sh`](serving/mtp-sweep.sh), 3 reps × 4 workloads:

| depth | decode | vs off | acceptance | code | diff | prose | json |
|---|---|---|---|---|---|---|---|
| off | 18.73 | 1.00× | — | 18.7 | 18.7 | 18.7 | 18.7 |
| 1 | 24.84 | 1.33× | 0.766 | 25.2 | 24.7 | 22.4 | 25.1 |
| **2** | **27.75** | **1.48×** | 0.640 | 28.6 | 27.9 | 23.3 | 27.6 |
| 3 *(default)* | 26.20 | 1.40× | 0.500 | 26.7 | 26.0 | 20.3 | 26.8 |
| 4 | 25.89 | 1.38× | 0.455 | 29.8 | 26.6 | 18.7 | 26.4 |
| 5 | 24.00 | 1.28× | 0.376 | 27.1 | 22.8 | 16.2 | 24.7 |

Acceptance falls faster than depth buys tokens. **By depth 4–5 the prose column
is at or below the no-MTP baseline** — wasted draft compute costs more than the
speculation saves.

MTP is genuinely active, not stubbed: every response carries
`timings.draft_n` / `draft_n_accepted`, and the off-arm reports `drafted=0`, so
"MTP disabled" can never be confused with "MTP enabled but never accepting".

### 3b. Stacking speculation types makes it worse, and n-grams alone do nothing

`--spec-type` takes a comma-separated **list**, so speculation strategies can be
combined, and the five `ngram-*` types need no draft model at all (only
`mtp`/`eagle3`/`dflash`/`dspark` download a sidecar). Adding one therefore costs
no memory — measured, all stacked arms load at exactly 90.8 GiB. It is a free
experiment, so it is worth knowing that it does not pay.

Swept with `serving/spec-sweep.sh`, 3 reps x 4 workloads:

| `--spec-type` | decode | acceptance | resident |
|---|---|---|---|
| **`draft-mtp`** | **26.90** | **0.625** | 90.8 GiB |
| `draft-mtp,ngram-map-k` | 26.46 | 0.559 | 90.8 GiB |
| `draft-mtp,ngram-cache` | 26.17 | 0.595 | 90.8 GiB |
| `draft-mtp,ngram-simple` | 25.66 | 0.537 | 90.8 GiB |
| `draft-mtp,ngram-mod` | 25.46 | 0.505 | 90.8 GiB |
| `ngram-mod` alone | **18.58** | 0.267 | 87.0 GiB |

Two things worth reading off this.

**Stacking dilutes acceptance.** Tokens drafted rises from ~1600 to ~2350 while
acceptance falls from 0.73 to ~0.50: the n-gram layer contributes drafts the
model then rejects, so you pay verification cost for tokens you throw away.
MTP's learned NextN head is simply a better predictor than prompt lookup here.

**Prompt-lookup speculation is worth nothing on this workload.** `ngram-mod`
alone measures **18.58 tok/s against 18.73 with speculation switched off
entirely** — the wasted drafts exactly cancel the gain. Agentic output is not as
repetitive as n-gram speculation assumes. It does save 3.8 GiB by not loading
the NextN heads, but giving up 31% of decode to reclaim memory we are not short
of is the wrong trade.

`draft-mtp` alone is what this repo ships, and nothing tried beat it.

### 3c. The noise floor is about 3%

Two runs of the *identical* speculation config (`draft-mtp`, depth 2) taken hours
apart measured **27.75** and **26.90** tok/s. That ~3% spread is the measurement
floor for decode on this box, and it is the number that decides which comparisons
above mean anything:

- `+ngram-map-k` at −1.6% is **inside** the floor — not better, but not shown worse
- `+ngram-mod` at −5.4% and `+ngram-simple` at −4.6% clear it — genuinely worse
- depth 2 over depth 3 at +5.9% clears it — that conclusion holds

Treat any single-digit-percent decode difference in this repo as unproven unless
it clears roughly 3%, and any tool-call difference as unproven unless it survives
a re-run at higher n — see the `min_p` episode in section 5b.

### 3d. The remaining MTP thresholds: all inside the noise floor

After depth (section 3), the knobs left govern *when* the drafter bothers rather
than how far it drafts. Swept with `serving/mtp-tune-sweep.sh`, 3 reps x 4
workloads:

| arm | decode | acceptance | vs baseline |
|---|---|---|---|
| baseline (`p-split 0.10`, `p-min 0.00`, `n-min 0`) | 28.21 | 0.596 | — |
| `--spec-draft-p-min 0.05` | 28.95 | 0.604 | +2.6% |
| `--spec-draft-p-split 0.05` | 28.60 | 0.629 | +1.4% |
| `--spec-draft-n-min 1` | 28.18 | 0.578 | −0.1% |
| `--spec-draft-p-split 0.20` | 27.91 | 0.579 | −1.1% |

**Every arm is inside the ±3% noise floor from section 3c, so none of them mean
anything.** Defaults kept.

This is the floor doing its job. `p-min 0.05` looks like a +2.6% win and it is
tempting to ship it; it is the same mistake as the n=5 `min_p` result in a
different costume. A number that cannot clear the measurement noise of the
harness that produced it is not a result.

### 4. `reasoning_effort` only accepts `low` and `high`

Straight from the chat template embedded in the GGUF:

```jinja
{%- set effective_reasoning_effort =
      reasoning_effort if reasoning_effort in ['low','high'] else 'max' -%}
```

Any other value — **including the literal string `"max"`** — falls through to
`'max'`, so an **unset** value means Max. Measured: unset → 5374 reasoning
chars; `low` → 35.

**This repo ships `high`, not Max — and that reversed the original assumption.**
More deliberation was supposed to mean better planning and tool selection. It
measured worse:

| `reasoning_effort` | full suite (5 trials) | focused re-run (12 trials) |
|---|---|---|
| unset (Max) | 38/50 strict | 16/24 |
| **`high`** | **44/50** | **21/24** |
| `low` | 43/50 | — |

Same direction in both runs, and the confirmation *strengthened* the effect
(p≈0.12 → ≈0.087) instead of dissolving it the way the `min_p` false positive
did — which is the only reason it was acted on. The failure mode is legible: on
a task whose file contents were already inline in the prompt, Max talked itself
into calling `read_file` "to verify" before patching. Extra caution on a task
with an unambiguous correct action is just a wasted turn. `high` is cheaper too,
being fewer reasoning tokens.

The honest caveat: p≈0.087 is suggestive, not conclusive. It is acted on because
it reproduced, the direction never flipped, and switching costs nothing.

**Tool-call safety is not affected by this knob at all.** Zero
`tool_choice=required` violations at Max, `high` *and* `low`, across every arm —
so the low-reasoning lane is safe to use, it simply is not the agent default.

Do **not** pin `low` globally; it starves the planning turns. Expose it per
request instead:

```json
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

`--reasoning-preserve` maps to template `clear_thinking=false` (keep all history
reasoning). `--no-reasoning-preserve` gives `clear_thinking=true`, dropping
reasoning from turns before the last user message — much cheaper on long runs.

### 5. Use `top_p=0.95`, not `1.0`

Unsloth's guide for **this GGUF** says `temperature=1.0, top_p=0.95` for most
tasks. The `top_p=1.0` in the *base model card's* benchmark footnotes is
zai-org's full-precision evaluation setting — **not** run guidance for a
quantized GGUF.

Measured here, same server, only `top_p` varied, 5 trials:

| scenario | top_p 1.0 | top_p 0.95 |
|---|---|---|
| long system prompt + required tool call | **1/5** | **5/5** |
| large tool set | 3/5 | 3/5 |

`top_p 1.0` produced duplicate `run_shell` calls.

### 5b. Declare `min_p` and `top_k` — llama.cpp applies them either way

`llama-server` defaults to `min_p=0.05` and `top_k=40` and applies both **on top
of** whatever `top_p` you set. A config that mentions only temperature and
top_p is not running "just those two" — it is running four samplers, two of
which it never chose and which can change between llama.cpp versions.

This repo sets them explicitly:

```
--temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 0
```

`min_p 0.01` is Zhipu's and Unsloth's documented llama.cpp line for GLM-5.3.
`top_k` is specified by neither, so it is disabled rather than left at 40.

**Measured, and the honest answer is that the values barely matter.** Three
arms, 50 trials each, identical server:

| min_p | top_k | strict | lenient |
|---|---|---|---|
| 0.05 | 40 | 0.78 | 0.88 |
| 0.01 | 40 | 0.84 | 0.88 |
| **0.01** | **0** | **0.86** | **0.90** |

The trend is monotonic but **not significant** (39/50 vs 43/50, p≈0.30). A
focused 12-trial re-run of the two scenarios that appeared to move found no
effect at all — scenario 06 scored **7/12 either way**. The 3/5→5/5 that looked
like a fix in the 5-trial run was noise.

So this change is made for **provenance, not performance**: it matches the model
authors' guidance and pins behaviour against llama.cpp changing its defaults.
It is not a speedup and not a quality win, and the repo should not claim it is.

The wider lesson, recorded because it nearly went the other way: a 5-trial
signal was about to be used to rewrite the "known model behaviours" table below
and declare scenario 06 fixed. Twelve trials said otherwise. **n=5 is not
evidence at this effect size.**

### 5c. `ubatch` 512: prefill keeps scaling, but 1024 breaks the watchdog

`--ubatch-size` is the physical batch used for prompt processing. Swept with
`serving/ubatch-sweep.sh`, 2 reps per depth, fresh nonce per prompt:

| ubatch | resident | MemAvailable | prefill @4k | @16k | @64k | decode |
|---|---|---|---|---|---|---|
| 128 | 89.53 GiB | 23.2 | 136 | 153 | 139 | 27.78 |
| 256 | 89.96 GiB | 23.3 | 200 | 216 | 180 | 25.92 |
| **512** | **90.83 GiB** | **22.0** | **283** | **282** | **215** | **27.43** |
| 1024 | 92.35 GiB | **19.4** | 360 | 343 | 243 | ~28 |

Two findings, one of which contradicts the reason the sweep was run.

**Prefill never plateaus.** This was set up expecting to find memory savings by
going *down*; instead every doubling buys 13–27% more prefill, and 1024 is still
climbing. 512 is not an over-large default — if anything it is conservative.

**Decode is untouched**, as theory predicts: `ubatch` governs prompt processing,
not generation. The 25.92 at ubatch 256 tracks that arm's lower MTP acceptance
(0.593), not the batch size, and sits inside the ~3% noise floor besides.

**512 stays the default anyway, and the reason is the watchdog, not throughput.**
1024 costs 1.52 GiB and puts MemAvailable at 19.4 GiB — permanently *below*
`gpu-mem-guard`'s 20 GiB floor. The server is exempt so it would not be killed,
but the guard would log `LOW MEM` every 3 seconds forever, and an alarm that is
always firing is an alarm nobody reads. Losing the ability to detect a real
memory event is a worse outcome than 27% slower prefill on cache misses —
especially since llama.cpp reuses prompt prefixes, so prefill cost falls mainly
on genuinely new content rather than on every turn.

Set `GLM53_UBATCH=1024` if your workload is prefill-dominated and you accept
that trade. Anything above 1024 would need the guard floor moved first.

**Note on the numbers above:** `bench_prefill.py` flags a run when a material
share of a prompt came from cache. On this sweep it fired on *every* arm — but
the cause was `cache_n = 9` on each request, the shared chat-template prefix,
which is 0.01% of a 73k prompt and identical across arms. The guard now
thresholds on a fraction (1%) rather than `> 0`, because a check that fires on
every run is one nobody heeds. The measurements themselves are uncontaminated.

### 5d. `--n-cpu-moe` frees nothing here — and `nvidia-smi` will lie to you about it

On a discrete GPU, moving MoE expert weights to the CPU trades VRAM for speed.
On GB10 the GPU's memory **is** system RAM, so there is no second pool to move
them to. Measured with `serving/moe-offload-sweep.sh`:

| `--n-cpu-moe` | `nvidia-smi` reports | **MemAvailable** | decode | acceptance |
|---|---|---|---|---|
| **0** | 90.83 GiB | **22.0 GiB** | **27.11** | 0.594 |
| 4 | 89.19 | 20.8 | 25.48 | 0.588 |
| 8 | 82.23 | 21.6 | 22.07 | 0.602 |
| 16 | **66.85** | **21.2** | **18.04** | 0.594 |

**`nvidia-smi` drops 24 GiB while real headroom does not move.** The bytes are
relocated from GPU allocations into CPU buffers; on unified memory that is the
same physical RAM, so `MemAvailable` sits at 21–22 GiB across every arm. You pay
33% of decode and receive nothing.

This is worth knowing precisely because it is an active trap: anyone sizing a
Spark by reading GPU memory would see 66.85 GiB used at `--n-cpu-moe 16`,
conclude ~50 GiB was free, and try to load a quant into memory that does not
exist. **On this hardware, `MemAvailable` is the number that means something and
`nvidia-smi` is not.** The same applies to `-cmoe` and to `-ot` overrides
targeting expert tensors.

MTP is unaffected — acceptance holds at ~0.59 across all arms — so the loss is
purely the cost of evaluating experts CPU-side.

### 6. Quant ceiling on 128 GB

| quant | weights | resident @128K | MemAvailable |
|---|---|---|---|
| **UD-IQ1_S** | 86.7 GiB | **90.9 GiB** | **22.1 GiB** |
| UD-IQ2_XXS | 94.8 GiB | 99.0 GiB | 13.3 GiB |
| UD-Q2_K_XL | 101.2 GiB | ~105 GiB | ~8 GiB |

Unsloth's own hardware table (total memory, unified): 1-bit **100 GB**, 2-bit
**115 GB**, 3-bit **128–150 GB**. This box has 128.5 GB, so 3-bit is out and
2-bit is tight.

**IQ1_S vs IQ2_XXS**, benchmarked identically (128K, q8 KV, MTP 2, 3 trials):

| | IQ1_S | IQ2_XXS |
|---|---|---|
| decode | 27.75 t/s | 27.5 t/s |
| MTP acceptance | 0.640 | **0.662** |
| agentic strict | 22/30 | **24/30** |
| agentic lenient | 25/30 | **27/30** |
| MemAvailable | **22.1 GiB** | 13.3 GiB |

**Stay on IQ1_S.** IQ2_XXS is better on every quality axis and identical on
speed, but at n=30 that difference is inside binomial noise — a hint, not a
result. The 8.8 GiB it costs is certain. Re-run with `--trials 15+` to settle it.

### 7. A newer llama.cpp is faster and breaks tool calls — stay pinned

The pin is `d07e71e`, and the branch head has moved on. `949f7ef` (PR #27754's
head, 14 commits ahead) was built into a separate worktree and measured
back-to-back against the pinned binary, same model, same config:

| | `d07e71e` (shipped) | `949f7ef` |
|---|---|---|
| load | 37 s | 46 s |
| resident | 93008 MiB | 93007 MiB |
| decode | 27.5 tok/s | **28.82** (+4.8%) |
| MTP acceptance | 0.568 | 0.587 |
| agentic strict | 42/50 | 42/50 |
| `required` violations | 0 | 0 |

+4.8% clears the ±3% floor, so the newer build genuinely is faster — plausibly
the CUDA MoE fast path (`mm_ids_helper` for any `n_expert_used`) that landed in
those commits, which matters for a model routing 8 of 288 experts per token.

**It is not shipped, because it produced a failure class the pinned build never
has.** Scenario 09 returned `arguments were not valid JSON` — malformed JSON
*inside* a tool call's arguments. Confirmed at 12 trials:

| scenario 09, 12 trials | result |
|---|---|
| `d07e71e` | **12/12 pass** |
| `949f7ef` | **9/12**, including invalid-JSON arguments |

Every scenario-09 failure on the pinned build across this entire campaign has
been a *wrong tool* — `read_file` instead of `apply_patch`. That is a judgement
call an agent loop recovers from. Malformed arguments are a serialisation fault:
the client receives a tool call it cannot parse at all, which is exactly what
breaks an agent session mid-run. Invalid JSON appeared in **both** `949f7ef` runs
and in **zero** runs of the pinned build.

4.8% decode does not buy that risk. The candidate build is kept on disk
(`~/llama.cpp-glm5-949f7ef`) so re-testing is only a `GLM53_BIN` change, and the
right moment to revisit is when `glm5next` **merges to upstream master** rather
than while it is a moving branch. Rebuild any candidate with
`serving/build-candidate.sh <commit>`, which uses a git worktree so the serving
tree and running binary are never touched.

### 7b. 256K works, but is not the default

Loads in 37 s, serves tool calls correctly, resident 94.2 GiB — but leaves only
**16.8 GiB** for the OS, containers, the agent and its shell commands. Opt in per
run with `GLM53_CTX=262144`. **Do not attempt 1M**: the KV alone would be ~13 GiB.

---

## Agentic reliability

[`benches/agentic_spec.py`](benches/agentic_spec.py) defines 10 scenarios: text,
code, JSON, one required tool call, multi-turn tool use, a 17-tool set,
tool-then-reasoning, a 2600-word system prompt, patch generation, shell
generation. Four are sent with `tool_choice="required"`.

**A 200 OK carrying zero tool calls on a `required` request is scored as a
serving failure, not a model miss** — separately reported, and it makes
`glm53-health.sh` return UNHEALTHY. Across every run here: **0 violations**.

Scoring is strict *and* lenient, because "called the wrong tool" and "called the
right tool plus a redundant one" are different problems.

### Known model behaviours (not fixable by configuration)

| scenario | result | note |
|---|---|---|
| tool-then-reasoning | **0/3 on both quants**, 0/5 at both top_p values | Told "explain in prose, do not call another tool" after a tool result, it calls `read_file` anyway. The one to watch in an agent loop. |
| large tool set | 2–3/3 | Picks the right tool every time, then tags on a spurious extra. Waste, not a wrong action. |
| patch generation | 1/3 (IQ1_S) · 3/3 (IQ2_XXS) | Sometimes re-reads a file whose contents were already inline. |

```bash
make test        # harness unit tests — no GPU, no server
make bench       # health + agentic + MTP
make mtp-sweep   # draft-depth sweep
```

## Layout

| path | |
|---|---|
| `serving/glm53.env` | every tunable; the only file you normally edit |
| `serving/glm53-up.sh` | builds the command line; what the unit execs |
| `serving/glm53-health.sh` | health check incl. the required-tool-call contract |
| `serving/glm53-build.sh` | reproducible llama.cpp build, pinned to the commit |
| `serving/preflight.sh` | refuses a manual launch with no headroom (`GLM53_PREFLIGHT=1`) |
| `serving/mtp-sweep.sh` | MTP draft-depth sweep |
| `benches/agentic_spec.py` | the 10 scenarios + verdict functions |
| `benches/bench_agentic.py` | runs them; separates serving from model failures |
| `benches/bench_mtp.py` | decode t/s + MTP acceptance from server timings |
| `deploy/` | host-level files (watchdog, systemd unit, OpenCode provider) |
| `results/` | raw JSON behind every number above |
| `LICENSE` | MIT |

## License

MIT — see [LICENSE](LICENSE).

The pinned llama.cpp fork and the GGUF weights are third-party and carry their
own licences: [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) (MIT,
following upstream ggml-org/llama.cpp) and
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
(MIT, per its model card). Neither is vendored here.

## Provenance

```
llama.cpp  https://github.com/unslothai/llama.cpp
           branch glm5next/upstream
           commit d07e71ede795b6ab60bb46d9212a6c584e4b2272  ("Add MTP support")
model      unsloth/GLM-5.3-Flash-GGUF  ·  UD-IQ1_S
hardware   NVIDIA DGX Spark (GB10), 128.5 GB unified, driver 580.126.09, CUDA 13.0
measured   2026-08-31
```

Upstream llama.cpp did not carry the `glm5next` NextN/MTP graph when this was
built; the unslothai branch does, and it is where `--spec-type draft-mtp` comes
from. The build pins the commit deliberately — the MTP graph is young code, and a
silent change shows up as an acceptance-rate regression, not a build error.
