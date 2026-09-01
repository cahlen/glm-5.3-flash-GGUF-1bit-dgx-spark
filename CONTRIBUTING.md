# Contributing

This repo is a **measured** configuration, not a collection of opinions. The
bar for changing a number is that you measured it on the hardware.

## The one rule

**Every performance or reliability claim must come with the run that produced
it.** `results/` holds the raw JSON behind every figure in the README. If you
change a default, add the result file that justifies it.

This is not ceremony. Two claims in this repo's own history were wrong because
they were calculated and presented as measured:

- q8_0 KV was documented as saving ~1.4 GiB. Measured, it saves **0.32 GiB**,
  and a matched A/B showed no speed or quality difference against f16 at all.
- `top_p=1.0` was taken from the base model card's benchmark footnotes. Those
  are the full-precision evaluation settings, not run guidance for a quantized
  GGUF — and 1.0 measurably *hurt* tool-call reliability (1/5 vs 5/5).

## Running the tests

```bash
uv sync --group dev
uv run --group dev pytest -q        # no GPU, no model, no server
```

On the machine actually running the server, also run the drift checks that
compare `deploy/` against the installed originals:

```bash
GLM53_DEPLOY_HOST=1 uv run --group dev pytest tests/test_deploy_sync.py
```

They are opt-in because an unrelated machine may have different files at those
paths, and comparing against them produces a baffling false failure.

## Things that will bite you

- **Never add `--spec-draft-model` / `-md`.** GLM-5.3-Flash carries its MTP head
  inside the same GGUF; pointing `-md` at the GGUF loads a *second* full copy of
  the weights and hard-locks the machine.
- **Don't "fix" a failing drift test** in `tests/test_deploy_sync.py` by editing
  the test. Copy whichever file is newer over the other.
- **Don't make tests read `$HOME`.** They should be self-contained to the repo;
  anything else is a false failure waiting to happen on someone else's box.
- **Re-run `serving/mtp-sweep.sh` if the model, quant, or llama.cpp commit
  changes.** The optimal draft depth is specific to all three.
- Flags belong in `serving/glm53.env`, never in the systemd unit — otherwise the
  launch the unit performs and the launch the benches perform drift apart.

## Scope

Everything here targets one machine shape: a single GB10 / DGX Spark with
~128 GB of unified memory. The context and quant conclusions are specific to
that. The watchdog, MTP-depth and `reasoning_effort` findings generalise
further.
