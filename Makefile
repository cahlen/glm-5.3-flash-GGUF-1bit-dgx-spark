SHELL := /bin/bash
# Recipes that pipe need pipefail, or a failed command still reports success.
.SHELLFLAGS := -eu -o pipefail -c

URL    ?= http://127.0.0.1:8090/v1
LABEL  ?= glm-5.3-flash
TRIALS ?= 3

.PHONY: setup test lint up down restart health agentic mtp mtp-sweep bench

setup:
	uv sync

# Harness unit tests. No GPU, no server, ~1s. Run this before trusting a number.
test:
	uv run --group dev pytest -q

# The exact command CI runs. Run it BEFORE pushing: three commits in this repo's
# history were pushed red because CI was used as the linter instead of this.
# shellcheck is a python package, so it needs no root: uv tool install shellcheck-py
lint:
	shellcheck -S warning serving/*.sh deploy/*.sh

up:      ; systemctl --user start glm53
down:    ; systemctl --user stop glm53
restart: ; systemctl --user restart glm53
health:  ; @./serving/glm53-health.sh

# The 10 agentic scenarios. Exits non-zero if any tool_choice=required request
# returned zero tool calls — a serving fault, not a model miss.
agentic:
	uv run python benches/bench_agentic.py --base-url $(URL) --label $(LABEL) --trials $(TRIALS)

# Decode throughput + MTP acceptance at whatever serving/glm53.env sets.
mtp:
	uv run python benches/bench_mtp.py --base-url $(URL) --label $(LABEL)

# Restarts the service once per arm; restores glm53.env on exit.
mtp-sweep:
	./serving/mtp-sweep.sh

# Correctness first, then speed.
bench: health agentic mtp
