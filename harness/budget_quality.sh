#!/usr/bin/env bash
# Does a tighter reasoning budget make WORSE decisions?
#
# 1024 measured 2x faster than 2048 with identical reliability (10/10 tool calls,
# 0 cap-outs each). But the replay only checks that a tool call comes back, not
# that it is the right one. Recommending 1024 on latency alone would repeat the
# mistake this whole investigation kept making: measuring the easy axis.
#
# Two measures per arm:
#   1. the agentic suite  — 11 scenarios, strict/lenient correctness, tool choice
#   2. the real agent loop — does it still build software whose tests pass
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HARNESS/.." && pwd)"
D="$HOME/.config/systemd/user/glm53.service.d"; mkdir -p "$D"
DROPIN="$D/99-budgetquality.conf"
TRIALS="${TRIALS:-5}"

cleanup() {
  echo "== restoring glm53.env =="
  rm -f "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
}
trap cleanup EXIT

for b in 1024 2048; do
  echo
  echo "################################ budget=$b ################################"
  printf '[Service]\nEnvironment="GLM53_REASONING_BUDGET=%s"\n' "$b" > "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
  ok=0
  for _ in $(seq 1 80); do
    [ "$(curl -s --max-time 2 http://127.0.0.1:8090/health 2>/dev/null)" = '{"status":"ok"}' ] && { ok=1; break; }
    sleep 3
  done
  [ "$ok" = 1 ] || { echo "SERVER FAILED at budget=$b"; continue; }
  echo "in force: $(ps -eo args | grep '[l]lama-server' | grep -oE -- '--reasoning-budget [0-9]+')"

  echo "--- agentic suite (correctness) ---"
  ( cd "$REPO" && uv run python benches/bench_agentic.py \
      --base-url http://127.0.0.1:8090/v1 --label "budget${b}-quality" \
      --trials "$TRIALS" )

  echo "--- real agent loop (does it build working software?) ---"
  python3 "$HARNESS/agent_loop.py" "${TMPDIR:-/tmp}/sandbox_b${b}" 16 2>&1 
done
