#!/usr/bin/env bash
# Settle 1024 vs 2048 with enough agent-loop runs to mean something.
#
# The earlier comparison was n=1 per arm: 2048 produced 16 passing tests, 1024
# produced 11, and 2048 was kept on that. One build is not evidence — the same
# error this investigation made repeatedly. Five runs per arm.
#
# Per run: turns used, wall time, budget-exhausted turns, and whether the
# software it built actually passes its own tests.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="$HOME/.config/systemd/user/glm53.service.d"; mkdir -p "$D"
DROPIN="$D/99-agentloopab.conf"
RUNS="${RUNS:-5}"

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

  for r in $(seq 1 "$RUNS"); do
    SB="${TMPDIR:-/tmp}/ab_b${b}_r${r}"
    out=$(python3 "$HARNESS/agent_loop.py" "$SB" 18 2>&1)
    turns=$(printf '%s\n' "$out" | grep -cE "^[0-9]+ +[0-9]+ +[0-9]+")
    wall=$(printf '%s\n' "$out" | grep -oE "total wall: [0-9.]+ min" | grep -oE "[0-9.]+")
    exh=$(printf '%s\n' "$out" | grep -oE "budget-exhausted turns: [0-9]+" | grep -oE "[0-9]+$")
    done_flag=$(printf '%s\n' "$out" | grep -c "Agent reported DONE" || true)
    # Did it build something that works?
    if [ -x "$SB/.venv/bin/python" ]; then
      pt=$(cd "$SB" && ./.venv/bin/python -m pytest -q 2>&1 | tail -1)
    else
      pt="(no venv)"
    fi
    passed=$(printf '%s\n' "$pt" | grep -oE "[0-9]+ passed" | grep -oE "^[0-9]+")
    failed=$(printf '%s\n' "$pt" | grep -oE "[0-9]+ failed" | grep -oE "^[0-9]+")
    printf 'budget=%-5s run=%s  turns=%-3s wall=%-6s exhausted=%-3s done=%-2s tests_passed=%-4s tests_failed=%-4s\n' \
      "$b" "$r" "${turns:-?}" "${wall:-?}" "${exh:-?}" "${done_flag:-0}" "${passed:-0}" "${failed:-0}"
    rm -rf "$SB"   # reclaim the venv (~50MB each)
  done
done
