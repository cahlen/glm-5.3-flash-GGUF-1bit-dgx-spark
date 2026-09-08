#!/usr/bin/env bash
# Find the FLOOR of --reasoning-budget.
#
# 1024/2048/4096 were all measured at the real 17,908-token prompt and all gave
# zero cap-outs. That establishes a ceiling of safety but never locates the
# cliff: somewhere below 1024 the cap must start cutting reasoning the model
# actually needed, and without knowing where, "2048 has margin" is an assumption.
#
# Raised by dipankarsarkar on the HF post, who correctly pointed out that the
# scenario suite could never have tested this — its largest completion is ~300
# tokens, so a 1024 cap cannot bind on it.
#
# A too-tight cap does NOT show up as a cap-out (the cap prevents those). It
# shows up as worse decisions: wrong tool, malformed arguments, or a first action
# that makes no sense. So this measures both the replay (does a sane tool call
# come back) and the agent loop (does it still build working software).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The captured request is NOT shipped with this repo -- it contains real source
# files from the session it was recorded in. Record your own and point at it:
CAPTURE="${GLM53_CAPTURE:-$HOME/glm53-capture/opencode_capture.jsonl}"
D="$HOME/.config/systemd/user/glm53.service.d"; mkdir -p "$D"
DROPIN="$D/99-floor.conf"
REPS="${REPS:-10}"
LOOPS="${LOOPS:-3}"

cleanup() {
  echo "== restoring glm53.env =="
  rm -f "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
}
trap cleanup EXIT

for b in 128 256 512; do
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

  echo "--- replay of the real 17,908-token request, n=$REPS ---"
  python3 "$HARNESS/replay_real.py" "$CAPTURE" 3 "$REPS"

  echo "--- agent loop x$LOOPS: does it still build working software? ---"
  for r in $(seq 1 "$LOOPS"); do
    SB="${TMPDIR:-/tmp}/floor_b${b}_r${r}"
    out=$(python3 "$HARNESS/agent_loop.py" "$SB" 20 2>&1)
    turns=$(printf '%s\n' "$out" | grep -cE "^[0-9]+ +[0-9]+ +[0-9]+")
    wall=$(printf '%s\n' "$out" | grep -oE "total wall: [0-9.]+ min" | grep -oE "[0-9.]+")
    exh=$(printf '%s\n' "$out" | grep -oE "budget-exhausted turns: [0-9]+" | grep -oE "[0-9]+$")
    dn=$(printf '%s\n' "$out" | grep -c "Agent reported DONE" || true)
    if [ -x "$SB/.venv/bin/python" ]; then
      pt=$(cd "$SB" && ./.venv/bin/python -m pytest -q 2>&1 | tail -1)
    else pt="(no venv)"; fi
    passed=$(printf '%s\n' "$pt" | grep -oE "[0-9]+ passed" | grep -oE "^[0-9]+")
    failed=$(printf '%s\n' "$pt" | grep -oE "[0-9]+ failed" | grep -oE "^[0-9]+")
    printf 'FLOOR budget=%-5s run=%s turns=%-3s wall=%-6s exhausted=%-3s done=%-2s tests_passed=%-4s tests_failed=%-4s\n' \
      "$b" "$r" "${turns:-?}" "${wall:-?}" "${exh:-?}" "${dn:-0}" "${passed:-0}" "${failed:-0}"
    rm -rf "$SB"
  done
done
