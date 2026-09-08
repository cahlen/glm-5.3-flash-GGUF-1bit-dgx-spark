#!/usr/bin/env bash
# Sweep --reasoning-budget. 2048 was picked because it worked, not because it
# was measured against alternatives. Too tight would truncate thinking on hard
# problems; too loose would let the runaway back in.
#
# Each arm replays the SAME captured production request n times, so the
# comparison is against the input that actually produced the failure.
set -euo pipefail
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The captured request is NOT shipped with this repo -- it contains real source
# files from the session it was recorded in. Record your own and point at it:
CAPTURE="${GLM53_CAPTURE:-$HOME/glm53-capture/opencode_capture.jsonl}"
D="$HOME/.config/systemd/user/glm53.service.d"; mkdir -p "$D"
DROPIN="$D/99-budgetsweep.conf"
REPS="${REPS:-10}"

cleanup() {
  echo "== restoring glm53.env =="
  rm -f "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
}
trap cleanup EXIT

for b in "$@"; do
  echo
  echo "################ reasoning-budget=$b ################"
  if [ "$b" = "off" ]; then
    printf '[Service]\nEnvironment="GLM53_REASONING_BUDGET="\n' > "$DROPIN"
  else
    printf '[Service]\nEnvironment="GLM53_REASONING_BUDGET=%s"\n' "$b" > "$DROPIN"
  fi
  systemctl --user daemon-reload
  systemctl --user restart glm53
  for _ in $(seq 1 80); do
    [ "$(curl -s --max-time 2 http://127.0.0.1:8090/health 2>/dev/null)" = '{"status":"ok"}' ] && break
    sleep 3
  done
  python3 "$HARNESS/replay_real.py" "$CAPTURE" 3 "$REPS" 2>&1
done
