#!/usr/bin/env bash
# Sweep MTP draft depth and record decode throughput + acceptance at each.
#
# Restarts glm53.service once per setting via a systemd drop-in, so the sweep
# measures the same code path production uses rather than a hand-rolled launch.
# The drop-in is removed on exit (including on Ctrl-C), leaving glm53.env in
# charge again.
#
#   ./mtp-sweep.sh                 sweep off,1,2,3,4,5
#   ./mtp-sweep.sh 0 3             sweep only "off" and n_max=3
set -euo pipefail

# uv lives in ~/.local/bin, which is not on a systemd unit's default PATH.
export PATH="$HOME/.local/bin:$PATH"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$DIR/.." && pwd)"
DROPIN_DIR="$HOME/.config/systemd/user/glm53.service.d"
DROPIN="$DROPIN_DIR/99-mtp-sweep.conf"
REPS="${REPS:-3}"
SETTINGS=("$@")
[ ${#SETTINGS[@]} -eq 0 ] && SETTINGS=(0 1 2 3 4 5)

cleanup() {
  echo "== restoring glm53.env configuration =="
  rm -f "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
}
trap cleanup EXIT

mkdir -p "$DROPIN_DIR"

for n in "${SETTINGS[@]}"; do
  if [ "$n" = "0" ]; then
    label="mtp-off"
    # Omit --spec-type entirely so the nextn heads are not even loaded; this is
    # the true MTP-disabled baseline, not "MTP on with depth 0".
    printf '[Service]\nEnvironment=GLM53_SPEC_TYPE=none\n' > "$DROPIN"
  else
    label="mtp-n${n}"
    printf '[Service]\nEnvironment=GLM53_SPEC_TYPE=draft-mtp\nEnvironment=GLM53_SPEC_DRAFT_N_MAX=%s\n' "$n" > "$DROPIN"
  fi

  echo
  echo "=== $label ==="
  systemctl --user daemon-reload
  systemctl --user restart glm53

  ( cd "$BENCH" && uv run python benches/bench_mtp.py \
      --base-url "http://127.0.0.1:${GLM53_PORT:-8090}/v1" \
      --label "$label" --reps "$REPS" )
done
