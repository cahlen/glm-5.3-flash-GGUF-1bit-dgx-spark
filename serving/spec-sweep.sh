#!/usr/bin/env bash
# Sweep --spec-type combinations and record decode throughput + acceptance.
#
# Unlike mtp-sweep.sh (which varies draft DEPTH for a fixed type), this varies
# the speculation STRATEGY. --spec-type takes a comma-separated list, so types
# can stack: draft-mtp uses the model's own NextN head, while the ngram-* types
# are prompt-lookup based and need no draft model at all, so they cost nothing
# to add. Whether they actually help on this workload is the question.
#
#   ./spec-sweep.sh                     sweep the default combo list
#   ./spec-sweep.sh draft-mtp ngram-mod sweep only these
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$DIR/.." && pwd)"
DROPIN_DIR="$HOME/.config/systemd/user/glm53.service.d"
DROPIN="$DROPIN_DIR/99-spec-sweep.conf"
REPS="${REPS:-3}"

COMBOS=("$@")
[ ${#COMBOS[@]} -eq 0 ] && COMBOS=("draft-mtp" "draft-mtp,ngram-mod" "draft-mtp,ngram-cache" "ngram-mod")

cleanup() {
  echo "== restoring glm53.env configuration =="
  rm -f "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
}
trap cleanup EXIT
mkdir -p "$DROPIN_DIR"

for c in "${COMBOS[@]}"; do
  label="spec-$(printf '%s' "$c" | tr ',' '+')"
  echo
  echo "=== $label ==="
  printf '[Service]\nEnvironment=GLM53_SPEC_TYPE=%s\n' "$c" > "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
  ( cd "$BENCH" && uv run python benches/bench_mtp.py \
      --base-url "http://127.0.0.1:${GLM53_PORT:-8090}/v1" \
      --label "$label" --reps "$REPS" )
done
