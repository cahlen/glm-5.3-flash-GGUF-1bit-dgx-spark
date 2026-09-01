#!/usr/bin/env bash
# The MTP knobs left after draft depth: acceptance thresholds, not depth.
#
#   --spec-draft-p-split  (default 0.10) branch probability for the draft tree
#   --spec-draft-p-min    (default 0.00) minimum probability to draft at all
#   --spec-draft-n-min    (default 0)    floor on draft length
#
# Depth was swept separately (mtp-sweep.sh, optimum 2). These control WHEN the
# drafter bothers, which is a different question: a stricter threshold drafts
# less often but accepts more of what it drafts.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$DIR/.." && pwd)"
DROPIN_DIR="$HOME/.config/systemd/user/glm53.service.d"
DROPIN="$DROPIN_DIR/99-mtptune.conf"
REPS="${REPS:-3}"

cleanup(){ echo "== restoring glm53.env =="; rm -f "$DROPIN"; systemctl --user daemon-reload; systemctl --user restart glm53; }
trap cleanup EXIT
mkdir -p "$DROPIN_DIR"

# label:extra-args
ARMS=(
  "baseline:"
  "psplit0.05:--spec-draft-p-split 0.05"
  "psplit0.20:--spec-draft-p-split 0.20"
  "pmin0.05:--spec-draft-p-min 0.05"
  "nmin1:--spec-draft-n-min 1"
)
for a in "${ARMS[@]}"; do
  lbl="${a%%:*}"; extra="${a#*:}"
  echo; echo "=== mtptune-$lbl  [${extra:-defaults}] ==="
  printf '[Service]\nEnvironment=GLM53_EXTRA=%s\n' "$extra" > "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
  ( cd "$BENCH" && uv run python benches/bench_mtp.py \
      --base-url "http://127.0.0.1:${GLM53_PORT:-8090}/v1" \
      --label "mtptune-$lbl" --reps "$REPS" )
done
