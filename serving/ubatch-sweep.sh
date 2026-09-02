#!/usr/bin/env bash
# Sweep --ubatch-size (and optionally --batch-size), recording BOTH what it buys
# and what it costs.
#
# ubatch is the physical batch llama.cpp processes at once. Larger keeps the GPU
# busier on long prompts but grows the compute buffers; smaller frees memory at
# some prefill cost. On this box the weights leave only ~22 GiB, so the memory
# side is not a rounding error — it is the difference between 128K and 256K
# context being comfortable.
#
# Records resident GPU memory per arm, because a prefill win that costs 2 GiB is
# not obviously a win here.
#
#   ./ubatch-sweep.sh              sweep 128,256,512,1024
#   ./ubatch-sweep.sh 256 512      sweep only these
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$DIR/.." && pwd)"
DROPIN_DIR="$HOME/.config/systemd/user/glm53.service.d"
DROPIN="$DROPIN_DIR/99-ubatch.conf"
DEPTHS="${DEPTHS:-4096,16384,65536}"

UBS=("$@")
[ ${#UBS[@]} -eq 0 ] && UBS=(128 256 512 1024)

cleanup() {
  echo "== restoring glm53.env configuration =="
  rm -f "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
}
trap cleanup EXIT
mkdir -p "$DROPIN_DIR"

for ub in "${UBS[@]}"; do
  echo
  echo "=== ubatch=$ub ==="
  printf '[Service]\nEnvironment=GLM53_UBATCH=%s\n' "$ub" > "$DROPIN"
  systemctl --user daemon-reload
  systemctl --user restart glm53
  # wait for readiness before sampling memory, or we measure a partial load
  for _ in $(seq 1 80); do
    [ "$(curl -s --max-time 2 "http://127.0.0.1:${GLM53_PORT:-8090}/health" 2>/dev/null)" = '{"status":"ok"}' ] && break
    sleep 3
  done
  sleep 2
  gpu=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1)
  avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  echo "memory: GPU=$(awk -v m="${gpu:-0}" 'BEGIN{printf "%.2f", m/1024}') GiB  MemAvailable=${avail} GiB"
  ( cd "$BENCH" && uv run python benches/bench_prefill.py \
      --base-url "http://127.0.0.1:${GLM53_PORT:-8090}/v1" \
      --label "ub${ub}" --depths "$DEPTHS" )
  # decode should be ~unaffected by ubatch; measured to confirm rather than assumed
  ( cd "$BENCH" && uv run python benches/bench_mtp.py \
      --base-url "http://127.0.0.1:${GLM53_PORT:-8090}/v1" \
      --label "ub${ub}" --reps 2 )
done
