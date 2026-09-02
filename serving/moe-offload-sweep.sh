#!/usr/bin/env bash
# Move MoE expert weights for the first N layers into CPU buffers.
#
# On GB10 the GPU's memory IS system RAM, so this does not reduce total bytes.
# What it changes is the ACCOUNTING: GPU-pinned allocations are unreclaimable,
# while CPU-buffer tensors backed by the mmap'd GGUF are page cache the kernel
# can evict. Memory headroom is the binding constraint on this box — it is what
# rules out 256K context and the 2-bit quants — so converting pinned bytes into
# reclaimable ones is worth measuring even at a throughput cost.
#
# Records BOTH sides: resident GPU memory and MemAvailable, plus decode.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$DIR/.." && pwd)"
DROPIN_DIR="$HOME/.config/systemd/user/glm53.service.d"
DROPIN="$DROPIN_DIR/99-moe.conf"
REPS="${REPS:-2}"

NS=("$@")
[ ${#NS[@]} -eq 0 ] && NS=(0 4 8 16)

cleanup(){ echo "== restoring glm53.env =="; rm -f "$DROPIN"; systemctl --user daemon-reload; systemctl --user restart glm53; }
trap cleanup EXIT
mkdir -p "$DROPIN_DIR"

for n in "${NS[@]}"; do
  echo; echo "=== n-cpu-moe=$n ==="
  if [ "$n" = "0" ]; then
    printf '[Service]\nEnvironment="GLM53_EXTRA="\n' > "$DROPIN"
  else
# NOTE: systemd Environment= splits on whitespace unless the WHOLE
# assignment is quoted. Without the quotes, GLM53_EXTRA=--n-cpu-moe 4
# reaches the server as just "--n-cpu-moe" and it exits with
# 'expected value for argument'. This silently invalidated a whole sweep.
    printf '[Service]\nEnvironment="GLM53_EXTRA=--n-cpu-moe %s"\n' "$n" > "$DROPIN"
  fi
  systemctl --user daemon-reload
  systemctl --user restart glm53
  ok=0
  for i in $(seq 1 100); do
    [ "$(curl -s --max-time 2 "http://127.0.0.1:${GLM53_PORT:-8090}/health" 2>/dev/null)" = '{"status":"ok"}' ] && { ok=1; break; }
    sleep 3
  done
  [ "$ok" = 1 ] || { echo "FAILED TO START at n-cpu-moe=$n — skipping"; continue; }
  sleep 2
  gpu=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "memory: GPU=$(awk -v m="${gpu:-0}" 'BEGIN{printf "%.2f", m/1024}') GiB  MemAvailable=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB"
  ( cd "$BENCH" && uv run python benches/bench_mtp.py \
      --base-url "http://127.0.0.1:${GLM53_PORT:-8090}/v1" \
      --label "moe-ncmoe${n}" --reps "$REPS" )
done
