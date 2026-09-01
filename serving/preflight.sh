#!/usr/bin/env bash
# Refuse to start a model server when this box has no real headroom.
#
# Both hard lockups came from something heavy running alongside a resident model
# on unified memory. install-gpu-mem-guard.sh watches MemAvailable, which counts
# reclaimable page cache, so on 2026-07-26 it read ~113G while MemFree was ~2G
# and the machine died anyway. Gate on MemFree.
#
# PREFLIGHT_SKIP=1        override entirely
# PREFLIGHT_FLOOR_GB=N    required MemFree (default 20)
# PREFLIGHT_MEMINFO=path  alternate meminfo source (tests)
# PREFLIGHT_SKIP_PROCS=1  skip the resident-process check
set -euo pipefail

MEMINFO="${PREFLIGHT_MEMINFO:-/proc/meminfo}"
FLOOR_GB="${PREFLIGHT_FLOOR_GB:-20}"

if [ "${PREFLIGHT_SKIP:-0}" = "1" ]; then
  echo "preflight: skipped by PREFLIGHT_SKIP=1"
  exit 0
fi

[ -r "$MEMINFO" ] || { echo "preflight: cannot read $MEMINFO" >&2; exit 1; }

field() { awk -v k="$1:" '$1==k {print $2; exit}' "$MEMINFO"; }
to_gb() { awk -v v="${1:-0}" 'BEGIN{printf "%.0f", v/1048576}'; }

free_kb="$(field MemFree)"
[ -n "$free_kb" ] || { echo "preflight: MemFree missing from $MEMINFO" >&2; exit 1; }

free_gb="$(to_gb "$free_kb")"
avail_gb="$(to_gb "$(field MemAvailable)")"
cached_gb="$(to_gb "$(field Cached)")"

echo "preflight: MemFree=${free_gb}G MemAvailable=${avail_gb}G Cached=${cached_gb}G floor=${FLOOR_GB}G"

if [ "$free_gb" -lt "$FLOOR_GB" ]; then
  echo "preflight: REFUSING to start — MemFree ${free_gb}G is below the ${FLOOR_GB}G floor." >&2
  echo "  MemAvailable reads ${avail_gb}G but ${cached_gb}G of that is page cache and will not" >&2
  echo "  survive a model load. Stop the resident server (serving/vllm-down.sh) first," >&2
  echo "  or set PREFLIGHT_SKIP=1 to override." >&2
  exit 1
fi

if [ "${PREFLIGHT_SKIP_PROCS:-0}" != "1" ]; then
  resident="$(ps -eo rss=,args= 2>/dev/null | awk '$1 > 8000000 {print; exit}' || true)"
  if [ -n "$resident" ]; then
    echo "preflight: REFUSING to start — a process already holds >8G RSS:" >&2
    echo "  ${resident}" >&2
    echo "  Two resident models do not fit in 119Gi. Set PREFLIGHT_SKIP=1 to override." >&2
    exit 1
  fi
fi

echo "preflight: ok"
