#!/usr/bin/env bash
# Run the agentic + MTP benches against the live server under one label.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
LABEL="${1:?label}"
TRIALS="${2:-3}"
# || exit matters here: this script runs `set -uo pipefail` without -e, so an
# unchecked cd would silently continue and bench whatever directory it landed in.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
echo "===== $LABEL start $(date -u +%FT%TZ) ====="
echo "--- config in force ---"
curl -s --max-time 10 http://127.0.0.1:8090/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print("n_ctx",d["default_generation_settings"]["n_ctx"])' 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1
awk '/MemAvailable/{printf "MemAvailable %.1f GiB\n", $2/1048576}' /proc/meminfo
echo "--- agentic ---"
uv run python benches/bench_agentic.py --base-url http://127.0.0.1:8090/v1 \
  --label "$LABEL" --trials "$TRIALS" 2>&1
echo "--- mtp ---"
uv run python benches/bench_mtp.py --base-url http://127.0.0.1:8090/v1 \
  --label "$LABEL" --reps 3 2>&1
echo "===== $LABEL done $(date -u +%FT%TZ) ====="
