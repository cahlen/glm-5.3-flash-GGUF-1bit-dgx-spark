#!/usr/bin/env bash
# Health check for the GLM-5.3-Flash server.
#
# Goes past "is the port open". A server that answers /health but returns a
# 200 with zero tool calls on tool_choice=required is broken for agentic use and
# must not be reported healthy — that is the failure this check exists to catch.
#
#   ./glm53-health.sh            human-readable, exit 0 healthy / 1 unhealthy
#   ./glm53-health.sh --quiet    exit code only
#   ./glm53-health.sh --json     machine-readable
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$DIR/glm53.env}"

# Match glm53-up.sh: a value already in the environment WINS over glm53.env, so
# checking a server launched with a systemd drop-in override (as mtp-sweep.sh
# and the 256K trial do) compares against what actually launched rather than
# against the file, which otherwise reports a bogus n_ctx mismatch.
_preset="$(env | grep -o '^GLM53_[A-Z0-9_]*' || true)"
declare -A _keep
for k in $_preset; do _keep[$k]="${!k}"; done
# ENV_FILE is configurable by design, so there is no constant path for
# shellcheck to follow; the directive has to sit on the `.` line itself.
if [ -r "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
fi
for k in "${!_keep[@]}"; do export "$k=${_keep[$k]}"; done

PORT="${GLM53_PORT:-8090}"
HOSTNAME_="${GLM53_HEALTH_HOST:-127.0.0.1}"
BASE="http://${HOSTNAME_}:${PORT}"
ALIAS="${GLM53_ALIAS:-glm-5.3-flash}"
MODE=human
case "${1:-}" in --quiet) MODE=quiet ;; --json) MODE=json ;; esac

fail=0
note() { [ "$MODE" = human ] && echo "$@"; return 0; }
bad()  { fail=1; [ "$MODE" = human ] && echo "  FAIL: $*"; return 0; }

# 1. unit ------------------------------------------------------------------
unit_state="$(systemctl --user is-active glm53 2>/dev/null || echo unknown)"
note "unit           : $unit_state"
[ "$unit_state" = active ] || bad "glm53.service is $unit_state"

# restart_count is the tell for the watchdog-kill loop this setup used to have.
restarts="$(systemctl --user show glm53 -p NRestarts --value 2>/dev/null || echo '?')"
note "restarts       : $restarts"

# 2. liveness --------------------------------------------------------------
health="$(curl -s --max-time 10 "$BASE/health" 2>/dev/null)"
if [ "${health}" = '{"status":"ok"}' ]; then
  note "health         : ok"
else
  bad "/health returned '${health:-<nothing>}'"
fi

# 3. configuration actually in force --------------------------------------
props="$(curl -s --max-time 10 "$BASE/props" 2>/dev/null)"
n_ctx="$(printf '%s' "$props" | python3 -c 'import sys,json;print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])' 2>/dev/null || echo '?')"
note "n_ctx          : $n_ctx (configured ${GLM53_CTX:-?})"
if [ -n "${GLM53_CTX:-}" ] && [ "$n_ctx" != "${GLM53_CTX}" ]; then
  bad "server n_ctx $n_ctx != configured ${GLM53_CTX} (--fit may have shrunk it)"
fi

# 4. memory headroom -------------------------------------------------------
avail_gb=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
free_gb=$(awk '/MemFree/{printf "%.1f", $2/1048576}' /proc/meminfo)
gpu_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
          | awk -F',' '{gsub(/ /,"",$2); s+=$2} END{print s+0}')
note "memory         : MemAvailable=${avail_gb}G MemFree=${free_gb}G GPU=$((gpu_mib/1024))G"
# The guard kills below 20G. Warn before it does.
awk -v a="$avail_gb" 'BEGIN{exit !(a < 22)}' && note "  WARN: MemAvailable ${avail_gb}G is near the 20G gpu-mem-guard floor"

# 5. the agentic contract: required tool call must produce a tool call ------
req='{"model":"'"$ALIAS"'","max_tokens":256,"tool_choice":"required",
 "messages":[{"role":"user","content":"Read the file app/client.py."}],
 "tools":[{"type":"function","function":{"name":"read_file",
   "description":"Read a file and return its contents.",
   "parameters":{"type":"object","properties":{"path":{"type":"string"}},
   "required":["path"],"additionalProperties":false}}}]}'
resp="$(curl -s --max-time 180 -H 'Content-Type: application/json' \
        -d "$req" "$BASE/v1/chat/completions" 2>/dev/null)"
ncalls="$(printf '%s' "$resp" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d["choices"][0]["message"].get("tool_calls") or []))
except Exception:
    print(-1)
' 2>/dev/null || echo -1)"
note "required tool   : $ncalls call(s)"
case "$ncalls" in
  -1) bad "tool-call probe did not return parseable JSON" ;;
   0) bad "tool_choice=required returned ZERO tool calls — server is not usable for agents" ;;
esac

if [ "$MODE" = json ]; then
  printf '{"healthy":%s,"unit":"%s","restarts":"%s","n_ctx":"%s","mem_available_gb":%s,"gpu_gib":%s,"required_tool_calls":%s}\n' \
    "$([ $fail -eq 0 ] && echo true || echo false)" "$unit_state" "$restarts" "$n_ctx" \
    "$avail_gb" "$((gpu_mib/1024))" "$ncalls"
fi

[ "$MODE" = human ] && { [ $fail -eq 0 ] && echo "RESULT: healthy" || echo "RESULT: UNHEALTHY"; }
exit $fail
