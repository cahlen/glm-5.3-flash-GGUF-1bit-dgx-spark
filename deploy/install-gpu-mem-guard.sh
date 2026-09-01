#!/usr/bin/env bash
# =============================================================================
# Install the GPU memory guard on a unified-memory host (e.g. Spark / GB10).
#
# Why: on GB10 the GPU's memory IS system RAM and CUDA allocations BYPASS the
# container memory cgroup (verified) -> a runaway GPU job can exhaust host RAM
# and wedge the box so no one can even SSH in. cgroup caps can't stop it.
#
# This watchdog (runs on the host) watches free RAM and, when it drops below a
# floor, SIGKILLs the single biggest GPU-memory process -> frees the unified
# memory and keeps the host loginable. Also marks sshd/tailscaled/lxd OOM-immune.
#
#   RUN AS:  sudo bash /home/cahlen/install-gpu-mem-guard.sh   (on the host)
#   Floor:   GPU_GUARD_FLOOR_GB=20 sudo -E bash ...   (default 20 GB)
# =============================================================================
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo/root"; exit 1; }
FLOOR_GB="${GPU_GUARD_FLOOR_GB:-20}"
# Units the guard must never kill. The intended resident model server is not a
# runaway job; killing it is an outage, not a save. See the DAEMON header for
# the 2026-08-31 incident (7 SIGKILLs of glm53.service in under two hours).
EXEMPT_UNITS="${GPU_GUARD_EXEMPT_UNITS:-glm53.service}"
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found on host"; exit 1; }

# ---- the watchdog daemon ----------------------------------------------------
install -m 0755 /dev/stdin /usr/local/bin/gpu-mem-guard.sh <<'DAEMON'
#!/usr/bin/env bash
# Kill the biggest RUNAWAY GPU-memory process before the host runs out of RAM.
#
# Gate on MemAvailable with a 20G floor. The original guard had the right metric
# and the wrong floor; do not "fix" this to MemFree (see below).
#
# Evidence, all measured on this machine:
#
#   when                     MemFree  MemAvailable  Cached   correct action
#   2026-07-26 21:30          13.1G       12.8G      625M    kill
#   2026-07-26 21:50          12.2G       12.2G       ~1G    kill (died 13s later)
#   2026-07-28 13:01 (load)   16.1G       ~67G       51.3G   leave alone
#
#   1. Why not MemFree: a legitimate large-model load transiently drops MemFree
#      below any useful floor while the page cache it will reclaim is still held.
#      A MemFree/20G gate killed a healthy vLLM weight load on 2026-07-28
#      ("-> KILL biggest GPU pid=166667 VLLM::EngineCore"). MemFree cannot
#      distinguish a load from a runaway job. MemAvailable can: it stayed at ~67G
#      through that load, and fell to 12.2G on its own in the real spiral.
#   2. Why 20G: MemAvailable read 12.2G in the last sample before the box died and
#      never crossed the old 12G floor. 20G matches
#      spark-bench/serving/preflight.sh.
#
# The real danger signature is not "free memory is low" but "there is nothing left
# to reclaim" — which is exactly what MemAvailable measures.
#
# EXEMPTIONS (added 2026-08-31). The guard exists to stop a *runaway* job from
# wedging the host. The intended resident model server is not a runaway job, and
# killing it is not a safe fallback — it is an outage. On 2026-08-31 this guard
# SIGKILLed glm53.service seven times between 12:10 and 14:07 (status=9/KILL, no
# core, no llama.cpp fault), because a 90G resident model legitimately parks
# MemAvailable near the floor and every long prompt pushed it under. Units named
# in GPU_GUARD_EXEMPT_UNITS are skipped; the next-biggest GPU process is killed
# instead. If every GPU process is exempt the guard logs and does nothing —
# at that point the fix is headroom (a smaller quant), not a kill.
#
# Test hooks (exercised by spark-bench/tests/test_mem_guard.py):
#   GPU_GUARD_MEMINFO=path   read memory stats from a file instead of /proc
#   GPU_GUARD_ONESHOT=1      evaluate once and exit
#   GPU_GUARD_DRY_RUN=1      report the decision, never kill
#   GPU_GUARD_APPS=path      read "pid, mem" rows from a file instead of nvidia-smi
#   GPU_GUARD_CGROUP_DIR=d   read <d>/<pid>.cgroup instead of /proc/<pid>/cgroup
FLOOR_GB="${GPU_GUARD_FLOOR_GB:-20}"
INTERVAL="${GPU_GUARD_INTERVAL:-3}"
MEMINFO="${GPU_GUARD_MEMINFO:-/proc/meminfo}"
ONESHOT="${GPU_GUARD_ONESHOT:-0}"
DRY_RUN="${GPU_GUARD_DRY_RUN:-0}"
EXEMPT_UNITS="${GPU_GUARD_EXEMPT_UNITS:-glm53.service}"
FLOOR_KB=$(( FLOOR_GB * 1024 * 1024 ))
log(){ echo "gpu-mem-guard: $*"; }
# pick_victim's stdout is the return value (consumed by $(...)), so anything
# it wants a human to read must go to stderr. systemd captures both.
logerr(){ echo "gpu-mem-guard: $*" >&2; }
field(){ awk -v k="$1:" '$1==k {print $2; exit}' "$MEMINFO"; }

# "pid, used_mem_mib" rows, biggest first.
apps(){
  if [ -n "${GPU_GUARD_APPS:-}" ]; then
    cat "$GPU_GUARD_APPS" 2>/dev/null
  else
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null
  fi
}

# The systemd unit owning a pid. User units live under .../app.slice/<unit>,
# system units under .../system.slice/<unit>; the unit is the last .service
# component of the cgroup path either way.
pid_unit(){
  local pid="$1" src
  if [ -n "${GPU_GUARD_CGROUP_DIR:-}" ]; then
    src="${GPU_GUARD_CGROUP_DIR}/${pid}.cgroup"
  else
    src="/proc/${pid}/cgroup"
  fi
  [ -r "$src" ] || return 1
  grep -o '[a-zA-Z0-9@_.\\-]*\.service' "$src" 2>/dev/null | tail -1
}

is_exempt(){
  local unit="$1" e rest="$EXEMPT_UNITS"
  [ -n "$unit" ] || return 1
  while [ -n "$rest" ]; do
    e="${rest%%,*}"
    [ "$e" = "$rest" ] && rest="" || rest="${rest#*,}"
    e="$(printf '%s' "$e" | tr -d ' ')"
    [ -n "$e" ] && [ "$unit" = "$e" ] && return 0
  done
  return 1
}

# Echo "pid mem unit" for the biggest non-exempt GPU process, or return 1.
pick_victim(){
  local row pid mem unit skipped=""
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    pid="$(printf '%s' "$row" | awk -F',' '{gsub(/[^0-9]/,"",$1); print $1}')"
    mem="$(printf '%s' "$row" | awk -F',' '{gsub(/[^0-9]/,"",$2); print $2}')"
    [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null || continue
    unit="$(pid_unit "$pid")"
    if is_exempt "$unit"; then
      skipped="$skipped $pid($unit)"
      continue
    fi
    [ -n "$skipped" ] && logerr "skipped exempt GPU process(es):$skipped"
    printf '%s %s %s\n' "$pid" "$mem" "${unit:-no-unit}"
    return 0
  done < <(apps | sort -t, -k2 -nr)
  if [ -n "$skipped" ]; then
    logerr "all GPU processes are exempt:$skipped — NOT killing. Free headroom instead (smaller quant / lower -ub), or edit GPU_GUARD_EXEMPT_UNITS."
  else
    logerr "no GPU process to kill (non-GPU pressure, e.g. page-cache writeback)"
  fi
  return 1
}

[ "$ONESHOT" = "1" ] || log "started: floor=${FLOOR_GB}G interval=${INTERVAL}s source=${MEMINFO} exempt=${EXEMPT_UNITS}"
while :; do
  avail_kb=$(field MemAvailable)
  free_kb=$(field MemFree)
  cached_kb=$(field Cached)
  stats="MemAvailable=$(( ${avail_kb:-0} / 1024 ))MB MemFree=$(( ${free_kb:-0} / 1024 ))MB Cached=$(( ${cached_kb:-0} / 1024 ))MB"
  if [ "${avail_kb:-999999999}" -lt "$FLOOR_KB" ]; then
    log "LOW MEM $stats < ${FLOOR_GB}G floor — nothing left to reclaim"
    if victim="$(pick_victim)"; then
      set -- $victim
      pid="$1"; mem="$2"; unit="$3"
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-140)
      log "-> KILL biggest non-exempt GPU pid=$pid (${mem}MiB GPU, unit=$unit) cmd=[$cmd]"
      if [ "$DRY_RUN" != "1" ]; then
        kill -9 "$pid" 2>/dev/null && log "killed $pid" || log "kill failed $pid"
        [ "$ONESHOT" = "1" ] || sleep 5
      fi
    fi
  else
    [ "$ONESHOT" != "1" ] || log "OK $stats >= ${FLOOR_GB}G floor"
  fi
  [ "$ONESHOT" != "1" ] || exit 0
  sleep "$INTERVAL"
done
DAEMON

# ---- systemd unit -----------------------------------------------------------
cat > /etc/systemd/system/gpu-mem-guard.service <<UNIT
[Unit]
Description=GPU memory guard (kill biggest GPU job before host OOM on unified memory)

[Service]
Type=simple
Environment=GPU_GUARD_FLOOR_GB=${FLOOR_GB}
Environment=GPU_GUARD_EXEMPT_UNITS=${EXEMPT_UNITS}
ExecStart=/usr/local/bin/gpu-mem-guard.sh
Restart=always
RestartSec=5
Nice=-5
OOMScoreAdjust=-1000

[Install]
WantedBy=multi-user.target
UNIT

# ---- keep the login path alive: mark critical services OOM-immune -----------
for svc in ssh.service tailscaled.service snap.lxd.daemon.service; do
  systemctl list-unit-files "$svc" >/dev/null 2>&1 || continue
  d="/etc/systemd/system/${svc}.d"; install -d "$d"
  printf '[Service]\nOOMScoreAdjust=-900\n' > "$d/10-oom-immune.conf"
  echo "  OOM-protected $svc"
done

systemctl daemon-reload
systemctl reset-failed gpu-mem-guard.service 2>/dev/null || true
systemctl enable gpu-mem-guard.service >/dev/null 2>&1 || true
systemctl restart gpu-mem-guard.service
sleep 2
echo
echo "Installed. Status: $(systemctl is-active gpu-mem-guard.service)"
echo "Recent log:"; timeout 6 journalctl -u gpu-mem-guard.service -n 3 --no-pager 2>/dev/null || true
echo "Exempt = ${EXEMPT_UNITS} (never killed; the next-biggest GPU process is chosen instead)."
echo "Floor = ${FLOOR_GB}G. Change: edit Environment=GPU_GUARD_FLOOR_GB in the unit, daemon-reload, restart."
