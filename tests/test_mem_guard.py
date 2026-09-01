"""The host watchdog must fire in a death spiral and stay quiet during a load.

Evidence from this machine, all three cases real:

  2026-07-26 21:30  MemFree 13.1G  MemAvailable 12.8G  Cached 625M   -> must kill
  2026-07-26 21:50  MemFree 12.2G  MemAvailable 12.2G  Cached ~1G    -> must kill
                    (last sample; box hard-locked 13s later, ~13.75h down)
  2026-07-28 13:01  MemFree 16.1G  MemAvailable ~67G   Cached 51.3G  -> must NOT kill

The third case is a legitimate vLLM weight load. A MemFree gate killed it
(`-> KILL biggest GPU pid=166667 VLLM::EngineCore`), which is why MemFree cannot
be the trigger: a large model load transiently drops it below any useful floor
and is indistinguishable from a runaway job.

MemAvailable is the right signal — it counts reclaimable page cache, which a load
genuinely can reclaim, and it collapsed on its own in the real spiral. The actual
defect in the original guard was the 12G floor: MemAvailable read 12.2G at the
moment of death and never crossed it.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

# Prefer the repo copy so these tests run on any checkout; fall back to the
# installed original, which is the source of truth on the deployment host.
# tests/test_deploy_sync.py fails if the two ever drift apart.
_REPO_COPY = Path(__file__).resolve().parent.parent / "deploy" / "install-gpu-mem-guard.sh"
_INSTALLED = Path.home() / "install-gpu-mem-guard.sh"
INSTALLER = _REPO_COPY if _REPO_COPY.is_file() else _INSTALLED
KB = 1024

pytestmark = pytest.mark.skipif(
    not INSTALLER.is_file(), reason="host guard installer not present"
)


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    """The daemon script as actually embedded in the installer."""
    text = INSTALLER.read_text()
    m = re.search(r"<<'DAEMON'\n(.*?)\nDAEMON\n", text, re.DOTALL)
    assert m, "could not find the DAEMON heredoc in the installer"
    p = tmp_path_factory.mktemp("guard") / "gpu-mem-guard.sh"
    p.write_text(m.group(1))
    p.chmod(0o755)
    return p


def _meminfo(tmp_path, *, free_gb, available_gb, cached_gb, name="meminfo"):
    p = tmp_path / name
    p.write_text(
        f"MemTotal:       {119 * 1024 * KB} kB\n"
        f"MemFree:        {int(free_gb * 1024 * KB)} kB\n"
        f"MemAvailable:   {int(available_gb * 1024 * KB)} kB\n"
        f"Cached:         {int(cached_gb * 1024 * KB)} kB\n"
    )
    return p


def _decide(daemon, meminfo, **env):
    e = {
        **os.environ,
        "GPU_GUARD_ONESHOT": "1",
        "GPU_GUARD_DRY_RUN": "1",
        "GPU_GUARD_MEMINFO": str(meminfo),
        **env,
    }
    r = subprocess.run(
        ["bash", str(daemon)], capture_output=True, text=True, env=e, timeout=30
    )
    return r.stdout + r.stderr


# ------------------------------------------------- must not kill a legit load


def test_legitimate_model_load_is_not_killed(daemon, tmp_path):
    """2026-07-28 13:01 — the state in which a MemFree gate killed vLLM."""
    out = _decide(daemon, _meminfo(tmp_path, free_gb=16.1, available_gb=67, cached_gb=51.3))
    assert "OK" in out
    assert "KILL" not in out


def test_low_memfree_with_ample_reclaimable_cache_is_not_killed(daemon, tmp_path):
    """Cache is a reserve *for a load* — the kernel will evict it."""
    out = _decide(daemon, _meminfo(tmp_path, free_gb=4, available_gb=80, cached_gb=76))
    assert "OK" in out


# ---------------------------------------------------- must kill a real spiral


def test_death_spiral_at_2130_is_caught(daemon, tmp_path):
    out = _decide(daemon, _meminfo(tmp_path, free_gb=13.1, available_gb=12.8, cached_gb=0.6))
    assert "LOW MEM" in out


def test_death_spiral_at_2150_is_caught(daemon, tmp_path):
    """The last sample before the box died. A 12G floor missed this."""
    out = _decide(daemon, _meminfo(tmp_path, free_gb=12.2, available_gb=12.2, cached_gb=1))
    assert "LOW MEM" in out


def test_a_12g_floor_would_have_missed_the_2150_state(daemon, tmp_path):
    """Regression witness for why the floor moved to 20G."""
    mem = _meminfo(tmp_path, free_gb=12.2, available_gb=12.2, cached_gb=1)
    assert "OK" in _decide(daemon, mem, GPU_GUARD_FLOOR_GB="12")
    assert "LOW MEM" in _decide(daemon, mem, GPU_GUARD_FLOOR_GB="20")


def test_exhausted_cache_with_no_headroom_is_caught(daemon, tmp_path):
    out = _decide(daemon, _meminfo(tmp_path, free_gb=2, available_gb=3, cached_gb=0.5))
    assert "LOW MEM" in out


# --------------------------------------------------------------- mechanics


def test_default_floor_is_20g(daemon, tmp_path):
    out = _decide(daemon, _meminfo(tmp_path, free_gb=30, available_gb=30, cached_gb=1))
    assert "20G" in out


def test_floor_is_configurable(daemon, tmp_path):
    mem = _meminfo(tmp_path, free_gb=25, available_gb=25, cached_gb=1)
    assert "OK" in _decide(daemon, mem, GPU_GUARD_FLOOR_GB="20")
    assert "LOW MEM" in _decide(daemon, mem, GPU_GUARD_FLOOR_GB="40")


def test_decision_reports_both_metrics_for_diagnosis(daemon, tmp_path):
    out = _decide(daemon, _meminfo(tmp_path, free_gb=16, available_gb=67, cached_gb=51))
    assert "MemAvailable" in out and "MemFree" in out


def test_dry_run_does_not_kill(daemon, tmp_path):
    out = _decide(daemon, _meminfo(tmp_path, free_gb=1, available_gb=2, cached_gb=0.5))
    assert "killed" not in out.lower()


def test_installer_gates_on_memavailable(daemon):
    """The kill trigger must be MemAvailable, not MemFree."""
    text = INSTALLER.read_text()
    body = re.search(r"<<'DAEMON'\n(.*?)\nDAEMON\n", text, re.DOTALL).group(1)
    gate = [ln for ln in body.splitlines() if "-lt" in ln and "FLOOR_KB" in ln]
    assert gate, "no floor comparison found"
    assert any("avail" in ln.lower() for ln in gate), "gate must use MemAvailable"
    assert not any(re.search(r"\bfree_kb\b", ln) for ln in gate), \
        "MemFree must not be the kill trigger — it fires on every large model load"
