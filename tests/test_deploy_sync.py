"""Files under deploy/ are copies of things that live outside this repo.

The host installs from the ORIGINAL paths: the guard installer must be at
~/install-gpu-mem-guard.sh because test_mem_guard.py reads the watchdog out of
that file's heredoc, and systemd loads user units from ~/.config/systemd/user/.
The copies here exist so those files are version-controlled at all.

Two copies of anything is a drift bug waiting to happen, so these tests fail the
moment they diverge. If one fails, copy the newer file over the older — do not
"fix" the test.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

PAIRS = [
    (DEPLOY / "install-gpu-mem-guard.sh", Path.home() / "install-gpu-mem-guard.sh"),
    (DEPLOY / "glm53.service", Path.home() / ".config/systemd/user/glm53.service"),
]


@pytest.mark.parametrize("copy,original", PAIRS, ids=lambda p: p.name)
def test_deploy_copy_matches_the_installed_original(copy, original):
    if not original.is_file():
        pytest.skip(f"{original} not present — not the deployment host")
    assert copy.is_file(), f"{copy} missing from deploy/"
    assert copy.read_text() == original.read_text(), (
        f"{copy.name} has drifted from {original}.\n"
        f"  cp {original} {copy}    (or the reverse, whichever is newer)"
    )


def test_guard_installer_exempts_the_service_this_repo_ships():
    """The exemption and the unit name must agree, or the kill loop returns."""
    installer = (DEPLOY / "install-gpu-mem-guard.sh").read_text()
    assert "GPU_GUARD_EXEMPT_UNITS" in installer
    assert "glm53.service" in installer


def test_shipped_unit_execs_the_launcher_not_a_raw_command():
    unit = (DEPLOY / "glm53.service").read_text()
    assert "glm53-up.sh" in unit
    assert "--ctx-size" not in unit, "flags belong in serving/glm53.env, not the unit"


def test_opencode_provider_does_not_pin_reasoning_effort_on_the_default_model():
    """Pinning 'low' globally starves planning and tool-selection turns — the
    exact misconfiguration this repo exists to document."""
    import json

    cfg = json.loads((DEPLOY / "opencode-provider.json").read_text())
    models = cfg["provider"]["spark-glm"]["models"]
    default_id = cfg["model"].split("/", 1)[1]
    assert "chat_template_kwargs" not in models[default_id]["options"]
    # ...but the opt-in low lane must still exist and be wired to small_model.
    low_id = cfg["small_model"].split("/", 1)[1]
    assert models[low_id]["options"]["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_opencode_context_limit_reserves_the_output_budget():
    """--parallel 1 means one slot holds prompt + generation, and context shift
    is off: overflow is a hard error, so the advertised context must leave room."""
    import json

    cfg = json.loads((DEPLOY / "opencode-provider.json").read_text())
    for m in cfg["provider"]["spark-glm"]["models"].values():
        assert m["limit"]["context"] + m["limit"]["output"] <= 131072
