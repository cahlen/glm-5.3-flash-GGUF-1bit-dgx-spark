"""The GLM-5.3 launcher must build the command the serving notes promise.

Three defects this guards against, all of them things that already went wrong on
this box:

  - `--spec-draft-model` pointed at the full GGUF loads a SECOND 87 GiB copy of
    the weights and hard-locks the machine (2026-08-30). The launcher must never
    emit it, whatever the config says.
  - The OpenCode provider config pinned chat_template_kwargs.reasoning_effort to
    "low" globally, starving planning and tool-selection turns of reasoning. The
    server default must stay unset (template default = Max).
  - `--spec-type none` must omit the MTP flags entirely rather than pass depth 0,
    or the mtp-off arm of the sweep is not actually measuring MTP off.
"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "serving" / "glm53-up.sh"
ENV_FILE = ROOT / "serving" / "glm53.env"

pytestmark = pytest.mark.skipif(not UP.is_file(), reason="glm53-up.sh not present")


def dry(**env):
    e = {**os.environ, "DRY_RUN": "1", "ENV_FILE": str(ENV_FILE), **env}
    r = subprocess.run(["bash", str(UP)], capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 0, f"dry run failed: {r.stderr}"
    return r.stdout


# ------------------------------------------------------------------ defaults

def test_dry_run_emits_a_llama_server_command():
    assert "llama-server" in dry()


def test_defaults_match_the_documented_target_configuration():
    out = dry()
    for flag in ("--ctx-size 131072", "--parallel 1", "--cache-type-k q8_0",
                 "--cache-type-v q8_0", "--flash-attn on", "--jinja",
                 "--spec-type draft-mtp", "--spec-draft-n-max 2",
                 "--n-gpu-layers 999"):
        assert flag in out, f"missing {flag}"


def test_min_p_and_top_k_are_declared_not_inherited():
    """llama.cpp applies min_p and top_k whether or not you set them.

    Its defaults (min_p=0.05, top_k=40) truncate on top of top_p, so a config
    that omits them is not "unfiltered" — it is at the mercy of whatever this
    build happens to default to, which can change between versions. Both must
    appear on the command line explicitly.
    """
    out = dry()
    assert "--min-p" in out, "min_p left to the llama.cpp default"
    assert "--top-k" in out, "top_k left to the llama.cpp default"
    assert "--min-p 0.01" in out   # Zhipu/Unsloth guidance for GLM-5.3
    assert "--top-k 0" in out      # specified by neither; disabled rather than 40


def test_sampling_defaults_match_unsloths_guide_for_this_gguf():
    """unsloth.ai/docs/models/glm-5.3-flash "Recommended Settings", Default:
    temperature=1.0, top_p=0.95 — and measured better here than top_p=1.0
    (scenario 08: 5/5 vs 1/5). The base model card's top_p=1.0 footnote is
    zai-org's full-precision eval setting, not GGUF run guidance."""
    out = dry()
    assert "--temp 1.0" in out
    assert "--top-p 0.95" in out


# --------------------------------------------------------- the fatal flag

def test_never_emits_a_separate_draft_model():
    """A second full copy of the weights does not fit in 119 GiB. It locked the box."""
    out = dry()
    assert "--spec-draft-model" not in out
    assert " -md " not in out
    assert "--model-draft" not in out


def test_draft_model_is_not_emitted_even_via_extra():
    """GLM53_EXTRA is passed through verbatim, so the guarantee is only as good
    as the config; assert the default config does not smuggle one in.

    Comments are stripped first: glm53.env documents --spec-draft-model at
    length precisely to warn against it, and that prose must not trip this.
    """
    live = "\n".join(
        ln for ln in ENV_FILE.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    )
    assert "spec-draft-model" not in live
    assert "-md " not in live
    assert "model-draft" not in live


# ------------------------------------------------------------------ reasoning

def test_reasoning_budget_is_capped_not_unrestricted():
    """Unrestricted reasoning fails 45% of real agentic turns.

    Measured by replaying a captured OpenCode request (17,908 prompt tokens,
    53 tools), n=20 per arm: unrestricted returned a tool call 11/20 and hit the
    output ceiling with nothing actionable 9/20; with a 2048 budget it was 20/20
    and 0/20. Median turn latency fell from 11.2 min to 1.6 min. This is the
    difference between the setup being usable and not.
    """
    out = dry()
    assert "--reasoning-budget" in out, "unrestricted reasoning caps out ~45% of turns"
    assert "--reasoning-budget 2048" in out


def test_reasoning_effort_ships_high_not_max():
    """'high' measured better than Max for agentic tool use, twice.

    The template honours only 'low' and 'high'; anything else (including the
    literal "max") falls through to 'max', so an UNSET value means Max. This
    repo used to ship unset on the assumption that more deliberation helps
    planning. Measurement disagreed — Max 38/50 vs high 44/50 on the full suite,
    reproduced at 16/24 vs 21/24 on a focused 12-trial re-run — so the value is
    now set explicitly.
    """
    out = dry()
    assert "--reasoning-effort high" in out, "should ship high, measured better than Max"


def test_reasoning_effort_can_be_set_per_deployment():
    assert "--reasoning-effort low" in dry(GLM53_REASONING_EFFORT="low")


def test_reasoning_preserve_maps_on_and_off():
    assert "--reasoning-preserve" in dry(GLM53_REASONING_PRESERVE="on")
    assert "--no-reasoning-preserve" in dry(GLM53_REASONING_PRESERVE="off")


def test_reasoning_preserve_rejects_a_typo():
    e = {**os.environ, "DRY_RUN": "1", "ENV_FILE": str(ENV_FILE),
         "GLM53_REASONING_PRESERVE": "yes"}
    r = subprocess.run(["bash", str(UP)], capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode != 0
    assert "must be on|off" in r.stderr


# ----------------------------------------------------------------- MTP arms

def test_spec_none_omits_every_mtp_flag():
    out = dry(GLM53_SPEC_TYPE="none")
    assert "--spec-type" not in out
    assert "--spec-draft-n-max" not in out


def test_draft_depth_is_overridable():
    assert "--spec-draft-n-max 5" in dry(GLM53_SPEC_DRAFT_N_MAX="5")


# ------------------------------------------------------------- env precedence

def test_command_line_env_beats_the_env_file():
    """mtp-sweep.sh depends on this: it sweeps one variable via systemd
    Environment= while glm53.env still holds the production value."""
    assert "--ctx-size 262144" in dry(GLM53_CTX="262144")


def test_env_file_supplies_values_not_given_on_the_command_line():
    out = dry(GLM53_CTX="262144")
    assert "--cache-type-k q8_0" in out


# --------------------------------------------------------------- unit wiring

def test_systemd_unit_execs_the_launcher_not_a_raw_command():
    unit = Path.home() / ".config/systemd/user/glm53.service"
    if not unit.is_file():
        pytest.skip("unit not installed")
    text = unit.read_text()
    assert "glm53-up.sh" in text
    assert "--ctx-size" not in text, "flags belong in glm53.env, not the unit"


def test_guard_exemption_names_the_actual_unit():
    """If the unit is renamed and the exemption is not, the kill loop returns.

    Checks the copy in deploy/, never $HOME: a machine may well have an
    unrelated file of that name, and comparing against it produces a confusing
    false failure on a fresh clone.
    """
    installer = ROOT / "deploy" / "install-gpu-mem-guard.sh"
    assert installer.is_file(), "deploy/install-gpu-mem-guard.sh is missing"
    text = installer.read_text()
    assert "GPU_GUARD_EXEMPT_UNITS" in text
    assert "glm53.service" in text



# ---------------------------------------------- the quoting bug, three times
# An unquoted value containing a space has broken this deployment three separate
# ways, each time crash-looping the server:
#   1. systemd  Environment=GLM53_EXTRA=--n-cpu-moe 4   -> arrived as "--n-cpu-moe"
#      with no value; the server exited "expected value for argument"
#   2. glm53-up.sh  CMD+=( ${GLM53_EXTRA} )  word-splits unquoted, so a
#      multi-word --reasoning-budget-message became separate argv entries
#   3. glm53.env is SOURCED BY BASH, so
#         GLM53_REASONING_BUDGET_MESSAGE=Reasoning budget reached. ...
#      parsed "budget" as a command. 40 restarts before it was caught.
# The fix is always quoting; the guard is here so there is no fourth time.

def test_env_file_parses_as_bash():
    """glm53.env is sourced, so it must be valid bash."""
    import subprocess
    r = subprocess.run(["bash", "-n", str(ENV_FILE)], capture_output=True, text=True)
    assert r.returncode == 0, f"glm53.env is not valid bash:\n{r.stderr}"


def test_env_values_containing_spaces_are_quoted():
    """An unquoted value with a space is a crash loop waiting to happen."""
    offenders = []
    for i, line in enumerate(ENV_FILE.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if not key.startswith("GLM53_"):
            continue
        if " " in val and not (val.startswith(('"', "'")) and val.endswith(('"', "'"))):
            offenders.append(f"{ENV_FILE.name}:{i}: {key} has an unquoted space")
    assert not offenders, "\n".join(offenders)


def test_launcher_quotes_the_budget_message():
    """It must reach argv as ONE argument, spaces intact."""
    out = dry(GLM53_REASONING_BUDGET_MESSAGE="stop now and act")
    assert "--reasoning-budget-message" in out
    # %q-escaped by the dry run, so spaces appear as backslash-escapes
    assert ("stop\\ now\\ and\\ act" in out) or ("'stop now and act'" in out), out
