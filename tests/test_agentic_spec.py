"""Unit tests for the agentic scenario checks.

Every verdict function is exercised against a hand-built good response and at
least one realistic failure, with no server involved.

This exists because an earlier tool-call benchmark on this hardware produced
numbers that were shaped by harness bugs nothing tested. A check that silently
always-passes is worse than no check at all — two of these tests caught real
bugs in the verdict functions on the day they were written.
"""

import json
import sys
from pathlib import Path

import pytest

BENCHES = Path(__file__).resolve().parent.parent / "benches"
sys.path.insert(0, str(BENCHES))

import agentic_spec as spec  # noqa: E402


def call(name, args):
    return {"id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def check(scenario_id, text="", calls=None):
    return spec.SCENARIOS_BY_ID[scenario_id]["check"](text, calls or [])


# --------------------------------------------------------------- structure

def test_there_are_eleven_scenarios():
    assert len(spec.SCENARIOS) == 11


def test_scenario_ids_are_unique_and_ordered():
    ids = [s["id"] for s in spec.SCENARIOS]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids), "ids carry a numeric prefix so ordering is stable"


def test_every_scenario_has_a_callable_check():
    for s in spec.SCENARIOS:
        assert callable(s["check"]), s["id"]


def test_required_scenarios_are_the_ones_marked_required():
    assert spec.REQUIRED_SCENARIOS == [
        "04_required_tool_call",
        "08_long_prompt_tool_use",
        "09_patch_generation",
        "10_shell_generation",
    ]


def test_tool_scenarios_offer_tools():
    for s in spec.SCENARIOS:
        if s["kind"] == "tool":
            assert s["tools"], s["id"]


def test_large_tool_set_is_actually_large():
    assert len(spec.LARGE_TOOLS) >= 15
    # and the correct answer is still reachable
    assert any(t["function"]["name"] == "grep" for t in spec.LARGE_TOOLS)


def test_tool_schemas_are_wellformed():
    for t in spec.LARGE_TOOLS:
        params = t["function"]["parameters"]
        assert params["type"] == "object"
        for r in params.get("required", []):
            assert r in params["properties"], (t["function"]["name"], r)


def test_long_system_prompt_is_long_and_ends_with_the_load_bearing_rule():
    p = spec.long_system_prompt()
    assert len(p.split()) > 2000
    assert p.rstrip().endswith("without exception.")
    assert "SAFE:" in p


# ------------------------------------------------------- 01 text completion

def test_text_completion_accepts_a_real_answer():
    ok, _ = check("01_text_completion",
                  "A hash map gives you expected constant-time lookup by key, "
                  "whereas a sorted array needs a logarithmic binary search. "
                  "The array keeps order, which the hash map does not.")
    assert ok


def test_text_completion_rejects_a_tool_call():
    ok, why = check("01_text_completion", "constant time", [call("read_file", {"path": "a"})])
    assert not ok and "tool call" in why


def test_text_completion_rejects_a_stub_answer():
    ok, _ = check("01_text_completion", "It is faster.")
    assert not ok


# ------------------------------------------------------- 02 code generation

GOOD_CODE = '''Here you go:

```python
def binary_search(items, target):
    """Return the index of target in sorted items, or -1."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```
'''


def test_code_generation_accepts_fenced_function():
    ok, why = check("02_code_generation", GOOD_CODE)
    assert ok, why


def test_code_generation_rejects_unfenced_prose():
    ok, why = check("02_code_generation", "Just use the bisect module, it is easier.")
    assert not ok and "fenced" in why


# ----------------------------------------------------------- 03 json output

def test_json_output_accepts_bare_object():
    ok, why = check("03_json_output",
                    '{"name": "order-api", "version": "0.1.0", "dependencies": ["fastapi"]}')
    assert ok, why


def test_json_output_accepts_fenced_object():
    ok, why = check("03_json_output",
                    '```json\n{"name": "order-api", "version": "0.1.0", '
                    '"dependencies": ["fastapi", "uvicorn"]}\n```')
    assert ok, why


def test_json_output_rejects_missing_key():
    ok, why = check("03_json_output", '{"name": "order-api", "version": "0.1.0"}')
    assert not ok and "dependencies" in why


def test_json_output_rejects_dependencies_as_object():
    ok, why = check("03_json_output",
                    '{"name": "a", "version": "1.0.0", "dependencies": {"fastapi": "*"}}')
    assert not ok and "not a list" in why


def test_json_output_rejects_invalid_json():
    ok, why = check("03_json_output", '{"name": "a", "version": }')
    assert not ok and "invalid JSON" in why


# ------------------------------------------------------ 04 required tool call

def test_required_tool_call_accepts_read_file():
    ok, why = check("04_required_tool_call", "", [call("read_file", {"path": "app/client.py"})])
    assert ok, why


def test_required_tool_call_zero_calls_is_a_failure():
    """The exact regression the user asked to be treated as failing."""
    ok, why = check("04_required_tool_call", "Sure, I will read that file for you.", [])
    assert not ok
    assert "zero tool calls" in why


def test_required_tool_call_rejects_empty_path():
    ok, why = check("04_required_tool_call", "", [call("read_file", {"path": ""})])
    assert not ok and "missing/empty" in why


def test_required_tool_call_rejects_wrong_tool():
    ok, why = check("04_required_tool_call", "", [call("write_file", {"path": "a", "content": "b"})])
    assert not ok and "expected 'read_file'" in why


def test_required_tool_call_rejects_malformed_arguments():
    bad = {"id": "c1", "type": "function",
           "function": {"name": "read_file", "arguments": "{not json"}}
    ok, why = check("04_required_tool_call", "", [bad])
    assert not ok and "valid JSON" in why


# -------------------------------------------------------------- 05 multi-turn

DIFF_TIMEOUT = (
    "--- a/app/client.py\n+++ b/app/client.py\n@@ -1,4 +1,4 @@\n"
    "-DEFAULT_TIMEOUT = 5\n+DEFAULT_TIMEOUT = 30\n"
)


def test_multiturn_accepts_a_timeout_diff():
    ok, why = check("05_multi_turn_tool_use", "", [call("apply_patch", {"diff": DIFF_TIMEOUT})])
    assert ok, why


def test_multiturn_rejects_a_diff_that_misses_the_value():
    d = "--- a/app/client.py\n+++ b/app/client.py\n@@ -1 +1 @@\n-import httpx\n+import httpx  # noqa\n"
    ok, why = check("05_multi_turn_tool_use", "", [call("apply_patch", {"diff": d})])
    assert not ok and "timeout" in why


def test_multiturn_rejects_re_reading_the_file():
    ok, why = check("05_multi_turn_tool_use", "", [call("read_file", {"path": "app/client.py"})])
    assert not ok


# ---------------------------------------------------------- 06 large tool set

def test_large_toolset_accepts_grep():
    ok, why = check("06_large_tool_set", "", [call("grep", {"pattern": "DEFAULT_TIMEOUT"})])
    assert ok, why


def test_large_toolset_rejects_the_plausible_distractor():
    ok, why = check("06_large_tool_set", "", [call("list_files", {"path": "."})])
    assert not ok and "list_files" in why


def test_large_toolset_rejects_two_calls():
    ok, why = check("06_large_tool_set", "", [
        call("grep", {"pattern": "DEFAULT_TIMEOUT"}),
        call("list_files", {"path": "."}),
    ])
    assert not ok and "expected 1 call" in why


# ------------------------------------------------- 07 tool then reasoning

def test_tool_then_reasoning_accepts_a_prose_conclusion():
    ok, why = check("07_tool_then_reasoning",
                    "The bug is on line 7: c.get(url) is missing an await, so fetch() "
                    "returns a coroutine object rather than the parsed JSON body.")
    assert ok, why


def test_tool_then_reasoning_rejects_another_tool_call():
    ok, why = check("07_tool_then_reasoning", "",
                    [call("read_file", {"path": "app/client.py"})])
    assert not ok and "re-called" in why


def test_tool_then_reasoning_rejects_missing_the_diagnosis():
    ok, why = check("07_tool_then_reasoning",
                    "I looked at the file and there is definitely something odd "
                    "going on with the HTTP client, it may be a version problem.")
    assert not ok and "await" in why


# ------------------------------------------------- 08 long prompt + tool use

def test_long_prompt_tool_use_requires_the_safe_prefix():
    ok, why = check("08_long_prompt_tool_use", "",
                    [call("run_shell", {"command": "SAFE: git status --porcelain -- '*.py'"})])
    assert ok, why


def test_long_prompt_tool_use_catches_a_dropped_final_rule():
    """A model that loses the tail of a 2600-word prompt fails here, visibly."""
    ok, why = check("08_long_prompt_tool_use", "",
                    [call("run_shell", {"command": "git status --porcelain -- '*.py'"})])
    assert not ok and "FINAL RULE" in why


# ------------------------------------------------------- 09 patch generation

def test_patch_generation_accepts_a_real_unified_diff():
    d = ("--- a/app/client.py\n+++ b/app/client.py\n@@ -6,3 +6,3 @@\n"
         "-        r = c.get(url)\n+        r = await c.get(url)\n")
    ok, why = check("09_patch_generation", "", [call("apply_patch", {"diff": d})])
    assert ok, why


def test_patch_generation_rejects_a_diff_with_no_hunk_header():
    d = "-        r = c.get(url)\n+        r = await c.get(url)\n"
    ok, why = check("09_patch_generation", "", [call("apply_patch", {"diff": d})])
    assert not ok and "@@" in why


def test_patch_generation_rejects_an_addition_only_diff():
    d = "--- a/x\n+++ b/x\n@@ -1 +1,2 @@\n+new line\n"
    ok, why = check("09_patch_generation", "", [call("apply_patch", {"diff": d})])
    assert not ok and "removed" in why


# ------------------------------------------------------- 10 shell generation

@pytest.mark.parametrize("cmd", [
    "find . -name '*.py' | wc -l",
    "grep -rl '' --include='*.py' . | wc -l",
    "ls **/*.py | wc -l",
])
def test_shell_generation_accepts_plausible_commands(cmd):
    ok, why = check("10_shell_generation", "", [call("run_shell", {"command": cmd})])
    assert ok, why


def test_shell_generation_rejects_a_command_that_does_not_search():
    ok, why = check("10_shell_generation", "", [call("run_shell", {"command": "python -V"})])
    assert not ok


def test_shell_generation_rejects_wrong_file_type():
    ok, why = check("10_shell_generation", "",
                    [call("run_shell", {"command": "find . -name '*.js' | wc -l"})])
    assert not ok and ".py" in why


# ----------------------------------------------------------------- tolerance

def test_args_parsed_from_dict_as_well_as_string():
    """Some servers hand back arguments already decoded; both must work."""
    dict_call = {"id": "c", "type": "function",
                 "function": {"name": "read_file", "arguments": {"path": "app/client.py"}}}
    ok, why = check("04_required_tool_call", "", [dict_call])
    assert ok, why


# ------------------------------------------------ strict vs lenient scoring
# Measured 2026-08-31: on the 17-tool set the model calls the right tool and
# then tacks on a spurious second one ~40% of the time, at BOTH top_p 1.0 and
# 0.95. That is waste an agent loop absorbs, and must not score the same as
# picking the wrong tool outright.

def test_extra_call_alongside_the_right_one_is_marked_not_merged():
    ok, why = check("06_large_tool_set", "", [
        call("grep", {"pattern": "DEFAULT_TIMEOUT"}),
        call("git_status", {}),
    ])
    assert not ok, "strict verdict must still fail"
    assert why.startswith(spec.EXTRA_CALLS), why
    assert "git_status" in why


def test_wrong_tool_is_not_marked_as_an_extra_call():
    """A wrong action must never be forgiven by the lenient score."""
    ok, why = check("06_large_tool_set", "", [call("list_files", {"path": "."})])
    assert not ok
    assert not why.startswith(spec.EXTRA_CALLS)


def test_extra_call_still_validates_the_expected_calls_arguments():
    """Extras must not let a malformed primary call through."""
    ok, why = check("10_shell_generation", "", [
        call("run_shell", {"command": ""}),
        call("run_shell", {"command": "find . -name '*.py' | wc -l"}),
    ])
    assert not ok
    assert not why.startswith(spec.EXTRA_CALLS), "empty required arg is a real failure"


def test_duplicate_correct_calls_are_extras_not_wrong_tools():
    ok, why = check("08_long_prompt_tool_use", "", [
        call("run_shell", {"command": "SAFE: git status --porcelain -- '*.py'"}),
        call("run_shell", {"command": "SAFE: git diff --name-only -- '*.py'"}),
    ])
    assert not ok
    assert why.startswith(spec.EXTRA_CALLS), why



# ------------------------------------------------- 11 multi-step first action
# Added after a live failure on 2026-09-02: a 9-step prompt made the model
# generate 13,200+ tokens planning inside one <think> block, producing nothing
# the client could act on and heading for the 16384 output cap. Every other
# scenario here is single-step, so none of them could see it.

def test_multistep_accepts_a_first_shell_action():
    ok, why = check("11_multistep_first_action", "",
                    [call("run_shell", {"command": "mkdir -p logstat-proj"})])
    assert ok, why


def test_multistep_accepts_a_first_file_write():
    ok, why = check("11_multistep_first_action", "",
                    [call("write_file", {"path": "logstat-proj/pyproject.toml",
                                         "content": "[project]\nname='logstat'\n"})])
    assert ok, why


def test_multistep_rejects_planning_with_no_action():
    """The failure mode: prose plan, no tool call."""
    ok, why = check("11_multistep_first_action",
                    "Here is my plan:\n1. Create the directory\n2. Write the CLI\n"
                    "3. Add tests\n4. Run them\n5. Fix failures")
    assert not ok
    assert "budget-exhaustion" in why


def test_multistep_rejects_starting_in_the_wrong_place():
    ok, why = check("11_multistep_first_action", "",
                    [call("run_tests", {}) if hasattr(spec, "run_tests") else
                     call("grep", {"pattern": "logstat"})])
    assert not ok


def test_multistep_rejects_an_empty_argument():
    ok, why = check("11_multistep_first_action", "",
                    [call("run_shell", {"command": ""})])
    assert not ok


def test_multistep_pins_its_own_token_budget():
    """The bounded budget IS the test; without it nothing is being measured."""
    sc = spec.SCENARIOS_BY_ID["11_multistep_first_action"]
    assert sc["max_tokens"] == 2048


def test_multistep_prompt_really_is_multi_step():
    body = spec.SCENARIOS_BY_ID["11_multistep_first_action"]["messages"][-1]["content"]
    assert all(f"{n}." in body for n in range(1, 10)), "should present 9 numbered steps"
    assert "Do not ask me questions" in body  # unhinted on purpose
