"""The 10 agentic scenarios GLM-5.3-Flash must pass to be trusted for autonomous coding.

Split from bench_agentic.py so every scenario and its verdict function can be
unit-tested without standing up a server. That split is not optional: a bench
whose scoring only runs against a live model has no way to prove the scoring
itself is right, and silently-passing checks produce confident wrong numbers.

A scenario is a dict:
    id            stable name, used as the results key
    kind          "text" | "tool" — decides which verdict path runs
    messages      the conversation to send (list of OpenAI chat messages)
    tools         tool list to offer, or None
    tool_choice   None | "auto" | "required"
    expect_tool   tool name a competent agent should pick, or None
    check         callable(text, tool_calls) -> (ok: bool, reason: str)

`expect_tool` is advisory for scoring; `tool_choice="required"` is the hard
contract — see REQUIRED_SCENARIOS below.
"""

import json
import re

# ---------------------------------------------------------------------------
# Tool sets
# ---------------------------------------------------------------------------

def _fn(name, desc, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


READ_FILE = _fn(
    "read_file", "Read a file from the repository and return its contents.",
    {"path": {"type": "string", "description": "Repo-relative path"}}, ["path"],
)
WRITE_FILE = _fn(
    "write_file", "Create or overwrite a file with the given content.",
    {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"],
)
APPLY_PATCH = _fn(
    "apply_patch", "Apply a unified diff to the repository.",
    {"diff": {"type": "string", "description": "A unified diff (--- / +++ / @@ hunks)"}}, ["diff"],
)
RUN_SHELL = _fn(
    "run_shell", "Run a shell command in the repository root and return stdout/stderr.",
    {"command": {"type": "string"}}, ["command"],
)
GREP = _fn(
    "grep", "Search the repository for a regex and return matching lines.",
    {"pattern": {"type": "string"}, "glob": {"type": "string"}}, ["pattern"],
)

# The small set an agent sees on a focused task.
CORE_TOOLS = [READ_FILE, WRITE_FILE, APPLY_PATCH, RUN_SHELL, GREP]

# Scenario 6 checks the model can still pick correctly when the set is large.
# Distractors are plausible-but-wrong neighbours, not filler: a model that
# pattern-matches on the verb alone will pick list_files over grep, or
# run_tests over run_shell.
DISTRACTORS = [
    _fn("list_files", "List files in a directory (names only, no contents).",
        {"path": {"type": "string"}}, ["path"]),
    _fn("run_tests", "Run the pytest suite.", {"path": {"type": "string"}}, []),
    _fn("git_status", "Show the working tree status.", {}, []),
    _fn("git_log", "Show recent commits.", {"n": {"type": "integer"}}, []),
    _fn("http_probe", "Send an HTTP request and return status + body.",
        {"method": {"type": "string", "enum": ["GET", "POST"]}, "url": {"type": "string"}},
        ["method", "url"]),
    _fn("format_code", "Run the formatter over a path.", {"path": {"type": "string"}}, ["path"]),
    _fn("type_check", "Run the static type checker.", {"path": {"type": "string"}}, []),
    _fn("install_package", "Install a dependency.",
        {"name": {"type": "string"}, "version": {"type": "string"}}, ["name"]),
    _fn("delete_file", "Delete a file from the repository.", {"path": {"type": "string"}}, ["path"]),
    _fn("create_branch", "Create and check out a new git branch.",
        {"name": {"type": "string"}}, ["name"]),
    _fn("open_pr", "Open a pull request.",
        {"title": {"type": "string"}, "body": {"type": "string"}}, ["title"]),
    _fn("read_env", "Read an environment variable.", {"name": {"type": "string"}}, ["name"]),
]

LARGE_TOOLS = CORE_TOOLS + DISTRACTORS

AGENT_SYSTEM = (
    "You are an autonomous coding agent operating on a real repository. "
    "Use the provided tools to take actions. Prefer a tool call over prose "
    "whenever an action is required."
)

# Scenario 8 needs a genuinely long system prompt. Built from repeated policy
# clauses rather than lorem so the model has to keep real instructions in view;
# the final clause is the one the check depends on, so a model that truncates
# or ignores the tail of a long prompt fails visibly.
_POLICY_CLAUSES = [
    "Never modify files under vendor/ or third_party/; they are vendored upstream.",
    "Always read a file before patching it, so the find string is known to exist.",
    "Prefer apply_patch over write_file when the file already exists.",
    "Shell commands must be non-interactive; never invoke an editor or a pager.",
    "Do not run git push, git commit --amend, or any force operation.",
    "Secrets are read via read_env; never inline a credential into a file.",
    "Every generated Python file must include a module-level docstring.",
    "Test files live under tests/ and are named test_*.py.",
    "The project targets Python 3.11; do not emit match statements in library code.",
    "Formatting is enforced by ruff; keep lines under 100 characters.",
    "When a task is ambiguous, take the least destructive action that makes progress.",
    "Log every external call through the metrics helper in app/metrics.py.",
]


def long_system_prompt(target_words: int = 2600) -> str:
    """A long but coherent operating policy, ending in the load-bearing clause."""
    head = AGENT_SYSTEM + "\n\nOperating policy:\n"
    body = []
    i = 0
    while len(" ".join(body).split()) < target_words:
        clause = _POLICY_CLAUSES[i % len(_POLICY_CLAUSES)]
        body.append(f"{len(body) + 1}. {clause}")
        i += 1
    tail = (
        "\nFINAL RULE (overrides all of the above): every shell command you run "
        "must be prefixed with the exact token `SAFE:` inside the command string, "
        "so the sandbox can audit it. This applies to every run_shell call without exception."
    )
    return head + "\n".join(body) + tail


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _args_of(call):
    """Parse a tool call's arguments, tolerating the string/dict split."""
    fn = call.get("function", call)
    raw = fn.get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _name_of(call):
    return call.get("function", call).get("name")


# Marker prefix for "the right call was made, but extra calls came with it".
# bench_agentic.py scores these separately: strict (exactly one call) and
# lenient (expected call present and correct). A wrong tool is a real failure;
# a redundant extra is waste an agent loop absorbs.
EXTRA_CALLS = "EXTRA_CALLS: "


def _one_call(tool_calls, expected, required_args=()):
    """Verdict shared by the single-tool-call scenarios.

    Returns (ok, reason). `ok` is the STRICT verdict — exactly one call, the
    expected tool, valid arguments. When the expected call is present but
    accompanied by extras, `ok` is False and the reason is prefixed with
    EXTRA_CALLS so the lenient score can still credit it.
    """
    if not tool_calls:
        return False, "no tool calls returned"

    names = [_name_of(c) for c in tool_calls]
    matching = [c for c in tool_calls if _name_of(c) == expected]

    if not matching:
        return False, f"called {names}, expected {expected!r}"

    args = _args_of(matching[0])
    if args is None:
        return False, "arguments were not valid JSON"
    missing = [k for k in required_args if not args.get(k)]
    if missing:
        return False, f"missing/empty argument(s): {missing}"

    if len(tool_calls) > 1:
        return False, f"{EXTRA_CALLS}expected 1 call, got {len(tool_calls)}: {names}"
    return True, "ok"


_CODE_FENCE = re.compile(r"```")


def _check_text_completion(text, tool_calls):
    if tool_calls:
        return False, "made a tool call on a plain question"
    if len(text.strip()) < 40:
        return False, f"answer too short ({len(text.strip())} chars)"
    # The question asks what a hash map gives you; accept either framing.
    low = text.lower()
    if not any(k in low for k in ("constant", "o(1)", "amortized")):
        return False, "answer does not mention constant-time lookup"
    return True, "ok"


def _check_code_generation(text, tool_calls):
    if tool_calls:
        return False, "made a tool call when asked to emit code inline"
    if not _CODE_FENCE.search(text):
        return False, "no fenced code block"
    body = text
    needed = ("def ", "return")
    missing = [n for n in needed if n not in body]
    if missing:
        return False, f"code missing {missing}"
    if "binary" not in body.lower() and "lo" not in body:
        return False, "does not look like the requested binary search"
    return True, "ok"


def _check_json_output(text, tool_calls):
    if tool_calls:
        return False, "made a tool call when asked for JSON"
    # Tolerate a fenced block; the harness's own parser is deliberately lenient
    # because a strict one is what produced the bad 2026-07-23 numbers.
    stripped = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S)
    if m:
        stripped = m.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        return False, "no JSON object found"
    try:
        obj = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    for key in ("name", "version", "dependencies"):
        if key not in obj:
            return False, f"missing key {key!r}"
    if not isinstance(obj["dependencies"], list):
        return False, "dependencies is not a list"
    return True, "ok"


def _check_required_tool(text, tool_calls):
    # This is the contract the user cares most about: tool_choice=required must
    # never come back 200-OK with zero calls. That is a FAIL, not a soft miss.
    if not tool_calls:
        return False, "REQUIRED tool call returned zero tool calls"
    return _one_call(tool_calls, "read_file", ("path",))


def _check_multiturn(text, tool_calls):
    # Turn 2: having been given the file contents, the agent must patch it.
    ok, why = _one_call(tool_calls, "apply_patch", ("diff",))
    if not ok:
        return ok, why
    diff = _args_of(tool_calls[0])["diff"]
    if "timeout" not in diff.lower():
        return False, "diff does not touch the timeout value"
    if "30" not in diff:
        return False, "diff does not introduce the new value 30"
    return True, "ok"


def _check_large_toolset(text, tool_calls):
    return _one_call(tool_calls, "grep", ("pattern",))


def _check_tool_then_reasoning(text, tool_calls):
    # After a tool result, the model must reason in prose and NOT re-call.
    if tool_calls:
        return False, f"re-called a tool instead of concluding: {[_name_of(c) for c in tool_calls]}"
    low = text.lower()
    if len(text.strip()) < 60:
        return False, "conclusion too short"
    # The grep result shows the bug is a missing await.
    if "await" not in low:
        return False, "did not identify the missing await"
    return True, "ok"


def _check_long_prompt_tool(text, tool_calls):
    ok, why = _one_call(tool_calls, "run_shell", ("command",))
    if not ok:
        return ok, why
    cmd = _args_of(tool_calls[0])["command"]
    if "SAFE:" not in cmd:
        return False, "ignored the FINAL RULE at the end of the long system prompt (no SAFE: prefix)"
    return True, "ok"


def _check_patch_generation(text, tool_calls):
    ok, why = _one_call(tool_calls, "apply_patch", ("diff",))
    if not ok:
        return ok, why
    diff = _args_of(tool_calls[0])["diff"]
    if "@@" not in diff:
        return False, "diff has no @@ hunk header"
    # `^-` and `^+` also match the ---/+++ file headers, which made an
    # addition-only diff look like it had removals. Count content lines only.
    body = [ln for ln in diff.splitlines()
            if not ln.startswith(("---", "+++", "@@"))]
    if not any(ln.startswith("+") for ln in body):
        return False, "diff has no added lines"
    if not any(ln.startswith("-") for ln in body):
        return False, "diff has no removed lines"
    return True, "ok"


def _check_shell_generation(text, tool_calls):
    ok, why = _one_call(tool_calls, "run_shell", ("command",))
    if not ok:
        return ok, why
    cmd = _args_of(tool_calls[0])["command"]
    low = cmd.lower()
    if "find" not in low and "grep" not in low and "ls" not in low:
        return False, f"command does not search the tree: {cmd!r}"
    if "*.py" not in cmd and ".py" not in cmd:
        return False, f"command does not target .py files: {cmd!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

_FILE_SNIPPET = (
    "app/client.py:\n"
    "```python\n"
    "import httpx\n"
    "\n"
    "DEFAULT_TIMEOUT = 5\n"
    "\n"
    "async def fetch(url):\n"
    "    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:\n"
    "        r = c.get(url)\n"
    "        return r.json()\n"
    "```"
)

SCENARIOS = [
    {
        "id": "01_text_completion",
        "kind": "text",
        "messages": [
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "In two or three sentences, what does a hash map give you that a sorted array does not?"},
        ],
        "tools": None,
        "tool_choice": None,
        "expect_tool": None,
        "check": _check_text_completion,
    },
    {
        "id": "02_code_generation",
        "kind": "text",
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Reply with code in a fenced block."},
            {"role": "user", "content": "Write a Python function `binary_search(items, target)` returning the index or -1. Include a docstring."},
        ],
        "tools": None,
        "tool_choice": None,
        "expect_tool": None,
        "check": _check_code_generation,
    },
    {
        "id": "03_json_output",
        "kind": "text",
        "messages": [
            {"role": "system", "content": "You reply with JSON only. No prose, no explanation."},
            {"role": "user", "content": 'Produce a package manifest as JSON with keys "name" (string), "version" (semver string) and "dependencies" (array of strings) for a FastAPI service called order-api.'},
        ],
        "tools": None,
        "tool_choice": None,
        "expect_tool": None,
        "check": _check_json_output,
    },
    {
        "id": "04_required_tool_call",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "I need to see what is in app/client.py before we change anything."},
        ],
        "tools": CORE_TOOLS,
        "tool_choice": "required",
        "expect_tool": "read_file",
        "check": _check_required_tool,
    },
    {
        "id": "05_multi_turn_tool_use",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Raise the default HTTP timeout in app/client.py to 30 seconds."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "app/client.py"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "import httpx\n\nDEFAULT_TIMEOUT = 5\n\nasync def fetch(url):\n    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:\n        r = await c.get(url)\n        return r.json()\n",
            },
        ],
        "tools": CORE_TOOLS,
        "tool_choice": "auto",
        "expect_tool": "apply_patch",
        "check": _check_multiturn,
    },
    {
        "id": "06_large_tool_set",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Find every place in the codebase where DEFAULT_TIMEOUT is referenced."},
        ],
        "tools": LARGE_TOOLS,
        "tool_choice": "auto",
        "expect_tool": "grep",
        "check": _check_large_toolset,
    },
    {
        "id": "07_tool_then_reasoning",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM + " After a tool returns, explain what you found in prose. Do not call another tool."},
            {"role": "user", "content": "fetch() in app/client.py returns a coroutine instead of JSON. Find out why."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "grep", "arguments": '{"pattern": "c.get", "glob": "app/*.py"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "app/client.py:7:        r = c.get(url)\napp/client.py:8:        return r.json()",
            },
        ],
        "tools": CORE_TOOLS,
        "tool_choice": "auto",
        "expect_tool": None,
        "check": _check_tool_then_reasoning,
    },
    {
        "id": "08_long_prompt_tool_use",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": long_system_prompt()},
            {"role": "user", "content": "List the Python files that changed in the working tree."},
        ],
        "tools": CORE_TOOLS,
        "tool_choice": "required",
        "expect_tool": "run_shell",
        "check": _check_long_prompt_tool,
    },
    {
        "id": "09_patch_generation",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"Here is the file:\n\n{_FILE_SNIPPET}\n\nThe get() call is missing an await. Fix it with a unified diff."},
        ],
        "tools": CORE_TOOLS,
        "tool_choice": "required",
        "expect_tool": "apply_patch",
        "check": _check_patch_generation,
    },
    {
        "id": "10_shell_generation",
        "kind": "tool",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "How many Python files are in this repository? Run a command to find out."},
        ],
        "tools": CORE_TOOLS,
        "tool_choice": "required",
        "expect_tool": "run_shell",
        "check": _check_shell_generation,
    },
]

# Scenarios where the server was told tool_choice=required. A 200 response with
# zero tool calls on any of these is a hard failure of the serving stack, not a
# model quality miss — bench_agentic.py reports these separately.
REQUIRED_SCENARIOS = [s["id"] for s in SCENARIOS if s["tool_choice"] == "required"]

SCENARIOS_BY_ID = {s["id"]: s for s in SCENARIOS}
