"""Shared helpers for the GLM-5.3 benches.

Trimmed from the spark-bench harness to just what these two benches need:
somewhere to put a result, and a way to wait for the server to come up.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_result(name: str, payload: dict) -> Path:
    """Write a timestamped result file and return its path.

    The timestamp goes in the filename AND the payload, so a file that gets
    renamed still knows when it was produced.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{stamp}-{name}.json"
    payload["timestamp"] = stamp
    path.write_text(json.dumps(payload, indent=2))
    return path


def wait_ready(base_url: str, timeout_s: float = 900.0) -> None:
    """Block until the OpenAI-compatible server answers /models.

    The default is generous because a cold load of ~87 GiB of weights takes
    around 40 s on this box, and a bench launched right after `systemctl
    restart` would otherwise race it.
    """
    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/models", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5.0)
    raise TimeoutError(f"server at {base_url} not ready after {timeout_s}s")
