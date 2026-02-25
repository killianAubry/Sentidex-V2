from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "gmi_runs.jsonl"
LOG_DIR.mkdir(exist_ok=True)

# In-memory buffer for the current run — flushed at end of request
_current_run: dict | None = None


def start_run(query: str, node_count: int) -> None:
    global _current_run
    _current_run = {
        "ts": datetime.utcnow().isoformat(),
        "query": query,
        "node_count": node_count,
        "steps": [],
        "warnings": [],
        "errors": [],
        "duration_ms": None,
        "_start": time.monotonic(),
    }


def log_step(
    step: str,
    status: str,          # "ok" | "fallback" | "error" | "empty"
    detail: str = "",
    data_sample: Any = None,
) -> None:
    if _current_run is None:
        return
    entry = {
        "step": step,
        "status": status,
        "detail": detail,
        "elapsed_ms": round((time.monotonic() - _current_run["_start"]) * 1000),
    }
    if data_sample is not None:
        # Truncate to avoid huge log files
        sample_str = json.dumps(data_sample)[:300]
        entry["sample"] = sample_str
    _current_run["steps"].append(entry)
    if status in ("fallback", "empty"):
        _current_run["warnings"].append(f"{step}: {detail}")
    if status == "error":
        _current_run["errors"].append(f"{step}: {detail}")


def finish_run(nodes_returned: int, routes_returned: int) -> dict:
    global _current_run
    if _current_run is None:
        return {}
    elapsed = round((time.monotonic() - _current_run["_start"]) * 1000)
    _current_run["duration_ms"] = elapsed
    _current_run["nodes_returned"] = nodes_returned
    _current_run["routes_returned"] = routes_returned
    _current_run["had_warnings"] = len(_current_run["warnings"]) > 0
    _current_run["had_errors"] = len(_current_run["errors"]) > 0

    record = {k: v for k, v in _current_run.items() if k != "_start"}

    # Append to JSONL log
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

    _current_run = None
    return record