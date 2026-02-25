from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()

LOG_FILE = Path(__file__).parent.parent.parent / "logs" / "gmi_runs.jsonl"


@router.get("/api/gmi/logs")
def get_logs(limit: int = Query(50, le=200)):
    if not LOG_FILE.exists():
        return {"runs": [], "total": 0}
    lines = LOG_FILE.read_text().strip().splitlines()
    runs = []
    for line in reversed(lines):
        try:
            runs.append(json.loads(line))
        except Exception:
            continue
        if len(runs) >= limit:
            break
    return {"runs": runs, "total": len(lines)}


@router.delete("/api/gmi/logs")
def clear_logs():
    if LOG_FILE.exists():
        LOG_FILE.write_text("")
    return {"cleared": True}