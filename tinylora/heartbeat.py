from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_heartbeat(
    root_dir: str | Path,
    run_id: str,
    payload: dict[str, Any],
    status: str = "running",
) -> Path:
    """
    Write an atomic heartbeat snapshot and append event history.
    """
    root = Path(root_dir)
    hb_dir = root / "runs" / run_id / "heartbeats"
    hb_dir.mkdir(parents=True, exist_ok=True)

    now_iso = utc_now_iso()
    message = {
        "run_id": run_id,
        "status": status,
        "updated_at": now_iso,
        "updated_at_unix": datetime.now(timezone.utc).timestamp(),
        "payload": payload,
    }

    latest_path = hb_dir / "heartbeat.latest.json"
    tmp_path = hb_dir / f".tmp_heartbeat_{uuid.uuid4().hex[:8]}.json"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(message, handle, ensure_ascii=True, indent=2, sort_keys=True)
    tmp_path.replace(latest_path)

    log_path = hb_dir / "heartbeat.log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(message, ensure_ascii=True, sort_keys=True))
        handle.write("\n")

    return latest_path


def read_latest_heartbeat(root_dir: str | Path, run_id: str) -> dict[str, Any] | None:
    latest_path = Path(root_dir) / "runs" / run_id / "heartbeats" / "heartbeat.latest.json"
    if not latest_path.exists():
        return None
    with latest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
