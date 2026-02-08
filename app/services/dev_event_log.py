from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID


_REDACT_KEYS = {
    # identity / sensitive
    "email",
    "user_name",
    "sex",
    "age",
    "skin_tone",
}


def _enabled() -> bool:
    value = os.getenv("DEV_EVENT_LOG", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _log_dir() -> Path:
    return Path(os.getenv("DEV_EVENT_LOG_DIR", "logs/user_events"))


def _safe_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if k in _REDACT_KEYS:
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def log_user_event(
    *,
    user_id: UUID | str | None,
    event: str,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """
    Lightweight developer audit log for debugging and optimization (JSONL).

    - Writes to `logs/user_events/<user_id>.jsonl` by default
    - Redacts sensitive keys if they appear in meta
    - Controlled by env var `DEV_EVENT_LOG` (default: enabled)
    """
    if not _enabled():
        return

    uid = str(user_id) if user_id else "anonymous"
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": uid,
        "event": event,
        "meta": _safe_meta(meta),
    }

    d = _log_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{uid}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

