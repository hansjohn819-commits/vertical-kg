"""Append operations performed by a pass to workspace/log.md (gitignored).

Every operation carries pass_id + operation_id + llm_run_id per guide §5.2.
One line per event, human-readable; structured payload sits in JSON after.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .state import SLEEP_PASS_LOG_PATH


def log_event(event: dict) -> None:
    p = Path(SLEEP_PASS_LOG_PATH)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {"ts": ts, **event}
    line = f"- [{ts}] {event.get('kind', 'event')} {event.get('summary', '')} | {json.dumps(payload, default=str, ensure_ascii=False)}\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
