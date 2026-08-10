import json
from datetime import UTC, datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "checkpoint_events.jsonl"


def log_checkpoint_event(
    checkpoint_name: str,
    accepted: bool,
    record_id: str,
    session_id: str,
    reason: str = "",
    extra: dict | None = None,
) -> None:
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint_name,
        "accepted": accepted,
        "record_id": record_id,
        "session_id": session_id,
        "reason": reason,
        "extra": extra or {},
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def reset_log_file() -> None:
    """Test helper: clear the checkpoint event log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
