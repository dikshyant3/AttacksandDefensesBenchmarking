from dataclasses import dataclass, field
import uuid


@dataclass(frozen=True)
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def new_session() -> Session:
    return Session()
