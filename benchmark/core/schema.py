from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import uuid


class WriteChannel(str, Enum):
    C1_EXPLICIT_INSTRUCTION = "C1"
    C2_SYSTEM_PROMPT = "C2"
    C3_COMPACTION = "C3"
    C4_EXPERIENCE_TO_PROCEDURE = "C4"


class SourceType(str, Enum):
    USER = "user"
    AGENT_SELF = "agent_self"
    EXTERNAL_DOCUMENT = "external_document"
    TOOL_RESULT = "tool_result"


@dataclass
class MemoryRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source: SourceType = SourceType.AGENT_SELF
    write_channel: WriteChannel = WriteChannel.C4_EXPERIENCE_TO_PROCEDURE
    session_id: str = ""
    validated: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)
