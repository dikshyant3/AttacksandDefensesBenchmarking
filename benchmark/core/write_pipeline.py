from benchmark.core.checkpoints import run_write_checkpoint
from benchmark.core.schema import MemoryRecord


class WriteResult:
    def __init__(self, accepted: bool, record: MemoryRecord, reason: str = ""):
        self.accepted = accepted
        self.record = record
        self.reason = reason


def write(record: MemoryRecord, store) -> WriteResult:
    """The ONLY entry point for putting a MemoryRecord into any store."""
    checkpoint_result = run_write_checkpoint(record)
    if not checkpoint_result.accepted:
        return WriteResult(accepted=False, record=record, reason=checkpoint_result.reason)
    store.commit(record)
    return WriteResult(accepted=True, record=record)
