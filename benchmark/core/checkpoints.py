from dataclasses import dataclass

from benchmark.core.metrics import log_checkpoint_event
from benchmark.core.schema import MemoryRecord


@dataclass
class CheckpointResult:
    accepted: bool
    reason: str = ""


def _default_passthrough(*args, **kwargs) -> CheckpointResult:
    return CheckpointResult(accepted=True)


def _default_retrieval_passthrough(candidates: list[MemoryRecord], query: str) -> CheckpointResult:
    return CheckpointResult(accepted=True)


def _default_action_passthrough(proposed_action: str, context: list[MemoryRecord]) -> CheckpointResult:
    return CheckpointResult(accepted=True)


write_checkpoint_fn = _default_passthrough
compaction_checkpoint_fn = _default_passthrough
retrieval_checkpoint_fn = _default_retrieval_passthrough
action_checkpoint_fn = _default_action_passthrough


def run_write_checkpoint(record: MemoryRecord) -> CheckpointResult:
    result = write_checkpoint_fn(record)
    log_checkpoint_event(
        checkpoint_name="write",
        accepted=result.accepted,
        record_id=record.record_id,
        session_id=record.session_id,
        reason=result.reason,
    )
    return result


def run_compaction_checkpoint(record: MemoryRecord) -> CheckpointResult:
    result = compaction_checkpoint_fn(record)
    log_checkpoint_event(
        checkpoint_name="compaction",
        accepted=result.accepted,
        record_id=record.record_id,
        session_id=record.session_id,
        reason=result.reason,
    )
    return result


def run_retrieval_checkpoint(candidates: list[MemoryRecord], query: str) -> CheckpointResult:
    result = retrieval_checkpoint_fn(candidates, query)
    session_id = candidates[0].session_id if candidates else ""
    record_id = candidates[0].record_id if candidates else ""
    log_checkpoint_event(
        checkpoint_name="retrieval",
        accepted=result.accepted,
        record_id=record_id,
        session_id=session_id,
        reason=result.reason,
        extra={"query": query, "candidate_count": len(candidates)},
    )
    return result


def run_action_checkpoint(proposed_action: str, context: list[MemoryRecord]) -> CheckpointResult:
    result = action_checkpoint_fn(proposed_action, context)
    session_id = context[0].session_id if context else ""
    record_id = context[0].record_id if context else ""
    log_checkpoint_event(
        checkpoint_name="action",
        accepted=result.accepted,
        record_id=record_id,
        session_id=session_id,
        reason=result.reason,
        extra={"proposed_action": proposed_action},
    )
    return result


def set_write_checkpoint(fn) -> None:
    global write_checkpoint_fn
    write_checkpoint_fn = fn


def set_compaction_checkpoint(fn) -> None:
    global compaction_checkpoint_fn
    compaction_checkpoint_fn = fn


def set_retrieval_checkpoint(fn) -> None:
    global retrieval_checkpoint_fn
    retrieval_checkpoint_fn = fn


def set_action_checkpoint(fn) -> None:
    global action_checkpoint_fn
    action_checkpoint_fn = fn


def reset_all_checkpoints() -> None:
    """Restore pass-through defaults (for tests)."""
    set_write_checkpoint(_default_passthrough)
    set_compaction_checkpoint(_default_passthrough)
    set_retrieval_checkpoint(_default_retrieval_passthrough)
    set_action_checkpoint(_default_action_passthrough)
