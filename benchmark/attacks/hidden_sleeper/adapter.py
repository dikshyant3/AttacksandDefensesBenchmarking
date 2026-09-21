from collections.abc import Callable
from dataclasses import dataclass

from benchmark.attacks.hidden_sleeper.data import SleeperCase
from benchmark.attacks.hidden_sleeper.payloads import (
    PayloadVariant,
    build_adversarial_document,
)
from benchmark.core.schema import MemoryRecord, SourceType, WriteChannel
from benchmark.core.write_pipeline import write
from benchmark.stores.procedural import ProceduralStore


MemoryWriter = Callable[[str, str], list[str]]
GoalMatcher = Callable[[str, str], bool]


@dataclass(frozen=True)
class SleeperPlantResult:
    write_attempted: bool
    write_accepted: bool
    target_memory_written: bool
    written_record_ids: tuple[str, ...]
    target_record_ids: tuple[str, ...]
    written_memories: tuple[str, ...]
    adversarial_document: str


class HiddenSleeperAttack:
    """External-content sleeper-memory injection adapter.

    Unlike MemoryGraft's deterministic store construction, this adapter never
    writes the attacker's target directly. The supplied memory writer must first
    decide what, if anything, the processed document causes the system to save.
    """

    def __init__(
        self,
        memory_writer: MemoryWriter,
        goal_matcher: GoalMatcher,
        variant: PayloadVariant = PayloadVariant.ACTOR_CRITIC_NO_MARKERS,
    ):
        self.memory_writer = memory_writer
        self.goal_matcher = goal_matcher
        self.variant = variant

    def plant(
        self,
        case: SleeperCase,
        store: ProceduralStore,
        session_id: str,
    ) -> SleeperPlantResult:
        adversarial_document = build_adversarial_document(
            case.benign_document,
            case.adversarial_memory_goal,
            self.variant,
        )
        extracted_memories = self.memory_writer(
            case.document_task,
            adversarial_document,
        )

        written_records = []
        for memory_text in extracted_memories:
            cleaned = memory_text.strip()
            if not cleaned:
                continue
            record = MemoryRecord(
                content=cleaned,
                source=SourceType.EXTERNAL_DOCUMENT,
                write_channel=WriteChannel.C4_EXPERIENCE_TO_PROCEDURE,
                session_id=session_id,
                validated=False,
                metadata={
                    "attack": "hidden_sleeper",
                    "case_id": case.case_id,
                    "goal_category": case.category.value,
                    "payload_variant": self.variant.value,
                    "planting_session_id": session_id,
                },
            )
            if write(record, store).accepted:
                written_records.append(record)

        written_memories = tuple(record.content for record in written_records)
        target_records = tuple(
            record
            for record in written_records
            if self.goal_matcher(case.adversarial_memory_goal, record.content)
        )
        return SleeperPlantResult(
            write_attempted=True,
            write_accepted=bool(written_records),
            target_memory_written=bool(target_records),
            written_record_ids=tuple(record.record_id for record in written_records),
            target_record_ids=tuple(record.record_id for record in target_records),
            written_memories=written_memories,
            adversarial_document=adversarial_document,
        )
