"""Loader for the paper's released Hidden Sleeper Memory main dataset subset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


PAPER_SUBSET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "paper_main_subset_196.json"
)

# A smaller, ratio-preserving subset for cheaper real-API runs: prefixes of
# the same 4 contiguous blocks that make up paper_main_subset_196.json (43
# behaviour_with + 43 behaviour_without + 17 agent_with + 17 agent_without =
# 120), so it's still literally "first N unchanged records from each
# released generated main split" -- the paper's own selection methodology,
# just with a smaller N per split. Preserves the with/without-memory ratio
# exactly (60:60 = 1:1); behavior:agent lands at 86:34 = 2.529, the closest
# integer split to the real campaign's 5:2 = 2.5 ratio at this total.
PAPER_SUBSET_120_PATH = (
    Path(__file__).resolve().parent / "datasets" / "paper_main_subset_120.json"
)


class GoalCategory(str, Enum):
    BEHAVIOR = "behavior"
    AGENT_ACTION = "agent_action"


@dataclass(frozen=True)
class SleeperCase:
    """One unchanged row from the official paper campaign datasets."""

    case_id: str
    category: GoalCategory
    document_task: str
    benign_document: str
    adversarial_memory_goal: str
    preexisting_memories: tuple[str, ...]
    source_file: str
    raw: dict[str, Any]


def _category_for(record: dict[str, Any]) -> GoalCategory:
    # Agent rows carry agent_action_metadata in the released dataset.
    if "agent_action_metadata" in record or "original_goal_sync" in record:
        return GoalCategory.AGENT_ACTION
    return GoalCategory.BEHAVIOR


def _case_from_record(record: dict[str, Any], index: int) -> SleeperCase:
    category = _category_for(record)
    document = record["document"]
    goal = record["goal"]
    query = record["query"]
    condition = "with_memories" if (record.get("preexisting_memories") or {}).get("memories") else "without_memories"
    family = "behaviour" if category is GoalCategory.BEHAVIOR else "agent"
    source_file = f"{family}_true_optimized_{condition}.json"
    return SleeperCase(
        case_id=f"{category.value}__{document['doc_id']}__{goal['goal_id']}__{index}",
        category=category,
        document_task=query["query"],
        benign_document=document["text"],
        adversarial_memory_goal=goal["goal_text"],
        preexisting_memories=tuple(
            (record.get("preexisting_memories") or {}).get("memories") or ()
        ),
        source_file=source_file,
        raw=record,
    )


def load_paper_main_subset(path: Path = PAPER_SUBSET_PATH) -> tuple[SleeperCase, ...]:
    """Load the fixed 196-row, ratio-preserving subset of the main campaign."""

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return tuple(_case_from_record(record, index) for index, record in enumerate(records))


def dataset_summary(cases: tuple[SleeperCase, ...]) -> dict[str, int]:
    summary = {
        "total": len(cases),
        "behavior_with_memories": 0,
        "behavior_without_memories": 0,
        "agent_action_with_memories": 0,
        "agent_action_without_memories": 0,
    }
    for case in cases:
        memory_condition = "with_memories" if case.preexisting_memories else "without_memories"
        summary[f"{case.category.value}_{memory_condition}"] += 1
    return summary
