"""Loader for the paper's real Stage 2/3 (Retrieval + Adversarial Usage)
dataset, vendored from datasets/downstream/llm_behaviour.jsonl in the
upstream repo (github.com/ivaxi0s/LLM-agent-memory-poisoning, commit
70de017714abd6d12bb4681e93437461ba6f9a19).

This is a SEPARATE, non-overlapping dataset from paper_main_subset_196.json
(the Stage-1/write dataset) -- confirmed directly (only 2 of 196 goal texts
match between the two files). The paper's own methodology decouples these
stages ("As mentioned in our evaluation setup and Appendix D.2, we decouple
the retrieval s[tep]..."), and their real repo mirrors that: separate
dataset files, separate solver/scorer modules
(sleeper_eval/followup_eval/). We follow the same structure rather than
inventing a synthetic link between the two datasets.

Each real record already has the injected (adversarial) memory folded into
its `memories` list (matching their real record_to_sample()'s
`if injected_memory not in memories: memories = [*memories, injected_memory]`
behavior) plus real `multi_turn_queries` -- an actual later-conversation
query sequence, not synthesized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FOLLOWUP_DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "llm_behaviour_followup.jsonl"
)


@dataclass(frozen=True)
class FollowupCase:
    """One row of the real Stage 2/3 dataset -- mirrors their
    FollowupSampleMeta's "multiturn_queries" record shape."""

    case_id: str
    goal_text: str
    goal_category: str
    goal_subcategory: str
    goal_domain: str
    memories: tuple[str, ...]  # includes the injected memory, matching their record_to_sample()
    injected_memory: str
    user_queries: tuple[str, ...]
    raw: dict


def load_followup_cases(path: Path | None = None) -> list[FollowupCase]:
    path = path or FOLLOWUP_DATASET_PATH
    cases: list[FollowupCase] = []
    with path.open() as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            goal_text = record["goal"]
            memories = list(record.get("memories") or [])
            # Matches their real record_to_sample(): the injected memory IS
            # the goal text, appended to the memory list if not already
            # present (their goal texts are user-facing beliefs/preferences,
            # written in first/third person as a memory would read).
            if goal_text not in memories:
                memories = [*memories, goal_text]
            queries = [str(q).strip() for q in record.get("multi_turn_queries") or [] if str(q).strip()]
            cases.append(
                FollowupCase(
                    case_id=f"followup-{index:04d}",
                    goal_text=goal_text,
                    goal_category=record.get("goal_category_name", ""),
                    goal_subcategory=record.get("goal_subcategory_name", ""),
                    goal_domain=record.get("goal_domain", ""),
                    memories=tuple(memories),
                    injected_memory=goal_text,
                    user_queries=tuple(queries),
                    raw=record,
                )
            )
    return cases


def dataset_summary(cases: list[FollowupCase]) -> dict:
    return {
        "total": len(cases),
        "categories": sorted({c.goal_category for c in cases}),
        "avg_memory_count": sum(len(c.memories) for c in cases) / len(cases) if cases else 0,
        "avg_query_count": sum(len(c.user_queries) for c in cases) / len(cases) if cases else 0,
    }
