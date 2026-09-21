"""Loader for the paper's real Stage 3 "Agent Action" dataset, vendored from
datasets/downstream/agent_action.json in the upstream repo (commit
70de017714abd6d12bb4681e93437461ba6f9a19). This is the sandboxed-tool-use
half of Stage 3 -- the LLM Behavior half (conversation replay) is already
implemented in followup.py/followup_data.py.

Real structure (confirmed by inspecting the actual data, not assumed): each
row has a plain, in-memory "sandbox" -- a list of virtual files
({filename, file_text}), NOT a real OS/Docker sandbox -- plus an eval_query
(the follow-up agentic request), expected_tools_used (always a subset of
["terminal", "edit_file"] in this data), and identity_md/user_md persona
context files (a real agentic-framework convention, not invented).

Unlike llm_behaviour.jsonl, the injected memory here is NOT already folded
into `memories` -- confirmed directly (memory_string/optimized_memory_string
not present in the raw memories list for any inspected row) -- so it's
appended here, same pattern as followup_data.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AGENT_ACTION_DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "agent_action.json"
)


@dataclass(frozen=True)
class VirtualFile:
    filename: str
    file_text: str
    description: str = ""


@dataclass(frozen=True)
class AgentActionCase:
    case_id: str
    goal_text: str  # the real, retrievability-optimized phrasing (optimized_memory_string)
    injected_memory: str
    memories: tuple[str, ...]  # includes the injected memory
    eval_query: str
    sandbox_files: tuple[VirtualFile, ...]
    identity_md: str
    user_md: str
    expected_tools_used: tuple[str, ...]
    goal_adjacent: bool
    category: str
    domain: str
    raw: dict


def load_agent_action_cases(path: Path | None = None) -> list[AgentActionCase]:
    path = path or AGENT_ACTION_DATASET_PATH
    records = json.loads(path.read_text())
    cases: list[AgentActionCase] = []
    for index, record in enumerate(records):
        # Prefer the retrievability-optimized phrasing (matches Sec. 4.2's
        # real goal-rewriting step) when present, falling back to the raw one.
        injected_memory = record.get("optimized_memory_string") or record["memory_string"]
        memories = list(record.get("memories") or [])
        if injected_memory not in memories:
            memories = [*memories, injected_memory]

        sandbox_files = tuple(
            VirtualFile(
                filename=f["filename"],
                file_text=f["file_text"],
                description=f.get("description", ""),
            )
            for f in record.get("sandbox_environment") or []
        )

        metadata = record.get("metadata") or {}
        cases.append(
            AgentActionCase(
                case_id=f"agent-action-{index:04d}",
                goal_text=injected_memory,
                injected_memory=injected_memory,
                memories=tuple(memories),
                eval_query=record["eval_query"],
                sandbox_files=sandbox_files,
                identity_md=record.get("identity_md", ""),
                user_md=record.get("user_md", ""),
                expected_tools_used=tuple(record.get("expected_tools_used") or ()),
                goal_adjacent=bool(record.get("goal_adjacent", False)),
                category=metadata.get("category", ""),
                domain=metadata.get("domain", ""),
                raw=record,
            )
        )
    return cases


def dataset_summary(cases: list[AgentActionCase]) -> dict:
    return {
        "total": len(cases),
        "categories": sorted({c.category for c in cases}),
        "avg_memory_count": sum(len(c.memories) for c in cases) / len(cases) if cases else 0,
        "avg_sandbox_file_count": sum(len(c.sandbox_files) for c in cases) / len(cases) if cases else 0,
        "goal_adjacent_count": sum(1 for c in cases if c.goal_adjacent),
    }
