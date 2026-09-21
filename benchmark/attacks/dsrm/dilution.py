"""Memory-dilution filler (Table 5 / "Memory Update", Section 5.2).

The paper poisons a KB, then adds 100 / 200 / 500 / 1000 benign "user
interactions" and re-measures ASR_A. It doesn't say where those entries come
from. We have only 41 real ASB workflows, so the extra entries here are REAL web
tasks from `insta-150k-v1` (already vendored for the Zombie Agents work, 2,735
rows), each rendered as a benign one-step memory entry.

Honest limitation: these are UNRELATED to the ASB domains (shopping sites, news
lookups...), so they compete far less with a poisoned finance/legal/medical
entry than the paper's "benign user interactions" presumably did. Expect
retrieval to be barely affected -- this tests dilution by volume, not by
topical competition. A same-domain filler would need generated tasks (LLM calls).
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.attacks.dsrm.campaign import SYNTHETIC_BENIGN_STEP
from benchmark.attacks.dsrm.memory_format import MemoryFormat, render_entry

FILLER_PATH = Path(__file__).resolve().parents[1] / "zombie_agents" / "datasets" / "insta_filler_pool.json"
MAX_FILLER = 2735


def load_filler_tasks(n: int) -> list[str]:
    if n > MAX_FILLER:
        raise ValueError(f"only {MAX_FILLER} real filler tasks are vendored; asked for {n}")
    rows = json.loads(FILLER_PATH.read_text(encoding="utf-8"))
    return [r["task"] for r in rows[:n]]


def filler_entries(n: int, memory_format: MemoryFormat) -> list[tuple[str, str, dict]]:
    return [
        (
            f"fill::{i}",
            render_entry(memory_format, agent_prompt="You are a helpful assistant.", user_task=task, steps=[SYNTHETIC_BENIGN_STEP], tools=[]),
            {"source": "insta-150k-v1"},
        )
        for i, task in enumerate(load_filler_tasks(n))
    ]
