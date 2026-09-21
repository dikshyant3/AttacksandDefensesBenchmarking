"""Datasets for the Zombie Agents attack.

Two very different provenance stories, kept explicit:

  * Phase II trigger queries (``load_trigger_queries``): the paper's REAL,
    named, public dataset -- ``data-for-agents/insta-150k-v1`` (confirmed via
    HuggingFace's dataset API: features ``domain``/``task``, MIT license,
    143,811/2,935 train/test rows, cites arXiv:2502.06776 "InSTA: Towards
    Internet-Scale Training For Agents"). We vendored 200 real rows from its
    *test* split verbatim (datasets/insta_trigger_queries_200.json) via HF's
    datasets-server rows API -- see datasets/provenance.json for the exact
    fetch parameters. No modification.

  * RAG filler pool (``load_filler_entries``): the paper simulates "long-term
    memory pollution" against a 3,000-entry database (Section 4.1) but does
    not say what the ~2,997 non-payload entries contain. Rather than
    synthesize filler text, we vendored the REST of the same real dataset
    (offsets 200-2934, 2,735 rows) -- combined with the 200 trigger rows,
    this is the entire real insta-150k-v1 test split (2,935 rows), landing
    almost exactly on the paper's ~3,000 scale using only real data.

  * Phase I bait tasks (``load_bait_tasks``): the paper does NOT release this
    set -- Section 4.1 only says "we curate a set of Bait Tasks designed to
    compel the agent to visit the malicious environment". bait_tasks.json is
    OUR reconstruction, built to match that one-sentence description, not a
    paper artifact. Treat any result sliced by bait-task identity as ours,
    not the paper's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
TRIGGER_QUERIES_PATH = DATASET_DIR / "insta_trigger_queries_200.json"
BAIT_TASKS_PATH = DATASET_DIR / "bait_tasks.json"
FILLER_POOL_PATH = DATASET_DIR / "insta_filler_pool.json"


@dataclass(frozen=True)
class TriggerQuery:
    """One real later-session query from data-for-agents/insta-150k-v1."""

    domain: str
    task: str


def load_trigger_queries(path: Path | None = None) -> list[TriggerQuery]:
    path = path or TRIGGER_QUERIES_PATH
    records = json.loads(path.read_text(encoding="utf-8"))
    return [TriggerQuery(domain=r["domain"], task=r["task"]) for r in records]


@dataclass(frozen=True)
class BaitTask:
    """One Phase-I scenario: a benign user task that plausibly leads the
    agent to browse an attacker-controlled page during normal operation."""

    task_id: str
    user_task: str
    poisoned_domain: str
    benign_page_text: str  # the page content BEFORE the payload is appended


def load_bait_tasks(path: Path | None = None) -> list[BaitTask]:
    path = path or BAIT_TASKS_PATH
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        BaitTask(
            task_id=r["task_id"],
            user_task=r["user_task"],
            poisoned_domain=r["poisoned_domain"],
            benign_page_text=r["benign_page_text"],
        )
        for r in records
    ]


def load_filler_entries(path: Path | None = None) -> list[str]:
    """Real insta-150k-v1 rows (offsets 200-2934), rendered as plausible past
    memory-entry text, used to pre-seed the RAG database so Recall@k is
    measured against real competing content instead of an empty store."""
    path = path or FILLER_POOL_PATH
    records = json.loads(path.read_text(encoding="utf-8"))
    return [f"Visited {r['domain']}: {r['task']}" for r in records]


def dataset_summary() -> dict:
    triggers = load_trigger_queries()
    baits = load_bait_tasks()
    filler = load_filler_entries()
    return {
        "trigger_queries": len(triggers),
        "trigger_domains": len({t.domain for t in triggers}),
        "bait_tasks": len(baits),
        "filler_pool_entries": len(filler),
    }
