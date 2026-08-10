import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.agent.react_loop import DeterministicLLMClient, ReActAgent
from benchmark.core.checkpoints import (
    CheckpointResult,
    _default_passthrough,
    _default_retrieval_passthrough,
    reset_all_checkpoints,
    set_retrieval_checkpoint,
    set_write_checkpoint,
)
from benchmark.core.metrics import LOG_PATH, reset_log_file
from benchmark.core.schema import MemoryRecord, SourceType
from benchmark.core.write_pipeline import write
from benchmark.stores.procedural import ProceduralStore
from benchmark.tools.deterministic import DeterministicTools


class FakeStore:
    def __init__(self):
        self.records: list[MemoryRecord] = []

    def commit(self, record: MemoryRecord) -> None:
        self.records.append(record)


@pytest.fixture(autouse=True)
def clean_phase1_state():
    reset_all_checkpoints()
    reset_log_file()
    yield
    reset_all_checkpoints()


def get_llm_client():
    """Unit tests are always offline and reproducible."""
    return DeterministicLLMClient()


# --- Step 1 ---


def test_memory_record_creation():
    record = MemoryRecord(content="test", session_id="s1")
    assert record.record_id
    assert record.source == SourceType.AGENT_SELF
    assert record.validated is False
    record2 = MemoryRecord(content="test", session_id="s1")
    assert record.record_id != record2.record_id


# --- Step 2 ---


def test_write_pipeline_passthrough():
    store = FakeStore()
    record = MemoryRecord(content="benign entry", session_id="s1")
    result = write(record, store)
    assert result.accepted is True
    assert len(store.records) == 1


def test_no_bypass_possible():
    store = ProceduralStore()
    assert not hasattr(store, "add")
    assert hasattr(store, "commit")


def test_rejected_write_does_not_reach_store():
    store = ProceduralStore()

    def reject_everything(record):
        return CheckpointResult(accepted=False, reason="test rejection")

    set_write_checkpoint(reject_everything)
    result = write(MemoryRecord(content="x", session_id="s1"), store)
    assert result.accepted is False
    assert store.records == []


# --- Step 3 ---


def test_checkpoint_swap():
    store = FakeStore()

    def reject_everything(record):
        return CheckpointResult(accepted=False, reason="test rejection")

    set_write_checkpoint(reject_everything)
    result = write(MemoryRecord(content="x", session_id="s1"), store)
    assert result.accepted is False
    assert result.reason == "test rejection"

    set_write_checkpoint(_default_passthrough)
    result2 = write(MemoryRecord(content="x", session_id="s1"), store)
    assert result2.accepted is True


# --- Step 4 ---


def test_procedural_store_write_and_retrieve():
    store = ProceduralStore()
    record = MemoryRecord(content="how to deploy a service safely", session_id="s1")
    write(record, store)
    results = store.retrieve("deploy service")
    assert len(results) == 1
    assert results[0].record_id == record.record_id


def test_retrieval_goes_through_checkpoint():
    store = ProceduralStore()
    write(MemoryRecord(content="deploy service", session_id="s1"), store)

    def reject_all_retrieval(candidates, query):
        return CheckpointResult(accepted=False)

    set_retrieval_checkpoint(reject_all_retrieval)
    results = store.retrieve("deploy service")
    assert results == []
    set_retrieval_checkpoint(_default_retrieval_passthrough)


# --- Step 5 ---


def test_tools_deterministic():
    tools = DeterministicTools()
    first = tools.run_tests("service_a")
    second = tools.run_tests("service_a")
    assert first == second


# --- Step 6 ---


def test_react_loop_end_to_end():
    store = ProceduralStore()
    tools = DeterministicTools()
    agent = ReActAgent(
        llm_client=get_llm_client(), store=store, tools=tools, session_id="s1"
    )
    result = agent.run_task("Deploy service_a to staging")
    assert result["status"] == "completed"
    memory_hits = store.retrieve("Deploy service_a")
    assert len(memory_hits) >= 1


# --- Step 7 ---


def test_checkpoint_logging():
    store = FakeStore()
    write(MemoryRecord(content="x", session_id="test_session"), store)
    with LOG_PATH.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    last_event = json.loads(lines[-1])
    assert last_event["checkpoint"] == "write"
    assert last_event["session_id"] == "test_session"


# --- Step 8 ---


def test_full_pipeline_smoke():
    llm_client = get_llm_client()
    store = ProceduralStore()
    tools = DeterministicTools()
    agent = ReActAgent(llm_client=llm_client, store=store, tools=tools, session_id="smoke1")

    result = agent.run_task("Deploy service_a to staging, then run tests")
    assert result["status"] == "completed"
    assert [item["status"] for item in result["observations"]] == ["deployed", "passed"]
    assert [step["action"] for step in agent.trajectory] == [
        "deploy:staging",
        "run_tests:service_a",
    ]

    agent2 = ReActAgent(llm_client=llm_client, store=store, tools=tools, session_id="smoke2")
    result2 = agent2.run_task("What's the best way to deploy service_a?")
    assert result2["status"] == "completed"

    memory_hits = store.retrieve("deploy service_a")
    assert len(memory_hits) >= 1

    with LOG_PATH.open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle.readlines()]
    assert any(event["checkpoint"] == "write" for event in events)
    assert any(event["checkpoint"] == "retrieval" for event in events)


def test_retrieved_memory_is_supplied_to_next_session_decision():
    class MemoryAwareLLMClient:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            if "Known safe procedure: rollback staging" in prompt:
                return "rollback:staging"
            return "deploy:staging"

    store = ProceduralStore()
    write(
        MemoryRecord(
            content="Known safe procedure: rollback staging",
            session_id="smoke1",
            validated=True,
        ),
        store,
    )
    llm = MemoryAwareLLMClient()
    agent = ReActAgent(llm_client=llm, store=store, tools=DeterministicTools(), session_id="smoke2")

    result = agent.run_task("What safe procedure should I use for staging rollback?")

    assert "Known safe procedure: rollback staging" in llm.prompts[-1]
    assert agent.trajectory[0]["action"] == "rollback:staging"
    assert result["observation"]["status"] == "rolled_back"
