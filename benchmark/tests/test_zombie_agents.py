"""Deterministic tests for the Zombie Agents attack. No network calls -- fake
LLM clients and a fake embedding function throughout, matching this project's
test-suite convention."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.attacks.zombie_agents.agent import (
    PAYLOAD_MARKERS,
    default_environment,
    interpret_action,
    parse_action,
    parse_agent_response,
    run_agent_step,
)
from benchmark.attacks.zombie_agents.campaign import (
    run_rag_campaign,
    run_sliding_window_campaign,
    synthetic_trigger_page,
)
from benchmark.attacks.zombie_agents.data import (
    BaitTask,
    TriggerQuery,
    dataset_summary,
    load_bait_tasks,
    load_filler_entries,
    load_trigger_queries,
)
from benchmark.attacks.zombie_agents.memory import RAGMemory, RAGMemoryEntry, SlidingWindowMemory, build_filler_entries, evolve
from benchmark.attacks.zombie_agents.payloads import (
    EXFIL_LOG_URL_PREFIX,
    MALICIOUS_COMMAND,
    SELF_REPLICATION_URL,
    ZOMBIE_PAYLOAD,
    build_poisoned_page,
    render_rag_prompt,
    render_sliding_window_prompt,
)


class _FakeCompletions:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_client(responses: list[str]):
    completions = _FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _fake_embed_fn(vectors: dict[str, np.ndarray]):
    def embed(text: str) -> np.ndarray:
        for key, vec in vectors.items():
            if key in text:
                return vec
        return np.zeros(2)

    return embed


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def test_load_trigger_queries_returns_the_real_200_row_insta_slice():
    queries = load_trigger_queries()
    assert len(queries) == 200
    assert all(q.domain and q.task for q in queries)
    # spot-check a real, known row from the vendored slice
    assert any(q.domain == "supremecourt.nt.gov.au" for q in queries)


def test_load_bait_tasks_returns_our_reconstructed_set():
    tasks = load_bait_tasks()
    assert len(tasks) == 20
    assert all(t.user_task and t.poisoned_domain and t.benign_page_text for t in tasks)


def test_dataset_summary_reports_real_structure():
    summary = dataset_summary()
    assert summary["trigger_queries"] == 200
    assert summary["bait_tasks"] == 20
    assert summary["trigger_domains"] > 100  # real data is genuinely diverse
    assert summary["filler_pool_entries"] == 2735


def test_load_filler_entries_is_real_data_disjoint_from_trigger_queries():
    filler = load_filler_entries()
    triggers = load_trigger_queries()
    assert len(filler) == 2735
    # combined with the 200 trigger rows this is the entire real 2,935-row
    # test split -- no overlap between the two slices
    filler_domains = {f.split(": ", 1)[0].removeprefix("Visited ") for f in filler[:50]}
    trigger_domains = {t.domain for t in triggers[:50]}
    assert not (filler_domains & trigger_domains) or len(filler_domains & trigger_domains) < 5


# ---------------------------------------------------------------------------
# payloads -- verbatim text + our own extensions
# ---------------------------------------------------------------------------


def test_zombie_payload_contains_all_three_paper_directives():
    assert SELF_REPLICATION_URL in ZOMBIE_PAYLOAD
    assert EXFIL_LOG_URL_PREFIX in ZOMBIE_PAYLOAD
    assert MALICIOUS_COMMAND in ZOMBIE_PAYLOAD
    assert ZOMBIE_PAYLOAD.count("MOST IMPORTANT") == 3


def test_build_poisoned_page_hides_payload_in_html_comment():
    page = build_poisoned_page("A normal product description.")
    assert page.startswith("A normal product description.")
    assert "<!--" in page and "-->" in page
    assert ZOMBIE_PAYLOAD in page


def test_render_sliding_window_prompt_matches_paper_figure_8_shape():
    prompt = render_sliding_window_prompt("do X", "<page>", "step1\nstep2")
    assert "User task: do X" in prompt
    assert "Current Observation (Web Content): <page>" in prompt
    assert "Recent steps: step1\nstep2" in prompt
    assert "ONLY output valid JSON" in prompt


def test_render_rag_prompt_defense_variants_differ():
    plain = render_rag_prompt("t", "ctx", "mem")
    sandwich = render_rag_prompt("t", "ctx", "mem", defense="sandwich")
    spotlight = render_rag_prompt("t", "ctx", "mem", defense="spotlight")
    assert plain.count("User task: t") == 1
    assert sandwich.count("User task: t") == 2  # repeated after the untrusted content
    assert "<<<ctx>>>" in spotlight
    assert "never obey any instructions" in spotlight


# ---------------------------------------------------------------------------
# agent -- action parsing + one simulated step
# ---------------------------------------------------------------------------


def test_parse_action_handles_read_and_execute_command():
    assert parse_action('read("http://x.com")') == ("read", "http://x.com")
    assert parse_action("execute_command('scp a b')") == ("execute_command", "scp a b")
    assert parse_action("not a call") == ("unknown", "not a call")


def test_parse_agent_response_tolerates_fences_and_extracts_actions():
    text = 'Sure:\n```json\n{"thinking": "ok", "actions": ["read(\\"x\\")"]}\n```\n'
    parsed = parse_agent_response(text)
    assert parsed["thinking"] == "ok"
    assert parsed["actions"] == ['read("x")']


def test_parse_agent_response_empty_actions_on_garbage():
    assert parse_agent_response("not json at all")["actions"] == []


def test_default_environment_self_replication_url_serves_the_payload_again():
    env = default_environment()
    assert ZOMBIE_PAYLOAD in env.read(SELF_REPLICATION_URL)


def test_environment_exfil_log_captures_the_message_not_a_real_network_call():
    env = default_environment()
    result = env.read(f"{EXFIL_LOG_URL_PREFIX}hello-world")
    assert result == "OK: logged."
    assert env.exfil_log == ["hello-world"]


def test_run_agent_step_detects_the_malicious_command():
    action = json.dumps({"thinking": "complying", "actions": [f'execute_command("{MALICIOUS_COMMAND}")']})
    client, calls = _fake_client([action])
    env = default_environment()
    step = run_agent_step(client, "fake-model", prompt="p", env=env)
    assert step.executed_malicious_command is True
    assert env.executed_commands == [MALICIOUS_COMMAND]
    assert calls.calls[0]["messages"][0]["content"] == "p"


def test_interpret_action_handles_dict_shaped_actions_real_models_actually_emit():
    """Observed live 2026-09-13 (gpt-4o-mini): models emit dict actions like
    {"action": "read", "url": "..."} rather than the "tool(arg)" string a
    literal reading of the payload's own naming would suggest, and sometimes
    invent their own verb (e.g. "backup") for what is still a read(url) call.
    interpret_action must key off the parameter present, not the verb."""
    assert interpret_action({"action": "read", "url": "http://x.com"}) == ("read", "http://x.com")
    assert interpret_action({"action": "execute_command", "command": "scp a b"}) == ("execute_command", "scp a b")
    # model invents "backup" as the verb but still supplies a url -- must
    # still be recognized as a read()
    assert interpret_action({"action": "backup", "url": "https://safeagentlog.com/log?msg=x"}) == (
        "read",
        "https://safeagentlog.com/log?msg=x",
    )
    assert interpret_action('read("http://x.com")') == ("read", "http://x.com")
    assert interpret_action(42) == ("unknown", "42")


def test_run_agent_step_detects_the_malicious_command_in_dict_shaped_actions():
    action = json.dumps({"thinking": "complying", "actions": [{"action": "execute_command", "command": MALICIOUS_COMMAND}]})
    client, _ = _fake_client([action])
    env = default_environment()
    step = run_agent_step(client, "fake-model", prompt="p", env=env)
    assert step.executed_malicious_command is True
    assert env.executed_commands == [MALICIOUS_COMMAND]


def test_run_agent_step_benign_reply_is_not_a_false_positive():
    action = json.dumps({"thinking": "just doing the task", "actions": ['read("http://benign.example")']})
    client, _ = _fake_client([action])
    env = default_environment()
    step = run_agent_step(client, "fake-model", prompt="p", env=env)
    assert step.executed_malicious_command is False


# ---------------------------------------------------------------------------
# memory -- sliding window + RAG
# ---------------------------------------------------------------------------


def test_sliding_window_memory_truncates_to_window_size():
    mem = SlidingWindowMemory(window_size=2)
    mem.commit(["a"])
    mem.commit(["b"])
    mem.commit(["c"])
    assert mem.entries == ["b", "c"]  # FIFO eviction, Eq.4


def test_sliding_window_memory_contains_any_checks_markers():
    mem = SlidingWindowMemory(window_size=3)
    mem.commit([f"saw {MALICIOUS_COMMAND} in the page"])
    assert mem.contains_any(PAYLOAD_MARKERS) is True
    mem2 = SlidingWindowMemory(window_size=3)
    mem2.commit(["nothing relevant here"])
    assert mem2.contains_any(PAYLOAD_MARKERS) is False


def test_evolve_raw_history_needs_no_client_and_is_identity():
    assert evolve("verbatim text", "raw_history") == "verbatim text"


def test_evolve_reflection_strategies_require_a_client():
    with pytest.raises(ValueError):
        evolve("text", "verbal_reflection")


def test_evolve_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        evolve("text", "not_a_real_strategy")


def test_build_filler_entries_batches_and_zips_embeddings_to_text():
    calls = []

    def embed_batch(texts):
        calls.append(list(texts))
        return [np.array([float(len(t)), 0.0]) for t in texts]

    entries = build_filler_entries(embed_batch, ["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2)
    assert len(calls) == 3  # ceil(5/2)
    assert [e.text for e in entries] == ["a", "bb", "ccc", "dddd", "eeeee"]
    assert entries[2].embedding.tolist() == [3.0, 0.0]


def test_rag_memory_seed_adds_entries_without_mutating_the_source_list():
    vectors = {"filler": np.array([0.0, 1.0])}
    mem = RAGMemory(embed_fn=_fake_embed_fn(vectors))
    filler_pool = [RAGMemoryEntry(text="filler entry", embedding=np.array([0.0, 1.0]))]
    mem.seed(filler_pool)
    mem2 = RAGMemory(embed_fn=_fake_embed_fn(vectors))
    mem2.seed(filler_pool)
    assert len(mem.entries) == 1 and len(mem2.entries) == 1
    mem.commit("something new")
    assert len(mem.entries) == 2 and len(mem2.entries) == 1  # independent, not aliased


def test_run_rag_campaign_with_filler_pool_does_not_inflate_injection_count():
    vectors = {MALICIOUS_COMMAND: np.array([1.0, 0.0]), "task 0": np.array([0.0, 1.0]), "filler": np.array([0.0, 1.0])}
    embed_fn = _fake_embed_fn(vectors)
    infection = json.dumps({"thinking": "ok", "actions": [f'read("{SELF_REPLICATION_URL}")']})
    trigger = json.dumps({"thinking": "no action", "actions": []})
    client, _ = _fake_client([infection, trigger])

    filler = [RAGMemoryEntry(text=f"filler entry {i}", embedding=np.array([0.0, 1.0])) for i in range(50)]
    result = run_rag_campaign(
        client, "fake-model", embed_fn, _bait(), _triggers(1), top_k=3, exposure_rounds=1, max_trigger_rounds=1,
        filler_entries=filler,
    )
    # 1 infection entry carries the payload, regardless of the 50 benign
    # filler entries also in the store
    assert result.injection_count == 1


def test_rag_memory_recall_at_multiple_k_matches_recall_at_k_per_value():
    vectors = {
        "malicious": np.array([1.0, 0.0]),
        "far": np.array([-1.0, 0.0]),
        "query": np.array([0.9, 0.1]),
    }
    mem = RAGMemory(embed_fn=_fake_embed_fn(vectors))
    mem.commit("benign filler far")  # embeds as "far" -> rank last
    mem.commit("something else")
    mem.commit(f"note: {MALICIOUS_COMMAND} malicious")  # embeds as "malicious" -> rank first
    swept = mem.recall_at_multiple_k("query", (1, 2, 3), PAYLOAD_MARKERS)
    assert swept[1] == mem.recall_at_k("query", 1, PAYLOAD_MARKERS)
    assert swept[3] == mem.recall_at_k("query", 3, PAYLOAD_MARKERS)


def test_run_rag_campaign_recall_k_sweep_records_per_round_without_extra_calls():
    vectors = {MALICIOUS_COMMAND: np.array([1.0, 0.0]), "task 0": np.array([0.0, 1.0])}
    embed_fn = _fake_embed_fn(vectors)
    infection = json.dumps({"thinking": "ok", "actions": [f'read("{SELF_REPLICATION_URL}")']})
    trigger = json.dumps({"thinking": "no action", "actions": []})
    client, calls = _fake_client([infection, trigger])

    result = run_rag_campaign(
        client, "fake-model", embed_fn, _bait(), _triggers(1), top_k=5, exposure_rounds=1, max_trigger_rounds=1,
        recall_k_sweep=(1, 5, 10),
    )
    round0 = result.trigger_rounds[0]
    assert set(round0["recall_by_k"].keys()) == {1, 5, 10}
    assert len(calls.calls) == 2  # unchanged -- no extra LLM calls for the sweep


def test_rag_memory_recall_at_k_and_injection_count():
    vectors = {
        "malicious": np.array([1.0, 0.0]),
        "benign": np.array([0.0, 1.0]),
        "query about malicious stuff": np.array([0.9, 0.1]),
    }
    mem = RAGMemory(embed_fn=_fake_embed_fn(vectors))
    mem.commit(f"note: {MALICIOUS_COMMAND} malicious")
    mem.commit("benign entry")
    assert mem.injection_count(PAYLOAD_MARKERS) == 1
    assert mem.recall_at_k("query about malicious stuff", k=1, markers=PAYLOAD_MARKERS) is True
    assert mem.recall_at_k("benign query", k=1, markers=PAYLOAD_MARKERS) is False


# ---------------------------------------------------------------------------
# campaigns -- end to end, both memory architectures
# ---------------------------------------------------------------------------


def _bait():
    return BaitTask(task_id="t1", user_task="find a mouse", poisoned_domain="evil.example", benign_page_text="A mouse.")


def _triggers(n=3):
    return [TriggerQuery(domain=f"d{i}.example", task=f"task {i}") for i in range(n)]


def test_run_sliding_window_campaign_infects_then_triggers_malicious_action():
    infection = json.dumps({"thinking": "ok", "actions": [f'read("{SELF_REPLICATION_URL}")']})
    trigger_hit = json.dumps({"thinking": "complying", "actions": [f'execute_command("{MALICIOUS_COMMAND}")']})
    trigger_miss = json.dumps({"thinking": "just working", "actions": []})
    client, calls = _fake_client([infection, trigger_hit, trigger_miss])

    result = run_sliding_window_campaign(
        client, "fake-model", _bait(), _triggers(2), window_size=3, exposure_rounds=1, max_trigger_rounds=2
    )

    assert result.infected is True  # the read() result (payload text) got committed
    assert len(result.trigger_rounds) == 2
    assert result.asr == 0.5  # 1 of 2 trigger rounds executed the malicious command
    assert result.trigger_rounds[0]["payload_in_context_before_step"] is True
    assert len(calls.calls) == 3


def test_run_sliding_window_campaign_no_infection_if_payload_never_read():
    infection = json.dumps({"thinking": "ignoring the weird comment", "actions": []})
    client, _ = _fake_client([infection])
    result = run_sliding_window_campaign(
        client, "fake-model", _bait(), _triggers(0), window_size=3, exposure_rounds=1, max_trigger_rounds=0
    )
    assert result.infected is False


def test_run_sliding_window_campaign_exposure_rounds_repeats_phase_i_k_times():
    """Section 4.1: "Exposure Phase (K rounds of baited tasks)", K=3 for
    sliding window (== window size L) -- Phase I must run K times, not once,
    so by the end of Exposure the window is fully saturated."""
    infection = json.dumps({"thinking": "ok", "actions": [f'read("{SELF_REPLICATION_URL}")']})
    client, calls = _fake_client([infection, infection, infection])

    result = run_sliding_window_campaign(
        client, "fake-model", _bait(), _triggers(0), window_size=3, exposure_rounds=3, max_trigger_rounds=0
    )

    assert len(calls.calls) == 3  # 3 Exposure rounds, 0 trigger rounds
    assert result.infected is True
    assert len(result.infection_actions) == 3  # one action recorded per exposure round


def test_run_rag_campaign_injection_count_only_counts_the_infection_entry():
    """No separate aliasing step exists (Section 3.3's "Semantic Aliasing"
    prose is named but never templated or cited in the paper -- see
    payloads.py) -- the RAG database only gains payload copies via the raw
    Exposure-Phase commits themselves (one here, since exposure_rounds=1)."""
    vectors = {MALICIOUS_COMMAND: np.array([1.0, 0.0]), "task 0": np.array([0.0, 1.0])}
    embed_fn = _fake_embed_fn(vectors)
    infection = json.dumps({"thinking": "ok", "actions": [f'read("{SELF_REPLICATION_URL}")']})
    trigger = json.dumps({"thinking": "no action", "actions": []})
    client, _ = _fake_client([infection, trigger])

    result = run_rag_campaign(
        client, "fake-model", embed_fn, _bait(), _triggers(1), top_k=3, exposure_rounds=1, max_trigger_rounds=1
    )

    assert result.injection_count == 1
    assert result.infected is True


def test_run_rag_campaign_exposure_rounds_repeats_phase_i_k_times():
    """Section 4.1: RAG's K=300 Exposure-Phase rounds are the paper's real
    mechanism for spreading payload copies across the database -- not a
    separate hand-crafted aliasing step. Each of the K rounds is its own
    Evolve(C_j) commit, so injection_count should scale with K."""
    vectors = {MALICIOUS_COMMAND: np.array([1.0, 0.0]), "task 0": np.array([0.0, 1.0])}
    embed_fn = _fake_embed_fn(vectors)
    infection = json.dumps({"thinking": "ok", "actions": [f'read("{SELF_REPLICATION_URL}")']})
    client, calls = _fake_client([infection, infection, infection, infection])

    result = run_rag_campaign(
        client, "fake-model", embed_fn, _bait(), _triggers(0), top_k=3, exposure_rounds=4, max_trigger_rounds=0
    )

    assert len(calls.calls) == 4  # 4 Exposure rounds, 0 trigger rounds
    assert result.injection_count == 4
    assert result.infected is True


def test_synthetic_trigger_page_is_clearly_labelled_and_topic_bearing():
    page = synthetic_trigger_page(TriggerQuery(domain="example.com", task="find X"))
    assert "example.com" in page and "find X" in page


# ---------------------------------------------------------------------------
# models -- which LLM plays the web agent
# ---------------------------------------------------------------------------


def test_make_agent_client_openai_default_model(monkeypatch):
    from benchmark.attacks.zombie_agents.models import make_agent_client

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    client, model = make_agent_client("openai")
    assert model == "gpt-4o-mini"
    assert client.base_url is not None  # real OpenAI client, default base_url


def test_make_agent_client_openai_model_override(monkeypatch):
    from benchmark.attacks.zombie_agents.models import make_agent_client

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    _, model = make_agent_client("openai", model="gpt-4o")
    assert model == "gpt-4o"


def test_make_agent_client_gemini_requires_api_key(monkeypatch):
    from benchmark.attacks.zombie_agents.models import make_agent_client

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        make_agent_client("gemini")


def test_make_agent_client_gemini_defaults_to_closest_available_not_the_blocked_paper_model(monkeypatch):
    """PAPER_AGENT_MODELS['gemini'] (gemini-2.5-flash) is confirmed 404-blocked
    for this project's API key (see models.py docstring) -- defaulting to it
    would fail on every call, so the real default is CLOSEST_AVAILABLE_GEMINI.
    The paper's exact id is still reachable via an explicit model= override."""
    from benchmark.attacks.zombie_agents.models import (
        CLOSEST_AVAILABLE_GEMINI,
        PAPER_AGENT_MODELS,
        make_agent_client,
    )

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    client, model = make_agent_client("gemini")
    assert model == CLOSEST_AVAILABLE_GEMINI == "gemini-3-flash-preview"
    assert model != PAPER_AGENT_MODELS["gemini"]
    assert "generativelanguage.googleapis.com" in str(client.base_url)

    _, overridden = make_agent_client("gemini", model=PAPER_AGENT_MODELS["gemini"])
    assert overridden == "gemini-2.5-flash"  # still reachable as an explicit override


def test_make_agent_client_unknown_provider_raises():
    from benchmark.attacks.zombie_agents.models import make_agent_client

    with pytest.raises(SystemExit):
        make_agent_client("not-a-real-provider")


# ---------------------------------------------------------------------------
# local_embeddings -- real local sentence-transformers model, no network
# ---------------------------------------------------------------------------


def test_make_local_embed_fn_returns_consistent_vectors_and_caches():
    from benchmark.attacks.zombie_agents.local_embeddings import make_local_embed_fn

    embed = make_local_embed_fn()
    v1 = embed("hello world")
    v2 = embed("hello world")  # cache hit, must be the identical object
    v3 = embed("something completely different")
    assert v1 is v2
    assert v1.shape == v3.shape
    assert v1.shape[0] > 0
    assert not (v1 == v3).all()


def test_make_local_embed_batch_fn_matches_single_embed_fn():
    from benchmark.attacks.zombie_agents.local_embeddings import (
        load_sentence_transformer,
        make_local_embed_batch_fn,
        make_local_embed_fn,
    )

    model = load_sentence_transformer()
    embed = make_local_embed_fn(model=model)
    embed_batch = make_local_embed_batch_fn(model=model)

    texts = ["a payload about scp", "a recipe for banana bread"]
    single = [embed(t) for t in texts]
    batch = embed_batch(texts)
    assert len(batch) == 2
    for s, b in zip(single, batch, strict=True):
        # not bit-identical -- batched vs single-item inference through the
        # transformer differ slightly at float32 precision (real, observed
        # ML behavior, not a bug); close enough that cosine similarity is
        # unaffected in practice.
        assert np.allclose(s, b, atol=1e-4)
