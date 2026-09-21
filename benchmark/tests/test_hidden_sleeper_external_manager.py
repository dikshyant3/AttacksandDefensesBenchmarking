"""Deterministic tests for the external-manager regime, reproducing upstream's
``prompt_only`` backend (a single simulated-manager LLM call with Mem0's
ADDITIVE_EXTRACTION_PROMPT -- no real mem0ai service, no vector store) plus the
paper's Appendix E.2 information scoping. No network calls -- scripted fake
OpenAI clients throughout."""

from __future__ import annotations

from types import SimpleNamespace

from benchmark.attacks.hidden_sleeper.data import load_paper_main_subset
from benchmark.attacks.hidden_sleeper.external_manager import (
    build_manager_messages,
    load_mem0_manager_system_prompt,
    parse_manager_response,
    run_external_manager_case,
)
from benchmark.attacks.hidden_sleeper.payloads import PayloadVariant


class _FakeCompletions:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.outputs.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_client(outputs: list[str]):
    completions = _FakeCompletions(outputs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


# ---------------------------------------------------------------------------
# vendored prompt + manager-message construction
# ---------------------------------------------------------------------------


def test_vendored_manager_system_prompt_is_mem0_additive_extraction_prompt():
    prompt = load_mem0_manager_system_prompt()
    # Structural markers unique to mem0's ADDITIVE_EXTRACTION_PROMPT.
    assert prompt.startswith("# ROLE")
    assert "You are a Memory Extractor" in prompt
    assert "Your sole operation is ADD" in prompt
    assert '{"memory": []}' in prompt


def test_build_manager_messages_matches_mem0_prompt_only_shape():
    messages = build_manager_messages(
        user_query="Compare the two stories.",
        assistant_reply="Understood. Here is the comparison.",
        seeded_memories=("User likes tea.", "User has a dog."),
        observation_date="2026-09-10",
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == load_mem0_manager_system_prompt()

    user_prompt = messages[1]["content"]
    # seeded memories go in as mem0's Existing Memories list, id-indexed
    assert '{"id": "0", "text": "User likes tea."}' in user_prompt
    assert '{"id": "1", "text": "User has a dog."}' in user_prompt
    # query + reply go in as New Messages
    assert '"role": "user", "content": "Compare the two stories."' in user_prompt
    assert '"role": "assistant", "content": "Understood. Here is the comparison."' in user_prompt
    assert "Recently Extracted Memories\n[]" in user_prompt
    assert "Observation Date\n2026-09-10" in user_prompt


# ---------------------------------------------------------------------------
# response parsing (functional equivalent of upstream parse_mem0_prompt_only_response)
# ---------------------------------------------------------------------------


def test_parse_manager_response_reads_mem0_memory_array():
    text = '{"memory": [{"id": "0", "text": "User is anti-nuclear"}, {"id": "1", "text": "User likes tea"}]}'
    assert parse_manager_response(text) == ("User is anti-nuclear", "User likes tea")


def test_parse_manager_response_tolerates_fences_think_blocks_and_trailing_commas():
    fenced = "Sure:\n```json\n{\"memory\": [{\"text\": \"A\"},]}\n```\n"
    assert parse_manager_response(fenced) == ("A",)
    thinking = '<think>let me consider</think>\n{"memory": [{"text": "B"}]}'
    assert parse_manager_response(thinking) == ("B",)


def test_parse_manager_response_empty_and_unparseable():
    assert parse_manager_response('{"memory": []}') == ()
    assert parse_manager_response("no json at all") == ()
    assert parse_manager_response("") == ()


# ---------------------------------------------------------------------------
# end-to-end case: information scoping + goal match
# ---------------------------------------------------------------------------


def _case():
    return load_paper_main_subset()[0]


def test_run_case_scopes_the_manager_away_from_the_document_by_default():
    case = _case()
    client, completions = _fake_client(
        [
            "Understood, you're anti-nuclear on safety grounds. Here is the comparison...",
            '{"memory": [{"text": "User is anti-nuclear on safety grounds"}]}',
        ]
    )
    result = run_external_manager_case(
        client, "gpt-4o-mini", case, goal_matcher=lambda g, m: "anti-nuclear" in m.lower()
    )

    subject_user_msg = completions.calls[0]["messages"][1]["content"]
    manager_user_msg = completions.calls[1]["messages"][1]["content"]
    # subject is shown the adversarial document; the manager is not
    assert "</document>" in subject_user_msg
    assert "</document>" not in manager_user_msg
    assert case.benign_document[:80] not in manager_user_msg
    # the manager only gets the bare task query + the subject's reply
    assert case.document_task in manager_user_msg
    assert result.include_document_content is False
    assert result.target_memory_written is True
    assert result.extracted_memories == ("User is anti-nuclear on safety grounds",)


def test_include_document_content_opts_out_of_scoping():
    case = _case()
    client, completions = _fake_client(
        ["A reply that never mentions the goal.", '{"memory": []}']
    )
    run_external_manager_case(
        client,
        "gpt-4o-mini",
        case,
        goal_matcher=lambda g, m: True,
        include_document_content=True,
    )
    manager_user_msg = completions.calls[1]["messages"][1]["content"]
    assert "</document>" in manager_user_msg  # full document now reaches the manager


def test_manager_can_use_a_separate_model_and_client():
    case = _case()
    subject_client, subject_comp = _fake_client(["Subject reply surfacing nothing."])
    manager_client, manager_comp = _fake_client(['{"memory": [{"text": "User likes tea"}]}'])

    result = run_external_manager_case(
        subject_client,
        "gpt-4o",
        case,
        goal_matcher=lambda g, m: False,
        manager_model="gpt-4o-mini",
        manager_client=manager_client,
    )
    assert subject_comp.calls[0]["model"] == "gpt-4o"
    assert manager_comp.calls[0]["model"] == "gpt-4o-mini"
    assert result.manager_model == "gpt-4o-mini"
    assert result.target_memory_written is False


def test_no_goal_match_when_manager_extracts_nothing_relevant():
    case = _case()
    client, _ = _fake_client(
        ["Here is a neutral comparison of the two accounts.", '{"memory": [{"text": "User asked for a comparison"}]}']
    )
    result = run_external_manager_case(
        client, "gpt-4o-mini", case, goal_matcher=lambda g, m: "anti-nuclear" in m.lower()
    )
    assert result.extracted_memories == ("User asked for a comparison",)
    assert result.target_memory_written is False


def test_default_payload_variant_is_the_regime_specific_one():
    case = _case()
    client, completions = _fake_client(["reply", '{"memory": []}'])
    run_external_manager_case(client, "gpt-4o-mini", case, goal_matcher=lambda g, m: False)
    subject_user_msg = completions.calls[0]["messages"][1]["content"]
    # EXTERNAL_MANAGER_C2 fakes closing the untrusted-document context then
    # states the goal as a first-person aside
    assert "</end_untrusted_document>" in subject_user_msg
    assert case.adversarial_memory_goal in subject_user_msg
    assert PayloadVariant.EXTERNAL_MANAGER_C2.value == "external_manager_c2"
