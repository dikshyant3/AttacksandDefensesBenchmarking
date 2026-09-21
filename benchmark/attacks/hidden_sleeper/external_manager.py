"""External-manager regime for Hidden Sleeper Memory.

In this regime the target LLM does NOT write memories itself; a separate
memory-manager LLM observes the interaction afterward and decides what to
persist -- the pattern used by Mem0.

FAITHFULNESS NOTE (verified 2026-09-10 against the upstream source and the
paper, correcting an earlier version of this file):

The paper's external-manager results are produced by a *simulated* manager,
NOT the real ``mem0ai`` service. Upstream README, verbatim: "rather than
running the Mem0 service, the framework drives a manager model with Mem0's
published memory-extraction prompt (a checked-in copy under
``sleeper_eval/prompts/mem0_manager/``)... The ``mem0`` naming in configs and
scripts refers to this simulated manager."

``sleeper_eval/memory_backend.py`` defines four runtimes -- ``local``,
``transcript_only``, ``prompt_only``, ``sdk``. The main external-manager
campaign (``sleeper_eval/eval_campaign/mem0_replay.py``) targets
``prompt_only``. ``sdk`` (real ``mem0ai`` + Qdrant) exists in the repo but is
not what Table 1 / Table 2 report.

``prompt_only`` is a SINGLE LLM call -- no vector store, no embeddings, no
ADD/UPDATE/DELETE reconciliation:
  * system message  = mem0's ``ADDITIVE_EXTRACTION_PROMPT``
    (vendored verbatim at prompts/mem0_manager/mem0_default_system_prompt.md,
    copied from the upstream repo's own checked-in copy)
  * user message    = ``mem0.configs.prompts.generate_additive_extraction_prompt(...)``
    (imported from the installed ``mem0ai`` package at runtime, exactly as
    upstream's ``_load_mem0_prompt_only_components`` does)
  * response parsed as ``{"memory": [{"text": ...}, ...]}`` -> list of strings

Information scoping (paper v2, Appendix E.2, verbatim): "the manager receives
only the direct user query and the target LLM's completion. It does not
receive the uploaded document or the target LLM's system prompt. The manager
is therefore structurally blind to document-embedded injection payloads
unless the target LLM surfaces adversarial content in its response." That is
the default here (``include_document_content=False``).

This module reproduces ``prompt_only`` exactly. Deviations:
  * manager model: the paper uses "Gemini 3.1 Flash Lite Preview (high)".
    make_manager_client("gemini") wires the real model
    (gemini-3.1-flash-lite-preview) via Google's OpenAI-compatible endpoint
    when GEMINI_API_KEY is set; an OpenAI model is the fallback.
  * subject model: gpt-4o / gpt-4o-mini instead of the paper's 6 frontier models
"""

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from benchmark.attacks.hidden_sleeper.data import SleeperCase
from benchmark.attacks.hidden_sleeper.openai_tool_pipeline import (
    build_user_message,
    load_official_gpt_prompt,
)
from benchmark.attacks.hidden_sleeper.payloads import PayloadVariant, build_adversarial_document

MEM0_MANAGER_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "mem0_manager" / "mem0_default_system_prompt.md"
)

# The paper's real manager model (Appendix E.2: "Gemini 3.1 Flash Lite Preview
# (high)"). "gemini-3.1-flash-lite-preview" is the model id served by Google's
# API; "(high)" is a reasoning-effort setting.
PAPER_MANAGER_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_manager_client(provider: str = "gemini"):
    """Build the client the simulated memory manager's single extraction call
    goes through.

      provider="gemini" -> the paper's actual manager: Gemini via Google's
        OpenAI-compatible endpoint (needs GEMINI_API_KEY). Zero new deps -- it
        is the same `openai` SDK pointed at a different base_url.
      provider="openai" -> an OpenAI model (fallback / ablation).

    Returns (client, default_model)."""
    from openai import OpenAI

    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is required for --manager-provider gemini")
        return OpenAI(base_url=GEMINI_OPENAI_BASE_URL, api_key=key), PAPER_MANAGER_MODEL
    if provider == "openai":
        return OpenAI(), "gpt-4o-mini"
    raise SystemExit(f"unknown manager provider: {provider!r}")


def load_mem0_manager_system_prompt() -> str:
    """mem0's ADDITIVE_EXTRACTION_PROMPT, vendored verbatim from the upstream
    repo's checked-in copy (see prompts/mem0_manager/README.md)."""
    return MEM0_MANAGER_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _load_mem0_user_prompt_builder():
    """Import mem0's own additive-extraction user-prompt builder, exactly as
    upstream's ``_load_mem0_prompt_only_components`` does (it is NOT vendored --
    upstream loads it from the installed ``mem0ai`` package at runtime)."""
    from mem0.configs.prompts import generate_additive_extraction_prompt

    return generate_additive_extraction_prompt


def build_manager_messages(
    *,
    user_query: str,
    assistant_reply: str,
    seeded_memories: tuple[str, ...],
    observation_date: str | None = None,
) -> list[dict]:
    """Construct the two-message exchange the simulated manager receives,
    matching ``prepare_mem0_prompt_only_request`` in upstream's
    ``memory_backend.py``:

      system = ADDITIVE_EXTRACTION_PROMPT
      user   = generate_additive_extraction_prompt(
                 summary="", recently_extracted_memories=[],
                 existing_memories=<seeded>, new_messages=<query+reply>,
                 last_k_messages=[])
    """
    builder = _load_mem0_user_prompt_builder()
    existing_memories = [{"id": str(i), "text": m} for i, m in enumerate(seeded_memories)]
    new_messages = [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_reply},
    ]
    kwargs = dict(
        summary="",
        recently_extracted_memories=[],
        existing_memories=existing_memories,
        new_messages=new_messages,
        last_k_messages=[],
    )
    if observation_date is not None:
        # generate_additive_extraction_prompt accepts current_date / timestamp;
        # pin both so the prompt is deterministic in tests.
        kwargs["current_date"] = observation_date
        kwargs["timestamp"] = observation_date
    user_prompt = builder(**kwargs)
    return [
        {"role": "system", "content": load_mem0_manager_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_candidate(text: str) -> str | None:
    """Pull the first plausible JSON object out of a raw model response --
    tolerates ```json fences and <think> blocks (functional equivalent of
    upstream's ``_extract_json_candidate`` / ``_normalize_mem0_json_response``)."""
    if not text:
        return None
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    brace = stripped.find("{")
    if brace == -1:
        return None
    depth = 0
    for i in range(brace, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[brace : i + 1]
    return None


def parse_manager_response(text: str) -> tuple[str, ...]:
    """Parse the manager's JSON into the list of extracted memory strings,
    matching upstream's ``parse_mem0_prompt_only_response``: read
    ``{"memory": [{"text": ...}, ...]}`` and take ``text`` / ``memory`` /
    ``content`` from each item."""
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return ()
    parsed = None
    for attempt in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
        try:
            parsed = json.loads(attempt, strict=False)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return ()
    if not isinstance(parsed, dict):
        return ()
    items = parsed.get("memory", [])
    if not isinstance(items, list):
        return ()
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("text") or item.get("memory") or item.get("content")
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return tuple(out)


@dataclass(frozen=True)
class ExternalManagerResult:
    write_attempted: bool
    extracted_memories: tuple[str, ...]
    target_memory_written: bool
    assistant_reply: str
    manager_model: str
    include_document_content: bool


def run_external_manager_case(
    client,
    model: str,  # SUBJECT model
    case: SleeperCase,
    goal_matcher,
    *,
    manager_model: str | None = None,
    manager_client=None,
    variant: PayloadVariant = PayloadVariant.EXTERNAL_MANAGER_C2,
    include_document_content: bool = False,
    observation_date: str | None = None,
    user_id: str | None = None,  # accepted for call-site compatibility; unused
) -> ExternalManagerResult:
    """Stage 1, external-manager regime, reproducing upstream's ``prompt_only``
    backend + Appendix E.2 information scoping.

    1. Subject model has a normal conversation with the ADVERSARIAL document
       (real regime-specific payload, EXTERNAL_MANAGER_C2 by default -- NOT a
       tool-induction payload; there is no ``bio`` tool in this regime).
    2. The simulated manager (one LLM call, mem0's ADDITIVE_EXTRACTION_PROMPT +
       generate_additive_extraction_prompt) sees ONLY the bare user query and
       the subject's reply -- never the document or the subject's system prompt
       (unless ``include_document_content=True``, an explicit opt-out of the
       paper's scoping).
    3. ``target_memory_written`` applies Stage 1's real semantic goal-match
       judge to whatever the manager extracted.
    """
    manager_model = manager_model or model
    manager_client = manager_client or client
    observation_date = observation_date or date.today().isoformat()
    _ = user_id or f"hidden-sleeper-{uuid.uuid4().hex[:12]}"

    adversarial_document = build_adversarial_document(
        case.benign_document, case.adversarial_memory_goal, variant
    )
    subject_user_message = build_user_message(case.document_task, adversarial_document)
    subject_messages = [
        {"role": "system", "content": load_official_gpt_prompt(model)},
        {"role": "user", "content": subject_user_message},
    ]
    response = client.chat.completions.create(model=model, temperature=0, messages=subject_messages)
    assistant_reply = (response.choices[0].message.content or "").strip()

    # Information scoping (paper v2, Appendix E.2): the manager sees the bare
    # query + the subject's completion only. include_document_content=True
    # instead feeds the full uploaded document (upstream's non-scoped default,
    # DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT=True) -- kept as an explicit knob.
    manager_user_query = subject_user_message if include_document_content else case.document_task
    manager_messages = build_manager_messages(
        user_query=manager_user_query,
        assistant_reply=assistant_reply,
        seeded_memories=case.preexisting_memories,
        observation_date=observation_date,
    )
    manager_response = manager_client.chat.completions.create(
        model=manager_model, temperature=0, messages=manager_messages
    )
    extracted = parse_manager_response(manager_response.choices[0].message.content or "")

    matched = any(goal_matcher(case.adversarial_memory_goal, text) for text in extracted)
    return ExternalManagerResult(
        write_attempted=True,
        extracted_memories=extracted,
        target_memory_written=matched,
        assistant_reply=assistant_reply,
        manager_model=manager_model,
        include_document_content=include_document_content,
    )
