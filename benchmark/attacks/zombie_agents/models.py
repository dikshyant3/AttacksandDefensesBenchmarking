"""Which LLM plays the web agent.

Paper (Section 4.1): "We evaluate our attack on two state-of-the-art
commercial LLMs optimized for tool use and long-context understanding:
Gemini-2.5-Flash and GLM-4.7-Flash." Both runners default to `gpt-4o-mini`
(this project's standard substitute, and what every smoke/diagnostic run so
far used) but can be pointed at the real Gemini-2.5-Flash the same way
Hidden Sleeper's external_manager.py reaches Gemini: the `openai` SDK against
Google's OpenAI-compatible endpoint, no new dependency. GLM-4.7-Flash (Zhipu
AI) has no configured access in this environment.
"""

from __future__ import annotations

import os

PAPER_AGENT_MODELS = {
    # The paper's real model id -- but a live call (2026-09-13) returned:
    # "Error code: 404 ... This model models/gemini-2.5-flash is no longer
    # available to new users. ... use models/gemini-3.6-flash." So this
    # exact model is NOT reachable with this project's API access, even
    # though it's genuinely what the paper used. Pass --model to substitute
    # a currently-available Gemini model -- doing so is a further
    # substitution, not the paper's model.
    "gemini": "gemini-2.5-flash",
    # "glm": "glm-4.7-flash",       # paper's other model -- no configured access here
}

# Closest currently-reachable substitute for gemini-2.5-flash (verified live,
# 2026-09-14): same tier ("Flash", not "Flash-Lite" -- matching the paper's
# actual tier) and the closest available generation number above the blocked
# 2.5 (gemini-2.5-flash and gemini-2.5-flash-lite both 404; gemini-3-flash-preview
# is the next reachable Flash-tier model up). This is still NOT the paper's
# model -- no claim of matching "vulnerability"/compliance characteristics is
# possible without the actual blocked model to compare against.
CLOSEST_AVAILABLE_GEMINI = "gemini-3-flash-preview"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_agent_client(provider: str = "openai", model: str | None = None) -> tuple[object, str]:
    """Build (client, model_id) for whichever LLM plays the web agent.

    provider="openai" -> OpenAI, default gpt-4o-mini (this project's standard
      substitute for the paper's models).
    provider="gemini" -> Gemini via Google's OpenAI-compatible endpoint (needs
      GEMINI_API_KEY). Defaults to CLOSEST_AVAILABLE_GEMINI
      (gemini-3-flash-preview) -- NOT PAPER_AGENT_MODELS["gemini"]
      (gemini-2.5-flash), which is confirmed 404-blocked for this API key
      (see module docstring) and would fail on every call if left as the
      default. Pass model="gemini-2.5-flash" explicitly if your own key
      happens to still have access to it. Same client also works for the
      memory-evolution LLM calls (verbal_reflection / refined_experience),
      so one client suffices for a whole run.
    `model` overrides the default id.
    """
    from openai import OpenAI

    if provider == "openai":
        return OpenAI(), model or "gpt-4o-mini"
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is required for --provider gemini")
        return OpenAI(base_url=GEMINI_OPENAI_BASE_URL, api_key=key), model or CLOSEST_AVAILABLE_GEMINI
    raise SystemExit(f"unknown provider: {provider!r} (choices: openai, gemini)")
