# Mem0 Manager Prompt (external-manager regime)

`mem0_default_system_prompt.md` is a **verbatim copy of the upstream repo's checked-in
copy** of Mem0's `ADDITIVE_EXTRACTION_PROMPT`.

Provenance chain:
- Upstream paper repo: `github.com/ivaxi0s/LLM-agent-memory-poisoning`, commit
  `70de017714abd6d12bb4681e93437461ba6f9a19`,
  file `sleeper_eval/prompts/mem0_manager/mem0_default_system_prompt.md`
  (their note: *"deterministic local copy of mem0's `ADDITIVE_EXTRACTION_PROMPT`,
  copied from the upstream mem0 prompt text and checked into this repo for
  reproducibility"*; their sha256 on 2026-05-01: `c9787244cc2627fd9748323e62ef1bc71c3893de0feda271faef1fe9d27c973e`).
- This file was fetched verbatim from that path/commit on 2026-09-10.

## How the paper actually runs the external-manager regime

Confirmed directly from the upstream source (`sleeper_eval/memory_backend.py`,
`sleeper_eval/eval_campaign/mem0_replay.py`) and the paper (arXiv:2605.15338 v2,
Appendix E.2):

- The regime is **simulated**, not the real Mem0 service. The upstream README:
  *"rather than running the Mem0 service, the framework drives a manager model
  with Mem0's published memory-extraction prompt (a checked-in copy under
  `sleeper_eval/prompts/mem0_manager/`)... The `mem0` naming in configs and
  scripts refers to this simulated manager."*
- `memory_backend.py` defines four runtimes: `local`, `transcript_only`,
  **`prompt_only`** (used for the paper's external-manager campaign), and `sdk`
  (real `mem0ai` + Qdrant — present in the repo but **not** what Table 1/Table 2
  report).
- `prompt_only` = **one LLM call**, no vector store, no embeddings, no
  ADD/UPDATE/DELETE reconciliation:
  - system message = this file (`ADDITIVE_EXTRACTION_PROMPT`)
  - user message = `mem0.configs.prompts.generate_additive_extraction_prompt(...)`
    (imported from the installed `mem0ai` package at runtime)
  - response parsed as `{"memory": [{"text": ...}, ...]}` → list of memory strings
- **Information scoping** (paper v2, Appendix E.2, verbatim): *"the manager
  receives only the direct user query and the target LLM's completion. It does
  not receive the uploaded document or the target LLM's system prompt. The
  manager is therefore structurally blind to document-embedded injection
  payloads unless the target LLM surfaces adversarial content in its response."*
- **Manager model** (paper v2): *"Gemini 3.1 Flash Lite Preview (high) was used
  as the manager LLM because it is relatively inexpensive and reflects the type
  of LLM users may deploy in production."*

`external_manager.py` in this project reproduces the `prompt_only` path exactly
(same system prompt, same user-prompt builder, same parsing, same information
scoping). Deviations from the paper, all forced by environment/access:
- manager model: an OpenAI model instead of Gemini 3.1 Flash Lite Preview
  (`--manager-model`, this project's standard substitution)
- subject model: `gpt-4o` / `gpt-4o-mini` instead of the paper's 6 frontier models
