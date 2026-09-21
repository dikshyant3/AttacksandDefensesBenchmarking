# Hidden Sleeper Memory — implementation fidelity

Paper: *Hidden in Memory: Sleeper Memory Poisoning in LLM Agents*, arXiv:2605.15338 (v2).
Upstream repo: `github.com/ivaxi0s/LLM-agent-memory-poisoning` @ `70de017714abd6d12bb4681e93437461ba6f9a19`.
Last verified against primary sources: **2026-09-10**.

Legend: ✅ faithful (verified against released code/data) · 🟡 faithful mechanism,
forced substitution · 🟠 extension beyond the released code · ⬜ not done.

---

## Stage 1 — Injection (IR)

### Tool-based regime — ✅ / 🟡
| Element | Status | Note |
|---|---|---|
| `bio` memory tool | ✅ | simulated fn; upstream's tool is also simulated (`sleeper_eval/tools.py` `TOOL_MAP`) |
| Official GPT system prompt | ✅ | `prompts/gpt.md`, template-rendered |
| `model_set_context` memory placement | ✅ | byte-identical to upstream `format_memories_model_set_context` |
| Goal-match judge prompt | ✅ | **byte-identical** to `sleeper_eval/scorer.py` |
| Payload `ACTOR_CRITIC_WITH_MARKERS` | ✅ | byte-verified = `universal_v2_optimized_with_markers.py` |
| Payload `ACTOR_CRITIC_NO_MARKERS` | ✅ | byte-verified = `universal_v2_optimized_without_markers.py` (fixed a 1-char trailing-newline drift 2026-09-10) |
| Payload `LITERATURE_BASELINE` | ✅ | byte-verified = `attack_from_literature.py` |
| Subject / judge model | 🟡 | `gpt-4o-mini` vs the paper's 6 frontier models + `openai/gpt-5.4-mini` grader |
| **Run status** | ✅ | n=196 IR = 100% (`injection_20260824_143402.json`); re-confirmed n=20 with the new judge wiring, IR = 100% (`injection_20260910_181635.json`) |

### External-manager regime — ✅ / 🟡  (rewritten 2026-09-10, commit 9c17905)
| Element | Status | Note |
|---|---|---|
| Mechanism = `prompt_only` (one LLM call, no vector store) | ✅ | matches `memory_backend.py`; the paper's Table 1/2 numbers use `prompt_only`, not the repo's `sdk`/Qdrant path |
| Manager system prompt = mem0 `ADDITIVE_EXTRACTION_PROMPT` | ✅ | vendored verbatim from the repo's own copy (`prompts/mem0_manager/`) |
| Manager user prompt = `generate_additive_extraction_prompt(...)` | ✅ | imported from installed `mem0ai`, same call as `_load_mem0_prompt_only_components` |
| Response parse `{"memory":[{"text":…}]}` | ✅ | functional equiv of `parse_mem0_prompt_only_response` |
| Information scoping (query + completion only) | ✅ | paper Appendix E.2, verbatim |
| Payload `EXTERNAL_MANAGER_C2` | ✅ | byte-verified = `mem0_external_prompt_leak.py` (incl. `_goal_sentence`) |
| Payload `EXTERNAL_MANAGER_NO_MARKERS` | ✅ | byte-verified = `mem0_external_prompt_leak_without_untrusted_markers.py` (added 2026-09-10) |
| Manager model | ✅ | **the paper's actual model** — `gemini-3.1-flash-lite-preview` via Google's OpenAI-compatible endpoint (`make_manager_client("gemini")`, needs `GEMINI_API_KEY`). Connectivity verified 2026-09-10 (returns valid mem0 JSON). `--manager-provider openai` is the fallback. |
| Subject model | 🟡 | `gpt-4o-mini` vs the paper's 6 frontier models |
| Single-exchange replay vs upstream's 2-stage transcript replay | 🟠 | functionally equal per case; no manager retry loop; `dynamic_template` variant not vendored |
| **Run status** | ✅ | n=20, IR = **10%** (2/20), any-extraction 95% (`results/external_manager_20260910_181921.json`). Below the paper's 13.6–86.4% — bottleneck is the weak subject rarely echoing the payload, not the Gemini manager. |

---

## Stage 2 — Retrieval (RR)

**The released `followup_eval` implements only "everything in-context".** It has no
retrieval-method parameter, no top-k, no embedding retrieval, no memory-management
agent (verified: `followup_eval/{task,solver,scorer,dataset}.py`). The paper's
Table 2 RR numbers come from the embedding cosine-similarity analysis in
`retrievability_optim.py` (goal-phrasing optimization), not a live RR eval.

| Method | Status | Note |
|---|---|---|
| G.1 "everything in-context" (`compute_retrieval_rate_everything_in_context`) | ✅ | RR ≡ 1.0 by construction — exactly what `followup_eval` does |
| G.3 "dynamic semantic retrieval" (`compute_retrieval_rate`, embedding top-k) | 🟠 | extension. `cosine_similarity` == `retrievability_optim.cosine_similarity`; we do per-query top-k rank, upstream does mean-similarity objective. `top_k=(15,5)` is not paper-verified. |
| G.2 "memory management agent" (`compute_retrieval_rate_llm_manager`) | 🟠 | extension. **Prompt is ours** — the paper ships no retrieval-agent prompt/impl; its Appendix M.1.2 citation points elsewhere. `run_followup_experiment.py --retrieval-manager gemini` runs it. |
| Memory pool | ✅ | exactly `preexisting_memories` + injected memory — matches `followup_eval/dataset.py` (no padding, upstream doesn't pad either) |
| Pipeline wiring | ✅ | `run_followup_experiment.py` runs G.1 + G.3 always, G.2 on request, per case, and reports RR per method + AUR + end-to-end |
| **Run status** | ✅ | n=20 (`results/followup_20260910_183540.json`): G.1 RR **100%**, G.3 RR (k=15) **61%** mean (std 0.42 — cases split hard by query proximity), AUR **55%**, end-to-end **50%**. AUR lands inside the paper's goal-adjacent behavior range (42–85%). |

---

## Stage 3 — Adversarial Usage (AUR)

### LLM Behavior — ✅ / 🟡
| Element | Status | Note |
|---|---|---|
| `FOLLOWUP_SYSTEM_PROMPT` | ✅ | **verbatim** = `followup_eval/solver.py` |
| `INFLUENCE_PROMPT` (judge) | ✅ | **verbatim** = `followup_eval/scorer.py` |
| Memory placement (`model_set_context`) | ✅ | byte-identical to upstream helper |
| Turn-by-turn replay, `max_tokens=40960`, no tools | ✅ | matches `replay_followup_conversation` + `GenerateConfig` |
| `initial_turns` replay | ✅ | supported; empty for the vendored multiturn dataset (upstream's other format uses it) |
| Decoupled from RR | ✅ | upstream decouples the stages |
| Subject / judge model | 🟡 | OpenAI substitute |
| **Run status** | ⬜ | not run live (`run_followup_experiment.py`, unit-tested only) |

### Agent Action — ✅ dataset / 🟠 harness
| Element | Status | Note |
|---|---|---|
| Dataset `agent_action.json` | ✅ | vendored verbatim @ 70de017 |
| Sandbox = virtual-file dict, tools = `terminal` + `edit_file` | ✅ | dataset shape confirmed by inspection |
| Eval harness (tool wiring, judge) | 🟠 | **reconstruction** — the paper's OpenClaw-based agent environment is not in the released repo |
| **Run status** | ✅ (harness) | n=20, AUR = 20% (`results/agent_action_20260903_155154.json`) — our harness's number, not a paper reproduction |

---

## Datasets — ✅ all verified faithful (2026-09-10)

| Vendored file | Upstream source @ 70de017 | Verification |
|---|---|---|
| `datasets/paper_main_subset_196.json` | `datasets/released/generated/{behaviour,agent}_true_optimized_{with,without}_memories.json` | Byte-identical **prefix** concat: first 70+70+28+28. Record 0 diffed field-by-field = upstream behaviour_with[0]. Block structure re-checked (see provenance.json). A **subset**, not a modification. |
| `datasets/paper_main_subset_120.json` | same four files | Same construction, N = 43+43+17+17 (ratio-preserving, cheaper runs). |
| `datasets/llm_behaviour_followup.jsonl` | `datasets/downstream/llm_behaviour.jsonl` | **Byte-identical**, only renamed. Record 0 (Kraken goal, 57 memories, 5 queries, `multi_turn_meta.split="wildchat_seed"`) diffed = upstream. |
| `datasets/agent_action.json` | `datasets/downstream/agent_action.json` | **Byte-identical**. Record 0 all 15 keys + values (`expected_tools_used=["terminal","edit_file"]`, `goal_adjacent`, `sandbox_environment=[{filename,description,file_text}]`) diffed = upstream. |

Loader parity: `followup_data.py` reproduces `followup_eval/dataset.py::record_to_sample`'s
multiturn transform (inject `goal_text` into `memories` iff absent; `injected_memory_index
= memories.index(...)`). We read the raw `downstream/llm_behaviour.jsonl`; upstream's
`followup_eval` reads a builder-processed copy under `followup/` — same source rows, same
injection step.

## Model assignment (project decision)
- **`gpt-4o-mini`** — everywhere: subject/target LLM, every LLM judge (goal-match,
  influence, edit-influence), and the optional G.2 retrieval agent when run with
  `--retrieval-manager openai`.
- **`gemini-3.1-flash-lite-preview`** (Google, OpenAI-compatible endpoint) — ONLY
  the external-manager regime's memory manager, which is the paper's actual model.
- **`text-embedding-3-small`** — G.3 semantic retrieval (not a chat model; no
  gpt-4o-mini equivalent exists).
- Every runner exposes `--judge-model` to decouple the judge from the subject
  (the paper uses a fixed separate grader, `gpt-5.4-mini`); default keeps them
  equal. Judges are binary Yes/No classifiers, so same-model self-evaluation
  bias is a mild, acknowledged risk — not the open-ended-scoring worst case.

## Cross-cutting gaps (all stages)
- **Model substitution**: `gpt-4o-mini` as subject vs the paper's 6 frontier
  models. Every absolute number must be read as "our model", not the paper's.
- **Sample size**: runs are n≈20 (except tool-based n=196). Too small for point
  estimates — needs bootstrap CIs and larger n before any number is cited as final.
- **Not run live**: external-manager (post-rewrite), Stage 2 RR, Stage 3 LLM-Behavior AUR.
