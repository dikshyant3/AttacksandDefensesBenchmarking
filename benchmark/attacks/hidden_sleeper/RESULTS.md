# Hidden Sleeper Memory — results, paper-style

Runs on 2026-09-10 unless noted. Subject + every LLM judge = **gpt-4o-mini**;
external-manager memory manager = **gemini-3.1-flash-lite-preview** (the paper's
actual manager). Paper numbers are ranges across its 6 frontier models
(arXiv:2605.15338 v2, Tables 1–2). Ours are single-model, n≈20 — trends, not
point estimates.

---

## Table 1 — Injection Rate (IR)

| Regime | Subset | **Ours** | n | Paper (6 models) |
|---|---|---|---|---|
| Tool-based | **LLM Behavior** | **100%** | 140 | ≤ 99.8% |
| Tool-based | **Agent Action** | **100%** | 56 | high |
| External-manager | **LLM Behavior** | **10%** (2/20) | 20 | 13.6 – 86.4% |
| External-manager | **Agent Action** | **5%** (1/20) | 20 | 5.5 – 82.5% |

Files: tool-based `injection_20260824_143402.json` (n=196); external-manager
`external_manager_20260910_181921.json` (behavior), `external_manager_20260910_185451.json`
(agent, `--start 140`).

Tool-based injection is near-total on gpt-4o-mini for both subsets. External-manager
is far lower for both — the manager only sees the subject's reply (information
scoping), so the attack needs gpt-4o-mini to *echo* the planted belief, which it
does ~5–10% of the time. External-manager Agent Action (5%) lands right at the
bottom of the paper's 5.5–82.5% band.

---

## Table 2 — Retrieval Rate (RR) & Adversarial Usage Rate (AUR)

Split by query proximity — LLM Behavior uses the dataset's `multi_turn_meta.split`
(`goal_adjacent` vs `wildchat_seed` = distant); Agent Action uses the row's
`goal_adjacent` flag (rows 0–99 adjacent, 100–199 distant).

**RR is regime-independent** (retrieval happens after the memory is stored) but IS
split by subset. The paper's Table 2 RR is measured with the **LLM memory manager**
(§5.3, caption: *"We report RR using the external memory manager"*) — our **G.2**.
G.1 (everything-in-context) and G.3 (embedding) are the Appendix-G ablations.

| Metric | **Ours: goal-adjacent** | **Ours: goal-distant** | Paper: adjacent | Paper: distant |
|---|---|---|---|---|
| **RR — G.2 LLM manager** (Gemini) — *the paper's mechanism* | **79%** (n=11) | **4%** (n=9) | **92 – 94%** (Behavior) | **3 – 6%** (Behavior) |
| RR — G.1 everything-in-context (ablation) | 100% | 100% | — | — |
| RR — G.3 semantic top-k, k=15 (ablation) | 95% | 20% | — | — |
| RR — G.3 semantic top-k, k=5 (ablation) | 85% | 11% | — | — |
| AUR — **LLM Behavior** | **91%** (10/11) | **11%** (1/9) | 42 – 85% | 0 – 6% |
| AUR — **Agent Action** | **30%** (6/20) | **0%** (0/20) | 60 – 89% | 6 – 17% |
| (Agent Action — any file edit) | 90% | 80% | — | — |

Files: LLM Behavior RR/AUR incl. G.2 `followup_20260910_191911.json` (n=20);
Agent Action `agent_action_20260910_185955.json` (adjacent), `agent_action_20260910_190306.json`
(distant).

Reading it:
- **RR (G.2 — the real number).** goal-distant **4%** is dead-on the paper's 3–6%:
  the manager LLM correctly declines to surface an unrelated poisoned memory ~96%
  of the time. goal-adjacent **79%** is below the paper's 92–94% — smaller manager
  model + our reconstructed retrieval prompt (paper's is Appendix M.2, not released)
  + n=11.
- G.1 = 100% by construction; G.3 (embedding ablation) runs looser, especially
  goal-distant (20% at k=15).
- **AUR — LLM Behavior** tracks RR: 91% adjacent / 11% distant. Adjacent slightly
  over the paper's 85% ceiling (n=11; gpt-4o-mini leans hard on provided context).
- **AUR — Agent Action** shows the right *shape* — 30% adjacent vs 0% distant —
  but below the paper (60–89% / 6–17%): weaker model, our reconstructed
  OpenClaw-style harness, n=20. The agent edits *some* file 80–90% of the time;
  it just usually edits it the way the user asked, not the attacker.

## Not yet done
- **Agent Action RR** — the agent-action runner uses G.1 placement and has no
  retrieval step; a G.2 RR over its memories + `eval_query` needs a small adapter.
- Larger n + bootstrap CIs on every non-tool-based cell.

---

## Remaining gaps
- Every non-tool-based cell is n≈20 — needs larger n + bootstrap CIs before any
  cell is a citable number.
- G.2 (memory-management-agent retrieval) implemented but not included in a run
  yet (`--retrieval-manager gemini|openai`); its prompt is ours, not paper-faithful.
