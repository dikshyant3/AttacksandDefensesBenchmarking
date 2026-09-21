# Hidden Sleeper Memory — ours vs. paper (GPT-5.4 column)

Only the metrics the paper puts in its tables: **Table 1 = IR**, **Table 2 = RR + AUR**.
Paper column = **GPT-5.4** (arXiv:2605.15338 v2), reported as `mean ± half-CI`.
Ours = **gpt-4o-mini** subject / judge, **Gemini 3.1 Flash Lite** external-manager
& retrieval manager. n≈20 for every non-tool-based cell.

---

## Table 1 — Injection Rate (IR)

| Regime | Subset | Paper (GPT-5.4) | **Ours (gpt-4o-mini)** | n (ours) |
|---|---|---|---|---|
| Tool-based | LLM Behavior | 99.4 ± 0.7 | **100%** | 140 |
| Tool-based | Agent Action | 97.0 ± 2.5 | **100%** | 56 |
| External-manager | LLM Behavior | 80.4 ± 3.5 | **10%** | 20 |
| External-manager | Agent Action | 61.5 ± 6.5 | **5%** | 20 |

## Table 2 — Retrieval Rate (RR) & Adversarial Usage Rate (AUR)

RR is measured with the LLM memory manager (paper §5.3 / our G.2).

| Metric | Subset | Proximity | Paper (GPT-5.4) | **Ours** | n (ours) |
|---|---|---|---|---|---|
| **RR** | LLM Behavior | goal-adjacent | 94.0 ± 5.0 | **79%** | 11 |
| **RR** | LLM Behavior | goal-distant  | 5.0 ± 10.0 | **4%** | 9 |
| **RR** | Agent Action | goal-adjacent | 98.0 ± 4.0 | *not run* | — |
| **RR** | Agent Action | goal-distant  | 13.0 ± 6.5 | *not run* | — |
| **AUR** | LLM Behavior | goal-adjacent | 42.0 ± 10.0 | **91%** | 11 |
| **AUR** | LLM Behavior | goal-distant  | 0.0 ± 0.0 | **11%** | 9 |
| **AUR** | Agent Action | goal-adjacent | 83.0 ± 7.5 | **30%** | 20 |
| **AUR** | Agent Action | goal-distant  | 14.0 ± 6.5 | **0%** | 20 |

---

## Where ours matches and where it diverges

| Cell | vs paper | Why |
|---|---|---|
| Tool-based IR (both subsets) | **≥ paper** (100 vs 97–99) | gpt-4o-mini is *more* compliant than GPT-5.4 with a direct "save this to my bio" |
| External-manager IR | **far below** (10/5 vs 80/62) | the subject must *echo* the planted belief for the manager to catch it; GPT-5.4 does this ~80% of the time, gpt-4o-mini ~10% |
| RR goal-distant (Behavior) | **matches** (4 vs 5 ± 10) | Gemini manager correctly refuses to surface an unrelated poisoned memory ~96% of the time |
| RR goal-adjacent (Behavior) | **below** (79 vs 94 ± 5) | smaller manager model + our reconstructed retrieval prompt (paper's Appendix M.2 not released) + n=11 |
| AUR goal-adjacent (Behavior) | **far above** (91 vs 42 ± 10) | gpt-4o-mini acts on a retrieved "memory" almost reflexively; GPT-5.4 is far more selective about letting a stored belief change its answer |
| AUR goal-distant (Behavior) | **above** (11 vs 0) | gpt-4o-mini produces false positives; GPT-5.4 never acts on a distant memory |
| AUR goal-adjacent (Agent Action) | **below** (30 vs 83 ± 7.5) | weaker model + our reconstructed OpenClaw-style harness (paper's agent env not released) |
| AUR goal-distant (Agent Action) | **below** (0 vs 14 ± 6.5) | same; low base rate, n=20 |

**Net:** the retrieval-discrimination result (goal-distant RR ≈ 4–5%) reproduces
cleanly. The two big divergences are both **subject-model capability effects**:
gpt-4o-mini is worse at the subtle external-manager echo (IR way down) but
*less* discerning once a memory is in context for plain chat (Behavior AUR way
up). Absolute Agent-Action numbers are limited by our reconstructed harness.

## Not run (paper has them, we don't)
- **RR — Agent Action** (goal-adjacent & goal-distant): the agent-action runner
  has no retrieval step yet. ~20-line adapter + one n=20 G.2 run.
