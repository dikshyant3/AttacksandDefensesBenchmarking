# Hidden Sleeper Memory -- results provenance

## Stage 1, tool-based regime
- `../../../results/hidden_sleeper_experiment_*.json` and `injection_*.json`
  (in this directory) -- IR via the real `bio` tool. `injection_20260824_143402.json`
  is the full n=196 run (IR = 100%). `injection_20260824_140059.json` is a 1-row
  smoke test.

## Stage 1, external-manager regime
### `external_manager_20260903_*.json` -- SUPERSEDED (wrong mechanism)
These four runs used a `mem0.Memory.from_config()` + local Chroma vector store
(the `sdk`-style backend). Verified 2026-09-10 against the upstream source: the
paper's external-manager numbers (Table 1/Table 2) come from the **`prompt_only`**
backend -- a single simulated-manager LLM call with Mem0's
`ADDITIVE_EXTRACTION_PROMPT`, **no** vector store, **no** real `mem0ai` service.
So these results measured a different (more sophisticated) system than the paper.
Kept for the record; do not cite as a paper comparison.
The `chroma_external_manager/` dir is the orphaned Chroma store from those runs.

### future `external_manager_*.json`
Runs from the rewritten `external_manager.py` carry
`"experiment": "external_manager_regime_prompt_only"` and
`"backend": "prompt_only (...)"` in the report -- those are the paper-faithful ones.

## Stage 3, agent-action half
- `agent_action_20260903_155154.json` -- n=20, AUR = 20%.

## Stage 2 (retrieval) + Stage 3 LLM-behavior half
- Not yet run live (`run_followup_experiment.py` exists, unit-tested only).
