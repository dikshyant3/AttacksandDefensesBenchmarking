"""Run the DSRM campaign (Table 2's core comparison: None / Naive-Attack /
DSRM) against real ASB scenarios.

The command is a DRY RUN unless --yes is supplied -- prints exactly what it
is about to do and an approximate real-API-call count first. This project's
standing rule (see FIDELITY.md, and the user's explicit instruction) is that
no real LLM API calls happen without permission for that specific run, even
after the code itself has been reviewed and approved.

Retrieval (MiniLM/DPR) is always local -- no API cost, only compute time.
Only the agent-decision calls (--agent-model) and, for the DSRM method, the
decision-construction calls (--decision-model, paper default gpt-4o) cost
real money.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.dsrm.campaign import (
    DSRM_STAGES,
    AttackMethod,
    CampaignMetrics,
    ScenarioResult,
    build_background_kb,
    build_negative_pool,
    build_scenarios,
    run_campaign,
)
from benchmark.attacks.dsrm.data import dataset_summary
from benchmark.attacks.dsrm.decision import (
    DEFAULT_DECISION_MODEL,
    DEFAULT_INITIAL_DECISION_RETRIES,
    DEFAULT_MAX_REFINE_ITERS,
    DEFAULT_REASONING_LENGTH_WORDS,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from benchmark.attacks.dsrm.llm_cache import CachedChatClient
from benchmark.attacks.dsrm.whitebox import (
    DEFAULT_CANDIDATES,
    DEFAULT_NEGATIVES,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    JsonlCache,
    WhiteBoxConfig,
    grad_encoder_for,
)
from benchmark.attacks.hidden_sleeper.stats import rate_with_ci

DEFAULT_AGENT_MODEL = "gpt-4o-mini"  # this project's standard substitute for the paper's open-source backbones
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "results" / "llm_cache.jsonl"
DEFAULT_WHITEBOX_CACHE_PATH = Path(__file__).resolve().parent / "results" / "whitebox_cache.jsonl"
WHITEBOX_METHODS = (AttackMethod.DSRM_WHITEBOX, AttackMethod.CORPUS_POISON)


def make_embedder(name: str):
    """Lazy so a dry run never loads model weights."""
    from benchmark.attacks.dsrm.retrieval import get_embedder

    return get_embedder(name)


def serialize_result(r: ScenarioResult) -> dict:
    """Per-scenario record. Keeps the ORDERED steps (not just the merged tool
    set) so an alternative success rule -- e.g. 'final step only' -- can be
    re-scored later from the saved report without re-running anything."""
    return {
        "agent_name": r.scenario.agent_name,
        "attack_tool": r.scenario.attack_tool.name,
        "retrieved": r.retrieved,
        "attack_succeeded": r.attack_succeeded,
        "selected_tools": sorted(r.agent_decision.selected_tools),
        "steps": [{"message": s.message, "tool_use": list(s.tool_use)} for s in r.agent_decision.steps],
        "top_k_doc_ids": list(r.top_k_doc_ids),
        "executed_attack": r.executed_attack,
        "called_tools": list(r.called_tools),
    }


def add_memory_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared by run_experiment and run_sweep."""
    parser.add_argument(
        "--memory-format",
        choices=["figure1", "asb"],
        default="asb",
        help="How entries and queries are written: 'figure1' (Query/Context{Pt,Ts,Rt}, query = task; what runs before 2026-09-20 used) "
        "or 'asb' (ASB's own 'Agent; Task; Workflow; Tools' entries, query = task + full tool-set JSON). See memory_format.py.",
    )
    parser.add_argument(
        "--background",
        choices=["synthetic", "real"],
        default="real",
        help="Benign background entries: 'synthetic' one-liners (our placeholder) or 'real' agent-generated workflows "
        "(needs datasets/background_workflows.json from generate_background.py).",
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay each plan step by step and score whether the attack tool was actually CALLED -- ASB's definition of success and the paper's wording "
        "(\"successfully executes the attacker's intended action using the attack tool\"). Default ON; --no-execute scores the plan only. "
        "Adds up to one extra agent-model call per plan step that lists a tool.",
    )
    parser.add_argument(
        "--agent-style",
        choices=["default", "asb"],
        default="default",
        help="How the agent is prompted. 'default' = our reconstruction (memory as background context in one user turn). "
        "'asb' = ASB's real layout (separate plan-instruction turn, memory as an ASSISTANT turn phrased as an obligation, task as the last turn) -- see agent.py.",
    )
    parser.add_argument("--asb-memory-entries", type=int, default=5, help="Retrieved workflows shown in the ASB-style memory turn (ASB itself uses 1; DSRM retrieves 5)")


def add_whitebox_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared by every runner that can use the white-box conditions."""
    g = parser.add_argument_group("white-box (Algorithm 2) / Corpus Poisoning -- local compute, no API cost")
    g.add_argument("--whitebox-steps", type=int, default=DEFAULT_STEPS, help="HotFlip steps K (paper: 30)")
    g.add_argument("--whitebox-candidates", type=int, default=DEFAULT_CANDIDATES, help="candidates per step (paper: 100)")
    g.add_argument("--whitebox-negatives", type=int, default=DEFAULT_NEGATIVES, help="negatives in Eq. 5 (paper: 40)")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for the optimizer's position choice (paper: 42)")
    g.add_argument("--whitebox-cache-path", type=Path, default=DEFAULT_WHITEBOX_CACHE_PATH, help="cache of finished optimizations")


def make_whitebox_config(args, embedder) -> WhiteBoxConfig:
    return WhiteBoxConfig(
        encoder=grad_encoder_for(embedder),
        negative_pool=build_negative_pool(args.memory_format),
        steps=args.whitebox_steps,
        num_candidates=args.whitebox_candidates,
        num_negatives=args.whitebox_negatives,
        seed=args.seed,
        cache=JsonlCache(args.whitebox_cache_path),
    )


def make_chat_client(args):
    """(client, is_offline). --offline = cache only: a miss raises, the API is never reached."""
    if args.offline:
        return CachedChatClient(None, args.cache_path)
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    raw = OpenAI()
    return raw if args.no_cache else CachedChatClient(raw, args.cache_path)


def load_background(args) -> list[dict] | None:
    if args.background == "real":
        from benchmark.attacks.dsrm.data import load_background_workflows

        return load_background_workflows()
    return None


def sample_scenarios_round_robin(scenarios: list, n: int) -> list:
    """Picks scenario[0] from every domain first, then scenario[1] from
    every domain, etc. -- NOT the paper's own method (it uses all 400), but
    a deliberate choice for OUR cost-limited sampling: n=10 covers all 10
    domains once each rather than concentrating on whichever domain sorts
    first, giving a more representative small-n qualitative read."""
    by_agent: dict[str, list] = {}
    for s in scenarios:
        by_agent.setdefault(s.agent_name, []).append(s)
    agent_order = list(by_agent.keys())
    result = []
    i = 0
    while len(result) < n:
        made_progress = False
        for agent in agent_order:
            if i < len(by_agent[agent]):
                result.append(by_agent[agent][i])
                made_progress = True
                if len(result) == n:
                    break
        if not made_progress:
            break
        i += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help="Limit to this many scenarios, sampled round-robin across all 10 domains (default: all 400).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[m.value for m in AttackMethod],
        default=["none", "naive", "dsrm"],
        help="Which conditions to run (default: none, naive, dsrm; also clean, ori_attack, dsrm_no_srm, dsrm_no_csrm).",
    )
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL, help="LLM playing the benign agent (Phase 1)")
    parser.add_argument(
        "--decision-model",
        default=DEFAULT_DECISION_MODEL,
        help="LLM used to construct DSRM's adversarial decisions (Section 5.1's own default: gpt-4o)",
    )
    parser.add_argument("--retriever", choices=["minilm", "dpr", "realm"], default="dpr", help="Embedding backend (all local, no API cost). Default dpr = the paper's default retriever. 'realm' is dropped/unsupported here.")
    parser.add_argument(
        "--attacker-embedder",
        choices=["minilm", "same"],
        default="minilm",
        help="Embedder the ATTACKER uses for Self-Refine's similarity check. Under the black-box threat model the attacker can't see the "
        "deployed retriever, so the default is a fixed 'minilm' regardless of --retriever ('same' = reuse the retriever's embedder, the "
        "earlier behavior). Identical for --retriever minilm.",
    )
    parser.add_argument("--background-size", type=int, default=None, help="Benign background entries in the KB (default all 41; 0 = empty KB)")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help="On-disk LLM response cache; cache hits make NO API call")
    parser.add_argument("--no-cache", action="store_true", help="Disable the response cache (every call goes to the API)")
    parser.add_argument("--top-k", type=int, default=5, help="K, Section 5.1 default")
    parser.add_argument("--metric", choices=["ip", "cos", "l2"], default="ip", help="Retrieval similarity metric, Section 5.1 default")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="tau, Self-Refine similarity threshold (paper: 0.6)")
    parser.add_argument("--max-refine-iters", type=int, default=DEFAULT_MAX_REFINE_ITERS, help="MaxIter for Self-Refine (NOT paper-specified, see FIDELITY.md)")
    parser.add_argument("--reasoning-length-words", type=int, default=DEFAULT_REASONING_LENGTH_WORDS, help="L, approximate word cap on CSRM's per-step reasoning (paper: 45)")
    parser.add_argument("--initial-decision-retries", type=int, default=DEFAULT_INITIAL_DECISION_RETRIES, help="Retries if the model won't cite the real attack tool (engineering safeguard, not paper-specified)")
    add_memory_arguments(parser)
    add_whitebox_arguments(parser)
    parser.add_argument("--rerank", choices=["none", "ppl"], default="none", help="Re-ranking defense (Table 7): reorder the retrieved top-K by ascending GPT-2 log-perplexity of each plan before the agent sees them.")
    parser.add_argument("--scorer-model", default="gpt2", help="causal LM used by --rerank ppl")
    parser.add_argument("--offline", action="store_true", help="Answer LLM requests ONLY from the cache (a miss raises); makes zero API calls, so no --yes is needed.")
    parser.add_argument("--retrieval-only", action="store_true", help="Stop after retrieval: measures RR and skips every agent call. Combine with --offline for a fully free run.")
    parser.add_argument("--yes", action="store_true", help="Make real, paid API calls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def _estimate_dsrm_calls_per_scenario(args: argparse.Namespace) -> tuple[int, int]:
    """(typical, worst_case) real gpt-4o calls per DSRM scenario. Typical:
    initial decision usually complies first try (1) + self-refine usually
    clears tau immediately (1 similarity check, 0 extra calls) + reasoning
    (1) = 2. Worst case: all initial-decision retries exhausted +
    max_refine_iters+1 refine calls + reasoning."""
    typical = 1 + 0 + 1
    worst = args.initial_decision_retries + (args.max_refine_iters + 1) + 1
    return typical, worst


def main() -> None:
    args = parse_args()
    all_scenarios = build_scenarios()
    scenarios = (
        sample_scenarios_round_robin(all_scenarios, args.n_scenarios)
        if args.n_scenarios is not None
        else all_scenarios
    )
    methods = [AttackMethod(m) for m in args.methods]

    print("DSRM -- Deceptive Semantic Reasoning Manipulation")
    print(json.dumps(dataset_summary(), indent=2))
    print(f"Scenarios this run: {len(scenarios)} (of {len(all_scenarios)} total)  |  Methods: {[m.value for m in methods]}")
    print(f"Agent model: {args.agent_model}  (this project's standard substitute for the paper's backbone sweep)")
    print(f"Decision model: {args.decision_model}" + ("  <- paper's own stated default" if args.decision_model == DEFAULT_DECISION_MODEL else "  <- override"))
    print(f"Retriever: {args.retriever} (local, no API cost)  |  top-k={args.top_k}  |  metric={args.metric}")
    print(f"Memory format: {args.memory_format}  |  background: {args.background}  |  scoring: plan-level" + (" + execution-level" if args.execute else " only"))
    print(f"Self-Refine: tau={args.threshold}  max_iters={args.max_refine_iters} (not paper-specified)  |  reasoning length~{args.reasoning_length_words} words")

    n = len(scenarios)
    typical, worst = _estimate_dsrm_calls_per_scenario(args)
    call_estimate_lines = []
    for m in methods:
        if m is AttackMethod.POISONEDRAG:
            call_estimate_lines.append(f"  {m.value}: {n} decision-model calls ({args.decision_model}, temp 1) + {n} agent calls")
        elif m is AttackMethod.ASB_ATTACK:
            call_estimate_lines.append(f"  {m.value}: {n} extra agent calls to plan under the injection + {n} agent calls")
        elif m in DSRM_STAGES:
            call_estimate_lines.append(
                f"  {m.value}: {n} agent calls ({args.agent_model}) + up to ~{n * typical}-{n * worst} decision calls ({args.decision_model})"
            )
        else:
            call_estimate_lines.append(f"  {m.value}: {n} agent calls ({args.agent_model})")
    if args.execute:
        call_estimate_lines.append(f"  + execution replay: up to ~{n * len(methods) * 4} more {args.agent_model} calls (~1 per tool-listing step, stops at first attack call)")
    print("Upper bound on real API calls (a cache hit, see --cache-path, makes none):")
    print("\n".join(call_estimate_lines))

    if args.retrieval_only:
        print("  (--retrieval-only: agent and execution calls are skipped; only decision/baseline construction can call an LLM)")
    if args.offline:
        print("OFFLINE: every LLM request must already be in the cache; nothing can reach an API.")
    elif not args.yes:
        print("\nDRY RUN ONLY: add --yes to execute paid API calls.")
        return

    client = make_chat_client(args)
    agent_client = decision_client = client

    embedder = make_embedder(args.retriever)
    rerank_scorer = None
    if args.rerank == "ppl":
        from benchmark.attacks.dsrm.defenses import LogPerplexityScorer, PlanPerplexityScorer

        rerank_scorer = PlanPerplexityScorer(LogPerplexityScorer(args.scorer_model))
    whitebox_cfg = make_whitebox_config(args, embedder) if any(m in WHITEBOX_METHODS or m is AttackMethod.DSRM_WHITEBOX for m in methods) else None
    attacker_embedder = embedder if args.attacker_embedder == "same" else make_embedder("minilm")

    print(f"\nBuilding background KB ({args.background_size if args.background_size is not None else 41} real benign entries, {args.retriever})...")
    background_kb = build_background_kb(
        embedder, max_entries=args.background_size, memory_format=args.memory_format, workflows=load_background(args)
    )

    all_results: dict[str, list[ScenarioResult]] = {}
    all_metrics: dict[str, CampaignMetrics] = {}
    for method in methods:
        print(f"\nRunning method={method.value} on {n} scenarios...")
        results, metrics = run_campaign(
            scenarios,
            method,
            background_kb,
            agent_client=agent_client,
            agent_model=args.agent_model,
            decision_client=decision_client,
            decision_model=args.decision_model,
            k=args.top_k,
            metric=args.metric,
            attacker_embedder=attacker_embedder,
            memory_format=args.memory_format,
            execute=args.execute and not args.retrieval_only,
            agent_style=args.agent_style,
            asb_memory_entries=args.asb_memory_entries,
            whitebox=whitebox_cfg,
            retrieval_only=args.retrieval_only,
            rerank_scorer=rerank_scorer,
            threshold=args.threshold,
            max_refine_iters=args.max_refine_iters,
            reasoning_length_words=args.reasoning_length_words,
            initial_decision_retries=args.initial_decision_retries,
        )
        all_results[method.value] = results
        all_metrics[method.value] = metrics
        asr_a_ci = rate_with_ci([1 if r.attack_succeeded else 0 for r in results])
        rr_ci = rate_with_ci([1 if r.retrieved else 0 for r in results])
        retrieved_bits = [1 if r.attack_succeeded else 0 for r in results if r.retrieved]
        asr_r_ci = rate_with_ci(retrieved_bits) if retrieved_bits else None
        line = f"  ASR_A(plan)={asr_a_ci['pretty']}  RR={rr_ci['pretty']}  ASR_R={asr_r_ci['pretty'] if asr_r_ci else '-'}"
        if args.retrieval_only:
            line = f"  RR={rr_ci['pretty']}  (retrieval only -- no agent was run, ASR not measured)"
        elif args.execute:
            exec_ci = rate_with_ci([1 if r.executed_attack else 0 for r in results])
            line += f"  |  ASR_A(executed)={exec_ci['pretty']}"
        print(line)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "dsrm_campaign",
        "n_scenarios": n,
        "n_total_scenarios": len(all_scenarios),
        "methods": [m.value for m in methods],
        "agent_model": args.agent_model,
        "decision_model": args.decision_model,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "metric": args.metric,
        "threshold": args.threshold,
        "max_refine_iters": args.max_refine_iters,
        "reasoning_length_words": args.reasoning_length_words,
        "metrics": {
            method: {
                "n": m.n,
                "asr_a": m.asr_a,
                "asr_r": m.asr_r,
                "rr": m.rr,
                "asr_a_exec": m.asr_a_exec,
                "asr_r_exec": m.asr_r_exec,
            }
            for method, m in all_metrics.items()
        },
        "scenarios": [
            {
                "agent_name": r.scenario.agent_name,
                "user_task": r.scenario.user_task,
                "attack_tool": r.scenario.attack_tool.name,
            }
            for r in all_results[methods[0].value]
        ],
        "attacker_embedder": args.attacker_embedder,
        "memory_format": args.memory_format,
        "background": args.background,
        "execute": args.execute,
        "agent_style": args.agent_style,
        "retrieval_only": args.retrieval_only,
        "rerank": args.rerank,
        "offline": args.offline,
        "whitebox": {"steps": args.whitebox_steps, "candidates": args.whitebox_candidates, "negatives": args.whitebox_negatives, "seed": args.seed} if whitebox_cfg else None,
        "background_size": args.background_size,
        "llm_cache": None if (args.no_cache and not args.offline) else {"path": str(args.cache_path), "hits": client.hits, "real_calls_made": client.misses},
        "results": {method: [serialize_result(r) for r in results] for method, results in all_results.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"campaign_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {output}")


if __name__ == "__main__":
    main()
