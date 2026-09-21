"""Seed stability (Table 8): repeat the same experiment under several seeds.

The paper reruns DSRM four times "each with a different, randomly chosen seed"
(42, 100, 1337, 2446) and reports ASR_A per seed. It doesn't say what a seed
controls. Ours controls the two real sources of variation in this harness:
  * WHICH scenarios are drawn (a seeded random sample of N of the 400 -- the paper
    always uses all 400, where this source vanishes), and
  * the LLM's sampling (an explicit --temperature plus OpenAI's best-effort `seed`;
    at our default temperature 0 a seed would change nothing).
--vary picks scenarios, sampling, or both.

Real calls: each seed builds its own decisions and agent plans (sampling changes
them), so nothing is shared between seeds. DRY RUN unless --yes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.dsrm.campaign import AttackMethod, build_background_kb, build_scenarios, run_campaign
from benchmark.attacks.dsrm.decision import DEFAULT_DECISION_MODEL
from benchmark.attacks.dsrm.llm_cache import CachedChatClient, SeededClient
from benchmark.attacks.dsrm.run_experiment import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CACHE_PATH,
    add_memory_arguments,
    load_background,
    make_embedder,
    sample_scenarios_round_robin,
)

PAPER_SEEDS = [42, 100, 1337, 2446]  # Table 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--n-scenarios", type=int, default=25)
    p.add_argument("--vary", choices=["scenarios", "sampling", "both"], default="both")
    p.add_argument("--temperature", type=float, default=1.0, help="LLM sampling temperature when sampling varies")
    p.add_argument("--methods", nargs="+", choices=[m.value for m in AttackMethod], default=["dsrm"])
    p.add_argument("--retriever", choices=["minilm", "dpr"], default="dpr")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    p.add_argument("--decision-model", default=DEFAULT_DECISION_MODEL)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--yes", action="store_true", help="make real, paid API calls")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    add_memory_arguments(p)
    return p.parse_args()


def scenarios_for_seed(all_scenarios: list, n: int, seed: int, vary: str) -> list:
    if vary == "sampling":  # same scenarios every seed
        return sample_scenarios_round_robin(all_scenarios, n)
    return random.Random(seed).sample(all_scenarios, n)


def main() -> None:
    args = parse_args()
    all_scenarios = build_scenarios()
    methods = [AttackMethod(m) for m in args.methods]
    print(f"Seed stability: seeds={args.seeds}  n={args.n_scenarios}/seed  vary={args.vary}  methods={[m.value for m in methods]}")
    if not args.yes:
        print(f"DRY RUN: up to ~{len(args.seeds) * args.n_scenarios * len(methods) * 6} real calls (decisions + agent + execution) -- nothing is shared across seeds. Add --yes.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    cached = CachedChatClient(OpenAI(), args.cache_path)
    embedder = make_embedder(args.retriever)
    kb = build_background_kb(embedder, memory_format=args.memory_format, workflows=load_background(args))

    per_seed = {}
    for seed in args.seeds:
        client = SeededClient(cached, seed, args.temperature) if args.vary in ("sampling", "both") else cached
        scenarios = scenarios_for_seed(all_scenarios, args.n_scenarios, seed, args.vary)
        per_seed[seed] = {}
        for method in methods:
            _, m = run_campaign(
                scenarios, method, kb, agent_client=client, agent_model=args.agent_model, decision_client=client,
                decision_model=args.decision_model, k=args.top_k, memory_format=args.memory_format, execute=args.execute,
            )
            per_seed[seed][method.value] = {"asr_a": m.asr_a, "asr_a_exec": m.asr_a_exec, "rr": m.rr}
            print(f"  seed {seed:>5}  {method.value}: ASR_A={m.asr_a * 100:.1f}%  RR={m.rr * 100:.0f}%")
    summary = {}
    for method in methods:
        vals = [per_seed[s][method.value]["asr_a"] * 100 for s in args.seeds]
        summary[method.value] = {"mean": statistics.mean(vals), "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0, "min": min(vals), "max": max(vals)}
        print(f"  {method.value}: mean {summary[method.value]['mean']:.2f}%  stdev {summary[method.value]['stdev']:.2f}  range [{min(vals):.1f}, {max(vals):.1f}]"
              "   (paper, LLaMA3-70b: 40.00-42.00, mean 41.18; GPT-4o: 40.75 +/- 1.19)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"seeds_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "vary": args.vary, "temperature": args.temperature, "n_scenarios": args.n_scenarios, "per_seed": per_seed, "summary": summary, "real_calls_made": cached.misses}, indent=2), encoding="utf-8")
    print(f"Real calls made: {cached.misses}  |  Report: {out}")


if __name__ == "__main__":
    main()
