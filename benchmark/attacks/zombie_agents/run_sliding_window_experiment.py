"""Execute the Zombie Agents attack against a sliding-window-memory web agent
(paper Section 3.3, Scenario A; Section 4.1: K=3, M=20).

For each bait task: run Phase I (Exposure, K rounds of the bait task), then
replay `--trigger-rounds` real insta-150k-v1 queries as Phase II (trigger)
sessions, measuring ASR and context-retention.

The command is a dry run unless --yes is supplied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.hidden_sleeper.stats import rate_with_ci
from benchmark.attacks.zombie_agents.campaign import (
    DEFAULT_SLIDING_WINDOW_EXPOSURE_ROUNDS,
    DEFAULT_SLIDING_WINDOW_SIZE,
    DEFAULT_TRIGGER_ROUNDS,
    run_sliding_window_campaign,
)
from benchmark.attacks.zombie_agents.data import dataset_summary, load_bait_tasks, load_trigger_queries
from benchmark.attacks.zombie_agents.models import CLOSEST_AVAILABLE_GEMINI, PAPER_AGENT_MODELS, make_agent_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help=f"Web-agent LLM provider. 'openai' (default) is this project's standard substitute "
        f"(gpt-4o-mini). 'gemini' is the paper's actual model ({PAPER_AGENT_MODELS['gemini']}), "
        f"needs GEMINI_API_KEY. The paper's other model, GLM-4.7-Flash, has no configured access here.",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default model id")
    parser.add_argument("--window-size", type=int, default=DEFAULT_SLIDING_WINDOW_SIZE, help="Sliding-window length L (paper default 3)")
    parser.add_argument(
        "--exposure-rounds",
        type=int,
        default=DEFAULT_SLIDING_WINDOW_EXPOSURE_ROUNDS,
        help="Exposure Phase rounds K -- how many times Phase I repeats the bait task before Phase II begins "
        "(paper default 3, Section 4.1: 'Exposure Phase (K rounds of baited tasks)'). Paper sets K=3=window size "
        "so the window is fully saturated with payload-bearing entries by the time Trigger begins.",
    )
    parser.add_argument("--trigger-rounds", type=int, default=DEFAULT_TRIGGER_ROUNDS, help="Trigger rounds M per bait task (paper default 20)")
    parser.add_argument("--num-bait-tasks", type=int, default=5, help="How many of the 20 bait tasks to run (each is a separate campaign)")
    parser.add_argument("--defense", choices=["none"], default="none", help="Sliding-window agent has no defense variant in this runner (see run_rag_experiment.py for sandwich/spotlight)")
    parser.add_argument("--yes", action="store_true", help="Make paid API calls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bait_tasks = load_bait_tasks()[: args.num_bait_tasks]
    trigger_queries = load_trigger_queries()

    print("Zombie Agents -- sliding-window memory")
    print(json.dumps(dataset_summary(), indent=2))
    print(f"Bait tasks: {len(bait_tasks)}  |  Exposure rounds K: {args.exposure_rounds}  |  Trigger rounds M: {args.trigger_rounds}  |  Window size L: {args.window_size}")
    intended_model = args.model or (CLOSEST_AVAILABLE_GEMINI if args.provider == "gemini" else "gpt-4o-mini")
    is_paper_model = args.provider == "gemini" and intended_model == PAPER_AGENT_MODELS["gemini"]
    print(f"Provider: {args.provider}  |  Model: {intended_model}" + ("  <- the paper's actual model" if is_paper_model else "  <- substitute"))
    total_calls = len(bait_tasks) * (args.exposure_rounds + args.trigger_rounds)
    print(f"About to make {total_calls} real chat calls if run for real (no embeddings, no judge -- ASR/retention are rule-based).")
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv

    load_dotenv()
    client, model = make_agent_client(args.provider, args.model)
    print(f"Model: {model}")

    campaigns = []
    for index, bait in enumerate(bait_tasks, start=1):
        result = run_sliding_window_campaign(
            client, model, bait, trigger_queries,
            window_size=args.window_size, exposure_rounds=args.exposure_rounds, max_trigger_rounds=args.trigger_rounds,
        )
        campaigns.append(result)
        print(
            f"[{index:02d}/{len(bait_tasks)}] {bait.task_id}: infected={int(result.infected)} "
            f"ASR={result.asr:.2f} retention={result.retention_rate:.2f}"
        )

    all_asr_bits = [1 if r["executed_malicious"] else 0 for c in campaigns for r in c.trigger_rounds]
    all_retention_bits = [1 if r["payload_in_context_before_step"] else 0 for c in campaigns for r in c.trigger_rounds]
    infection_bits = [1 if c.infected else 0 for c in campaigns]

    metrics = {
        "infection_rate": rate_with_ci(infection_bits),
        "attack_success_rate": rate_with_ci(all_asr_bits),
        "context_retention_rate": rate_with_ci(all_retention_bits),
    }
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "zombie_agents_sliding_window",
        "provider": args.provider,
        "model": model,
        "window_size": args.window_size,
        "exposure_rounds": args.exposure_rounds,
        "trigger_rounds_per_task": args.trigger_rounds,
        "num_bait_tasks": len(bait_tasks),
        "metrics": metrics,
        "campaigns": [asdict(c) for c in campaigns],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"sliding_window_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v["pretty"] for k, v in metrics.items()}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
