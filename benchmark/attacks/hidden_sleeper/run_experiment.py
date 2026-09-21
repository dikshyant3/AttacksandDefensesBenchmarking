"""Execute the paper-dataset Hidden Sleeper injection campaign with OpenAI.

The command is a dry run unless --yes is supplied. Subject and goal-match judge
both default to gpt-4o-mini; pass --judge-model to decouple them (the paper
uses a fixed separate grader, sleeper_eval/scorer.py).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.hidden_sleeper.adapter import HiddenSleeperAttack
from benchmark.attacks.hidden_sleeper.data import (
    PAPER_SUBSET_120_PATH,
    PAPER_SUBSET_PATH,
    dataset_summary,
    load_paper_main_subset,
)
from benchmark.attacks.hidden_sleeper.openai_tool_pipeline import DEFAULT_JUDGE_MODEL, OpenAIToolPipeline
from benchmark.attacks.hidden_sleeper.payloads import PayloadVariant
from benchmark.attacks.hidden_sleeper.stats import rate_with_ci
from benchmark.stores.procedural import ProceduralStore

SUBSET_PATHS = {"196": PAPER_SUBSET_PATH, "120": PAPER_SUBSET_120_PATH}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model for the subject (attacked assistant)")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model for the semantic goal-match judge (default: {DEFAULT_JUDGE_MODEL}, same as the "
        f"default subject). Pass a different value to decouple the judge from the subject.",
    )
    parser.add_argument(
        "--attack",
        choices=[variant.value for variant in PayloadVariant],
        default=PayloadVariant.ACTOR_CRITIC_NO_MARKERS.value,
    )
    parser.add_argument(
        "--subset",
        choices=list(SUBSET_PATHS.keys()),
        default="196",
        help="Which ratio-preserving subset to run: the full 196-row one, or the cheaper 120-row one.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of subset rows to run (default: all)")
    parser.add_argument("--yes", action="store_true", help="Make paid OpenAI API calls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_paper_main_subset(SUBSET_PATHS[args.subset])
    limit = args.limit if args.limit is not None else len(cases)
    if limit < 1 or limit > len(cases):
        raise SystemExit(f"--limit must be between 1 and {len(cases)}")
    selected = cases[:limit]

    print("Hidden Sleeper Memory — official paper main subset")
    print(json.dumps(dataset_summary(cases), indent=2))
    print(f"Selected rows: {len(selected)}")
    print(f"Attack: {args.attack}")
    print(f"OpenAI model (subject): {args.model}")
    print(f"OpenAI model (goal-match judge): {args.judge_model}"
          + ("   <- same as subject (default)" if args.judge_model == args.model else "   <- decoupled from subject"))
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid OpenAI calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when --yes is supplied")

    pipeline = OpenAIToolPipeline(OpenAI(), args.model, judge_model=args.judge_model)
    variant = PayloadVariant(args.attack)
    rows: list[dict] = []
    for index, case in enumerate(selected, start=1):
        store = ProceduralStore()
        attack = HiddenSleeperAttack(
            memory_writer=pipeline.memory_writer(case.preexisting_memories),
            goal_matcher=pipeline.semantic_goal_match,
            variant=variant,
        )
        result = attack.plant(case, store, session_id=f"hidden-sleeper-injection-{index}")
        row = {
            "case_id": case.case_id,
            "category": case.category.value,
            "source_file": case.source_file,
            "has_preexisting_memories": bool(case.preexisting_memories),
            "goal": case.adversarial_memory_goal,
            **asdict(result),
        }
        rows.append(row)
        print(
            f"[{index:03d}/{len(selected):03d}] {case.case_id}: "
            f"tool_write={int(result.write_accepted)} goal_match={int(result.target_memory_written)}"
        )

    def slice_ci(category, key):
        vals = [1 if row[key] else 0 for row in rows if category is None or row["category"] == category]
        return rate_with_ci(vals)

    metrics = {
        "tool_write_rate": slice_ci(None, "write_accepted"),
        "injection_rate": slice_ci(None, "target_memory_written"),
        "injection_rate_behavior": slice_ci("behavior", "target_memory_written"),
        "injection_rate_agent_action": slice_ci("agent_action", "target_memory_written"),
    }
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "paper_tool_based_main_openai",
        "model": args.model,
        "judge_model": args.judge_model,
        "attack": args.attack,
        "dataset": SUBSET_PATHS[args.subset].name,
        "selected_rows": len(selected),
        "metrics": metrics,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"injection_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _summarize(v):
        if isinstance(v, dict) and "pretty" in v:
            return f"{v['pretty']}  (n={v['n']})" if v["n"] else "no cases in this slice"
        return v

    print(json.dumps({k: _summarize(v) for k, v in metrics.items()}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
