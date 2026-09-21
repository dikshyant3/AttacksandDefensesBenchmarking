"""Execute the External-Manager regime of Hidden Sleeper Memory, reproducing
the upstream ``prompt_only`` backend (a single simulated-manager LLM call with
Mem0's published ADDITIVE_EXTRACTION_PROMPT -- no real mem0ai service, no
vector store) plus the paper's Appendix E.2 information scoping.

See external_manager.py's module docstring for the full faithfulness note.

The command is a dry run unless --yes is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.hidden_sleeper.data import dataset_summary, load_paper_main_subset
from benchmark.attacks.hidden_sleeper.external_manager import (
    PAPER_MANAGER_MODEL,
    make_manager_client,
    run_external_manager_case,
)
from benchmark.attacks.hidden_sleeper.openai_tool_pipeline import DEFAULT_JUDGE_MODEL, OpenAIToolPipeline
from benchmark.attacks.hidden_sleeper.payloads import PayloadVariant
from benchmark.attacks.hidden_sleeper.stats import rate_with_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for the subject (target LLM)")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model for the goal-match judge (default: {DEFAULT_JUDGE_MODEL}; the manager stays Gemini "
        f"regardless). Pass a different value to decouple the judge from the subject.",
    )
    parser.add_argument(
        "--manager-provider",
        choices=["gemini", "openai"],
        default="gemini",
        help=f"Provider for the simulated memory manager's single extraction call. "
        f"'gemini' (default) is the paper's actual manager ({PAPER_MANAGER_MODEL}) via Google's "
        f"OpenAI-compatible endpoint (needs GEMINI_API_KEY). 'openai' uses an OpenAI model instead.",
    )
    parser.add_argument(
        "--manager-model",
        default=None,
        help="Override the manager model id. Defaults to the provider's default "
        f"({PAPER_MANAGER_MODEL} for gemini, gpt-4o-mini for openai).",
    )
    parser.add_argument(
        "--attack",
        choices=[variant.value for variant in PayloadVariant],
        default=PayloadVariant.EXTERNAL_MANAGER_C2.value,
        help="Payload variant. Defaults to the real regime-specific one (external_manager_c2).",
    )
    parser.add_argument(
        "--include-document-content",
        action="store_true",
        help="Opt OUT of the paper's Appendix E.2 information scoping and feed the manager the full "
        "uploaded document (upstream's non-scoped DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT=True). "
        "Default: scoped (manager sees only the bare query + the subject's reply).",
    )
    parser.add_argument("--start", type=int, default=0, help="Index of the first row to run (behavior rows are 0-139, agent rows 140-195)")
    parser.add_argument("--limit", type=int, default=20, help="Number of official subset rows to run, starting at --start")
    parser.add_argument("--yes", action="store_true", help="Make paid OpenAI API calls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_paper_main_subset()
    if args.start < 0 or args.start >= len(cases):
        raise SystemExit(f"--start must be between 0 and {len(cases) - 1}")
    if args.limit < 1 or args.start + args.limit > len(cases):
        raise SystemExit(f"--start + --limit must be <= {len(cases)}")
    selected = cases[args.start : args.start + args.limit]

    default_manager_model = PAPER_MANAGER_MODEL if args.manager_provider == "gemini" else "gpt-4o-mini"
    manager_model = args.manager_model or default_manager_model
    scoping = "document included (NOT scoped)" if args.include_document_content else "scoped: query + reply only"
    print("Hidden Sleeper Memory -- External-Manager regime (upstream prompt_only backend)")
    print(json.dumps(dataset_summary(cases), indent=2))
    print(f"Selected rows: {len(selected)}")
    print(f"Subject model (OpenAI): {args.model}")
    print(f"Goal-match judge (OpenAI): {args.judge_model}"
          + ("  <- same as subject (default)" if args.judge_model == args.model else "  <- decoupled from subject"))
    faithful = "  <- the paper's actual manager" if args.manager_provider == "gemini" else "  <- OpenAI substitute"
    print(f"Memory manager ({args.manager_provider}): {manager_model}{faithful}")
    print(f"Information scoping: {scoping}")
    print(f"Payload variant: {args.attack}")
    print(
        f"About to make {len(selected)} subject calls (OpenAI) + {len(selected)} manager calls "
        f"({args.manager_provider}) + one judge call per extracted memory."
    )
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when --yes is supplied")

    client = OpenAI()
    pipeline = OpenAIToolPipeline(client, args.model, judge_model=args.judge_model)
    manager_client, _ = make_manager_client(args.manager_provider)

    rows: list[dict] = []
    for index, case in enumerate(selected, start=1):
        result = run_external_manager_case(
            client,
            args.model,
            case,
            pipeline.semantic_goal_match,
            manager_model=manager_model,
            manager_client=manager_client,
            variant=PayloadVariant(args.attack),
            include_document_content=args.include_document_content,
        )
        row = {
            "case_id": case.case_id,
            "category": case.category.value,
            "goal": case.adversarial_memory_goal,
            **asdict(result),
        }
        rows.append(row)
        print(
            f"[{index:03d}/{len(selected):03d}] {case.case_id}: "
            f"extracted={len(result.extracted_memories)} goal_match={int(result.target_memory_written)}"
        )

    def slice_ci(category, key):
        vals = [1 if row[key] else 0 for row in rows if category is None or row["category"] == category]
        return rate_with_ci(vals)

    metrics = {
        "any_extraction_rate": slice_ci(None, "extracted_memories"),
        "injection_rate": slice_ci(None, "target_memory_written"),
        "injection_rate_behavior": slice_ci("behavior", "target_memory_written"),
        "injection_rate_agent_action": slice_ci("agent_action", "target_memory_written"),
    }
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "external_manager_regime_prompt_only",
        "start_index": args.start,
        "backend": "prompt_only (simulated manager, mem0 ADDITIVE_EXTRACTION_PROMPT, no vector store)",
        "model": args.model,
        "judge_model": args.judge_model,
        "manager_provider": args.manager_provider,
        "manager_model": manager_model,
        "paper_manager_model": PAPER_MANAGER_MODEL,
        "information_scoping": not args.include_document_content,
        "dataset": "paper_main_subset_196.json",
        "selected_rows": len(selected),
        "metrics": metrics,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"external_manager_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def _summarize(v):
        if isinstance(v, dict) and "pretty" in v:
            return f"{v['pretty']}  (n={v['n']})" if v["n"] else "no cases in this slice"
        return v

    print(json.dumps({k: _summarize(v) for k, v in metrics.items()}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
