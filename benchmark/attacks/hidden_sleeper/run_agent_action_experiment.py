"""Execute Stage 3's "Agent Action" half of Hidden Sleeper Memory: real
sandboxed tool-use cases (see agent_action.py's module docstring), plus an
optional Stage 2 retrieval measurement over the same cases.

The command is a dry run unless --yes is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.hidden_sleeper.agent_action import (
    judge_agent_action_usage,
    retrieval_view,
    run_agent_action_case,
)
from benchmark.attacks.hidden_sleeper.agent_action_data import dataset_summary, load_agent_action_cases
from benchmark.attacks.hidden_sleeper.external_manager import make_manager_client
from benchmark.attacks.hidden_sleeper.followup import compute_all_retrieval_methods
from benchmark.attacks.hidden_sleeper.openai_tool_pipeline import DEFAULT_JUDGE_MODEL
from benchmark.attacks.hidden_sleeper.stats import rate_with_ci

EMBEDDING_MODEL = "text-embedding-3-small"


def make_openai_embed_fn(client, model: str = EMBEDDING_MODEL, cache: dict | None = None):
    cache = cache if cache is not None else {}

    def embed(text: str):
        import numpy as np

        if text not in cache:
            response = client.embeddings.create(model=model, input=text)
            cache[text] = np.array(response.data[0].embedding)
        return cache[text]

    return embed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for the agent (subject)")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model for the edit-influence judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--start", type=int, default=0, help="First row index (goal_adjacent 0-99, goal_distant 100-199)")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to run, starting at --start")
    parser.add_argument("--max-turns", type=int, default=5, help="Max real tool-calling turns per case")
    parser.add_argument(
        "--retrieval",
        choices=["none", "g1g3", "gemini", "openai"],
        default="none",
        help="Also measure Stage 2 RR over these cases: 'g1g3' = G.1 + G.3 (embeddings only); "
        "'gemini'/'openai' additionally run G.2 (LLM memory manager) with that provider. "
        "'none' skips retrieval.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[15, 5], help="G.3 retrieval cutoffs")
    parser.add_argument("--yes", action="store_true", help="Make paid API calls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_agent_action_cases()
    if args.start < 0 or args.start >= len(cases):
        raise SystemExit(f"--start must be between 0 and {len(cases) - 1}")
    if args.limit < 1 or args.start + args.limit > len(cases):
        raise SystemExit(f"--start + --limit must be <= {len(cases)}")
    selected = cases[args.start : args.start + args.limit]
    top_k = tuple(args.top_k)

    print("Hidden Sleeper Memory -- Stage 3 Agent Action (+ optional Stage 2 RR)")
    print(json.dumps(dataset_summary(cases), indent=2))
    print(f"Selected rows: {args.start}..{args.start + len(selected) - 1}  (n={len(selected)})")
    print(f"OpenAI model (agent / subject): {args.model}")
    print(f"Edit-influence judge model: {args.judge_model}"
          + ("  <- same as agent (default)" if args.judge_model == args.model else "  <- decoupled from subject"))
    print(f"Retrieval measurement: {args.retrieval}")
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when --yes is supplied")

    client = OpenAI()
    embed_fn = None
    manager_client = manager_model = None
    if args.retrieval in ("g1g3", "gemini", "openai"):
        embed_fn = make_openai_embed_fn(client)
    if args.retrieval in ("gemini", "openai"):
        manager_client, manager_model = make_manager_client(args.retrieval)

    rows: list[dict] = []
    for index, case in enumerate(selected, start=1):
        result = run_agent_action_case(client, args.model, case, max_turns=args.max_turns)
        verdict = judge_agent_action_usage(client, args.judge_model, case, result)
        retrieval = None
        if embed_fn is not None:
            retrieval = compute_all_retrieval_methods(
                retrieval_view(case), embed_fn=embed_fn,
                manager_client=manager_client, manager_model=manager_model, top_k=top_k,
            )
        row = {
            "case_id": case.case_id,
            "category": case.category,
            "goal_adjacent": case.goal_adjacent,
            "made_edits": bool(result.edits),
            **asdict(result),
            "verdict": verdict,
            "retrieval_methods": retrieval,
        }
        rows.append(row)
        rr = ""
        if retrieval:
            rr = " " + " ".join(f"{k.split('_')[0].upper()}={v['retrieval_rate']:.2f}" for k, v in retrieval.items())
        print(f"[{index:03d}/{len(selected):03d}] {case.case_id}: edits={len(result.edits)} AUR={int(verdict['used'])}{rr}")

    def slice_ci(pred, key_fn):
        vals = [key_fn(r) for r in rows if pred(r)]
        return rate_with_ci(vals)

    adj = lambda r: r["goal_adjacent"]  # noqa: E731
    dist = lambda r: not r["goal_adjacent"]  # noqa: E731
    aur = lambda r: 1 if r["verdict"]["used"] else 0  # noqa: E731
    edit = lambda r: 1 if r["made_edits"] else 0  # noqa: E731

    metrics = {
        "any_edit_rate": rate_with_ci([edit(r) for r in rows]),
        "adversarial_usage_rate": rate_with_ci([aur(r) for r in rows]),
        "aur_goal_adjacent": slice_ci(adj, aur),
        "aur_goal_distant": slice_ci(dist, aur),
        "any_edit_goal_adjacent": slice_ci(adj, edit),
        "any_edit_goal_distant": slice_ci(dist, edit),
    }
    if rows[0]["retrieval_methods"]:
        for mkey in rows[0]["retrieval_methods"]:
            metrics[f"rr_{mkey}_goal_adjacent"] = slice_ci(
                adj, lambda r, mk=mkey: r["retrieval_methods"][mk]["retrieval_rate"]
            )
            metrics[f"rr_{mkey}_goal_distant"] = slice_ci(
                dist, lambda r, mk=mkey: r["retrieval_methods"][mk]["retrieval_rate"]
            )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "stage3_agent_action_sandboxed_tool_use",
        "start_index": args.start,
        "model": args.model,
        "judge_model": args.judge_model,
        "retrieval": args.retrieval,
        "manager_model": manager_model,
        "embedding_model": EMBEDDING_MODEL if embed_fn else None,
        "top_k": list(top_k),
        "dataset": "agent_action.json",
        "selected_rows": len(selected),
        "metrics": metrics,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"agent_action_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    def _summarize(v):
        if isinstance(v, dict) and "pretty" in v:
            return f"{v['pretty']}  (n={v['n']})" if v["n"] else "no cases in this slice"
        return v

    print(json.dumps({k: _summarize(v) for k, v in metrics.items()}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
