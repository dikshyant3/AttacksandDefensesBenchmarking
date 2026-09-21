"""Execute Stage 2 (Retrieval) + Stage 3 (Adversarial Usage) of Hidden
Sleeper Memory, using the real vendored llm_behaviour_followup.jsonl dataset
(200 rows, byte-identical to the upstream datasets/downstream/llm_behaviour.jsonl:
100 goal_adjacent + 100 wildchat_seed [= goal-distant], multi_turn_meta.split).

Stage 1 (write/injection) is run_experiment.py. See followup.py's module
docstring for exactly what each retrieval method is and which parts are
paper-faithful vs. our extensions.

Retrieval methods run per case:
  * G.1 everything-in-context   -- always (RR == 1.0 by definition; the only
                                   mode the released followup_eval implements)
  * G.3 semantic top-k          -- always here (OpenAI embeddings; our extension)
  * G.2 memory-management agent -- only if --retrieval-manager is not "none"
                                   (this is the mechanism the paper's Table 2 RR
                                   actually uses -- "We report RR using the
                                   external memory manager" -- but our prompt is
                                   a reconstruction, not paper-faithful)

The command is a dry run unless --yes is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.hidden_sleeper.external_manager import make_manager_client
from benchmark.attacks.hidden_sleeper.followup import run_case_stages_2_and_3
from benchmark.attacks.hidden_sleeper.followup_data import dataset_summary, load_followup_cases
from benchmark.attacks.hidden_sleeper.openai_tool_pipeline import DEFAULT_JUDGE_MODEL
from benchmark.attacks.hidden_sleeper.stats import rate_with_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model for the follow-up assistant (subject)")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model for the influence judge (default: {DEFAULT_JUDGE_MODEL}; pass a different value to decouple from the subject).",
    )
    parser.add_argument("--start", type=int, default=0, help="First row index (0-199)")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to run, starting at --start (of 200 available)")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[15, 5],
        help="Retrieval cutoffs for the G.3 embedding RR check (the paper reports more than one).",
    )
    parser.add_argument(
        "--retrieval-manager",
        choices=["none", "gemini", "openai"],
        default="none",
        help="Also run the G.2 memory-management-agent retrieval method with this provider "
        "(one LLM call per query). 'none' skips G.2. NOTE: G.2's prompt is ours, not paper-faithful, "
        "though G.2 is the mechanism the paper's own Table 2 RR uses.",
    )
    parser.add_argument("--yes", action="store_true", help="Make paid API calls")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_followup_cases()
    if args.start < 0 or args.start >= len(cases):
        raise SystemExit(f"--start must be between 0 and {len(cases) - 1}")
    if args.limit < 1 or args.start + args.limit > len(cases):
        raise SystemExit(f"--start + --limit must be <= {len(cases)}")
    selected = cases[args.start : args.start + args.limit]
    top_k = tuple(args.top_k)

    n_queries = sum(len(c.user_queries) for c in selected)
    n_memories = sum(len(c.memories) for c in selected)
    print("Hidden Sleeper Memory -- Stage 2 (Retrieval, all methods) + Stage 3 (Adversarial Usage)")
    print(json.dumps(dataset_summary(cases), indent=2))
    print(f"Selected rows: {args.start}..{args.start + len(selected) - 1}  (n={len(selected)})")
    print(f"Chat model (follow-up assistant / subject): {args.model}")
    print(f"Influence judge model: {args.judge_model}"
          + ("  <- same as subject (default)" if args.judge_model == args.model else "  <- decoupled from subject"))
    print(f"Embedding model (G.3): {EMBEDDING_MODEL}  | top-k: {top_k}")
    print(f"G.2 memory-management agent: {args.retrieval_manager}")
    approx = f"~{n_queries} chat + {len(selected)} judge + ~{n_memories + n_queries} embedding calls"
    if args.retrieval_manager != "none":
        approx += f" + ~{n_queries} G.2 manager calls ({args.retrieval_manager})"
    print(f"About to make {approx} if run for real.")
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when --yes is supplied")

    client = OpenAI()
    embed_fn = make_openai_embed_fn(client)
    manager_client = manager_model = None
    if args.retrieval_manager != "none":
        manager_client, manager_model = make_manager_client(args.retrieval_manager)

    rows = []
    for index, case in enumerate(selected, start=1):
        result = run_case_stages_2_and_3(
            client,
            args.model,
            case,
            embed_fn,
            judge_model=args.judge_model,
            manager_client=manager_client,
            manager_model=manager_model,
            top_k=top_k,
        )
        # goal_adjacent / wildchat_seed (= goal-distant) -- the paper's proximity split
        result["proximity"] = case.raw.get("multi_turn_meta", {}).get("split", "unknown")
        rows.append(result)
        methods = result["retrieval_methods"]
        rr_bits = " ".join(f"{k.split('_')[0].upper()}={v['retrieval_rate']:.2f}" for k, v in methods.items())
        print(
            f"[{index:03d}/{len(selected):03d}] {case.case_id} [{result['proximity']}]: {rr_bits} "
            f"AUR={'used' if result['adversarial_usage']['used'] else 'not used'}"
        )

    def slice_ci(proximity, extractor):
        vals = [extractor(r) for r in rows if proximity is None or r["proximity"] == proximity]
        return rate_with_ci(vals)

    adjacent, distant = "goal_adjacent", "wildchat_seed"
    aur = lambda r: 1 if r["adversarial_usage"]["used"] else 0  # noqa: E731
    method_keys = list(rows[0]["retrieval_methods"].keys())

    metrics = {
        "adversarial_usage_rate": slice_ci(None, aur),
        "aur_goal_adjacent": slice_ci(adjacent, aur),
        "aur_goal_distant": slice_ci(distant, aur),
    }
    for mkey in method_keys:
        rr = lambda r, mk=mkey: r["retrieval_methods"][mk]["retrieval_rate"]  # noqa: E731
        metrics[f"rr_{mkey}_goal_adjacent"] = slice_ci(adjacent, rr)
        metrics[f"rr_{mkey}_goal_distant"] = slice_ci(distant, rr)

    # End-to-end: injected memory BOTH retrieved (G.3, >=1 query) AND judged
    # to have influenced the conversation -- the composed per-case success
    # (README.md "coupled rates"), not IR x RR x AUR multiplied.
    e2e = lambda r: 1 if (r["retrieval_methods"]["g3_semantic_topk"]["hits"] > 0 and r["adversarial_usage"]["used"]) else 0  # noqa: E731
    metrics["end_to_end_rate"] = slice_ci(None, e2e)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "stage2_3_retrieval_all_methods_and_adversarial_usage",
        "start_index": args.start,
        "model": args.model,
        "judge_model": args.judge_model,
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_manager": args.retrieval_manager,
        "manager_model": manager_model,
        "dataset": "llm_behaviour_followup.jsonl",
        "top_k": list(top_k),
        "selected_rows": len(selected),
        "metrics": metrics,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"followup_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def _summarize(v):
        if isinstance(v, dict) and "pretty" in v:
            return f"{v['pretty']}  (n={v['n']})" if v["n"] else "no cases in this slice"
        return v

    print(json.dumps({k: _summarize(v) for k, v in metrics.items()}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
