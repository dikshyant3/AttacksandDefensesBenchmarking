"""Memory dilution (Table 5): poison the KB, add N benign entries, re-measure.

The paper adds 100 / 200 / 500 / 1000 benign "user interactions" and finds ASR_A
declines only gently (GPT-4o-mini: 36.25 -> 34.25). Our filler is real but
UNRELATED web tasks (see dilution.py), so this tests dilution by volume, not by
topical competition.

--retrieval-only measures only RR and, with --offline, costs nothing (DSRM
decisions come from the cache). Otherwise the agent runs and ASR_A is reported
(real calls; needs --yes).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.dsrm.campaign import AttackMethod, build_background_kb, build_scenarios, run_campaign
from benchmark.attacks.dsrm.dilution import filler_entries
from benchmark.attacks.dsrm.run_experiment import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CACHE_PATH,
    add_memory_arguments,
    load_background,
    make_chat_client,
    make_embedder,
    sample_scenarios_round_robin,
)
from benchmark.attacks.dsrm.decision import DEFAULT_DECISION_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[0, 100, 200, 500, 1000], help="benign entries added (Table 5: 0 100 200 500 1000)")
    p.add_argument("--n-scenarios", type=int, default=25)
    p.add_argument("--methods", nargs="+", choices=[m.value for m in AttackMethod], default=["dsrm"])
    p.add_argument("--retriever", choices=["minilm", "dpr"], default="dpr")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    p.add_argument("--decision-model", default=DEFAULT_DECISION_MODEL)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--offline", action="store_true", help="cache only; never calls an API (no --yes needed)")
    p.add_argument("--retrieval-only", action="store_true", help="measure RR only; skip the agent")
    p.add_argument("--yes", action="store_true", help="make real, paid API calls")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    add_memory_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = sample_scenarios_round_robin(build_scenarios(), args.n_scenarios)
    methods = [AttackMethod(m) for m in args.methods]
    free = args.offline and args.retrieval_only
    print(f"Dilution: sizes={args.sizes}  methods={[m.value for m in methods]}  n={len(scenarios)}  " + ("(free: offline + retrieval-only)" if free else ""))
    if not (args.offline or args.yes):
        agent_calls = "0" if args.retrieval_only else f"~{len(scenarios) * len(methods) * len(args.sizes)}"
        print(f"DRY RUN: would make up to {agent_calls} agent calls (+ decision calls not already cached). Add --yes, or use --offline --retrieval-only for the free RR study.")
        return

    client = make_chat_client(args)
    embedder = make_embedder(args.retriever)
    workflows = load_background(args)
    rows = []
    for size in args.sizes:
        kb = build_background_kb(embedder, memory_format=args.memory_format, workflows=workflows, extra_entries=filler_entries(size, args.memory_format))
        line = f"  +{size:>4} benign (KB={len(kb):>4}):"
        row = {"added": size, "kb_size": len(kb), "methods": {}}
        for method in methods:
            _, m = run_campaign(
                scenarios, method, kb, agent_client=client, agent_model=args.agent_model, decision_client=client,
                decision_model=args.decision_model, k=args.top_k, memory_format=args.memory_format,
                execute=args.execute and not args.retrieval_only, retrieval_only=args.retrieval_only,
            )
            row["methods"][method.value] = {"rr": m.rr, "asr_a": None if args.retrieval_only else m.asr_a, "asr_a_exec": m.asr_a_exec}
            line += f"  {method.value}: RR={m.rr * 100:.0f}%" + ("" if args.retrieval_only else f" ASR_A={m.asr_a * 100:.0f}%")
        print(line)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"dilution_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "n_scenarios": len(scenarios), "rows": rows, "llm_cache": {"hits": getattr(client, "hits", None), "real_calls_made": getattr(client, "misses", None)}}, indent=2, default=str), encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
