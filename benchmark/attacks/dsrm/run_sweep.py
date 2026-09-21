"""One-at-a-time config sweeps around a base configuration -- the paper's
sensitivity studies: top-k (Fig. 4), reasoning length L (Fig. 5), similarity
metric (Table 10), retriever (Tables 2/4), decision-generation LLM (Table 11),
plus the SRM threshold tau, SRM MaxIter, agent backbone, and background-KB
size. Module ablations (Table 9) are not a sweep dimension -- they are
campaign methods: --methods ori_attack dsrm_no_srm dsrm_no_csrm dsrm.

    python -m benchmark.attacks.dsrm.run_sweep --sweep top_k=1,3,5,10 --sweep metric=ip,cos,l2

Each --sweep varies ONE parameter while all others stay at the base config.

DRY RUN unless --yes. Real calls go through llm_cache.CachedChatClient, so a
sweep only pays for prompts it has not sent before: sweeping top_k / metric /
retriever / L / background_size leaves the attacker's decision-construction
prompts unchanged (all cache hits), and only agent calls whose retrieved
context actually changed are new. The run report records exactly how many
real calls were made.

NOT swept (needs a decision from the user, see FIDELITY.md): Table 5's memory
dilution to 1000 entries (only 41 real benign entries exist) and Table 8's
random seeds (temperature-0 + cache makes a seed a no-op here).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.dsrm.campaign import (
    DSRM_STAGES,
    AttackMethod,
    build_background_kb,
    build_scenarios,
    run_campaign,
)
from benchmark.attacks.dsrm.decision import (
    DEFAULT_DECISION_MODEL,
    DEFAULT_INITIAL_DECISION_RETRIES,
    DEFAULT_MAX_REFINE_ITERS,
    DEFAULT_REASONING_LENGTH_WORDS,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from benchmark.attacks.dsrm.llm_cache import CachedChatClient
from benchmark.attacks.dsrm.run_experiment import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CACHE_PATH,
    add_memory_arguments,
    load_background,
    make_embedder,
    sample_scenarios_round_robin,
    serialize_result,
)

SWEEPABLE: dict[str, type] = {
    "top_k": int,
    "metric": str,
    "retriever": str,
    "reasoning_length_words": int,
    "threshold": float,
    "max_refine_iters": int,
    "decision_model": str,
    "agent_model": str,
    "background_size": int,
}
_CHOICES = {"metric": {"ip", "cos", "l2"}, "retriever": {"minilm", "dpr", "realm"}}


def parse_sweep(spec: str) -> tuple[str, list]:
    """'top_k=1,3,5' -> ('top_k', [1, 3, 5]). Raises ValueError on a bad spec."""
    name, sep, raw_values = spec.partition("=")
    name = name.strip()
    if not sep or name not in SWEEPABLE:
        raise ValueError(f"bad --sweep {spec!r}: expected NAME=v1,v2,... with NAME in {sorted(SWEEPABLE)}")
    values = [SWEEPABLE[name](v.strip()) for v in raw_values.split(",") if v.strip()]
    if not values:
        raise ValueError(f"--sweep {spec!r} has no values")
    if name in _CHOICES:
        bad = [v for v in values if v not in _CHOICES[name]]
        if bad:
            raise ValueError(f"--sweep {name}: {bad} not in {sorted(_CHOICES[name])}")
    return name, values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="append", required=True, metavar="NAME=v1,v2,...", help=f"parameter to sweep; NAME in {sorted(SWEEPABLE)}. Repeatable.")
    p.add_argument("--n-scenarios", type=int, default=10, help="Scenarios per point, round-robin across domains (default 10)")
    p.add_argument("--methods", nargs="+", choices=[m.value for m in AttackMethod], default=["none", "naive", "dsrm"])
    p.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    p.add_argument("--decision-model", default=DEFAULT_DECISION_MODEL)
    p.add_argument("--retriever", choices=sorted(_CHOICES["retriever"]), default="dpr")
    p.add_argument("--attacker-embedder", choices=["minilm", "same"], default="minilm")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--metric", choices=sorted(_CHOICES["metric"]), default="ip")
    p.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    p.add_argument("--max-refine-iters", type=int, default=DEFAULT_MAX_REFINE_ITERS)
    p.add_argument("--reasoning-length-words", type=int, default=DEFAULT_REASONING_LENGTH_WORDS)
    p.add_argument("--initial-decision-retries", type=int, default=DEFAULT_INITIAL_DECISION_RETRIES)
    p.add_argument("--background-size", type=int, default=None)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--no-cache", action="store_true")
    add_memory_arguments(p)
    p.add_argument("--yes", action="store_true", help="Make real, paid API calls")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return p.parse_args()


def base_config(args: argparse.Namespace) -> dict:
    return {name: getattr(args, name) for name in SWEEPABLE}


def sweep_points(base: dict, sweeps: list[tuple[str, list]]) -> list[tuple[str, object, dict]]:
    """Flatten to (sweep_name, value, full_config) rows: one row per value,
    every other parameter held at its base value."""
    return [(name, value, {**base, name: value}) for name, values in sweeps for value in values]


def main() -> None:
    args = parse_args()
    try:
        sweeps = [parse_sweep(s) for s in args.sweep]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    methods = [AttackMethod(m) for m in args.methods]
    base = base_config(args)
    points = sweep_points(base, sweeps)
    scenarios = sample_scenarios_round_robin(build_scenarios(), args.n_scenarios)
    n = len(scenarios)

    print(f"Sweeps: {', '.join(f'{name}={values}' for name, values in sweeps)}")
    print(f"Base config: {base}")
    print(f"{len(points)} sweep points x {len(methods)} methods x {n} scenarios = {len(points) * len(methods) * n} scenario runs")
    decision_methods = [m for m in methods if m in DSRM_STAGES]
    print(
        f"Upper bound on real calls: {len(points) * len(methods) * n} agent calls"
        + (f" + up to {len(points) * len(decision_methods) * n * (args.initial_decision_retries + args.max_refine_iters + 2)} decision calls" if decision_methods else "")
        + " -- but cache hits make none, and retrieval-only sweeps reuse every decision."
    )
    if not args.yes:
        print("\nDRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    raw = OpenAI()
    client = raw if args.no_cache else CachedChatClient(raw, args.cache_path)

    embedders: dict[str, object] = {}
    kbs: dict[tuple, object] = {}

    def embedder_for(name: str):
        if name not in embedders:
            embedders[name] = make_embedder(name)
        return embedders[name]

    rows = []
    for name, value, cfg in points:
        kb_key = (cfg["retriever"], cfg["background_size"])
        if kb_key not in kbs:
            kbs[kb_key] = build_background_kb(
                embedder_for(cfg["retriever"]),
                max_entries=cfg["background_size"],
                memory_format=args.memory_format,
                workflows=load_background(args),
            )
        attacker = embedder_for(cfg["retriever"] if args.attacker_embedder == "same" else "minilm")
        print(f"\n--- {name}={value} ---")
        row = {"sweep": name, "value": value, "config": cfg, "methods": {}}
        for method in methods:
            results, metrics = run_campaign(
                scenarios,
                method,
                kbs[kb_key],
                agent_client=client,
                agent_model=cfg["agent_model"],
                decision_client=client,
                decision_model=cfg["decision_model"],
                k=cfg["top_k"],
                metric=cfg["metric"],
                attacker_embedder=attacker,
                memory_format=args.memory_format,
                execute=args.execute,
                agent_style=args.agent_style,
                asb_memory_entries=args.asb_memory_entries,
                threshold=cfg["threshold"],
                max_refine_iters=cfg["max_refine_iters"],
                reasoning_length_words=cfg["reasoning_length_words"],
                initial_decision_retries=args.initial_decision_retries,
            )
            asr_r = "-" if metrics.asr_r is None else f"{metrics.asr_r * 100:.0f}%"
            exec_part = f"  ASR_A(exec)={metrics.asr_a_exec * 100:5.1f}%" if metrics.asr_a_exec is not None else ""
            print(f"  {method.value:13s} ASR_A(plan)={metrics.asr_a * 100:5.1f}%{exec_part}  RR={metrics.rr * 100:5.1f}%  ASR_R={asr_r}")
            row["methods"][method.value] = {
                "n": metrics.n,
                "asr_a": metrics.asr_a,
                "asr_r": metrics.asr_r,
                "rr": metrics.rr,
                "asr_a_exec": metrics.asr_a_exec,
                "asr_r_exec": metrics.asr_r_exec,
                "results": [serialize_result(r) for r in results],
            }
        rows.append(row)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "dsrm_sweep",
        "n_scenarios": n,
        "methods": [m.value for m in methods],
        "base_config": base,
        "attacker_embedder": args.attacker_embedder,
        "memory_format": args.memory_format,
        "background": args.background,
        "execute": args.execute,
        "llm_cache": None if args.no_cache else {"path": str(args.cache_path), "hits": client.hits, "real_calls_made": client.misses},
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"sweep_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not args.no_cache:
        print(f"\nReal API calls made: {client.misses}  |  cache hits: {client.hits}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
