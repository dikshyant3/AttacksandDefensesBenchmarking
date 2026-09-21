"""Stealth evaluation (Section 5.3): can perplexity or an LLM tell poisoned
entries from benign ones?

Perplexity detection (Fig. 3, Table 6's FPR/FNR) is LOCAL: entries are built
from the cached decisions (offline, no API calls), retrieved, and the retrieved
plans are scored with GPT-2. LLM-based detection (Table 6) calls a model once per
distinct entry -- it runs only with --llm-detect --yes.

  python -m benchmark.attacks.dsrm.run_defenses --n-scenarios 25 --memory-format asb --background real
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.dsrm.campaign import AttackMethod, build_background_kb, build_scenarios, run_campaign
from benchmark.attacks.dsrm.defenses import (
    DEFAULT_THRESHOLDS,
    LogPerplexityScorer,
    collect_detection_samples,
    detection_rates,
    llm_detect,
    roc_auc,
    threshold_sweep,
)
from benchmark.attacks.dsrm.llm_cache import CachedChatClient
from benchmark.attacks.dsrm.run_experiment import (
    DEFAULT_CACHE_PATH,
    WHITEBOX_METHODS,
    add_memory_arguments,
    add_whitebox_arguments,
    load_background,
    make_embedder,
    make_whitebox_config,
    sample_scenarios_round_robin,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-scenarios", type=int, default=25)
    p.add_argument("--methods", nargs="+", choices=[m.value for m in AttackMethod], default=["dsrm"])
    p.add_argument("--retriever", choices=["minilm", "dpr"], default="dpr")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--scorer-model", default="gpt2", help="causal LM for log-perplexity (the paper doesn't name one)")
    p.add_argument("--llm-detect", action="store_true", help="also run ASB's LLM-based detector (real API calls; needs --yes)")
    p.add_argument("--detect-model", default="gpt-4o-mini")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--yes", action="store_true", help="allow the real LLM-detector calls")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    add_memory_arguments(p)
    add_whitebox_arguments(p)
    args = p.parse_args()
    args.offline = True  # entries always come from the cache here
    return args


def main() -> None:
    args = parse_args()
    scenarios = sample_scenarios_round_robin(build_scenarios(), args.n_scenarios)
    methods = [AttackMethod(m) for m in args.methods]
    print(f"Detection evaluation: methods={[m.value for m in methods]}  n={len(scenarios)}  format={args.memory_format}  background={args.background}")
    if args.llm_detect and not args.yes:
        print("The LLM detector would make ~1 real call per distinct entry (benign entries + one per planted entry). Add --yes to run it.")
        args.llm_detect = False

    offline = CachedChatClient(None, args.cache_path)  # any cache miss raises -- decisions are never regenerated here
    embedder = make_embedder(args.retriever)
    whitebox = make_whitebox_config(args, embedder) if any(m in WHITEBOX_METHODS for m in methods) else None
    kb = build_background_kb(embedder, memory_format=args.memory_format, workflows=load_background(args))
    print("Loading the perplexity scorer...")
    scorer = LogPerplexityScorer(args.scorer_model)
    detector_client = None
    if args.llm_detect:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        detector_client = CachedChatClient(OpenAI(), args.cache_path)

    report = {"timestamp": datetime.now(UTC).isoformat(), "experiment": "dsrm_detection", "n_scenarios": len(scenarios), "scorer": args.scorer_model, "methods": {}}
    for method in methods:
        results, metrics = run_campaign(
            scenarios, method, kb, agent_client=offline, agent_model="unused", decision_client=offline,
            k=args.top_k, memory_format=args.memory_format, whitebox=whitebox, retrieval_only=True,
        )
        samples = collect_detection_samples(results)
        pos = [scorer.score(s.text) for s in samples if s.poisoned]
        neg = [scorer.score(s.text) for s in samples if not s.poisoned]
        print(f"\n=== {method.value}: {len(pos)} poisoned + {len(neg)} distinct benign retrieved entries (RR={metrics.rr * 100:.0f}%) ===")
        if not pos:
            print("  no poisoned entry was retrieved -- nothing to detect")
            continue
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        auc = roc_auc(pos, neg)
        print(f"  mean log-perplexity: poisoned {mean(pos):.2f} | benign {mean(neg):.2f}   ROC AUC = {auc:.2f}  (0.50 = chance; paper: 0.49)")
        rows = threshold_sweep(pos, neg, DEFAULT_THRESHOLDS)
        print("  threshold   FPR    FNR")
        for r in rows:
            print(f"    {r['threshold']:.1f}     {r['fpr']:.2f}   {r['fnr']:.2f}")
        entry = {"n_poisoned": len(pos), "n_benign": len(neg), "rr": metrics.rr, "mean_logppl_poisoned": mean(pos), "mean_logppl_benign": mean(neg), "auc": auc, "sweep": rows}
        if detector_client is not None:
            flags_pos = llm_detect(detector_client, args.detect_model, [s.text for s in samples if s.poisoned])
            flags_neg = llm_detect(detector_client, args.detect_model, [s.text for s in samples if not s.poisoned])
            entry["llm_detection"] = detection_rates(flags_pos, flags_neg)
            print(f"  LLM detector ({args.detect_model}): {entry['llm_detection']}  (paper Table 6 averages FNR 0.804 / FPR 0.088)")
        report["methods"][method.value] = entry

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"detection_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if detector_client is not None:
        print(f"\nReal LLM-detector calls: {detector_client.misses} (cache hits {detector_client.hits})")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
