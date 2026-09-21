"""Execute the Zombie Agents attack against a retrieval-augmented-memory
(RAG) web agent (paper Section 3.3, Scenario B; Section 4.1).

For each bait task: run Phase I (Exposure, K=300 rounds of the bait task by
default -- Section 4.1's real, paper-verbatim mechanism for spreading the
payload across embedding space; see --exposure-rounds), then replay
`--trigger-rounds` real insta-150k-v1 queries as Phase II (trigger) sessions,
measuring ASR, Injection Count, and Recall@k. Section 3.3's "Semantic
Aliasing" prose is named in the paper but given no separate template,
algorithm, or citation, so it is not reconstructed as its own step here --
see payloads.py.

The command is a dry run unless --yes is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.hidden_sleeper.stats import rate_with_ci
from benchmark.attacks.zombie_agents.campaign import DEFAULT_RAG_EXPOSURE_ROUNDS, DEFAULT_TRIGGER_ROUNDS, run_rag_campaign
from benchmark.attacks.zombie_agents.data import (
    dataset_summary,
    load_bait_tasks,
    load_filler_entries,
    load_trigger_queries,
)
from benchmark.attacks.zombie_agents.local_embeddings import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    load_sentence_transformer,
    make_local_embed_batch_fn,
    make_local_embed_fn,
)
from benchmark.attacks.zombie_agents.memory import build_filler_entries
from benchmark.attacks.zombie_agents.models import CLOSEST_AVAILABLE_GEMINI, PAPER_AGENT_MODELS, make_agent_client

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


def make_openai_embed_batch_fn(client, model: str = EMBEDDING_MODEL):
    def embed_batch(texts: list[str]):
        import numpy as np

        response = client.embeddings.create(model=model, input=texts)
        return [np.array(d.embedding) for d in response.data]

    return embed_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help=f"Web-agent LLM provider. 'openai' (default) is this project's standard substitute "
        f"(gpt-4o-mini). 'gemini' is the paper's actual model ({PAPER_AGENT_MODELS['gemini']}), "
        f"needs GEMINI_API_KEY. The paper's other model, GLM-4.7-Flash, has no configured access here. "
        f"Embeddings always use OpenAI regardless of this choice.",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default agent model id")
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "sentence-transformers"],
        default="openai",
        help="Embedding backend for RAG retrieval. 'openai' (default) is text-embedding-3-small, this "
        "project's consistent choice. 'sentence-transformers' runs a real HuggingFace model locally "
        f"(default {DEFAULT_SENTENCE_TRANSFORMER_MODEL!r}) -- no API key, no per-call cost -- to test "
        "whether embedding-model choice explains part of our Recall@k gap vs. the paper (which names "
        "no embedding model either).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding backend's default model id (e.g. a different sentence-transformers model name).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k actually shown to the agent (the 'real' k)")
    parser.add_argument(
        "--recall-k-sweep",
        type=int,
        nargs="*",
        default=[],
        help="Additionally report Recall@k for these k values (e.g. 5 10 20 50), computed from the same "
        "ranking pass as --top-k at no extra API cost -- use this to see how sensitive Recall@k is to k "
        "without paying for a separate run per k.",
    )
    parser.add_argument("--trigger-rounds", type=int, default=DEFAULT_TRIGGER_ROUNDS, help="Trigger rounds M per bait task (paper default 20)")
    parser.add_argument(
        "--exposure-rounds",
        type=int,
        default=DEFAULT_RAG_EXPOSURE_ROUNDS,
        help="Exposure Phase rounds K -- how many times Phase I repeats the bait task before Phase II begins "
        "(paper default 300, Section 4.1: 'to simulate long-term memory pollution... we set K = 300'). This is "
        "the paper's actual mechanism for spreading the payload across embedding space -- K distinct natural "
        "exposure sessions, not a separate hand-crafted aliasing step. Costs K extra chat + embedding calls per "
        "bait task, so this is by far the most expensive knob here -- lower it for a quick/cheap smoke test.",
    )
    parser.add_argument("--num-bait-tasks", type=int, default=5, help="How many of the 20 bait tasks to run (each is a separate campaign)")
    parser.add_argument(
        "--evolve-strategy",
        choices=["raw_history", "verbal_reflection", "refined_experience"],
        default="raw_history",
        help="Memory-write compression strategy (Section 4.2). raw_history is paper-described and needs no extra "
        "LLM call. verbal_reflection uses a Reflexion-grounded (Shinn et al. 2023), adapted-not-verbatim prompt; "
        "refined_experience uses an uncited RECONSTRUCTED prompt (see memory.py). Both cost one extra call per commit.",
    )
    parser.add_argument("--defense", choices=["none", "sandwich", "spotlight"], default="none", help="Prompt-level defense (Figures 10/12)")
    parser.add_argument(
        "--filler-pool-size",
        type=int,
        default=2735,
        help="Pre-seed the RAG database with this many real insta-150k-v1 filler entries (max 2735) before each "
        "campaign, approximating the paper's ~3,000-entry 'long-term memory pollution' scale (Section 4.1). "
        "0 disables seeding (Recall@k is then measured against an almost-empty store, NOT paper-comparable). "
        "Embedded once, in batches, and reused across all bait tasks.",
    )
    parser.add_argument("--yes", action="store_true", help="Make paid API calls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bait_tasks = load_bait_tasks()[: args.num_bait_tasks]
    trigger_queries = load_trigger_queries()
    defense = None if args.defense == "none" else args.defense

    print("Zombie Agents -- retrieval-augmented (RAG) memory")
    print(json.dumps(dataset_summary(), indent=2))
    print(f"Bait tasks: {len(bait_tasks)}  |  Exposure rounds K: {args.exposure_rounds}  |  Trigger rounds M: {args.trigger_rounds}  |  top-k: {args.top_k}")
    intended_model = args.model or (CLOSEST_AVAILABLE_GEMINI if args.provider == "gemini" else "gpt-4o-mini")
    is_paper_model = args.provider == "gemini" and intended_model == PAPER_AGENT_MODELS["gemini"]
    print(f"Provider: {args.provider}  |  Model: {intended_model}" + ("  <- the paper's actual model" if is_paper_model else "  <- substitute"))
    print(f"Evolve strategy: {args.evolve_strategy}  |  Defense: {args.defense}")
    embed_model_name = args.embedding_model or (DEFAULT_SENTENCE_TRANSFORMER_MODEL if args.embedding_provider == "sentence-transformers" else EMBEDDING_MODEL)
    print(f"Embeddings: {args.embedding_provider} ({embed_model_name})" + ("  <- local, free, no API key" if args.embedding_provider == "sentence-transformers" else ""))
    print(f"Filler pool: {args.filler_pool_size} real entries (embedded once, batched, shared across bait tasks)")
    n_commits_per_campaign = args.exposure_rounds + args.trigger_rounds
    extra_evolve_calls = f" + {n_commits_per_campaign} evolve calls/task ({args.evolve_strategy})" if args.evolve_strategy != "raw_history" else ""
    total_chat = len(bait_tasks) * (args.exposure_rounds + args.trigger_rounds)
    filler_batches = -(-min(args.filler_pool_size, 2735) // 500)  # ceil div, batch_size=500
    print(
        f"About to make ~{total_chat} chat calls + embedding calls per commit/query + "
        f"{filler_batches} batched filler-embedding calls{extra_evolve_calls} if run for real."
    )
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client, model = make_agent_client(args.provider, args.model)
    print(f"Model: {model}")

    # Embeddings: OpenAI (API, paid) or sentence-transformers (local, free) --
    # independent of --provider (the paper names no embedding model either;
    # this project's consistent choice has been OpenAI, sentence-transformers
    # is here to test whether embedding-model choice explains part of the
    # Recall@k gap vs the paper -- FIDELITY.md's hypothesis #1).
    if args.embedding_provider == "sentence-transformers":
        st_model = load_sentence_transformer(embed_model_name)  # loaded once, shared below
        embed_fn = make_local_embed_fn(embed_model_name, cache={}, model=st_model)
        embed_batch_fn_factory = lambda: make_local_embed_batch_fn(embed_model_name, model=st_model)  # noqa: E731
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required for OpenAI embeddings (pass --embedding-provider sentence-transformers to avoid this)")
        embed_client = OpenAI()
        embed_fn = make_openai_embed_fn(embed_client, model=embed_model_name)
        embed_batch_fn_factory = lambda: make_openai_embed_batch_fn(embed_client, model=embed_model_name)  # noqa: E731
    evolve_client = client if args.evolve_strategy != "raw_history" else None
    evolve_model = model if args.evolve_strategy != "raw_history" else None

    filler_entries = []
    if args.filler_pool_size > 0:
        filler_texts = load_filler_entries()[: args.filler_pool_size]
        filler_entries = build_filler_entries(embed_batch_fn_factory(), filler_texts)
        print(f"Filler pool embedded: {len(filler_entries)} entries.")

    campaigns = []
    for index, bait in enumerate(bait_tasks, start=1):
        result = run_rag_campaign(
            client,
            model,
            embed_fn,
            bait,
            trigger_queries,
            top_k=args.top_k,
            max_trigger_rounds=args.trigger_rounds,
            exposure_rounds=args.exposure_rounds,
            evolve_strategy=args.evolve_strategy,
            evolve_client=evolve_client,
            evolve_model=evolve_model,
            defense=defense,
            filler_entries=filler_entries,
            recall_k_sweep=tuple(args.recall_k_sweep),
        )
        campaigns.append(result)
        sweep_bits = " ".join(f"R@{k}={result.recall_at_k_rate() if k == args.top_k else sum(1 for r in result.trigger_rounds if r['recall_by_k'][k]) / len(result.trigger_rounds):.2f}" for k in sorted({args.top_k, *args.recall_k_sweep})) if result.trigger_rounds else ""
        print(
            f"[{index:02d}/{len(bait_tasks)}] {bait.task_id}: infected={int(result.infected)} "
            f"injection_count={result.injection_count} ASR={result.asr:.2f} {sweep_bits}"
        )

    all_asr_bits = [1 if r["executed_malicious"] else 0 for c in campaigns for r in c.trigger_rounds]
    all_recall_bits = [1 if r["recalled"] else 0 for c in campaigns for r in c.trigger_rounds]
    infection_bits = [1 if c.infected else 0 for c in campaigns]
    injection_counts = [c.injection_count for c in campaigns]

    metrics = {
        "infection_rate": rate_with_ci(infection_bits),
        "attack_success_rate": rate_with_ci(all_asr_bits),
        f"recall_at_{args.top_k}": rate_with_ci(all_recall_bits),
        "mean_injection_count": {"n": len(injection_counts), "mean": sum(injection_counts) / len(injection_counts) if injection_counts else 0},
    }
    for k in args.recall_k_sweep:
        if k == args.top_k:
            continue
        bits = [1 if r["recall_by_k"][k] else 0 for c in campaigns for r in c.trigger_rounds]
        metrics[f"recall_at_{k}"] = rate_with_ci(bits)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "zombie_agents_rag",
        "provider": args.provider,
        "model": model,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": args.top_k,
        "recall_k_sweep": args.recall_k_sweep,
        "evolve_strategy": args.evolve_strategy,
        "defense": args.defense,
        "filler_pool_size": len(filler_entries),
        "exposure_rounds": args.exposure_rounds,
        "trigger_rounds_per_task": args.trigger_rounds,
        "num_bait_tasks": len(bait_tasks),
        "metrics": metrics,
        "campaigns": [asdict(c) for c in campaigns],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rag_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def _pretty(v):
        return v.get("pretty", v.get("mean")) if isinstance(v, dict) else v

    print(json.dumps({k: _pretty(v) for k, v in metrics.items()}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
