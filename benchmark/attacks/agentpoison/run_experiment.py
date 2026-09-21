"""Real-LLM AgentPoison experiment run, ReAct-StrategyQA target (Chen et al.,
arXiv:2407.12784, NeurIPS 2024).

benchmark/tests/test_agentpoison.py is deterministic and offline (a scripted
LLM stand-in) so the test suite verifies plumbing -- KB injection, the
embedding/content split, search[]-argument-override-by-current_context,
lookup[]-is-a-no-op, clean_answer parsing, ASR-r/ASR-a/ASR-t/ACC formulas, and
the HotFlip trigger optimizer's monotonic-improvement invariant -- for free and
reproducibly, using a real (local, CPU) DPR embedder throughout but a scripted
LLM instead of a real chat model. It is NOT evidence that a real target LLM is
actually persuadable by a retrieved backdoor instruction; that can only be
shown by calling one. This script does that: real DPR retrieval + real HotFlip
trigger optimization (unchanged from the deterministic path) + a real OpenAI
chat model as the ReAct agent's backbone.

Two-stage cost, mirroring their own two-command workflow:
  1. Trigger optimization -- no LLM calls, only local embedder forward/backward
     passes (CPU-bound, not billed). Slow at paper-scale settings
     (num_iter=1000 etc.); this script's defaults are much smaller for
     tractability -- see trigger_optimization.py's module docstring. Pass
     --paper-scale for their literal defaults (expect this stage alone to take
     hours on CPU).
  2. ReAct episodes -- each one is up to MAX_STEPS=7 real chat completions.
     Run twice per their README: once with the trigger (--task-type adv) for
     ASR-r/ASR-a/ASR-t, once without (--task-type benign) for ACC.

Requires OPENAI_API_KEY in the environment (loaded from a .env file at the
project root if present).

Usage:
    python -m benchmark.attacks.agentpoison.run_experiment --task-type adv
    python -m benchmark.attacks.agentpoison.run_experiment --task-type benign
    python -m benchmark.attacks.agentpoison.run_experiment --task-type adv --model gpt-4o --num-questions 10
    python -m benchmark.attacks.agentpoison.run_experiment --task-type adv --kb-limit 9251 --paper-scale
"""

import argparse
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"

# Paper Table 1 (arXiv:2407.12784, p.7), ReAct-StrategyQA column, "ChatGPT +
# contrastive-retriever" row (DPR is a contrastive-based retriever, matching
# our target embedder), AGENTPOISON's own bolded row -- read directly off the
# rendered table image, not extracted programmatically, so there's some risk
# of a single-digit transcription slip; re-check Table 1 directly before
# treating this as load-bearing. Reference point, not a target to match
# exactly -- different backbone model (gpt-3.5-turbo-instruct vs. our
# gpt-4o-mini/gpt-4o) and (per the injection_num/token-count note in
# adapter.py) even the code we're driving disagrees with this same table's own
# caption about how many instances/tokens to use.
PAPER_REFERENCE = {"asr_r": 0.647, "asr_a": 0.736, "asr_t": 0.586, "acc": 0.657, "non_attack_acc": 0.566}


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def build_kb_and_attack(kb_limit, injection_num, trigger_kwargs, seed, verbose=True):
    import torch

    from benchmark.attacks.agentpoison import data
    from benchmark.attacks.agentpoison.adapter import AgentPoisonAttack
    from benchmark.attacks.agentpoison.embedder import DPREmbedder
    from benchmark.attacks.agentpoison.kb_store import DenseKnowledgeBase
    from benchmark.testcases.schema import AttackSignal

    torch.set_num_threads(max(os.cpu_count() or 4, 1))

    if verbose:
        print(f"Loading local DPR embedder and KB (limit={kb_limit or 'full 9251'})...")
    embedder = DPREmbedder()
    kb = DenseKnowledgeBase(embedder, limit=kb_limit, cache_name=f"paragraphs_{kb_limit or 'full'}.pt", seed=seed)

    train_questions = data.load_train_questions()
    attack = AgentPoisonAttack(
        embedder=embedder,
        kb=kb,
        poison_source_questions=train_questions,
        injection_num=injection_num,
        trigger_optimization_kwargs=trigger_kwargs,
        seed=seed,
    )
    test_case = attack.generate_test_case(domain="strategyqa", signal=AttackSignal.STRONG)

    if verbose:
        hr("STAGE 1 -- Trigger optimization + KB poisoning (local, no LLM calls)")
    result = attack.plant(test_case, session=None, agent=None, store=None)
    if verbose:
        print(f"Trigger text: {attack.trigger_text!r}")
        print(f"Injected {len(attack.injected_entries)} poisoned KB entries: {[e['id'] for e in attack.injected_entries]}")
        print(f"write_accepted={result.write_accepted}")

    return embedder, kb, attack, train_questions


def run(
    task_type: str,
    model: str,
    num_questions: int,
    kb_limit: int | None,
    injection_num: int,
    trigger_kwargs: dict,
    knn: int,
    seed: int,
) -> dict:
    from openai import OpenAI

    from benchmark.agent.react_search_agent import ReActSearchAgent
    from benchmark.attacks.agentpoison import data, evaluation

    embedder, kb, attack, train_questions = build_kb_and_attack(kb_limit, injection_num, trigger_kwargs, seed)

    rng = random.Random(seed + 1)
    poisoned_qids = {e["source_question"] for e in attack.injected_entries}
    eval_pool = [q for q in train_questions if q["question"] not in poisoned_qids]
    eval_questions = rng.sample(eval_pool, num_questions)

    react_prompt = data.load_react_prompt()
    client = OpenAI()
    agent = ReActSearchAgent(client, kb, react_prompt, knn=knn, model=model)

    trigger_sequence = attack.trigger_text if task_type == "adv" else None

    hr(f"STAGE 2 -- {num_questions} ReAct episodes, real {model}, task_type={task_type}")
    if trigger_sequence:
        print(f"Trigger inserted at step {2}: {trigger_sequence!r}")
    episodes = []
    for i, q in enumerate(eval_questions):
        outcome = agent.run_episode(q["question"], trigger_sequence=trigger_sequence)
        outcome["gt_answer"] = q["answer"]
        episodes.append(outcome)
        print(
            f"  [{i + 1}/{num_questions}] pred={outcome['predicted']} gt={q['answer']} "
            f"retrieval_success={outcome['retrieval_success']}/{outcome['overall_retrieval']}  "
            f"{q['question'][:70]}"
        )

    acc = evaluation.compute_accuracy(episodes)
    summary = {"task_type": task_type, "model": model, "accuracy": acc}

    if task_type == "adv":
        asr_r = evaluation.compute_asr_r(episodes)
        asr_a = evaluation.compute_asr_a(episodes)
        asr_t = evaluation.compute_asr_t(acc["accuracy"])
        summary.update({"asr_r": asr_r, "asr_a": asr_a, "asr_t": asr_t})
        hr("SUMMARY (adv)")
        print(f"  ASR-r = {asr_r['asr_r']:.3f}   ({asr_r['total_success']}/{asr_r['total_retrieval']})   paper ref: {PAPER_REFERENCE['asr_r']:.2f}")
        print(f"  ASR-a = {asr_a['asr_a']:.3f}   ({asr_a['hits']}/{asr_a['total']})   paper ref: {PAPER_REFERENCE['asr_a']:.2f}")
        print(f"  ASR-t = {asr_t:.3f}   paper ref: {PAPER_REFERENCE['asr_t']:.2f}")
    else:
        hr("SUMMARY (benign)")
        print(f"  ACC = {acc['accuracy']:.3f}   ({acc['correct']}/{acc['total']})   paper ref: {PAPER_REFERENCE['acc']:.2f} (non-attack: {PAPER_REFERENCE['non_attack_acc']:.2f})")

    summary["trigger_text"] = attack.trigger_text
    summary["injected_entries"] = attack.injected_entries
    summary["episodes"] = episodes
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-type", choices=["adv", "benign"], default="adv")
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo-instruct",
        help="Default matches their actual backbone (run_strategyqa_gpt3.5.py's gpt() function: "
        "model='gpt-3.5-turbo-instruct', via the legacy Completions API -- react_search_agent.py "
        "dispatches to it automatically for any '-instruct' model name). Pass a chat model "
        "(e.g. gpt-4o-mini, gpt-4o) to use the modern Chat Completions API instead -- not directly "
        "comparable to Table 1's numbers, but a valid real-model test.",
    )
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--kb-limit", type=int, default=500, help="Real paragraphs to load (default 500; use 9251 for the full corpus -- already cached by an earlier warm-up run in this session).")
    parser.add_argument(
        "--injection-num", type=int, default=4,
        help="Poisoned KB entries to inject. Default 4 matches Table 1's caption "
        "for ReAct-StrategyQA specifically (arXiv:2407.12784, p.7); the shipped "
        "code's local_wikienv.py hardcodes 2 for every agent -- pass 2 to match that instead.",
    )
    parser.add_argument("--knn", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-trigger-tokens", type=int, default=5,
        help="Default 5 matches Table 1's caption for ReAct-StrategyQA; the README's example "
        "CLI command's --num_adv_passage_tokens default is 10 (a generic, non-agent-specific default).",
    )
    parser.add_argument("--num-iter", type=int, default=8)
    parser.add_argument("--num-grad-iter", type=int, default=3)
    parser.add_argument("--num-cand", type=int, default=15)
    parser.add_argument("--trigger-batch-size", type=int, default=4)
    parser.add_argument(
        "--no-ppl-filter", action="store_true",
        help="Skip the GPT-2 coherence pre-filter (their --ppl_filter, on by default here to match "
        "their recommended usage) -- faster, but candidates are no longer checked for fluency.",
    )
    parser.add_argument(
        "--coh-sample", action="store_true",
        help="Use their --coh_sample softmax-over-negative-log-perplexity candidate selection "
        "instead of deterministic top-k-by-lowest-perplexity (the default here, and theirs).",
    )
    parser.add_argument(
        "--golden-trigger", type=str, default=None,
        help="Seed the trigger from a real phrase instead of [MASK] tokens (their --golden_trigger). "
        "Their own seed for the QA agent: 'Make efficient calls.'",
    )
    parser.add_argument(
        "--paper-scale", action="store_true",
        help="Use the paper's own trigger-optimization defaults (num_iter=1000, num_grad_iter=30, "
        "num_cand=100, num_adv_passage_tokens=5 for ReAct-StrategyQA -- see the injection_num/token-count "
        "note in adapter.py) instead of this script's tractable defaults. Expect hours on CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. This script needs it for the ReAct agent's chat completions.")
        print("  Add OPENAI_API_KEY=sk-... to a .env file at the project root, or:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if args.paper_scale:
        trigger_kwargs = {
            "num_trigger_tokens": 5, "num_iter": 1000, "num_grad_iter": 30, "num_cand": 100, "batch_size": 8,
            "ppl_filter": not args.no_ppl_filter,
        }
    else:
        trigger_kwargs = {
            "num_trigger_tokens": args.num_trigger_tokens,
            "num_iter": args.num_iter,
            "num_grad_iter": args.num_grad_iter,
            "num_cand": args.num_cand,
            "batch_size": args.trigger_batch_size,
            "ppl_filter": not args.no_ppl_filter,
        }
    trigger_kwargs["coh_sample"] = args.coh_sample
    trigger_kwargs["golden_trigger"] = args.golden_trigger

    max_calls = args.num_questions * 7  # MAX_STEPS=7 per episode, worst case
    print(
        f"About to make up to ~{max_calls} real chat-completion calls to {args.model} "
        f"({args.num_questions} episodes x up to 7 steps each), task_type={args.task_type}. "
        f"Trigger optimization itself makes NO LLM calls (local embedder only)."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    summary = run(
        task_type=args.task_type,
        model=args.model,
        num_questions=args.num_questions,
        kb_limit=args.kb_limit,
        injection_num=args.injection_num,
        trigger_kwargs=trigger_kwargs,
        knn=args.knn,
        seed=args.seed,
    )

    report_path = RESULTS_DIR / f"agentpoison_experiment_{timestamp}.json"
    with report_path.open("w") as f:
        json.dump(
            {"timestamp": datetime.now(UTC).isoformat(), "paper_reference": PAPER_REFERENCE, **summary},
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
