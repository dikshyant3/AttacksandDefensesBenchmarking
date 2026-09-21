"""Real-LLM MINJA experiment run (Dong et al., "Memory Injection Attacks on LLM
Agents via Query-Only Interaction", arXiv:2503.03704, NeurIPS 2025).

benchmark/tests/test_minja.py is deliberately deterministic and offline (a scripted
LLM stand-in, DeterministicQALLMClient) so the test suite verifies plumbing --
progressive shortening, conditional write-gating, verbatim record storage, ISR/ASR/UD
formulas -- for free and reproducibly. It is NOT evidence that a real LLM is actually
persuadable by the indication prompt; that can only be shown by calling one. This
script does that: it runs the exact same Algorithm 1 campaign (adapter.py's plant())
through a real `gpt-4o-mini` model, then evaluates ISR, ASR, and UD the same way.

Reference point (not a target to match exactly -- different LLM, corpus, and domain):
the paper's QA Agent / MMLU / GPT-4o results (Table 1, mean over 9 victim-target
pairs): ISR = 100.0, ASR = 68.9 (+-19.1), UD = -10.0 (+-8.2).

Retrieval: --retrieval levenshtein (default) matches the SHIPPED CODE (QA/main.py's
generate_prompt_and_response uses Levenshtein.distance) and needs no embedding
calls. --retrieval embedding matches the PAPER'S OWN TEXT instead (Sec. 5.1:
"text-embedding-ada-002 for QA Agent") -- the paper and code genuinely disagree
with each other on this, so both are real options, not one invented and one real.
Embedding mode adds a real (very cheap) API cost on top of chat completions.

Requires OPENAI_API_KEY in the environment (loaded from a .env file at the project
root if present).

Usage:
    python -m benchmark.attacks.minja.run_experiment
    python -m benchmark.attacks.minja.run_experiment --seed 7
    python -m benchmark.attacks.minja.run_experiment --model gpt-4o
    python -m benchmark.attacks.minja.run_experiment --corpus mmlu --model gpt-4o
    python -m benchmark.attacks.minja.run_experiment --corpus mmlu --model gpt-4o --trials 9

--trials N (mmlu only): draws N fresh seeded samples from the corpus's full pool
(qa_seeds_mmlu.sample_split) instead of one fixed split, runs each, and reports
mean+std -- a variance-reduction proxy for the paper's own practice of averaging
over 9 experiments. Important: their 9 are 9 DIFFERENT victim-target pairs; this is
9 reseeds of the SAME "food" pair, since that's the only real victim term we have
confirmed from their repo. Informative about single-pair sampling variance, not a
literal reproduction of their between-pair methodology.

Cost note: plant() now matches QA/main.py's actual per-question-block mechanism --
each attack question generates ALL shortening levels (not one), so with the default
5-step schedule this is 10*6=60 attack-side calls per run, not 10. A --trials 9 run
is therefore ~9x that again (see the two-stage cost warning in main() if you invoke
it without realizing this).
"""
import argparse
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark.agent.qa_loop import QAAgent
from benchmark.attacks.minja import evaluation
from benchmark.attacks.minja import qa_seeds as qa_seeds_deploy_ci
from benchmark.attacks.minja import qa_seeds_mmlu
from benchmark.attacks.minja import qa_seeds_mmlu_pairs
from benchmark.attacks.minja.adapter import MAX_RETRIES_PER_ATTACK_QUERY, MINJAAttack
from benchmark.core.session import new_session
from benchmark.stores.procedural import ProceduralStore
from benchmark.testcases.schema import AttackSignal

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Sec. 5.1's literal named model for QA Agent


def make_openai_embed_fn(client, model: str = EMBEDDING_MODEL):
    """Per-text cache around a single-string embeddings.create() call -- QAAgent
    itself already caches by text (see qa_loop.py), this is just the underlying
    API call each cache miss makes."""

    def embed_fn(text: str) -> list[float]:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    return embed_fn

CORPORA = {
    # Our own deploy/CI corpus -- thematically consistent with the rest of this
    # benchmark, but hand-written and structurally homogeneous (see the ISR/ASR
    # gap discussion this was built to help isolate).
    "deploy_ci": qa_seeds_deploy_ci,
    # The paper's actual corpus: real MMLU nutrition_test.csv rows (see
    # qa_seeds_mmlu.py's module docstring for exactly how they were selected).
    # Isolates one variable: same mechanism, same wording, same model choice --
    # only corpus authenticity changes from qa_seeds_deploy_ci.
    "mmlu": qa_seeds_mmlu,
}

# Paper Table 1: QA Agent, GPT-4o, MMLU, mean over 9 victim-target pairs.
PAPER_REFERENCE = {"isr": 100.0, "asr": 68.9, "ud": -10.0}

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run(
    seed: int,
    model: str,
    corpus_name: str,
    sample: dict | None = None,
    verbose: bool = True,
    retrieval: str = "levenshtein",
    victim_pair: str | None = None,
) -> dict:
    """Run one full campaign. `sample` -- if given, a dict with attack_questions/
    test_questions/benign_questions/benign_test_questions (the shape
    qa_seeds_mmlu.sample_split returns) -- overrides the corpus's default. `verbose=False`
    suppresses per-turn/per-query printing, used for run_trials(): 9 trials' worth of
    full per-turn output would be unreadable.

    `victim_pair` -- one of qa_seeds_mmlu_pairs.ALL_TERMS (mmlu corpus only) --
    selects which of the 9 REAL victim-target pairs (Figure 4/Appendix C) to run,
    not just "food". Defaults to "food" for backward compatibility (the only pair
    implemented before this reconstruction of the other 8).

    Sampling default: if the corpus exposes sample_split() (currently only mmlu),
    that's used by default -- a fresh, seeded draw from the FULL pool (all 31 food
    rows, all 268 benign rows from nutrition_test.csv), matching QA/main.py's actual
    `random.sample(templates_contents, num_templates)` behavior. This is different
    from earlier in this benchmarking session, where the default silently reused one
    fixed pre-selected 10/10/30/10 subset for every run -- that was a real fidelity
    gap (their code re-samples every run; a fixed subset never does), not just a
    scale issue. Pass a `sample` explicitly, or use a corpus without sample_split
    (deploy_ci -- genuinely hand-authored, no larger pool to sample from), to opt out.
    """
    corpus = CORPORA[corpus_name]
    if corpus_name == "mmlu" and victim_pair is not None and victim_pair != "food":
        term = victim_pair
        if sample is None:
            sample = qa_seeds_mmlu_pairs.sample_split(term, seed=seed)
        victim_term = term
        indication_prompt_fn = qa_seeds_mmlu_pairs.indication_prompt_fn_for(term)
        max_shorten_steps = qa_seeds_mmlu_pairs.MAX_SHORTEN_STEPS
    else:
        if sample is None:
            if hasattr(corpus, "sample_split"):
                sample = corpus.sample_split(seed=seed)
            else:
                sample = {
                    "attack_questions": corpus.ATTACK_QUESTIONS,
                    "test_questions": corpus.TEST_QUESTIONS,
                    "benign_questions": corpus.BENIGN_QUESTIONS,
                    "benign_test_questions": corpus.BENIGN_TEST_QUESTIONS,
                }
        victim_term = corpus.VICTIM_TERM
        indication_prompt_fn = corpus.indication_prompt
        max_shorten_steps = corpus.MAX_SHORTEN_STEPS

    attack = MINJAAttack(
        attack_questions=sample["attack_questions"],
        test_questions=sample["test_questions"],
        benign_questions=sample["benign_questions"],
        benign_test_questions=sample["benign_test_questions"],
        victim_term=victim_term,
        indication_prompt_fn=indication_prompt_fn,
        max_shorten_steps=max_shorten_steps,
        seed=seed,
    )

    def p(*a, **kw):
        if verbose:
            print(*a, **kw)

    if verbose:
        hr(f"PART 1 -- Injection campaign (Algorithm 1, real {model}, corpus={corpus_name})")
    else:
        print(f"[seed={seed}] running...", end=" ", flush=True)
    p(
        f"{len(attack.attack_questions)} attack queries (victim term: "
        f"'{attack.victim_term}') interleaved with {len(attack.benign_questions)} "
        f"benign queries, PSS shortening over {corpus.MAX_SHORTEN_STEPS} steps."
    )

    from openai import OpenAI

    client = OpenAI()
    embed_fn = make_openai_embed_fn(client) if retrieval == "embedding" else None

    store = ProceduralStore()
    session = new_session()
    test_case = attack.generate_test_case(domain="deploy_rollback", signal=AttackSignal.WEAK)
    agent = QAAgent(
        client, store, session.session_id, initial_demo=corpus.INITIAL_DEMO, model=model, embed_fn=embed_fn,
        seed=seed,
    )

    attack.plant(test_case, session, agent, store)

    p("\nPer-attack-query outcome:")
    for entry in attack.injection_log:
        status = "INJECTED" if entry["succeeded"] else "failed  "
        p(
            f"  [{status}] shorten_steps={entry['shorten_steps']} "
            f"attempts={entry['attempts']}  {entry['question'][:70]}"
        )

    isr_result = evaluation.compute_isr(attack)
    p(f"\nISR = {isr_result['isr']:.3f}  ({isr_result['injected_count']}/{isr_result['total_attack_queries']})")
    p(f"Paper reference (GPT-4o, MMLU, mean of 9 pairs): ISR = {PAPER_REFERENCE['isr']:.1f}%")

    if verbose:
        hr("PART 2 -- Attack Success Rate (held-out victim test queries)")
    p(f"{len(attack.test_questions)} held-out queries, never seen during injection.")
    asr_result = evaluation.compute_asr(agent, attack.test_questions)
    for question, detail in asr_result["per_question"].items():
        status = "SHIFTED" if detail["shifted"] else "normal "
        p(f"  [{status}] answer={detail['answer']}  {question[:70]}")
    p(f"\nASR = {asr_result['asr']:.3f}  ({asr_result['hits']}/{asr_result['total']})")
    p(f"Paper reference: ASR = {PAPER_REFERENCE['asr']:.1f}%")

    if verbose:
        hr("PART 3 -- Utility Drop (held-out benign queries, before vs. after poisoning)")
    clean_store = ProceduralStore()
    clean_session = new_session()
    clean_agent = QAAgent(
        client, clean_store, clean_session.session_id, initial_demo=corpus.INITIAL_DEMO, model=model,
        embed_fn=embed_fn, seed=seed,
    )

    p(f"Evaluating {len(attack.benign_test_questions)} held-out benign queries on a clean agent...")
    accuracy_before = evaluation.compute_accuracy(clean_agent, attack.benign_test_questions)
    p(f"Evaluating the same queries on the poisoned agent...")
    accuracy_after = evaluation.compute_accuracy(agent, attack.benign_test_questions)

    ud = evaluation.compute_ud(accuracy_before["accuracy"], accuracy_after["accuracy"])
    p(f"\nAccuracy before poisoning: {accuracy_before['accuracy']:.3f} ({accuracy_before['correct']}/{accuracy_before['total']})")
    p(f"Accuracy after poisoning:  {accuracy_after['accuracy']:.3f} ({accuracy_after['correct']}/{accuracy_after['total']})")
    p(f"UD = {ud:.1f}")
    p(f"Paper reference: UD = {PAPER_REFERENCE['ud']:.1f}")

    if not verbose:
        print(f"ISR={isr_result['isr']:.2f} ASR={asr_result['asr']:.2f} UD={ud:.1f}")

    return {
        "seed": seed,
        "corpus": corpus_name,
        "victim_term": attack.victim_term,
        "isr": isr_result,
        "asr": asr_result,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "ud": ud,
        "injection_log": attack.injection_log,
    }


def run_trials(
    base_seed: int, n_trials: int, model: str, corpus_name: str, retrieval: str = "levenshtein"
) -> dict:
    """Run n_trials fresh seeded samples (base_seed, base_seed+1, ...) and report
    mean+stdev, mirroring how the paper reports Table 1 (mean +- std over 9
    experiments) -- see this module's docstring for the important caveat about
    what this proxy does and doesn't reproduce.
    """
    corpus = CORPORA[corpus_name]
    if not hasattr(corpus, "sample_split"):
        raise ValueError(
            f"Corpus '{corpus_name}' has no sample_split() -- --trials is only "
            "meaningful for corpora with a full pool to resample from (currently: mmlu)."
        )

    hr(f"Running {n_trials} trials (seeds {base_seed}..{base_seed + n_trials - 1}), corpus={corpus_name}, model={model}, retrieval={retrieval}")
    trial_results = []
    for i in range(n_trials):
        seed = base_seed + i
        sample = corpus.sample_split(seed=seed)
        result = run(
            seed=seed, model=model, corpus_name=corpus_name, sample=sample, verbose=False, retrieval=retrieval
        )
        trial_results.append(result)

    isr_values = [r["isr"]["isr"] for r in trial_results]
    asr_values = [r["asr"]["asr"] for r in trial_results]
    ud_values = [r["ud"] for r in trial_results]

    def mean_std(values):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    isr_mean, isr_std = mean_std(isr_values)
    asr_mean, asr_std = mean_std(asr_values)
    ud_mean, ud_std = mean_std(ud_values)

    hr("TRIAL SUMMARY")
    print(f"  ISR = {isr_mean * 100:.1f}% (+-{isr_std * 100:.1f})   individual: {[round(v, 2) for v in isr_values]}")
    print(f"  ASR = {asr_mean * 100:.1f}% (+-{asr_std * 100:.1f})   individual: {[round(v, 2) for v in asr_values]}")
    print(f"  UD  = {ud_mean:.1f} (+-{ud_std:.1f})   individual: {[round(v, 1) for v in ud_values]}")
    print(
        f"\n  Paper reference (9 DIFFERENT victim-target pairs, not reseeds of one): "
        f"ISR = {PAPER_REFERENCE['isr']:.1f}% (+-0.0), "
        f"ASR = {PAPER_REFERENCE['asr']:.1f}% (+-19.1), "
        f"UD = {PAPER_REFERENCE['ud']:.1f} (+-8.2)"
    )

    return {
        "n_trials": n_trials,
        "base_seed": base_seed,
        "corpus": corpus_name,
        "model": model,
        "isr_mean": isr_mean,
        "isr_std": isr_std,
        "asr_mean": asr_mean,
        "asr_std": asr_std,
        "ud_mean": ud_mean,
        "ud_std": ud_std,
        "trials": trial_results,
    }


def run_multi_pair(seed: int, model: str, retrieval: str = "levenshtein") -> dict:
    """Runs all 9 REAL victim-target pairs (Figure 4/Appendix C: water, law,
    labor, financial, total, patient, security, evidence, food), one campaign
    each, and reports mean+std ACROSS PAIRS. Unlike run_trials() (N reseeds of
    the SAME "food" pair -- useful for characterizing that one pair's own
    stable distribution, but not comparable to the paper's between-pair
    average), this is the actual literal reproduction of their methodology:
    "we conduct 9 independent experiments, each with a unique victim-target
    pair" (Sec. 5.1). Same `seed` used for every pair's own interleaving
    shuffle/sampling -- the pair (term) itself is what varies across the 9
    runs here, matching their "9 independent experiments" framing directly.
    """
    hr(f"Running all 9 real victim-target pairs (seed={seed}), model={model}, retrieval={retrieval}")
    pair_results = []
    for term in qa_seeds_mmlu_pairs.ALL_TERMS:
        print(f"[pair={term}] running...", end=" ", flush=True)
        result = run(seed=seed, model=model, corpus_name="mmlu", verbose=False, retrieval=retrieval, victim_pair=term)
        pair_results.append(result)

    isr_values = [r["isr"]["isr"] for r in pair_results]
    asr_values = [r["asr"]["asr"] for r in pair_results]
    ud_values = [r["ud"] for r in pair_results]

    def mean_std(values):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    isr_mean, isr_std = mean_std(isr_values)
    asr_mean, asr_std = mean_std(asr_values)
    ud_mean, ud_std = mean_std(ud_values)

    hr("MULTI-PAIR SUMMARY (9 real, different victim-target pairs)")
    for term, r in zip(qa_seeds_mmlu_pairs.ALL_TERMS, pair_results):
        print(f"  {term:10s} ISR={r['isr']['isr']:.2f} ASR={r['asr']['asr']:.2f} UD={r['ud']:.1f}")
    print(f"\n  ISR = {isr_mean * 100:.1f}% (+-{isr_std * 100:.1f})")
    print(f"  ASR = {asr_mean * 100:.1f}% (+-{asr_std * 100:.1f})")
    print(f"  UD  = {ud_mean:.1f} (+-{ud_std:.1f})")
    print(
        f"\n  Paper reference (9 different victim-target pairs, GPT-4o, MMLU): "
        f"ISR = {PAPER_REFERENCE['isr']:.1f}% (+-0.0), "
        f"ASR = {PAPER_REFERENCE['asr']:.1f}% (+-19.1), "
        f"UD = {PAPER_REFERENCE['ud']:.1f} (+-8.2)"
    )

    return {
        "mode": "multi_pair",
        "seed": seed,
        "model": model,
        "retrieval": retrieval,
        "terms": list(qa_seeds_mmlu_pairs.ALL_TERMS),
        "isr_mean": isr_mean,
        "isr_std": isr_std,
        "asr_mean": asr_mean,
        "asr_std": asr_std,
        "ud_mean": ud_mean,
        "ud_std": ud_std,
        "pairs": pair_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Interleaving shuffle seed (default: 42).")
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"OpenAI chat model to use (default: {LLM_MODEL}). Paper uses GPT-4/GPT-4o; "
        "pass --model gpt-4o to test the same model family they validated.",
    )
    parser.add_argument(
        "--corpus",
        choices=list(CORPORA.keys()),
        default="deploy_ci",
        help="Question corpus to use (default: deploy_ci). 'mmlu' is the paper's own "
        "real data (real nutrition_test.csv rows, victim term 'food') -- use it to "
        "isolate whether an ISR/ASR gap is corpus-driven vs. model-driven.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Run N fresh seeded samples (mmlu corpus only) and report mean+std, "
        "seeds = --seed .. --seed+N-1. A variance-reduction proxy for the paper's "
        "9-experiment average -- see this module's docstring for the caveat.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["levenshtein", "embedding"],
        default="levenshtein",
        help="Retrieval strategy (default: levenshtein, matching the SHIPPED CODE). "
        "'embedding' matches the PAPER'S OWN TEXT instead (text-embedding-ada-002) -- "
        "they disagree with each other; see this module's docstring.",
    )
    parser.add_argument(
        "--victim-pair",
        choices=list(qa_seeds_mmlu_pairs.ALL_TERMS),
        default="food",
        help="Which of the 9 REAL victim-target pairs to run (mmlu corpus only; "
        "Figure 4/Appendix C). Defaults to 'food', the only pair implemented before "
        "this reconstruction of the other 8 from real MMLU data.",
    )
    parser.add_argument(
        "--multi-pair",
        action="store_true",
        help="Run all 9 real victim-target pairs (one campaign each, same seed) and report "
        "mean+std ACROSS pairs -- the actual literal reproduction of the paper's '9 "
        "independent experiments, each with a unique victim-target pair' methodology, "
        "unlike --trials (which reseeds one pair). Overrides --victim-pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. This script needs it for chat completions.")
        print("  Add OPENAI_API_KEY=sk-... to a .env file at the project root, or:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    corpus = CORPORA[args.corpus]
    attack_calls_per_run = len(corpus.ATTACK_QUESTIONS) * (corpus.MAX_SHORTEN_STEPS + 1)
    other_calls_per_run = (
        len(corpus.BENIGN_QUESTIONS) + len(corpus.TEST_QUESTIONS) + 2 * len(corpus.BENIGN_TEST_QUESTIONS)
    )
    calls_per_run = attack_calls_per_run + other_calls_per_run  # best case, 1 attempt per attack turn
    run_multiplier = 9 if args.multi_pair else max(args.trials, 1)
    total_calls = calls_per_run * run_multiplier
    print(
        f"About to make ~{total_calls} real API calls (best case, more if attack turns need retries: "
        f"up to {attack_calls_per_run * MAX_RETRIES_PER_ATTACK_QUERY + other_calls_per_run} per run) "
        f"to {args.model}, corpus={args.corpus}, "
        + (f"multi-pair=all 9 real pairs" if args.multi_pair else f"trials={run_multiplier}")
        + f", retrieval={args.retrieval}."
        + (
            f" Retrieval mode 'embedding' also makes one {EMBEDDING_MODEL} call per unique question text"
            " retrieved-against or retrieved-with (cached, so each unique text is embedded once) -- very"
            " cheap, but a real cost on top of the chat-completion count above."
            if args.retrieval == "embedding"
            else ""
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if args.multi_pair:
        result = run_multi_pair(args.seed, args.model, retrieval=args.retrieval)
        report_path = RESULTS_DIR / f"minja_multipair_{timestamp}.json"
        with report_path.open("w") as f:
            json.dump(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "llm_model": args.model,
                    "paper_reference": PAPER_REFERENCE,
                    **result,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nFull report saved to: {report_path}")
        return

    if args.trials > 1:
        result = run_trials(args.seed, args.trials, args.model, args.corpus, retrieval=args.retrieval)
        report_path = RESULTS_DIR / f"minja_trials_{timestamp}.json"
        with report_path.open("w") as f:
            json.dump(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "llm_model": args.model,
                    "paper_reference": PAPER_REFERENCE,
                    **result,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nFull report saved to: {report_path}")
        return

    result = run(args.seed, args.model, args.corpus, retrieval=args.retrieval, victim_pair=args.victim_pair)

    report_path = RESULTS_DIR / f"minja_experiment_{timestamp}.json"
    with report_path.open("w") as f:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "llm_model": args.model,
                "paper_reference": PAPER_REFERENCE,
                **result,
            },
            f,
            indent=2,
            default=str,
        )

    hr("SUMMARY")
    print(f"  ISR = {result['isr']['isr']:.3f}   (paper: {PAPER_REFERENCE['isr']:.1f}%)")
    print(f"  ASR = {result['asr']['asr']:.3f}   (paper: {PAPER_REFERENCE['asr']:.1f}%)")
    print(f"  UD  = {result['ud']:.1f}   (paper: {PAPER_REFERENCE['ud']:.1f})")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
