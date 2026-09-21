"""Our results laid out like the paper's Table 2 (gpt-4o-mini columns), next to
the paper's own numbers -- read from saved campaign reports, offline.

Paper values are Table 2, GPT-4o-mini columns (ASR_A / ASR_R; "-" = undefined),
copied from the PDF's text layer. Ours are EXECUTION-level (the attack tool was
actually called), the paper's and ASB's definition of success; plan-level ASR_A
(tool merely named in the plan) is shown alongside for reference.

    python -m benchmark.attacks.dsrm.paper_comparison --n 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.attacks.hidden_sleeper.stats import rate_with_ci

RESULTS = Path(__file__).resolve().parent / "results"

# Table 2, GPT-4o-mini: retriever -> method -> (ASR_A, ASR_R)
PAPER = {
    "dpr": {"none": (5.00, None), "naive": (10.25, 10.00), "poisonedrag": (29.50, 29.00), "asb_attack": (10.75, 10.00), "dsrm": (36.25, 33.75)},
    "minilm": {"none": (5.75, None), "naive": (7.00, 6.75), "poisonedrag": (27.00, 26.50), "asb_attack": (11.00, 10.25), "dsrm": (34.00, 27.75)},
}
LABELS = {"none": "None", "naive": "Naive", "poisonedrag": "PoisonedRAG", "asb_attack": "ASB", "dsrm": "DSRM"}
ORDER = ["none", "naive", "poisonedrag", "asb_attack", "dsrm"]


def latest_report(retriever: str, n: int, needs: str = "asb_attack") -> dict | None:
    """The newest campaign report for this retriever and n that includes `needs`."""
    best = None
    for path in sorted(RESULTS.glob("campaign_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("retriever") == retriever and r.get("n_scenarios") == n and needs in r.get("methods", []) and r.get("execute") and not r.get("retrieval_only"):
            best = {**r, "_path": path.name}
    return best


def cell(bits: list[int]) -> str:
    if not bits:
        return "-"
    ci = rate_with_ci(bits)
    return f"{100 * ci['mean']:.1f} +/-{100 * ci['half_width']:.0f}"


def rows_for(report: dict, retriever: str) -> list[dict]:
    rows = []
    for m in ORDER:
        res = report["results"].get(m)
        if res is None:
            continue
        ex = [1 if r["executed_attack"] else 0 for r in res]
        plan = [1 if r["attack_succeeded"] else 0 for r in res]
        retrieved = [r for r in res if r["retrieved"]]
        rr = len(retrieved) / len(res)
        asr_r = [1 if r["executed_attack"] else 0 for r in retrieved]
        paper_a, paper_r = PAPER[retriever][m]
        ours_a = 100 * sum(ex) / len(ex)
        rows.append({
            "method": LABELS[m], "paper_a": paper_a, "paper_r": paper_r,
            "ours_a": cell(ex), "ours_r": "-" if m == "none" else cell(asr_r), "plan_a": cell(plan),
            "rr": f"{100 * rr:.0f}%", "ratio": (ours_a / paper_a) if paper_a else float("nan"), "ours_a_num": ours_a,
        })
    return rows


def render(retriever: str, report: dict) -> str:
    rows = rows_for(report, retriever)
    head = f"{'Retrieval':<9}{'Method':<13}| {'Paper ASR_A':>11} {'ASR_R':>7} | {'Ours ASR_A (executed)':>22} {'ASR_R':>13} | {'Ours ASR_A (plan)':>18} | {'RR':>5} | {'ours/paper':>10}"
    lines = [head, "-" * len(head)]
    for i, r in enumerate(rows):
        pr = "-" if r["paper_r"] is None else f"{r['paper_r']:.2f}"
        lines.append(f"{retriever.upper() if i == 0 else '':<9}{r['method']:<13}| {r['paper_a']:>11.2f} {pr:>7} | {r['ours_a']:>22} {r['ours_r']:>13} | {r['plan_a']:>18} | {r['rr']:>5} | {r['ratio']:>9.1f}x")
    return "\n".join(lines)


def ordering_note(retriever: str, report: dict) -> str:
    rows = {r["method"]: r["ours_a_num"] for r in rows_for(report, retriever)}
    paper = {LABELS[m]: v[0] for m, v in PAPER[retriever].items()}
    rank = lambda d: " > ".join(k for k, _ in sorted(d.items(), key=lambda kv: -kv[1]))  # noqa: E731
    return f"  ordering  paper: {rank(paper)}\n            ours:  {rank(rows)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--retrievers", nargs="+", default=["dpr", "minilm"])
    args = p.parse_args()
    print(f"Table 2 layout, GPT-4o-mini, n={args.n} scenarios (paper: 400).  Ours = mean +/- 95% bootstrap half-width, EXECUTION-level.\n")
    for retriever in args.retrievers:
        report = latest_report(retriever, args.n)
        if report is None:
            print(f"[{retriever}] no matching report yet\n")
            continue
        print(f"[{retriever}]  source: {report['_path']}  real API calls in that run: {report['llm_cache']['real_calls_made']}")
        print(render(retriever, report))
        print(ordering_note(retriever, report) + "\n")


if __name__ == "__main__":
    main()
