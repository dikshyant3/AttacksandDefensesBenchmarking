"""Generate REAL benign background workflows for the knowledge base.

The paper (Section 5.1) says the tasks NOT selected as scenarios were "treated
as benign background interactions, and their execution histories were
incorporated into the knowledge base". Our first background entries were
synthetic one-liners ("Completed this task using the standard tools", no tool
used). This produces the real thing: run the benign agent (agent.run_agent_step,
no memory, that agent's own normal tools only -- no attack tool) on each of the
41 leftover ASB tasks and keep the workflow it plans.

Cost: 41 calls to --agent-model (default gpt-4o-mini), cached. DRY RUN unless
--yes. Output: datasets/background_workflows.json (a GENERATED file, marked as
such -- not vendored ASB data).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark.attacks.dsrm.agent import run_agent_step, to_tool_specs
from benchmark.attacks.dsrm.data import BACKGROUND_WORKFLOWS_PATH, load_agent_domains, load_normal_tools, remaining_tasks_per_domain
from benchmark.attacks.dsrm.llm_cache import CachedChatClient
from benchmark.attacks.dsrm.run_experiment import DEFAULT_AGENT_MODEL, DEFAULT_CACHE_PATH


def generate(client, model: str, domains=None) -> list[dict]:
    domains = domains if domains is not None else load_agent_domains()
    prompt_by_agent = {d.agent_name: d.system_prompt for d in domains}
    normal_by_agent: dict[str, list] = {}
    for nt in load_normal_tools():
        normal_by_agent.setdefault(nt.corresponding_agent, []).append(nt)
    workflows = []
    for agent_name, task in remaining_tasks_per_domain(domains):
        decision = run_agent_step(
            client,
            model,
            system_prompt=prompt_by_agent[agent_name],
            user_task=task,
            tools=to_tool_specs(normal_by_agent.get(agent_name, [])),
            retrieved_context=[],
        )
        workflows.append(
            {
                "agent_name": agent_name,
                "task": task,
                "steps": [{"message": s.message, "tool_use": list(s.tool_use)} for s in decision.steps],
            }
        )
    return workflows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=BACKGROUND_WORKFLOWS_PATH)
    parser.add_argument("--yes", action="store_true", help="Make real, paid API calls")
    args = parser.parse_args()

    n = len(remaining_tasks_per_domain())
    print(f"About to make up to {n} real calls to {args.agent_model} (cache hits make none).")
    if not args.yes:
        print("DRY RUN ONLY: add --yes to execute paid API calls.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = CachedChatClient(OpenAI(), args.cache_path)
    workflows = generate(client, args.agent_model)
    empty = [w for w in workflows if not w["steps"]]
    record = {
        "generated_by": args.agent_model,
        "generated_on": datetime.now(UTC).isoformat(),
        "note": "GENERATED, not vendored: benign agent workflows (no memory, normal tools only) for the 41 non-selected ASB tasks.",
        "workflows": workflows,
    }
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {len(workflows)} workflows to {args.output}  ({len(empty)} empty)  |  real calls: {client.misses}, cache hits: {client.hits}")


if __name__ == "__main__":
    main()
