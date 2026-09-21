"""Loaders for the vendored Agent Security Bench (ASB) data DSRM evaluates on.

See datasets/provenance.json for exact source commit, license, and the
field-by-field mapping to DSRM's own Appendix B tables. Every dataclass here
mirrors the ASB JSON schema as closely as possible (renamed to snake_case,
values otherwise unmodified) rather than reshaping it around DSRM's own
P_t/T_s/R_t vocabulary -- that assembly happens in the (not yet built)
decision-construction module, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
AGENT_TASK_PATH = DATASET_DIR / "asb_agent_task.jsonl"
ATTACK_TOOLS_PATH = DATASET_DIR / "asb_all_attack_tools.jsonl"
NORMAL_TOOLS_PATH = DATASET_DIR / "asb_all_normal_tools.jsonl"
AGENT_CONFIGS_PATH = DATASET_DIR / "asb_agent_configs.json"
# GENERATED (not vendored) by generate_background.py: real agent workflows for the 41 non-selected tasks.
BACKGROUND_WORKFLOWS_PATH = DATASET_DIR / "background_workflows.json"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass(frozen=True)
class NormalTool:
    """One legitimate tool T_n. Corresponds to ASB's all_normal_tools.jsonl
    and DSRM's Table B.6."""

    name: str
    description: str
    expected_achievements: str
    corresponding_agent: str


@dataclass(frozen=True)
class AttackTool:
    """One attacker tool T_a, with the attack instruction I the paper feeds
    into the Table A.1 initial-decision prompt. Corresponds to ASB's
    all_attack_tools.jsonl and DSRM's Table B.7."""

    name: str
    instruction: str
    description: str
    goal: str
    attack_type: str
    corresponding_agent: str
    aggressive: bool


@dataclass(frozen=True)
class AgentDomain:
    """One of the 10 ASB scenarios: an agent's system prompt P_s, its
    legitimate tool subset T_n, and the benign user tasks Q for that domain.
    Corresponds to ASB's agent_task.jsonl + per-agent config.json, and
    DSRM's Table B.4/B.5."""

    agent_name: str
    system_prompt: str
    normal_tool_names: tuple[str, ...]
    tasks: tuple[str, ...]


def load_normal_tools(path: Path | None = None) -> list[NormalTool]:
    path = path or NORMAL_TOOLS_PATH
    return [
        NormalTool(
            name=r["Tool Name"],
            description=r["Description"],
            expected_achievements=r["Expected Achievements"],
            corresponding_agent=r["Corresponding Agent"],
        )
        for r in _read_jsonl(path)
    ]


def load_attack_tools(path: Path | None = None) -> list[AttackTool]:
    path = path or ATTACK_TOOLS_PATH
    return [
        AttackTool(
            name=r["Attacker Tool"],
            instruction=r["Attacker Instruction"],
            description=r["Description"],
            goal=r["Attack goal"],
            attack_type=r["Attack Type"],
            corresponding_agent=r["Corresponding Agent"],
            aggressive=r["Aggressive"] == "True",
        )
        for r in _read_jsonl(path)
    ]


def load_agent_domains(
    task_path: Path | None = None, config_path: Path | None = None
) -> list[AgentDomain]:
    """Join agent_task.jsonl (tasks) with agent_configs.json (system prompt +
    legitimate tools) on agent_name. Order follows agent_task.jsonl."""
    task_path = task_path or AGENT_TASK_PATH
    config_path = config_path or AGENT_CONFIGS_PATH

    configs_by_name = {}
    with config_path.open(encoding="utf-8") as f:
        for cfg in json.load(f):
            configs_by_name[cfg["name"]] = cfg

    domains = []
    for row in _read_jsonl(task_path):
        name = row["agent_name"]
        cfg = configs_by_name[name]
        domains.append(
            AgentDomain(
                agent_name=name,
                system_prompt=cfg["description"][0]
                if isinstance(cfg["description"], list)
                else cfg["description"],
                normal_tool_names=tuple(cfg["tools"]),
                tasks=tuple(row["tasks"]),
            )
        )
    return domains


def attack_tools_for_agent(agent_name: str, attack_tools: list[AttackTool] | None = None) -> list[AttackTool]:
    attack_tools = attack_tools if attack_tools is not None else load_attack_tools()
    return [t for t in attack_tools if t.corresponding_agent == agent_name]


def first_task_per_domain(domains: list[AgentDomain] | None = None) -> list[tuple[str, str]]:
    """DSRM Section 5.1: "We selected the first task from each of the 10
    domains to ensure balanced coverage and generalizibility." Returns
    (agent_name, task) pairs in agent_task.jsonl order."""
    domains = domains if domains is not None else load_agent_domains()
    return [(d.agent_name, d.tasks[0]) for d in domains]


def remaining_tasks_per_domain(domains: list[AgentDomain] | None = None) -> list[tuple[str, str]]:
    """The tasks NOT selected as each domain's 'first task' (DSRM Section
    5.1: 'The remaining tasks were treated as benign background
    interactions, and their execution histories were incorporated into the
    knowledge base to simulate a realistic and noisy retrieval
    environment'). Real ASB task text -- same source/provenance as the
    selected scenarios, just the other 41 of the 51 vendored tasks."""
    domains = domains if domains is not None else load_agent_domains()
    return [(d.agent_name, task) for d in domains for task in d.tasks[1:]]


def load_background_workflows(path: Path | None = None) -> list[dict]:
    """Agent-generated workflows for the 41 non-selected tasks, produced by
    generate_background.py (one real LLM call each). Records:
    {agent_name, task, steps: [{message, tool_use}], generated_by}."""
    path = path or BACKGROUND_WORKFLOWS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- generate it with: python -m benchmark.attacks.dsrm.generate_background --yes")
    return json.loads(path.read_text(encoding="utf-8"))["workflows"]


def dataset_summary() -> dict:
    domains = load_agent_domains()
    attack_tools = load_attack_tools()
    normal_tools = load_normal_tools()
    return {
        "domains": len(domains),
        "total_tasks": sum(len(d.tasks) for d in domains),
        "attack_tools": len(attack_tools),
        "normal_tools": len(normal_tools),
        "selected_scenarios": len(first_task_per_domain(domains))
        * (len(attack_tools) // len(domains)),
    }
