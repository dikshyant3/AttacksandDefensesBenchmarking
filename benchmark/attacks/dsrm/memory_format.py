"""How a memory entry and the retrieval query are written down.

Two formats, selectable per run:

  figure1 -- what Phases 3-4 used: entry "Query: <task>\\nContext:{Pt, Ts, Rt}\\n
             Please generate a workflow..." (shape read off the paper's
             Figure 1), retrieval query = the task alone.

  asb     -- ASB's own format (pyopenagi/agents/react_agent_attack.py, read
             directly): entry "Agent: <role>; Task: <task>; Workflow: <steps
             json>; Tools: <tool-set json>", and the retrieval query is
             "<task>; <tool-set json>" -- the task plus the agent's FULL tool
             set, attack tool included. The paper's `R = Q (+) T_m` most
             plausibly means exactly this (T_m = the merged tool set, Section
             3.2). Still an inference: the paper never shows the format.

`tools` is always the tool set the agent had when the entry was written / the
query is issued (a benign entry: that agent's normal tools; a poisoned entry
and a NONE/NAIVE/DSRM query: normal tools + the attack tool; a CLEAN query:
normal tools only).
"""

from __future__ import annotations

import json
from typing import Literal

from benchmark.attacks.dsrm.agent import ToolSpec, tool_call_schema
from benchmark.attacks.dsrm.decision import DecisionStep, render_figure1_entry

MemoryFormat = Literal["figure1", "asb"]
MEMORY_FORMATS = ("figure1", "asb")


def tools_json(tools: list[ToolSpec]) -> str:
    return json.dumps([tool_call_schema(t) for t in tools])


def workflow_json(steps: list[DecisionStep]) -> str:
    return json.dumps([s.to_json_obj() for s in steps])


def render_entry(
    memory_format: MemoryFormat,
    *,
    agent_prompt: str,
    user_task: str,
    steps: list[DecisionStep],
    tools: list[ToolSpec],
) -> str:
    if memory_format == "figure1":
        return render_figure1_entry(user_task, steps)
    if memory_format == "asb":
        return f"Agent: {agent_prompt}; Task: {user_task}; Workflow: {workflow_json(steps)}; Tools: {tools_json(tools)}"
    raise ValueError(f"unknown memory_format {memory_format!r} (choices: {MEMORY_FORMATS})")


def render_query(memory_format: MemoryFormat, user_task: str, tools: list[ToolSpec]) -> str:
    if memory_format == "figure1":
        return user_task
    if memory_format == "asb":
        return f"{user_task}; {tools_json(tools)}"
    raise ValueError(f"unknown memory_format {memory_format!r} (choices: {MEMORY_FORMATS})")
