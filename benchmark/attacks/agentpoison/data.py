"""Real data for the AgentPoison ReAct-StrategyQA target (Chen et al.,
"AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases",
arXiv:2407.12784, NeurIPS 2024, github.com/AI-secure/AgentPoison).

Vendored directly into this repo (benchmark/attacks/agentpoison/data/), not
fetched over the network at runtime -- these are real files from the paper's
own repo (checked into git there, no Google Drive needed), so there's no
reason for this benchmark's test suite or experiment runs to depend on their
GitHub repository staying online or unchanged. Same real-data-over-fabricated
principle used for MINJA's qa_seeds_mmlu.py, just vendored as raw files here
instead of hand-transcribed into a .py module, since the corpus (9,251
paragraphs) is far too large to reasonably write out as Python literals.

Three files, each used for exactly what's named:
- strategyqa_train_paragraphs.json -- 9,251 real Wikipedia paragraphs, keyed
  by "{title}-{para_index}". This IS the retrieval knowledge base
  local_wikienv.py's WikiEnv builds its dense index from.
- strategyqa_train.json -- 2,290 real StrategyQA training examples (question,
  boolean answer, supporting facts). Their own wrappers.py::StrategyQAWrapper
  actually loads a *different* path (ReAct/data/strategyqa/strategyqa_train.json)
  for held-out eval questions, but that directory isn't in their repo
  (confirmed: 404) -- only ReAct/database's copy is real and fetchable, so
  it's used here for both roles (eval question source AND the pool the
  poisoned KB entries are built from), matching local_wikienv.py's own use of
  this same file for the latter role.
- sqa_react_prompt.txt -- the actual few-shot ReAct demonstration prompt their
  run_strategyqa_gpt3.5.py uses verbatim, extracted from prompts.json's
  "sqa_react" field. Their prompts.json also ships hotpotqa_*/sqa_standard/
  sqa_cot/mmlu_* prompt variants for other datasets/agent styles this
  benchmark's ReAct-StrategyQA target never touches -- only sqa_react is
  vendored, not the other ~21KB of unrelated prompt text.

Re-vendoring (only needed if these ever need refreshing from upstream):
    python -c "
    import json, urllib.request
    base = 'https://raw.githubusercontent.com/AI-secure/AgentPoison/master/ReAct'
    urllib.request.urlretrieve(f'{base}/database/strategyqa_train_paragraphs.json', 'data/strategyqa_train_paragraphs.json')
    urllib.request.urlretrieve(f'{base}/database/strategyqa_train.json', 'data/strategyqa_train.json')
    prompts = json.load(urllib.request.urlopen(f'{base}/prompts/prompts.json'))
    open('data/sqa_react_prompt.txt', 'w').write(prompts['sqa_react'])
    "
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_paragraphs() -> dict:
    """paragraph_id -> {"title", "section", "headers", "para_index", "content"}."""
    return json.loads((DATA_DIR / "strategyqa_train_paragraphs.json").read_text())


def load_train_questions() -> list[dict]:
    """Each entry: qid, term, description, question, answer (bool), facts, ..."""
    return json.loads((DATA_DIR / "strategyqa_train.json").read_text())


def load_react_prompt() -> str:
    """Verbatim few-shot ReAct-StrategyQA demonstrations, exactly as
    run_strategyqa_gpt3.5.py builds sqa_react_prompt from prompt_dict['sqa_react']."""
    return (DATA_DIR / "sqa_react_prompt.txt").read_text()
