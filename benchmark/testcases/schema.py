from dataclasses import dataclass, field
from enum import Enum


class AttackType(str, Enum):
    MEMORYGRAFT = "memorygraft"
    MINJA = "minja"
    AGENTPOISON = "agentpoison"


class AttackSignal(str, Enum):
    WEAK = "weak"
    STRONG = "strong"


class CapabilityTier(str, Enum):
    T1_BENIGN_INGESTION = "T1"
    # MINJA's attacker has strictly less access than T1: no ingestion/document
    # channel at all, only ordinary conversational queries like any regular user
    # (Dong et al., arXiv:2503.03704, Sec. 3: "the attacker behaves like a regular
    # user and cannot directly manipulate any part of the agent beyond what is
    # accessible to them").
    T2_QUERY_ONLY_INJECTION = "T2"
    # AgentPoison's attacker has MORE access than T1, not less: a T1 attacker submits
    # a benign-looking document and hopes the agent's own ingestion pipeline writes
    # it; AgentPoison's attacker writes a small number of entries directly into the
    # shared retrieval knowledge base (no agent decision in the loop at all -- see
    # local_wikienv.py's load_db()), AND has white-box gradient access to the
    # retriever's embedding model to optimize a short trigger phrase offline before
    # ever touching the live system (Chen et al., arXiv:2407.12784, Sec. 3: the
    # attacker "has access to a portion of the target RAG database" and "can query
    # the target retriever to obtain the embeddings/gradients"). Distinct from T1
    # (direct write, no embedder access) and T2 (no write access at all).
    T3_WHITEBOX_RETRIEVAL_BACKDOOR = "T3"


@dataclass
class TestCase:
    attack_type: AttackType
    attack_signal: AttackSignal
    capability_tier: CapabilityTier
    domain: str
    adversarial_goal: str
    user_query: str
    context: str
    expected_memory: str
    retrieval_query: str
    # Attack-specific structured extras that don't fit the generic fields above
    # (e.g. MINJA's victim/target/indication-prompt campaign). Empty for attacks
    # that don't need it, e.g. MemoryGraft.
    metadata: dict = field(default_factory=dict)
