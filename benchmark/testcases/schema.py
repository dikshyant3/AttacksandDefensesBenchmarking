from dataclasses import dataclass, field
from enum import Enum


class AttackType(str, Enum):
    MEMORYGRAFT = "memorygraft"
    MINJA = "minja"


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
