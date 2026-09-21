"""Hidden Sleeper Memory benchmark implementation.

This package is intentionally self-contained so it can be evaluated without
changing the existing MemoryGraft, MINJA, or AgentPoison implementations.
"""

from benchmark.attacks.hidden_sleeper.adapter import HiddenSleeperAttack

__all__ = ["HiddenSleeperAttack"]
