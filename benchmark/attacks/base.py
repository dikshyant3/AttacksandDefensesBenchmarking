from abc import ABC, abstractmethod
from dataclasses import dataclass

from benchmark.testcases.schema import AttackSignal, TestCase


@dataclass
class PlantResult:
    write_attempted: bool
    write_accepted: bool
    record_id: str | None = None


class AttackAdapter(ABC):
    @abstractmethod
    def generate_test_case(self, domain: str, signal: AttackSignal) -> TestCase:
        """Build a benchmark test case for this attack."""

    @abstractmethod
    def plant(self, test_case: TestCase, session, agent, store) -> PlantResult:
        """Attempt to introduce the attack payload into memory."""
