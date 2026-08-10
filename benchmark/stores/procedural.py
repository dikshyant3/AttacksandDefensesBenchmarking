from benchmark.core.checkpoints import run_retrieval_checkpoint
from benchmark.core.schema import MemoryRecord


class ProceduralStore:
    def __init__(self):
        self.records: list[MemoryRecord] = []

    def commit(self, record: MemoryRecord) -> None:
        """Low-level insert only. Called exclusively by core.write_pipeline.write()."""
        self.records.append(record)

    def retrieve(self, query: str, k: int = 5) -> list[MemoryRecord]:
        candidates = [
            record
            for record in self.records
            if any(word.lower() in record.content.lower() for word in query.split())
        ][:k]
        checkpoint_result = run_retrieval_checkpoint(candidates, query)
        if not checkpoint_result.accepted:
            return []
        return candidates
