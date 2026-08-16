from collections.abc import Callable

import numpy as np
from rank_bm25 import BM25Okapi


class UnionRetriever:
    def __init__(self, records: list[dict], embed_fn: Callable[[str], list[float]]):
        if any("text" not in record for record in records):
            raise ValueError("Every record passed to UnionRetriever must have a 'text' key")

        self.records = records
        self.embed_fn = embed_fn
        tokenized = [record["text"].lower().split() for record in records]
        self.bm25 = BM25Okapi(tokenized) if records else None
        self.embeddings = [np.asarray(embed_fn(record["text"]), dtype=float) for record in records]

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if not self.records:
            return []

        k = min(k, len(self.records))
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_idx = set(np.argsort(bm25_scores)[-k:])

        query_embedding = np.asarray(self.embed_fn(query), dtype=float)
        similarities = [
            float(
                np.dot(query_embedding, embedding)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-9)
            )
            for embedding in self.embeddings
        ]
        vector_top_idx = set(np.argsort(similarities)[-k:])

        union_idx = bm25_top_idx | vector_top_idx
        return [self.records[index] for index in sorted(union_idx)]
