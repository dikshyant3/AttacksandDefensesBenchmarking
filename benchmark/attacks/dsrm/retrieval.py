"""The retriever layer: R(Q, D, K) in Eq. 1 -- embed a knowledge base D,
embed a query Q, return its top-K most similar entries.

Three retrievers named in the paper (Section 5.1, Table 1): DPR
(Karpukhin et al., 2020), RealM (Guu et al., 2020), MiniLM (Wang et al.,
2020). DPR and MiniLM run in this environment. RealMEmbedder is implemented
against HuggingFace's RealmEmbedder, but that class only exists in
transformers 4.x -- this environment has 5.x, so constructing it raises a
clear RuntimeError here. (An earlier version of this file claimed REALM has
"no released checkpoint"; that was wrong -- the limit is the installed
transformers version, not the model. See FIDELITY.md.)

Similarity metrics (Table 10: ip / cos / L2) are all exposed uniformly as
"higher score = better match" -- L2 is implemented as *negative* squared
distance so the top-k argmax logic never needs a metric-specific sign flip.
`cosine_similarity` is also Eq. 4's S(P_i, Q) = E(P_i).E(Q) / (||E(P_i)||
||E(Q)||), reused as-is by the (not yet built) Self-Refine Module in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

Metric = Literal["ip", "cos", "l2"]


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)


def neg_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -float(np.linalg.norm(a - b))


SIMILARITY_FNS: dict[Metric, Callable[[np.ndarray, np.ndarray], float]] = {
    "ip": dot_product,
    "cos": cosine_similarity,
    "l2": neg_l2_distance,
}


class Embedder:
    """Common interface. embed_query/embed_document are separate methods
    because real retrievers split into two camps: DPR is ASYMMETRIC (a
    question tower and a context/passage tower trained separately -- Eq. 1
    of Karpukhin et al. 2020), while MiniLM-style sentence embedders are
    SYMMETRIC (one tower encodes both). A symmetric embedder just points
    both methods at the same encode call."""

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_document(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_documents_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_document(t) for t in texts]


DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedder(Embedder):
    """Wang et al., 2020 (MiniLM) -- symmetric sentence embedding via
    sentence-transformers, same library/model family already used in this
    repo's zombie_agents/local_embeddings.py. Loaded lazily so importing this
    module doesn't require the model until it's actually used."""

    def __init__(self, model_name: str = DEFAULT_MINILM_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text))

    def embed_document(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text))

    def embed_documents_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(v) for v in self.model.encode(texts)]


DEFAULT_DPR_QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
DEFAULT_DPR_CTX_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"


class DPREmbedder(Embedder):
    """Karpukhin et al., 2020 -- real dual-encoder checkpoints, question tower
    and context/passage tower loaded separately (asymmetric, per the paper's
    own architecture: two independently-trained BERT encoders, not a shared
    one). This is exactly the retriever type DSRM's threat model describes
    tampering with when white-box access is assumed (Algorithm 2)."""

    def __init__(
        self,
        question_model: str = DEFAULT_DPR_QUESTION_MODEL,
        ctx_model: str = DEFAULT_DPR_CTX_MODEL,
        device: str = "cpu",
    ):
        import torch
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizerFast,
        )

        self.device = device
        self.torch = torch
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(question_model)
        self.q_model = DPRQuestionEncoder.from_pretrained(question_model).to(device).eval()
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(ctx_model)
        self.ctx_model = DPRContextEncoder.from_pretrained(ctx_model).to(device).eval()

    def embed_query(self, text: str) -> np.ndarray:
        inputs = self.q_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = self.q_model(**inputs).pooler_output
        return out[0].cpu().numpy()

    def embed_document(self, text: str) -> np.ndarray:
        inputs = self.ctx_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = self.ctx_model(**inputs).pooler_output
        return out[0].cpu().numpy()


DEFAULT_REALM_CHECKPOINT = "google/realm-cc-news-pretrained-embedder"


def _import_realm_classes():
    """ReaLM's HuggingFace classes exist only in transformers 4.x (top-level,
    later under transformers.models.deprecated.realm); transformers 5.x
    removed them entirely. Kept as its own function so tests can simulate
    either environment without touching the installed package."""
    try:
        from transformers import RealmEmbedder, RealmTokenizer

        return RealmEmbedder, RealmTokenizer
    except (ImportError, AttributeError):
        pass
    try:
        from transformers.models.deprecated.realm import RealmEmbedder, RealmTokenizer

        return RealmEmbedder, RealmTokenizer
    except (ImportError, AttributeError) as exc:
        import transformers

        raise RuntimeError(
            f"ReaLM (Guu et al., 2020) is not available in the installed transformers "
            f"{transformers.__version__}: the RealmEmbedder/RealmTokenizer classes were removed "
            f"in transformers 5.x and only exist in 4.x. This is an ENVIRONMENT limit, not a "
            f"missing checkpoint -- see FIDELITY.md ('ReaLM') for the options (separate 4.x "
            f"environment, or vendoring the deprecated modeling code)."
        ) from exc


class RealMEmbedder(Embedder):
    """Guu et al., 2020 (REALM) via HuggingFace's `RealmEmbedder`, which
    projects the BERT [CLS] state to a 128-d retrieval vector
    (`projected_score`) -- the vector REALM's own retriever ranks by inner
    product. Symmetric here: the same embedder encodes queries and documents.

    The paper says nothing about how it ran ReaLM (no checkpoint, no library,
    no max length -- only one descriptive sentence and the observation that
    ReaLM is "almost unaffected by variations in L", Section 5.2). The
    checkpoint below is OUR choice (REALM's own CC-News-pretrained embedder),
    configurable via `checkpoint`; the choice of embedder over the scorer or
    the ORQA-finetuned variants is likewise an unverified assumption.
    Requires an environment where `_import_realm_classes` succeeds."""

    def __init__(self, checkpoint: str = DEFAULT_REALM_CHECKPOINT, device: str = "cpu"):
        import torch

        realm_embedder_cls, realm_tokenizer_cls = _import_realm_classes()
        self.torch = torch
        self.device = device
        self.tokenizer = realm_tokenizer_cls.from_pretrained(checkpoint)
        self.model = realm_embedder_cls.from_pretrained(checkpoint).to(device).eval()

    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs).projected_score
        return out[0].cpu().numpy()

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(text)


EMBEDDER_REGISTRY: dict[str, Callable[..., Embedder]] = {
    "minilm": MiniLMEmbedder,
    "dpr": DPREmbedder,
    "realm": RealMEmbedder,
}


def get_embedder(name: str, **kwargs) -> Embedder:
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(f"unknown retriever {name!r} (choices: {sorted(EMBEDDER_REGISTRY)})")
    return EMBEDDER_REGISTRY[name](**kwargs)


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    """D in R(Q, D, K): a flat set of (id, text, metadata, embedding)
    entries. Deliberately in-memory / brute-force cosine-scan, not an ANN
    index -- the paper's own scale (a handful of planted decisions plus a
    few hundred benign entries, Table 5's memory-update experiment tops out
    at 1000) doesn't need one, and this repo's other attacks
    (agentpoison/kb_store.py) use the same brute-force pattern."""

    embedder: Embedder
    _ids: list[str] = field(default_factory=list)
    _texts: list[str] = field(default_factory=list)
    _metadata: list[dict] = field(default_factory=list)
    _embeddings: list[np.ndarray] = field(default_factory=list)

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self._ids.append(doc_id)
        self._texts.append(text)
        self._metadata.append(metadata or {})
        self._embeddings.append(self.embedder.embed_document(text))

    def add_batch(self, entries: list[tuple[str, str, dict]]) -> None:
        texts = [text for _, text, _ in entries]
        embeddings = self.embedder.embed_documents_batch(texts)
        for (doc_id, text, metadata), emb in zip(entries, embeddings):
            self._ids.append(doc_id)
            self._texts.append(text)
            self._metadata.append(metadata or {})
            self._embeddings.append(emb)

    def __len__(self) -> int:
        return len(self._ids)

    def copy(self) -> "KnowledgeBase":
        """Cheap shallow copy -- shares the embedder and reuses already-
        computed embeddings (no re-embedding). Used by campaign.py to seed
        many per-scenario KBs from one shared, expensive-to-build background
        pool without recomputing its embeddings each time."""
        kb = KnowledgeBase(embedder=self.embedder)
        kb._ids = list(self._ids)
        kb._texts = list(self._texts)
        kb._metadata = list(self._metadata)
        kb._embeddings = list(self._embeddings)
        return kb

    def retrieve(self, query: str, k: int = 5, metric: Metric = "ip") -> list[RetrievedDoc]:
        if not self._ids:
            return []
        query_emb = self.embedder.embed_query(query)
        sim_fn = SIMILARITY_FNS[metric]
        scored = [
            RetrievedDoc(doc_id=doc_id, text=text, score=sim_fn(query_emb, emb), metadata=meta)
            for doc_id, text, meta, emb in zip(self._ids, self._texts, self._metadata, self._embeddings)
        ]
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]

    def contains_doc_in_topk(self, query: str, doc_id: str, k: int = 5, metric: Metric = "ip") -> bool:
        """Used for the Retrieval Rate (RR) metric: was a specific planted
        entry inside the top-K results for this query?"""
        return any(d.doc_id == doc_id for d in self.retrieve(query, k=k, metric=metric))
