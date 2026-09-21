"""White-box retrieval-text optimization: DSRM's Algorithm 2 (Eq. 5) and the
HotFlip machinery it shares with the Corpus Poisoning baseline.

Algorithm 2, lines 16-21: start from R(0) = Q (+) T_m, then for K steps
update R by gradient-guided token flips so that R (+) D_attack becomes more
similar to the target query Q_pos and less similar to negatives Q_neg:

    L = -log exp(sim(E(R (+) D), E(Q_pos)))
              / (exp(sim(E(R (+) D), E(Q_pos))) + sum_i exp(sim(E(R (+) D), E(Q_neg_i))))     (Eq. 5)

Hyperparameters, Section 5.1: HotFlip (Ebrahimi et al., 2018), 30 steps, 40
negatives, a random token per step, top-100 candidates, a greedy choice of the
replacement that helps most, the final tokens decoded back to text. sim is the
inner product (the paper's default retrieval metric). Candidate scoring reuses
agentpoison.trigger_optimization.hotflip_attack (first-order, dot product of the
gradient with every vocabulary embedding).

Interpretation choices (the paper is ambiguous; see FIDELITY.md):
  * Only the R tokens are optimized; D_attack is fixed. Section 5.1 says "only
    tokens in the reasoning and workflow portions" are modified, which we read as
    the R = Q (+) T_m text, since Eq. 5 is over R' (+) D_attack with D fixed.
  * A flip is accepted only if it lowers the loss (monotone), otherwise skipped.
  * Order is R then D (Eq. 5). Algorithm 2 line 21 writes D (+) R -- inconsistent
    with its own Eq. 5; we store what was optimized.
  * Negatives are the n most similar OTHER queries from a pool (the ASB tasks).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from benchmark.attacks.agentpoison.trigger_optimization import hotflip_attack
from benchmark.attacks.dsrm.retrieval import DPREmbedder, Embedder, MiniLMEmbedder

DEFAULT_STEPS = 30  # Section 5.1
DEFAULT_CANDIDATES = 100  # Section 5.1
DEFAULT_NEGATIVES = 40  # Section 5.1
DEFAULT_SEED = 42  # Section 5.1


class GradEncoder:
    """A retriever's DOCUMENT encoder, differentiable w.r.t. its input token
    embeddings, plus its (no-grad) query encoder. `doc_embedding` must match the
    corresponding Embedder.embed_document numerically (tested)."""

    tokenizer = None
    max_length: int = 512
    embedding_matrix: torch.Tensor

    def doc_embedding(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def query_embedding(self, text: str) -> torch.Tensor:
        raise NotImplementedError


def _freeze(module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


class MiniLMGradEncoder(GradEncoder):
    """all-MiniLM-L6-v2: BERT -> mean pooling -> L2 normalize (its three
    sentence-transformers modules), re-run from input embeddings."""

    def __init__(self, embedder: MiniLMEmbedder):
        st = embedder.model
        self.embedder = embedder
        self.tokenizer = st.tokenizer
        self.bert = st[0].auto_model
        self.max_length = int(st.max_seq_length)
        _freeze(self.bert)
        self.embedding_matrix = self.bert.embeddings.word_embeddings.weight.detach()

    def doc_embedding(self, inputs_embeds, attention_mask):
        hidden = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def query_embedding(self, text):
        return torch.as_tensor(self.embedder.embed_query(text), dtype=torch.float32)


class DPRGradEncoder(GradEncoder):
    """DPR context encoder (`pooler_output`) for documents; the question
    encoder (no grad) for queries."""

    def __init__(self, embedder: DPREmbedder):
        self.embedder = embedder
        self.tokenizer = embedder.ctx_tokenizer
        self.model = embedder.ctx_model
        self.max_length = 512
        _freeze(self.model)
        self.embedding_matrix = self.model.ctx_encoder.bert_model.embeddings.word_embeddings.weight.detach()

    def doc_embedding(self, inputs_embeds, attention_mask):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).pooler_output

    def query_embedding(self, text):
        return torch.as_tensor(self.embedder.embed_query(text), dtype=torch.float32)


def grad_encoder_for(embedder: Embedder) -> GradEncoder:
    if isinstance(embedder, MiniLMEmbedder):
        return MiniLMGradEncoder(embedder)
    if isinstance(embedder, DPREmbedder):
        return DPRGradEncoder(embedder)
    raise TypeError(f"no white-box gradient access implemented for {type(embedder).__name__} (MiniLM and DPR only)")


def contrastive_loss_fn(positive: torch.Tensor, negatives: torch.Tensor):
    """Eq. 5, per document: -log softmax over [pos, neg_1..neg_n] of the inner
    products with the document embedding."""

    def loss(doc_emb: torch.Tensor) -> torch.Tensor:
        s_pos = doc_emb @ positive
        s_all = torch.cat([s_pos.unsqueeze(1), doc_emb @ negatives.T], dim=1)
        return -(s_pos - torch.logsumexp(s_all, dim=1))

    return loss


def mean_similarity_loss_fn(queries: torch.Tensor):
    """Corpus Poisoning's objective: maximize the average inner product with a
    set of queries (query-agnostic -- 'retrieved indiscriminately')."""

    def loss(doc_emb: torch.Tensor) -> torch.Tensor:
        return -(doc_emb @ queries.T).mean(dim=1)

    return loss


def select_negatives(encoder: GradEncoder, positive_text: str, pool: list[str], n: int = DEFAULT_NEGATIVES) -> list[str]:
    """The n queries most similar to the positive but different from it."""
    candidates = [t for t in dict.fromkeys(pool) if t != positive_text]
    if not candidates:
        raise ValueError("negative pool has no query different from the positive")
    pos = encoder.query_embedding(positive_text)
    embs = torch.stack([encoder.query_embedding(t) for t in candidates])
    order = (embs @ pos).argsort(descending=True)[:n]
    return [candidates[i] for i in order.tolist()]


def wrap_ids(tokenizer, prefix: list[int], suffix: list[int], max_length: int) -> tuple[list[int], int]:
    """[CLS] prefix suffix [SEP], truncated to max_length by cutting the
    suffix's tail (the optimized prefix is always kept). BERT-family layout
    (MiniLM and DPR are both BERT); built from the CLS/SEP ids directly because
    transformers 5.x tokenizers no longer expose build_inputs_with_special_tokens.
    Returns (ids, offset of the first prefix token = 1)."""
    prefix = prefix[: max_length - 2]
    suffix = suffix[: max_length - 2 - len(prefix)]
    return [tokenizer.cls_token_id] + prefix + suffix + [tokenizer.sep_token_id], 1


@dataclass
class HotFlipOutcome:
    prefix_ids: list[int]
    initial_loss: float
    loss_trace: list[float]  # loss after each step
    accepted: int


def hotflip_optimize(
    encoder: GradEncoder,
    prefix_ids: list[int],
    suffix_ids: list[int],
    loss_fn,
    *,
    steps: int = DEFAULT_STEPS,
    num_candidates: int = DEFAULT_CANDIDATES,
    seed: int = DEFAULT_SEED,
    batch_size: int = 50,
) -> HotFlipOutcome:
    """Each step: pick a random prefix position, take the loss gradient at that
    token's embedding, score every vocabulary row by its dot product with it
    (hotflip_attack), evaluate the top-`num_candidates` replacements for real,
    and keep the best if it lowers the loss."""
    tok = encoder.tokenizer
    rng = random.Random(seed)
    prefix = list(prefix_ids)[: encoder.max_length - 2]
    if not prefix:
        return HotFlipOutcome([], 0.0, [], 0)
    banned = set(tok.all_special_ids) | {i for t, i in tok.get_vocab().items() if t.startswith("[unused")}
    ids, offset = wrap_ids(tok, prefix, list(suffix_ids), encoder.max_length)

    def loss_of(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mask = torch.ones_like(batch)
            return loss_fn(encoder.doc_embedding(encoder.embedding_matrix[batch], mask))

    current = float(loss_of(torch.tensor([ids]))[0])
    initial, trace, accepted = current, [], 0
    for _ in range(steps):
        pos = rng.randrange(len(prefix))
        emb = encoder.embedding_matrix[torch.tensor(ids)].detach().clone().requires_grad_(True)
        loss = loss_fn(encoder.doc_embedding(emb.unsqueeze(0), torch.ones(1, len(ids), dtype=torch.long)))[0]
        loss.backward()
        top = hotflip_attack(emb.grad[offset + pos], encoder.embedding_matrix, increase_loss=False, num_candidates=num_candidates * 3)
        cands = [c for c in top.tolist() if c not in banned and c != prefix[pos]][:num_candidates]
        best_loss, best_token = current, None
        for i in range(0, len(cands), batch_size):
            chunk = cands[i : i + batch_size]
            batch = torch.tensor([ids] * len(chunk))
            batch[:, offset + pos] = torch.tensor(chunk)
            losses = loss_of(batch)
            j = int(losses.argmin())
            if float(losses[j]) < best_loss:
                best_loss, best_token = float(losses[j]), chunk[j]
        if best_token is not None:
            prefix[pos] = best_token
            ids[offset + pos] = best_token
            current, accepted = best_loss, accepted + 1
        trace.append(current)
    return HotFlipOutcome(prefix, initial, trace, accepted)


def loss_of_text(encoder: GradEncoder, text: str, loss_fn) -> float:
    """Loss of a finished text through the same encoder (post-decode check)."""
    ids = encoder.tokenizer(text, truncation=True, max_length=encoder.max_length)["input_ids"]
    with torch.no_grad():
        batch = torch.tensor([ids])
        return float(loss_fn(encoder.doc_embedding(encoder.embedding_matrix[batch], torch.ones_like(batch)))[0])


@dataclass
class WhiteBoxOutcome:
    retrieval_text: str  # decoded R(K)
    entry_text: str  # R(K) (+) D_attack -- what gets stored
    initial_loss: float
    final_loss: float  # recomputed from the decoded TEXT, not the token ids
    accepted: int
    n_negatives: int


class JsonlCache:
    """Tiny keyed on-disk cache for expensive local optimizations."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def optimize_whitebox_entry(
    encoder: GradEncoder,
    *,
    r_init_text: str,
    d_text: str,
    positive_query: str,
    negative_queries: list[str],
    steps: int = DEFAULT_STEPS,
    num_candidates: int = DEFAULT_CANDIDATES,
    seed: int = DEFAULT_SEED,
    cache: JsonlCache | None = None,
    cache_tag: str = "",
) -> WhiteBoxOutcome:
    """Algorithm 2 for one scenario: optimize R (starting from Q (+) T_m) so
    that R (+) D_attack ranks for the target query."""
    tok = encoder.tokenizer
    pos = encoder.query_embedding(positive_query)
    negs = torch.stack([encoder.query_embedding(q) for q in negative_queries])
    loss_fn = contrastive_loss_fn(pos, negs)
    key = _key(cache_tag, r_init_text, d_text, positive_query, negative_queries, steps, num_candidates, seed)
    if cache is not None and (hit := cache.get(key)) is not None:
        entry = hit["entry_text"]
        return WhiteBoxOutcome(hit["retrieval_text"], entry, hit["initial_loss"], loss_of_text(encoder, entry, loss_fn), hit["accepted"], len(negative_queries))

    prefix = tok(r_init_text, add_special_tokens=False)["input_ids"]
    suffix = tok(d_text, add_special_tokens=False)["input_ids"]
    result = hotflip_optimize(encoder, prefix, suffix, loss_fn, steps=steps, num_candidates=num_candidates, seed=seed)
    r_text = tok.decode(result.prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    entry_text = f"{r_text} {d_text}"
    outcome = WhiteBoxOutcome(r_text, entry_text, result.initial_loss, loss_of_text(encoder, entry_text, loss_fn), result.accepted, len(negative_queries))
    if cache is not None:
        cache.put(key, {"retrieval_text": r_text, "entry_text": entry_text, "initial_loss": result.initial_loss, "accepted": result.accepted})
    return outcome


DEFAULT_CORPUS_TOKENS = 50  # Zhong et al. default: --num_adv_passage_tokens 50
DEFAULT_CORPUS_STEPS = 200  # deliberately > DSRM's 30: query-agnostic, optimized once (the original runs 5000)


def optimize_corpus_passage(
    encoder: GradEncoder,
    queries: list[str],
    *,
    num_tokens: int = DEFAULT_CORPUS_TOKENS,
    steps: int = DEFAULT_CORPUS_STEPS,
    num_candidates: int = DEFAULT_CANDIDATES,
    seed: int = DEFAULT_SEED,
) -> str:
    """Corpus Poisoning Attack (Zhong et al., 2023) as released: initialize
    `num_tokens` [MASK] tokens (NOT random characters, as DSRM's text says --
    read from their attack_poison.py) and HotFlip them to maximize the average
    similarity to a query set. One passage, independent of any single target."""
    tok = encoder.tokenizer
    q = torch.stack([encoder.query_embedding(t) for t in queries])
    result = hotflip_optimize(
        encoder, [tok.mask_token_id] * num_tokens, [], mean_similarity_loss_fn(q),
        steps=steps, num_candidates=num_candidates, seed=seed,
    )
    return tok.decode(result.prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


@dataclass
class WhiteBoxConfig:
    """Everything the white-box conditions need at run time."""

    encoder: GradEncoder
    negative_pool: list[str]
    steps: int = DEFAULT_STEPS
    num_candidates: int = DEFAULT_CANDIDATES
    num_negatives: int = DEFAULT_NEGATIVES
    seed: int = DEFAULT_SEED
    corpus_tokens: int = DEFAULT_CORPUS_TOKENS
    corpus_steps: int = DEFAULT_CORPUS_STEPS
    cache: JsonlCache | None = None
    _corpus_passage: str | None = field(default=None, repr=False)

    def corpus_passage(self) -> str:
        """Optimized once and reused for every scenario (it is query-agnostic)."""
        if self._corpus_passage is None:
            self._corpus_passage = optimize_corpus_passage(
                self.encoder, self.negative_pool, num_tokens=self.corpus_tokens,
                steps=self.corpus_steps, num_candidates=self.num_candidates, seed=self.seed,
            )
        return self._corpus_passage
