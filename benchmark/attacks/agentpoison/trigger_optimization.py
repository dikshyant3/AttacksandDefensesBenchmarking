"""HotFlip-style, gradient-guided discrete trigger optimization, matching the
actual `--algo ap` code path in algo/trigger_optimization.py (Chen et al.,
"AgentPoison", arXiv:2407.12784) -- traced from their main loop directly
(lines ~540-700 of their file), not inferred from the paper's prose or from
function definitions that turn out to be unused by that path.

An earlier version of this module used a DIFFERENT objective (Maximum Mean
Discrepancy) and swept every trigger position on every iteration. Both were
wrong: `compute_fitness` (MMD-based) is dead code for `--algo ap` in their own
file -- it's commented out at the call site
(`# loss, _, _ = compute_fitness(query_embeddings, db_embeddings)`) and
replaced by `compute_avg_cluster_distance`. And their loop only ever flips ONE
randomly-chosen trigger position per outer iteration
(`token_to_flip = random.randrange(args.num_adv_passage_tokens)`), relying on
many outer iterations (their default: 1000) to eventually cover every
position, not a full sweep each time. Both are fixed.

The actual algorithm, per iteration:
1. Pick one random trigger position to flip this iteration.
2. Accumulate the gradient of `compute_avg_cluster_distance` (query embeddings'
   average distance to the database's GMM cluster centers, minus a variance
   term) w.r.t. that position's token embedding, over several freshly-sampled
   real-query batches (drawn without replacement per epoch -- see
   `_EpochSampler`).
3. HotFlip: score every vocabulary row by dot product with that gradient, take
   the top candidates.
4. Coherence pre-filter: decode each candidate's resulting trigger text and
   score it with a real GPT-2 model; keep only the most fluent ones (their
   `candidate_filter`, `--ppl_filter`) -- this happens BEFORE candidates are
   ever evaluated for retrieval fitness, not as a tie-breaker afterward.
5. Re-evaluate each surviving candidate's real retrieval fitness, averaged
   over `num_grad_iter` fresh batches (matching their double loop -- more
   averaging than scoring against a single batch); keep it only if it
   improves on the current trigger's averaged fitness.

Tractability note: the paper's own defaults (num_iter=1000, num_grad_iter=30,
num_cand=100) assume GPU throughput; on CPU that's hours. Defaults here are
much smaller so a full run finishes in minutes -- same technique and
structure, smaller search budget, matching this benchmark's established
pattern of exposing corpus/trial-count knobs (MINJA's --trials, MemoryGraft's
corpus size) rather than substituting a different, cheaper algorithm. Pass
--paper-scale (run_experiment.py) for their literal values.

Initialization: default is `[MASK]` tokens, matching their actual default
(`if not args.golden_trigger: adv_passage_ids = [tokenizer.mask_token_id] *
args.num_adv_passage_tokens`). `golden_trigger` reproduces their
`--golden_trigger` option: seed from a real, task-relevant phrase instead
(their own per-agent choice for "qa" is "Make efficient calls.").

Target-generation-loss guidance (`--target_gradient_guidance` / `--use_gpt`)
is deliberately NOT implemented -- and after checking algo/utils.py, that's
the faithful choice, not a gap: `target_word_prob`/`target_asr` (the functions
that would compute it) are hardcoded to Agent-Driver's data shape
(`data["ego"]`, `data["perception"]`, "Driving Plan:") with no
StrategyQA/QA-agent equivalent anywhere in their repo, and the README's
example command using `--target_gradient_guidance` is `--agent ad`, not `qa`.
There is no real, working target-loss-guided optimization for the QA agent in
their own shipped code to be faithful *to*.

Implementation detail NOT present in their code as such: the trigger here is
appended after a variable-length real query (`query + " " + trigger`), not
after a fixed-length passage, so trigger token positions vary per query.
Rather than decode-then-re-tokenize the trigger into each query text (which
can silently change the trigger's own token ids at the query/trigger boundary
due to subword re-merging -- a real correctness pitfall for HotFlip), inputs
are built directly at the token-id level (`_build_input_ids`): [CLS] +
query_ids + trigger_ids + [SEP] + padding, with trigger_start/trigger_end
tracked exactly. This keeps the trigger's token ids stable and unambiguous
across the whole optimization, which their code gets for free because its
passages are fixed-length by construction.
"""

import random
import time

import torch
from sklearn.mixture import GaussianMixture


class _EpochSampler:
    """Draws query batches without replacement within a shuffled pass over
    `query_pool`, reshuffling and starting a new pass once exhausted --
    matches their `train_dataloader`/`DataLoader` iteration (shuffled,
    epoch-based) more closely than repeatedly calling `random.sample` with
    replacement on every draw, which can (and, for a small pool relative to
    num_iter*num_grad_iter, will) redraw the same queries far more often than
    a real epoch-based pass would."""

    def __init__(self, pool: list[str], batch_size: int, rng: random.Random):
        self.pool = list(pool)
        self.batch_size = min(batch_size, len(pool))
        self.rng = rng
        self._order: list[str] = []

    def next_batch(self) -> list[str]:
        if len(self._order) < self.batch_size:
            self._order = list(self.pool)
            self.rng.shuffle(self._order)
        batch, self._order = self._order[: self.batch_size], self._order[self.batch_size :]
        return batch


def compute_variance(embeddings: torch.Tensor) -> torch.Tensor:
    mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
    distances = torch.norm(embeddings - mean_embedding, dim=1)
    return torch.mean(distances)


def fit_cluster_centers(db_embeddings: torch.Tensor, n_clusters: int = 5, seed: int = 0) -> torch.Tensor:
    """Ported verbatim: `GaussianMixture(n_components=5, covariance_type='full',
    random_state=0).fit(db_embeddings); cluster_centers = gmm.means_`. Returns
    the (1, n_clusters, hidden) "expanded_cluster_centers" tensor their code
    passes into compute_avg_cluster_distance."""
    n_clusters = min(n_clusters, db_embeddings.shape[0])
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=seed)
    gmm.fit(db_embeddings.detach().cpu().numpy())
    centers = torch.tensor(gmm.means_, dtype=db_embeddings.dtype, device=db_embeddings.device)
    return centers.unsqueeze(0)


def compute_avg_cluster_distance(query_embeddings: torch.Tensor, expanded_cluster_centers: torch.Tensor) -> torch.Tensor:
    """Ported verbatim from trigger_optimization.py: average L2 distance of each
    query embedding to the GMM cluster centers of the real corpus, minus
    0.1x embedding variance. This is the ACTUAL objective their `--algo ap`
    path optimizes -- pulling trigger-bearing queries away from the corpus's
    own cluster structure while keeping them tightly grouped together, so the
    poisoned entries (embedded the same way -- real question + trigger) sit in
    that same carved-out region and dominate retrieval for any future
    trigger-bearing query."""
    expanded_query_embeddings = query_embeddings.unsqueeze(1)
    distances = torch.norm(expanded_query_embeddings - expanded_cluster_centers, dim=2)
    avg_distances = torch.mean(distances, dim=1)
    overall_avg_distance = torch.mean(avg_distances)
    variance = compute_variance(query_embeddings)
    return overall_avg_distance - 0.1 * variance


def hotflip_attack(
    averaged_grad: torch.Tensor,
    embedding_matrix: torch.Tensor,
    increase_loss: bool = True,
    num_candidates: int = 10,
) -> torch.Tensor:
    """Ported verbatim from trigger_optimization.py's hotflip_attack: score every
    vocabulary row by its dot product with the gradient, return the top-k ids
    that would most increase (or decrease) the loss if substituted in."""
    with torch.no_grad():
        scores = torch.matmul(embedding_matrix, averaged_grad)
        if not increase_loss:
            scores = scores * -1
        k = min(num_candidates, scores.numel())
        _, top_k_ids = scores.topk(k)
    return top_k_ids


class PerplexityFilter:
    """Wraps a real GPT-2 model for the coherence pre-filter (their
    `candidate_filter`, `--ppl_filter`). Loaded lazily so runs that don't want
    it (`ppl_filter=False`) never download/load GPT-2 at all."""

    def __init__(self, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device

    def perplexity(self, text: str) -> float:
        encoded = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if encoded.numel() < 2:
            return float("inf")
        with torch.no_grad():
            loss = self.model(encoded, labels=encoded).loss
        return float(torch.exp(loss))

    def filter_candidates(
        self,
        candidates: list[int],
        token_to_flip: int,
        trigger_ids: list[int],
        src_tokenizer,
        num_candidates: int,
        sample: bool = False,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> list[int]:
        """`sample=False` (default, matches their default): deterministic top-k
        by lowest perplexity. `sample=True`: their `--coh_sample` mode, flagged
        in their own file as the more literally Eq.10/Algorithm-1-line-7-faithful
        variant -- draw `num_candidates` distinct tokens via
        softmax(-log_ppl / temperature), so lower perplexity is more likely but
        not guaranteed, preserving some candidate diversity instead of always
        collapsing to the objectively most fluent options."""
        ppl_values = []
        for candidate in candidates:
            trial_ids = list(trigger_ids)
            trial_ids[token_to_flip] = candidate
            text = src_tokenizer.decode(trial_ids, skip_special_tokens=True)
            ppl_values.append(self.perplexity(text))
        ppl_tensor = torch.tensor(ppl_values)
        k = min(num_candidates, len(candidates))

        if not sample:
            _, top_idx = (-ppl_tensor).topk(k)
            return [candidates[i] for i in top_idx.tolist()]

        log_ppl = torch.log(ppl_tensor.clamp_min(1e-6))
        probs = torch.softmax(-log_ppl / max(temperature, 1e-6), dim=0)
        generator = torch.Generator().manual_seed(rng.randrange(2**31)) if rng else None
        chosen = torch.multinomial(probs, k, replacement=False, generator=generator)
        return [candidates[i] for i in chosen.tolist()]


def _build_input_ids(tokenizer, query: str, trigger_ids: list[int], max_length: int):
    query_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    budget = max(max_length - 2 - len(trigger_ids), 0)
    query_ids = query_ids[:budget]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    ids = [cls_id] + query_ids + list(trigger_ids) + [sep_id]
    trigger_start = 1 + len(query_ids)
    trigger_end = trigger_start + len(trigger_ids)

    pad_len = max_length - len(ids)
    attention = [1] * len(ids) + [0] * pad_len
    ids = ids + [pad_id] * pad_len
    return ids, attention, trigger_start, trigger_end


def _non_special_vocab_ids(tokenizer, vocab_size: int) -> list[int]:
    special = set(tokenizer.all_special_ids)
    return [i for i in range(vocab_size) if i not in special]


def optimize_trigger(
    embedder,
    query_pool: list[str],
    db_embeddings: torch.Tensor,
    num_trigger_tokens: int = 6,
    num_iter: int = 8,
    num_grad_iter: int = 3,
    num_cand: int = 15,
    batch_size: int = 4,
    n_clusters: int = 5,
    ppl_filter: bool = True,
    coh_sample: bool = False,
    coh_temperature: float = 1.0,
    golden_trigger: str | None = None,
    seed: int = 0,
    initial_trigger_ids: list[int] | None = None,
    verbose: bool = False,
) -> dict:
    """Runs the HotFlip + GMM-cluster-distance search described in this module's
    docstring. Returns {"trigger_ids", "trigger_text", "fitness_history"}.

    Initialization precedence: `initial_trigger_ids` (explicit ids) >
    `golden_trigger` (a seed phrase, tokenized -- their `--golden_trigger`) >
    default `[MASK]` tokens (their actual default).
    """
    rng = random.Random(seed)
    tokenizer = embedder.tokenizer

    if initial_trigger_ids is not None:
        trigger_ids = list(initial_trigger_ids)
    elif golden_trigger is not None:
        trigger_ids = tokenizer(golden_trigger, add_special_tokens=False)["input_ids"][:num_trigger_tokens]
        num_trigger_tokens = len(trigger_ids)
    else:
        mask_id = tokenizer.mask_token_id
        trigger_ids = [mask_id] * num_trigger_tokens

    cluster_centers = fit_cluster_centers(db_embeddings, n_clusters=n_clusters, seed=seed)
    ppl = PerplexityFilter(device=embedder.device) if ppl_filter else None
    sampler = _EpochSampler(query_pool, batch_size, rng)

    def avg_fitness_over_batches(ids: list[int], batches: list[list[str]]) -> float:
        with torch.no_grad():
            scores = []
            for queries in batches:
                full_texts = [f"{q} {tokenizer.decode(ids)}".strip() for q in queries]
                pooled = embedder.embed_texts(full_texts, batch_size=len(full_texts))
                scores.append(float(compute_avg_cluster_distance(pooled, cluster_centers)))
            return sum(scores) / len(scores)

    fitness_history: list[float] = []
    current_fitness = avg_fitness_over_batches(trigger_ids, [sampler.next_batch()])
    fitness_history.append(current_fitness)

    for iteration in range(num_iter):
        iter_start = time.time()
        # ONE randomly-chosen position per outer iteration -- matches
        # `token_to_flip = random.randrange(args.num_adv_passage_tokens)`, not a
        # full sweep of every position.
        token_to_flip = rng.randrange(len(trigger_ids))
        if verbose:
            print(
                f"[trigger_optimization] iter {iteration + 1}/{num_iter} starting -- "
                f"flip_pos={token_to_flip}, accumulating gradient over {num_grad_iter} batches...",
                flush=True,
            )

        accumulated_grad = None
        grad_batches = []
        for grad_step in range(num_grad_iter):
            batch_queries = sampler.next_batch()
            grad_batches.append(batch_queries)
            batch_input_ids, batch_attention, batch_spans = [], [], []
            for q in batch_queries:
                ids, attn, start, end = _build_input_ids(tokenizer, q, trigger_ids, embedder.max_length)
                batch_input_ids.append(ids)
                batch_attention.append(attn)
                batch_spans.append((start, end))

            input_ids = torch.tensor(batch_input_ids, device=embedder.device)
            attention_mask = torch.tensor(batch_attention, device=embedder.device)
            pooled, input_embeds = embedder.embed_ids_with_grad(input_ids, attention_mask)
            fitness = compute_avg_cluster_distance(pooled, cluster_centers)
            loss = -fitness
            embedder.model.zero_grad(set_to_none=True)
            loss.backward()

            grad = input_embeds.grad  # (batch, seq, hidden)
            for row, (start, _end) in enumerate(batch_spans):
                token_grad = grad[row, start + token_to_flip, :]  # just the flipped position
                accumulated_grad = token_grad if accumulated_grad is None else accumulated_grad + token_grad

            if verbose and (grad_step + 1) % max(num_grad_iter // 5, 1) == 0:
                print(
                    f"[trigger_optimization]   grad batch {grad_step + 1}/{num_grad_iter} "
                    f"({time.time() - iter_start:.1f}s elapsed)",
                    flush=True,
                )

        if verbose:
            print(
                f"[trigger_optimization]   gradient done ({time.time() - iter_start:.1f}s), "
                f"generating {num_cand * 10 if ppl_filter else num_cand} raw candidates...",
                flush=True,
            )

        raw_num_candidates = num_cand * 10 if ppl_filter else num_cand
        raw_candidates = hotflip_attack(
            accumulated_grad, embedder.embedding_matrix, increase_loss=True, num_candidates=raw_num_candidates
        ).tolist()

        if verbose and ppl_filter:
            print(
                f"[trigger_optimization]   PPL-filtering {len(raw_candidates)} candidates down to "
                f"{num_cand} ({time.time() - iter_start:.1f}s elapsed)...",
                flush=True,
            )

        candidates = (
            ppl.filter_candidates(
                raw_candidates, token_to_flip, trigger_ids, tokenizer, num_cand,
                sample=coh_sample, temperature=coh_temperature, rng=rng,
            )
            if ppl_filter
            else raw_candidates
        )

        if verbose:
            print(
                f"[trigger_optimization]   evaluating {len(candidates)} candidates x {num_grad_iter} "
                f"batches each ({time.time() - iter_start:.1f}s elapsed)...",
                flush=True,
            )

        # Score each candidate against `num_grad_iter` fresh batches, matching
        # their double loop (more averaging than a single-batch estimate).
        eval_batches = [sampler.next_batch() for _ in range(num_grad_iter)]
        best_id = trigger_ids[token_to_flip]
        best_fitness = current_fitness
        for cand_idx, cand_id in enumerate(candidates):
            trial_ids = list(trigger_ids)
            trial_ids[token_to_flip] = cand_id
            trial_fitness = avg_fitness_over_batches(trial_ids, eval_batches)
            if trial_fitness > best_fitness:
                best_fitness = trial_fitness
                best_id = cand_id
            if verbose and (cand_idx + 1) % max(len(candidates) // 5, 1) == 0:
                print(
                    f"[trigger_optimization]   candidate {cand_idx + 1}/{len(candidates)} "
                    f"best_so_far={best_fitness:.4f} ({time.time() - iter_start:.1f}s elapsed)",
                    flush=True,
                )
        trigger_ids[token_to_flip] = best_id
        current_fitness = best_fitness

        fitness_history.append(current_fitness)
        if verbose:
            print(
                f"[trigger_optimization] iter {iteration + 1}/{num_iter} DONE in "
                f"{time.time() - iter_start:.1f}s -- flip_pos={token_to_flip} fitness={current_fitness:.4f} "
                f"trigger={tokenizer.decode(trigger_ids).strip()!r}",
                flush=True,
            )

    trigger_text = tokenizer.decode(trigger_ids).strip()
    return {"trigger_ids": trigger_ids, "trigger_text": trigger_text, "fitness_history": fitness_history}
