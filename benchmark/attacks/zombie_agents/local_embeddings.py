"""Local (HuggingFace sentence-transformers) embeddings, as an alternative to
the OpenAI API embeddings used elsewhere in this project.

Why this exists: the paper never names an embedding model for the RAG setup.
We've been using OpenAI's `text-embedding-3-small` throughout (this project's
consistent default); this module lets a run test whether the *embedding
model itself* is part of why our Recall@k trails the paper (FIDELITY.md's
diagnosed-but-untested hypothesis #1) -- a real, different embedding
geometry could rank the infection entry differently relative to real
queries. Runs fully locally: no API key, no per-call cost.

Model choice: `all-MiniLM-L6-v2` -- sentence-transformers' own flagship
default and the most commonly deployed general-purpose sentence embedding
model in practice (the default in most RAG tutorials/frameworks). The paper
names no embedding model, so there's nothing to match here beyond "a
standard, widely-used choice" -- pass `--embedding-model` to use a different
one (e.g. the larger `all-mpnet-base-v2`).
"""

from __future__ import annotations

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


def load_sentence_transformer(model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def make_local_embed_fn(model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL, cache: dict | None = None, model=None):
    """Single-text embed_fn, matching the shape run_rag_campaign expects
    (embed_fn(text) -> np.ndarray). Cached per unique text, same convention
    as the OpenAI embed_fn elsewhere in this module. Pass `model` to reuse an
    already-loaded SentenceTransformer instead of loading a second copy."""
    model = model or load_sentence_transformer(model_name)
    cache = cache if cache is not None else {}

    def embed(text: str):
        if text not in cache:
            cache[text] = model.encode(text)
        return cache[text]

    return embed


def make_local_embed_batch_fn(model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL, model=None):
    """Batch embed_fn for memory.build_filler_entries -- encodes a whole
    batch in one local call (sentence-transformers batches internally on
    CPU/GPU), no network round-trip at all. Pass `model` to reuse an
    already-loaded SentenceTransformer instead of loading a second copy."""
    model = model or load_sentence_transformer(model_name)

    def embed_batch(texts: list[str]):
        return list(model.encode(texts))

    return embed_batch
