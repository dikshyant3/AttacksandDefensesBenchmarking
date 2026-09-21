"""Local, differentiable retriever embedder for AgentPoison.

HotFlip-style trigger optimization (trigger_optimization.py) needs gradients of a
retrieval-objective loss with respect to input token embeddings -- an API-only
embedding endpoint (e.g. OpenAI's, used elsewhere in this benchmark for
MemoryGraft) cannot provide that; only a locally-loaded, differentiable encoder
can. Defaults to facebook/dpr-ctx_encoder-single-nq-base, AgentPoison's own
paper/README default target embedder (their `--model dpr-ctx_encoder-single-nq-base`).
"""

import torch
from transformers import AutoTokenizer, DPRContextEncoder


class DPREmbedder:
    def __init__(
        self,
        model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        device: str | None = None,
        max_length: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DPRContextEncoder.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        # (vocab_size, hidden_size) -- HotFlip scores every vocab row against a
        # gradient via a single matmul (trigger_optimization.py's hotflip_attack).
        self.embedding_matrix = self.model.get_input_embeddings().weight

    def tokenize(self, text: str):
        return self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length
        ).to(self.device)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        """Batched, no-grad embedding for KB indexing and plain retrieval queries --
        mirrors local_wikienv.py's load_db()/local_retrieve_step() embedding calls."""
        if not texts:
            return torch.empty(0, self.model.config.hidden_size, device=self.device)
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length
            ).to(self.device)
            pooled = self.model(**encoded).pooler_output
            chunks.append(pooled)
        return torch.cat(chunks, dim=0)

    def embed_ids_with_grad(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Differentiable forward pass. `input_ids` is embedded via a leaf
        `inputs_embeds` tensor so `.grad` populates after backward() -- mechanically
        equivalent to the reference script's `GradientStorage` backward-hook on the
        embedding layer's output (both capture d(loss)/d(embedding_output) at the
        relevant positions), just via a leaf-tensor path instead of a module hook,
        which is more robust across transformers versions than reaching into
        `ctx_encoder.bert_model.embeddings...` internals directly.

        Returns (pooled_output, input_embeds) -- caller must call
        pooled_output-derived-loss.backward() then read input_embeds.grad.
        """
        input_embeds = self.model.get_input_embeddings()(input_ids).detach().clone()
        input_embeds.requires_grad_(True)
        pooled = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask).pooler_output
        return pooled, input_embeds
