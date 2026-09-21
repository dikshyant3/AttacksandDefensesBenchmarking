"""DSRM (Deceptive Semantic Reasoning Manipulation): memory-poisoning attack
on RAG-based LLM agents, reproducing Jing, Li, Dong, Zhou, Liu, "Memory
poisoning attacks on retrieval-augmented Large Language Model agents via
deceptive semantic reasoning", Engineering Applications of Artificial
Intelligence 167 (2026) 113968.

No DSRM-specific code repository accompanies the paper -- the paper's own
Ethical Considerations section states they deliberately withhold executable
attack scripts. Evaluated on the real, public Agent Security Bench (ASB;
Zhang et al., arXiv:2410.02644, github.com/agiresearch/ASB, MIT-licensed) --
see datasets/provenance.json for the exact vendored files and commit, and
FIDELITY.md for what is transcribed verbatim from the paper vs. reconstructed.
"""
