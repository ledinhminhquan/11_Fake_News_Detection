"""Fact-check layer: claim extraction, evidence retrieval (TF-IDF/dense over the
evidence corpus), stance/NLI scoring per evidence, and verdict aggregation
(classifier prior ⊕ stance votes → real / fake / unverified + citations)."""
