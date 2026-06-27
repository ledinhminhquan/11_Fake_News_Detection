"""Fake News & Misinformation Detection.

Classify whether a news article / short claim is **fake** vs **real** (with a
calibrated confidence), AND produce an agentic, evidence-based **fact-check**:
extract the claim → retrieve evidence → judge stance (NLI: support / refute /
neutral) per evidence → aggregate to a verdict with **citations** → **abstain**
when uncertain. A trainable transformer classifier (with a TF-IDF + LogReg
baseline) provides the prior; an agentic finite-state machine wraps the
classify → retrieve → stance → verdict → abstain pipeline. The system **flags
content for human review — it never auto-removes** and always shows its evidence.
"""

__version__ = "1.0.0"
