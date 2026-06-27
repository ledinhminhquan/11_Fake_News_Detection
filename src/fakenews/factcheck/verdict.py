"""Verdict aggregation: combine the per-evidence stances with the classifier prior
into a final verdict (real / fake / unverified) with a confidence and a rationale.

Semantics: the *claim* is what is being checked. Evidence that **supports** the
claim is a signal the claim is true → the news is **real**; evidence that
**refutes** it → the news is **fake**. Evidence dominates; the classifier prior
(P(fake)) only nudges the decision (it captures style/source, not truth). When the
evidence margin is too small or coverage is insufficient, the verdict is
**unverified** (abstain) — a first-class, safe outcome.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..config import AgentConfig


def aggregate_verdict(prior_fake: float, evidence: List[Dict[str, Any]], cfg: AgentConfig
                      ) -> Dict[str, Any]:
    """``evidence`` items carry ``stance`` (support/refute/neutral), ``score``
    (stance confidence) and ``relevance``. Returns the verdict dict."""
    support_w = 0.0
    refute_w = 0.0
    n_support = n_refute = 0
    for e in evidence:
        w = float(e.get("score", 0.0)) * float(e.get("relevance", 0.0))
        st = e.get("stance", "neutral")
        if st == "support":
            support_w += w
            n_support += 1
        elif st == "refute":
            refute_w += w
            n_refute += 1

    total = support_w + refute_w
    # evidence signal toward FAKE in [-1, 1]
    ev_fake = (refute_w - support_w) / total if total > 1e-9 else 0.0
    prior_signal = (float(prior_fake) - 0.5) * 2.0 if prior_fake is not None else 0.0
    # blend: evidence dominates, prior is a soft nudge
    net_fake = (1.0 - cfg.prior_weight) * ev_fake + cfg.prior_weight * prior_signal
    margin = abs(net_fake)

    enough_evidence = (n_support + n_refute) >= 1 and total > 1e-6
    if not enough_evidence or margin < cfg.verdict_margin:
        return {"verdict": "unverified", "confidence": round(0.5 + margin / 2, 3),
                "abstained": True, "n_support": n_support, "n_refute": n_refute,
                "rationale": ("No sufficiently relevant evidence found — flagged for human review."
                              if not enough_evidence else
                              "Evidence is mixed / inconclusive — flagged for human review."),
                "net_fake": round(net_fake, 3)}

    verdict = "fake" if net_fake > 0 else "real"
    confidence = round(min(0.99, 0.5 + margin / 2), 3)
    if verdict == "fake":
        rationale = (f"{n_refute} evidence passage(s) refute the claim"
                     + (f" (classifier prior P(fake)={prior_fake:.2f})." if prior_fake is not None else "."))
    else:
        rationale = (f"{n_support} evidence passage(s) support the claim"
                     + (f" (classifier prior P(fake)={prior_fake:.2f})." if prior_fake is not None else "."))
    return {"verdict": verdict, "confidence": confidence, "abstained": False,
            "n_support": n_support, "n_refute": n_refute, "rationale": rationale,
            "net_fake": round(net_fake, 3)}


__all__ = ["aggregate_verdict"]
