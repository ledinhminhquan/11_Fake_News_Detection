"""Decision-point logic for the fake-news / fact-check agent (pure, testable).

Five explicit decision points act on the model's own intermediate outputs:
* **D1** input / claim routing — short claim vs full article; extract the central claim.
* **D2** check-worthiness / classifier-confidence gate — skip the (costly) retrieval
  fact-check when the classifier is very confident on a non-claim article; otherwise fact-check.
* **D3** evidence-coverage gate — too little / low-relevance evidence → abstain (or widen).
* **D4** stance-aggregation / verdict gate — combine evidence stances + the classifier
  prior into real / fake / unverified (delegated to ``factcheck.verdict``).
* **D5** confidence / abstain gate — emit a verdict only above the confidence floor,
  else "unverified — needs human review".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..config import AgentConfig

_SENT = re.compile(r"[^.!?]*[.!?]")


def detect_input(text: str, requested_mode: str, cfg: AgentConfig) -> Dict:
    """D1 — is this a short claim or a full article? Extract the central claim."""
    n_words = len((text or "").split())
    is_claim = n_words <= cfg.short_claim_words
    if is_claim:
        claim = text.strip()
    else:
        # central claim heuristic: the first non-trivial sentence (often the lede/headline-claim)
        sents = [s.strip() for s in _SENT.findall(text) if len(s.split()) >= 4]
        claim = sents[0] if sents else text[:200].strip()
    branch = "claim" if is_claim else "article"
    return {"is_claim": is_claim, "claim": claim, "branch": branch, "n_words": n_words}


def checkworthy_gate(clf_prob: float, is_claim: bool, requested_mode: str, cfg: AgentConfig) -> Dict:
    """D2 — decide whether to run the evidence fact-check."""
    if requested_mode == "classify":
        return {"factcheck": False, "branch": "classify_only"}
    if requested_mode == "factcheck":
        return {"factcheck": True, "branch": "forced_factcheck"}
    # auto: fact-check a specific claim always; for an article, skip when the classifier is very confident
    if is_claim:
        return {"factcheck": True, "branch": "claim_factcheck"}
    if clf_prob is not None and clf_prob >= cfg.skip_factcheck_confidence:
        return {"factcheck": False, "branch": "confident_skip"}
    return {"factcheck": cfg.factcheck_in_auto, "branch": "auto_factcheck"}


def coverage_gate(evidence: List[Dict], cfg: AgentConfig) -> Dict:
    """D3 — is there enough relevant evidence to judge?"""
    relevant = [e for e in evidence if float(e.get("relevance", 0.0)) >= cfg.min_evidence_relevance]
    enough = len(relevant) >= cfg.min_evidence
    return {"enough": enough, "n_relevant": len(relevant), "n_total": len(evidence),
            "branch": "ok" if enough else "insufficient"}


def abstain_gate(verdict_result: Dict, cfg: AgentConfig) -> Dict:
    """D5 — final confidence / abstain decision."""
    if verdict_result.get("abstained"):
        return {"branch": "abstain", "abstained": True}
    conf = verdict_result.get("confidence", 0.0)
    if conf < cfg.min_verdict_confidence:
        return {"branch": "abstain", "abstained": True}
    return {"branch": verdict_result.get("verdict", "real"), "abstained": False}


__all__ = ["detect_input", "checkworthy_gate", "coverage_gate", "abstain_gate"]
