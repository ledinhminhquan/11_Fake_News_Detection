"""Agent tools — each operates on the JobState and returns it.

Tools run against the classifier, the evidence retriever and the stance model.
They work with the TF-IDF + LogReg classifier + the lexical stance over the seed
evidence, so the whole classify → retrieve → stance → verdict pipeline runs
offline for tests/CI. The orchestrator wraps each call with timing/trace; tools
never raise past it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import AppConfig
from ..factcheck.verdict import aggregate_verdict
from ..logging_utils import get_logger
from ..models.classifier import LABELS
from . import policy
from .state import Decision, JobState, JobStatus

logger = get_logger(__name__)


def tool_parse(job: JobState, cfg: AppConfig) -> JobState:
    """D1 — input / claim routing."""
    content = (job.title + ". " + job.text).strip(". ") if job.title else job.text
    route = policy.detect_input(content, job.requested_mode, cfg.agent)
    job.is_claim = route["is_claim"]
    job.claim = route["claim"]
    job.add_decision(Decision("D1", "claim_routing", route["branch"],
                              detail=f"{route['n_words']} words"))
    return job


def tool_classify(job: JobState, cfg: AppConfig, *, classifier) -> JobState:
    """Run the fake-news classifier (prior) + D2 check-worthiness gate."""
    content = (job.title + ". " + job.text).strip(". ") if job.title else job.text
    label, prob = classifier.predict(content)
    job.clf_label = LABELS.get(label, "real")
    job.clf_prob = round(float(prob), 4)
    job.clf_prior_fake = round(float(classifier.predict_proba(content)), 4)
    job.model_versions["classifier"] = getattr(classifier, "name", "?") + ":" + getattr(classifier, "version", "?")
    gate = policy.checkworthy_gate(job.clf_prob, job.is_claim, job.requested_mode, cfg.agent)
    job.check_worthy = gate["factcheck"]
    job.add_decision(Decision("D2", "checkworthy_gate", gate["branch"], score=job.clf_prior_fake,
                              detail=f"clf={job.clf_label}({job.clf_prob:.2f}), factcheck={gate['factcheck']}"))
    return job


def tool_factcheck(job: JobState, cfg: AppConfig, *, retriever, stance) -> JobState:
    """D3 evidence-coverage gate + D4 stance aggregation → verdict."""
    if not job.check_worthy:
        # classifier-only path: verdict follows the prior, no evidence retrieved
        job.verdict = job.clf_label
        job.confidence = job.clf_prob
        job.rationale = f"Classifier-only signal ({job.clf_label}, P={job.clf_prob:.2f}); no evidence retrieved."
        job.add_decision(Decision("D3", "coverage_gate", "skipped", detail="classifier-only"))
        job.add_decision(Decision("D4", "verdict_gate", "classifier_only", score=job.clf_prior_fake))
        return job
    # retrieve evidence for the claim
    ev = retriever.retrieve(job.claim, cfg.retrieval.top_k)
    cov = policy.coverage_gate(ev, cfg.agent)
    job.add_decision(Decision("D3", "coverage_gate", cov["branch"],
                              detail=f"relevant={cov['n_relevant']}/{cov['n_total']}"))
    # score stance for each evidence
    scored: List[Dict[str, Any]] = []
    for e in ev:
        st, sc = stance.score(job.claim, e["text"])
        scored.append({**e, "stance": st, "score": sc})
    job.evidence = scored
    job.model_versions["stance"] = getattr(stance, "name", "?") + ":" + getattr(stance, "version", "?")
    if not cov["enough"]:
        job.verdict = "unverified"
        job.confidence = 0.5
        job.abstained = True
        job.rationale = "No sufficiently relevant evidence found — flagged for human review."
        job.add_decision(Decision("D4", "verdict_gate", "insufficient_evidence"))
        return job
    result = aggregate_verdict(job.clf_prior_fake, scored, cfg.agent)
    job.verdict = result["verdict"]
    job.confidence = result["confidence"]
    job.abstained = result["abstained"]
    job.rationale = result["rationale"]
    job.n_support = result["n_support"]
    job.n_refute = result["n_refute"]
    job.add_decision(Decision("D4", "verdict_gate", result["verdict"], score=result.get("net_fake"),
                              detail=f"support={result['n_support']}, refute={result['n_refute']}"))
    return job


def tool_present(job: JobState, cfg: AppConfig, *, brain=None) -> JobState:
    """D5 — confidence / abstain gate + finalize."""
    gate = policy.abstain_gate({"abstained": job.abstained, "confidence": job.confidence,
                                "verdict": job.verdict}, cfg.agent)
    if gate["abstained"]:
        job.verdict = "unverified"
        job.abstained = True
        if not job.rationale:
            job.rationale = "Low confidence — flagged for human review."
    job.add_decision(Decision("D5", "abstain_gate", gate["branch"], score=job.confidence,
                              detail=f"abstained={gate['abstained']}"))
    job.status = JobStatus.UNVERIFIED if job.abstained else JobStatus.COMPLETED
    return job


__all__ = ["tool_parse", "tool_classify", "tool_factcheck", "tool_present"]
