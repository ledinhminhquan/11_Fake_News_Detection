"""The fake-news / fact-check agent — a deterministic FSM that turns an article or
a claim into a classifier signal + an evidence-grounded verdict.

    parse (D1 claim routing) -> classify (D2 check-worthiness gate)
        -> factcheck: retrieve + stance (D3 coverage gate, D4 verdict gate)
        -> present (D5 abstain gate)

Holds a classifier, an evidence retriever and a stance model (loaded once). Runs
fully offline (TF-IDF + LogReg classifier + lexical stance over the seed evidence)
and upgrades when fine-tuned models + an NLI model are present. Every step is timed
and traced; the system **flags for review and shows its evidence — it never
auto-removes**. Same input + same models + brain disabled ⇒ identical output.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class FakeNewsAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True,
                 classifier=None, retriever=None, stance=None):
        self.cfg = cfg or AppConfig()
        if classifier is None:
            from ..models.classifier import load_classifier
            classifier = load_classifier(self.cfg.classifier,
                                         prefer="transformer" if load_model else "tfidf")
        if retriever is None:
            from ..data.dataset import load_evidence
            from ..factcheck.retriever import EvidenceRetriever
            retriever = EvidenceRetriever.from_corpus(self.cfg.retrieval, load_evidence(self.cfg))
        if stance is None:
            from ..factcheck.stance import load_stance
            stance = load_stance(self.cfg.stance, prefer="nli" if load_model else "lexical")
        self.classifier = classifier
        self.retriever = retriever
        self.stance = stance
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.request_log_path) if self.cfg.serving.log_requests else None

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def run(self, text: str = "", *, title: str = "", mode: str = "auto", save: bool = True) -> JobState:
        job = JobState(text=(text or "").strip(), title=(title or "").strip(), requested_mode=mode)
        if not (job.text or job.title):
            job.status = JobStatus.FAILED
            return job
        t0 = time.perf_counter()
        job = self._step(job, "parse", lambda: tools.tool_parse(job, self.cfg),
                         summary="claim routing (D1)")
        job = self._step(job, "classify", lambda: tools.tool_classify(job, self.cfg, classifier=self.classifier),
                         summary="classify + check-worthiness (D2)")
        job = self._step(job, "factcheck", lambda: tools.tool_factcheck(job, self.cfg,
                                                                        retriever=self.retriever, stance=self.stance),
                         summary="retrieve + stance + verdict (D3,D4)")
        job = self._step(job, "present", lambda: tools.tool_present(job, self.cfg, brain=self.brain),
                         summary="abstain gate (D5)")
        # optional brain rationale (advisory)
        if self.brain.available() and not job.abstained and job.evidence:
            expl = self.brain.explain(job.claim, job.verdict, job.evidence)
            if expl:
                job.rationale = expl
                job.metrics["brain_used"] = True
        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if save and self._log is not None:
            try:
                self._log.log("decision", mode=job.requested_mode, verdict=job.verdict,
                              clf_label=job.clf_label, abstained=job.abstained, metrics=job.metrics)
            except Exception:
                pass
        return job

    def classify(self, text: str, title: str = "") -> dict:
        """Fast classifier-only path (no fact-check)."""
        job = self.run(text, title=title, mode="classify", save=False)
        return {"label": job.clf_label, "probability": job.clf_prob, "prior_fake": job.clf_prior_fake,
                "model_version": job.model_versions.get("classifier", "?")}


_AGENT: Optional[FakeNewsAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> FakeNewsAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = FakeNewsAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["FakeNewsAgent", "get_agent"]
