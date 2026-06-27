"""Shared state types for the fake-news / fact-check agent.

The agent is a deterministic finite-state machine; ``JobState`` is its context
object, carrying the input text, the classifier's prior, the extracted claim, the
retrieved evidence with per-item stance, the aggregated verdict, and a full audit
trail (``Decision`` records for D1–D5 + ``ToolTrace`` per tool call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    CLASSIFIED = "classified"
    CHECKED = "checked"
    COMPLETED = "completed"
    UNVERIFIED = "unverified"     # abstain gate fired: not enough evidence for a confident verdict
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str               # D1..D5
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class Evidence:
    text: str
    source: str = ""
    stance: str = "neutral"        # support | refute | neutral
    score: float = 0.0             # stance confidence
    relevance: float = 0.0         # retrieval relevance

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "source": self.source, "stance": self.stance,
                "score": round(float(self.score), 3), "relevance": round(float(self.relevance), 3)}


@dataclass
class JobState:
    # ---- inputs --------------------------------------------------------------
    text: str = ""                 # the article body / claim
    title: str = ""
    requested_mode: str = "auto"   # auto | classify | factcheck
    # ---- derived -------------------------------------------------------------
    status: JobStatus = JobStatus.PENDING
    is_claim: bool = False         # short claim vs full article
    claim: str = ""                # the central check-worthy claim
    # classifier prior
    clf_label: str = ""            # fake | real
    clf_prob: Optional[float] = None       # P(predicted label)
    clf_prior_fake: Optional[float] = None  # P(fake)
    check_worthy: bool = False
    # fact-check
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    n_support: int = 0
    n_refute: int = 0
    # ---- outputs -------------------------------------------------------------
    verdict: str = ""              # real | fake | unverified
    confidence: Optional[float] = None
    abstained: bool = False
    rationale: str = ""
    # ---- audit ---------------------------------------------------------------
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title, "requested_mode": self.requested_mode, "is_claim": self.is_claim,
            "claim": self.claim, "status": self.status.value,
            "clf_label": self.clf_label, "clf_prob": self.clf_prob, "clf_prior_fake": self.clf_prior_fake,
            "check_worthy": self.check_worthy,
            "verdict": self.verdict, "confidence": self.confidence, "abstained": self.abstained,
            "rationale": self.rationale,
            "evidence": self.evidence, "n_support": self.n_support, "n_refute": self.n_refute,
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "Evidence", "JobState"]
