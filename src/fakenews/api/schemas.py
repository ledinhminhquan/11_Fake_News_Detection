"""Pydantic request/response schemas for the Fake News Detection API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field("", description="Article body or claim text")
    title: str = Field("", description="Optional headline / title")


class ClassifyResponse(BaseModel):
    label: str                     # fake | real
    probability: float             # P(predicted label)
    prior_fake: float              # P(fake)
    model_version: str
    note: str = "This is a style/probability signal, not a verdict. Verify with the fact-check."


class FactCheckRequest(BaseModel):
    claim: str = Field(..., description="The claim (or article) to fact-check")
    title: str = Field("", description="Optional title")
    mode: str = Field("auto", description="auto | classify | factcheck")


class EvidenceOut(BaseModel):
    text: str
    source: str = ""
    stance: str = "neutral"        # support | refute | neutral
    score: float = 0.0
    relevance: float = 0.0


class FactCheckResponse(BaseModel):
    verdict: str                   # real | fake | unverified
    confidence: Optional[float] = None
    classifier_prior_fake: Optional[float] = None
    classifier_label: str = ""
    abstained: bool = False
    rationale: str = ""
    claim: str = ""
    evidence: List[EvidenceOut] = []
    n_support: int = 0
    n_refute: int = 0
    decisions: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    model_versions: Dict[str, str] = {}
    disclaimer: str = "Advisory only — flags content for human review. It never auto-removes content."


class HealthResponse(BaseModel):
    status: str
    classifier: str
    stance: str
    version: str


__all__ = ["ClassifyRequest", "ClassifyResponse", "FactCheckRequest", "EvidenceOut",
           "FactCheckResponse", "HealthResponse"]
