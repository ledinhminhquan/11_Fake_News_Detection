"""FastAPI service for the Fake News Detection system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /classify``   – text -> {label, probability}  (fast style/probability signal)
* ``POST /factcheck``  – claim -> {verdict, evidence, rationale}  (agentic, evidence-grounded)

The system **flags content for human review and always shows its evidence — it
never auto-removes content.**
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import (ClassifyRequest, ClassifyResponse, EvidenceOut, FactCheckRequest,
                      FactCheckResponse, HealthResponse)

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", classifier=getattr(agent.classifier, "name", "?"),
                          stance=getattr(agent.stance, "name", "?"), version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "classifier": getattr(agent.classifier, "version", "?"),
            "stance": getattr(agent.stance, "version", "?"),
            "retriever": "bm25+dense", "model_version": cfg.serving.model_version}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    if not (req.text.strip() or req.title.strip()):
        raise HTTPException(status_code=422, detail="provide text or a title")
    agent = get_agent()
    r = agent.classify(req.text, title=req.title)
    return ClassifyResponse(label=r["label"], probability=r["probability"],
                            prior_fake=r["prior_fake"], model_version=r["model_version"])


@app.post("/factcheck", response_model=FactCheckResponse)
def factcheck(req: FactCheckRequest) -> FactCheckResponse:
    if not (req.claim.strip() or req.title.strip()):
        raise HTTPException(status_code=422, detail="provide a claim or a title")
    agent = get_agent()
    job = agent.run(req.claim, title=req.title, mode=req.mode, save=True)
    sd = job.to_dict()
    return FactCheckResponse(
        verdict=sd["verdict"], confidence=sd["confidence"], classifier_prior_fake=sd["clf_prior_fake"],
        classifier_label=sd["clf_label"], abstained=sd["abstained"], rationale=sd["rationale"],
        claim=sd["claim"], evidence=[EvidenceOut(**e) for e in sd["evidence"]],
        n_support=sd["n_support"], n_refute=sd["n_refute"], decisions=sd["decisions"], trace=sd["trace"],
        metrics=sd["metrics"], model_versions=sd["model_versions"])


__all__ = ["app"]
