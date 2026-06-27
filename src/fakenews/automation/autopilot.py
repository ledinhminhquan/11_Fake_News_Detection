"""One-button autopilot: data → train → evaluate → analysis → report + slides + grade + bundle.

Runs the whole pipeline behind a single call. Each stage is isolated in its own
try/except and never aborts the run: it records a per-stage ``ok``/``error``/``skipped``
status with wall-clock seconds, then keeps going. A submission bundle is written to
``artifacts/submission/submission-<stamp>/`` (manifest + report.pdf + slides.pptx)
and zipped.

Heavy/optional stages (the transformer classifier, which needs torch +
transformers + datasets) degrade gracefully: when those libraries are unavailable
the training step is *skipped* (logged, not errored) and the system evaluates the
TF-IDF + LogReg baseline / lexical fallback instead. The sklearn baseline is always
trained because it is fast and dependency-light. Every dependency is lazy-imported
so this module imports with only the standard library.
"""

from __future__ import annotations

import json
import time
import zipfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import AppConfig, artifacts_dir, ensure_dirs
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)


def _step(steps: List[Dict], name: str, fn: Callable[[], Any], skip: bool = False) -> Optional[Any]:
    """Run ``fn`` as one isolated pipeline stage; never raises past this boundary."""
    if skip:
        logger.info("autopilot step %s skipped", name)
        steps.append({"step": name, "status": "skipped", "seconds": 0.0})
        return None
    t0 = time.perf_counter()
    try:
        out = fn()
        steps.append({"step": name, "status": "ok", "seconds": round(time.perf_counter() - t0, 2)})
        return out
    except Exception as exc:
        logger.warning("autopilot step %s failed: %s", name, exc)
        steps.append({"step": name, "status": "error", "error": str(exc),
                      "seconds": round(time.perf_counter() - t0, 2)})
        return None


def _training_available() -> bool:
    """True only when torch + transformers + datasets are all importable."""
    return all(find_spec(m) is not None for m in ("torch", "transformers", "datasets"))


def _demo_agent(cfg: AppConfig) -> Dict:
    """Run the rules-only agent on 2 seed claims, capturing the D1–D5 decisions + verdict."""
    from ..agent.fakenews_agent import FakeNewsAgent
    from ..data import samples
    agent = FakeNewsAgent(cfg, load_model=False)
    out: List[Dict] = []
    for c in samples.claims()[:2]:
        claim = c.get("claim", "")
        job = agent.run(claim, mode="factcheck", save=False)
        sd = job.to_dict()
        out.append({"claim": claim,
                    "gold": c.get("verdict", ""),
                    "status": sd["status"],
                    "verdict": sd["verdict"],
                    "confidence": sd["confidence"],
                    "n_support": sd["n_support"],
                    "n_refute": sd["n_refute"],
                    "decisions": [(x["id"], x["branch"]) for x in sd["decisions"]]})
    return {"demos": out}


def run_autopilot(cfg: AppConfig, title: str = None, author: str = None,
                  train: bool = True, limit: Optional[int] = None) -> Dict:
    """Run every stage end-to-end and write a zipped submission bundle.

    Returns ``{steps, submission_dir, zip, report, slides, grade_summary}``.
    Defaults for ``title``/``author`` come from ``cfg``.
    """
    ensure_dirs()
    title = title or cfg.project_title
    author = author or cfg.author
    steps: List[Dict] = []

    # (1) data — ensure dirs (above) + best-effort dataset prep.
    _step(steps, "prepare_data", lambda: __import__(
        "fakenews.data.download_dataset", fromlist=["download_all"]).download_all(cfg))

    # (2) baseline — always (sklearn TF-IDF + LogReg; fast, dependency-light).
    _step(steps, "train_baseline", lambda: __import__(
        "fakenews.training.train_baseline", fromlist=["train_baseline"]).train_baseline(
        cfg, limit=limit, save=True))

    # (3) classifier — optional; skip cleanly when disabled or heavy deps missing.
    can_train = train and _training_available()
    if train and not can_train:
        logger.info("training requested but torch/transformers/datasets unavailable — skipping")
    _step(steps, "train_classifier", lambda: __import__(
        "fakenews.training.train_classifier", fromlist=["train_classifier"]).train_classifier(
        cfg, limit=limit), skip=not can_train)

    # (4) evaluation.
    _step(steps, "evaluate", lambda: __import__(
        "fakenews.training.evaluate", fromlist=["evaluate"]).evaluate(cfg, save=True))

    # (5-6) analysis.
    _step(steps, "error_analysis", lambda: __import__(
        "fakenews.analysis.error_analysis", fromlist=["error_analysis"]).error_analysis(cfg, save=True))
    _step(steps, "benchmark", lambda: __import__(
        "fakenews.analysis.latency", fromlist=["benchmark"]).benchmark(cfg, n=12, warmup=2, save=True))

    # (7) agent demo on seed claims.
    _step(steps, "demo_agent", lambda: _demo_agent(cfg))

    # (10) monitoring / drift.
    _step(steps, "monitoring", lambda: __import__(
        "fakenews.monitoring.drift_report", fromlist=["monitoring_report"]).monitoring_report(cfg))

    # (8-9) report + slides, written straight into the submission dir.
    stamp = utc_stamp()
    sub = artifacts_dir() / "submission" / f"submission-{stamp}"
    sub.mkdir(parents=True, exist_ok=True)
    report = _step(steps, "report", lambda: __import__(
        "fakenews.autoreport.report_pdf", fromlist=["generate_report"]).generate_report(
        cfg, title=title, author=author, out_path=sub / "report.pdf"))
    slides = _step(steps, "slides", lambda: __import__(
        "fakenews.autoreport.slides_pptx", fromlist=["generate_slides"]).generate_slides(
        cfg, title=title, author=author, out_path=sub / "slides.pptx"))

    # (11) grading checklist against the repo root.
    repo_root = Path(__file__).resolve().parents[3]
    checklist = _step(steps, "grading", lambda: __import__(
        "fakenews.grading.checklist", fromlist=["build_checklist"]).build_checklist(repo_root))

    # Bundle: manifest + copied report/slides, then zip.
    manifest = {"generated_at": utc_now_iso(), "title": title, "author": author,
                "student_id": cfg.student_id, "steps": steps, "grading_checklist": checklist}
    try:
        (sub / "submission_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("manifest write failed: %s", exc)

    zip_path = None
    try:
        zip_path = sub / "submission_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sub.iterdir():
                if f.is_file() and f.name != "submission_bundle.zip":
                    z.write(f, f.name)
    except Exception as exc:
        logger.warning("bundle zip failed: %s", exc)
        zip_path = None

    logger.info("Autopilot done -> %s", sub)
    return {"steps": steps,
            "submission_dir": str(sub),
            "zip": str(zip_path) if zip_path else None,
            "report": str(report) if report else None,
            "slides": str(slides) if slides else None,
            "grade_summary": (checklist or {}).get("summary")}


__all__ = ["run_autopilot"]
