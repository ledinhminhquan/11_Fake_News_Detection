"""Collect generated run artifacts into one dict for the report + slides generators.

Reads the JSON written under ``run_dir()`` by the training / analysis / monitoring
layers — the held-out classification + fact-check eval (``eval/latest.json`` :
accuracy / macro-F1 / per-class P/R/F1 / ROC-AUC / ECE for the fake-news classifier
vs the TF-IDF+LogReg baseline and the majority-class floor, plus fact-check verdict
accuracy / coverage / abstain-rate for the agentic evidence-grounded checker), a
latency benchmark, an error analysis (false-positive / false-negative / correct
breakdown), and a drift/monitoring snapshot — plus the trained-classifier metadata
from the model registry. Every read is defensive: a missing or malformed file
yields ``None`` (or ``{}``) so the report/slides degrade to placeholders instead of
crashing. No heavy deps; pure stdlib + config + registry.

Internal label convention: 0 = real, 1 = fake.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, run_dir
from ..models.model_registry import read_metadata, resolve_latest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_artifacts(cfg: AppConfig) -> Dict[str, Any]:
    """Best-effort load of every run artifact. Missing files -> ``None``/``{}``."""
    rd = run_dir()
    arts: Dict[str, Any] = {
        # held-out eval: model{accuracy,macro_f1,per_class,roc_auc,ece}, baseline,
        # majority, factcheck{accuracy,abstain_rate,selective_accuracy,coverage}, summary
        "eval": _load_json(rd / "eval" / "latest.json"),
        # error modes for the classifier (false positives / false negatives / correct)
        "error_analysis": (_load_json(rd / "error_analysis" / "latest.json")
                           or _load_json(rd / "analysis" / "error_analysis.json")),
        # latency benchmark: classify / retrieve / stance / total p50,p95
        "benchmark": (_load_json(rd / "benchmark" / "latest.json")
                      or _load_json(rd / "analysis" / "latency.json")
                      or _load_json(rd / "latency" / "latest.json")),
        # hyper-parameter / threshold tuning best trial
        "tune": (_load_json(rd / "tune" / "tune.json")
                 or _load_json(rd / "tune" / "best.json")),
        # drift / monitoring snapshot
        "monitoring": (_load_json(rd / "monitoring" / "latest.json")
                       or _load_json(rd / "monitoring" / "drift.json")),
    }
    # trained-classifier metadata (version, base_model, metrics) from the registry
    try:
        latest = resolve_latest(cfg.classifier.output_dir)
        arts["model_meta"] = read_metadata(latest) if latest else {}
    except Exception:
        arts["model_meta"] = {}
    # stance / NLI model metadata (optional — usually pretrained zero-shot, no registry entry)
    try:
        latest_s = resolve_latest(cfg.stance.output_dir)
        arts["stance_meta"] = read_metadata(latest_s) if latest_s else {}
    except Exception:
        arts["stance_meta"] = {}
    return arts


# ─────────────────────────────────────────────────────────────────────────────
# Safe getters — every one tolerates a missing artifact and returns a default.
# ─────────────────────────────────────────────────────────────────────────────

def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def clf_metric(arts: Dict[str, Any], stage: str, key: str) -> Optional[float]:
    """Pull a single classification metric, e.g. ``clf_metric(arts, "model", "macro_f1")``.

    ``stage`` ∈ {"model", "baseline", "majority"}; ``key`` ∈ {"accuracy","macro_f1",
    "roc_auc","ece","n"}. Returns ``None`` when the eval artifact or the requested
    metric is absent.
    """
    block = (arts.get("eval") or {}).get(stage) or {}
    return _num(block.get(key))


def clf_per_class(arts: Dict[str, Any], stage: str, cls: str, key: str) -> Optional[float]:
    """Per-class P/R/F1, e.g. ``clf_per_class(arts, "model", "fake", "recall")``.

    ``stage`` ∈ {"model","baseline","majority"}; ``cls`` ∈ {"real","fake"};
    ``key`` ∈ {"precision","recall","f1"}.
    """
    block = (arts.get("eval") or {}).get(stage) or {}
    pc = (block.get("per_class") or {}).get(cls) or {}
    return _num(pc.get(key))


def factcheck_metric(arts: Dict[str, Any], key: str) -> Optional[float]:
    """Fact-check verdict metric, e.g. ``factcheck_metric(arts, "accuracy")``.

    ``key`` ∈ {"accuracy","abstain_rate","selective_accuracy","coverage","n"}.
    """
    block = (arts.get("eval") or {}).get("factcheck") or {}
    return _num(block.get(key))


def has_eval(arts: Dict[str, Any]) -> bool:
    ev = arts.get("eval") or {}
    return bool(ev.get("model") or (ev.get("summary") or {}).get("classifier"))


def has_factcheck(arts: Dict[str, Any]) -> bool:
    fc = (arts.get("eval") or {}).get("factcheck") or {}
    return _num(fc.get("accuracy")) is not None


def summary_stat(arts: Dict[str, Any], key: str) -> Any:
    return ((arts.get("eval") or {}).get("summary") or {}).get(key)


def beats_baseline(arts: Dict[str, Any]) -> Optional[bool]:
    v = summary_stat(arts, "beats_baseline")
    return bool(v) if isinstance(v, bool) else None


def classifier_name(arts: Dict[str, Any]) -> str:
    s = summary_stat(arts, "classifier")
    if s:
        return str(s)
    return str((arts.get("eval") or {}).get("classifier_name")
               or "tfidf-logreg (offline fallback)")


def model_version(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("version") or "untrained (TF-IDF + LogReg fallback)")


def base_model(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("base_model") or "distilbert-base-uncased")


def stance_name(arts: Dict[str, Any]) -> str:
    mv = arts.get("stance_meta") or {}
    return str(mv.get("base_model")
               or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (NLI; lexical fallback)")


def latency(arts: Dict[str, Any], key: str = "total", pct: str = "p50") -> Optional[float]:
    """Latency percentile for a stage, e.g. ``latency(arts, "classify", "p95")``.

    Tolerates a few benchmark shapes: ``{"total": {"p50":..}}`` or
    ``{"timings_ms": {"total": {"p50":..}}}`` or a flat ``{"total_p50":..}``.
    """
    b = arts.get("benchmark") or {}
    block = b.get("timings_ms") or b
    stage = block.get(key)
    if isinstance(stage, dict):
        return _num(stage.get(pct))
    return _num(b.get(f"{key}_{pct}"))


def error_breakdown(arts: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Classifier error modes for the error chart / report.

    Tolerates a few shapes: ``{"false_positive":.., "false_negative":.., "correct":..}``
    or a confusion-matrix-style ``{"fp":.., "fn":.., "tp":.., "tn":..}``.
    """
    ea = arts.get("error_analysis") or {}
    def _g(*names: str) -> Optional[float]:
        for n in names:
            if n in ea:
                return _num(ea.get(n))
        return None
    fp = _g("false_positive", "fp")
    fn = _g("false_negative", "fn")
    correct = _g("correct")
    if correct is None:
        tp, tn = _g("tp"), _g("tn")
        if tp is not None or tn is not None:
            correct = (tp or 0.0) + (tn or 0.0)
    return {"false_positive": fp, "false_negative": fn, "correct": correct}


def read_doc(name: str) -> str:
    """Load a docs/*.md section if present (fakenews keeps content self-contained,
    so this usually returns '' — the report supplies built-in section text)."""
    p = repo_root() / "docs" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


__all__ = [
    "load_artifacts", "read_doc", "repo_root",
    "clf_metric", "clf_per_class", "factcheck_metric", "has_eval", "has_factcheck",
    "summary_stat", "beats_baseline", "classifier_name", "model_version",
    "base_model", "stance_name", "latency", "error_breakdown",
]
