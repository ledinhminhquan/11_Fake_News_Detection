"""Lightweight hyper-parameter search for the classifier.

A small grid over the learning rate. Each trial fine-tunes on a capped slice and
scores macro-F1; the best LR is reported. Heavy training is delegated to
``train_classifier`` (lazy imports there).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_LRS: List[float] = [1e-5, 2e-5, 3e-5]


def tune_classifier(cfg: AppConfig, n_trials: int = 3, limit: Optional[int] = 4000,
                    lrs: Optional[List[float]] = None) -> Dict:
    from .train_classifier import train_classifier
    lrs = (lrs or _DEFAULT_LRS)[:n_trials]
    trials: List[Dict] = []
    best = None
    for lr in lrs:
        cfg.classifier.learning_rate = float(lr)
        try:
            res = train_classifier(cfg, limit=limit, resume=False)
            score = (res.get("metrics") or {}).get("eval_macro_f1") \
                or (res.get("metrics") or {}).get("macro_f1") or 0.0
        except Exception as exc:
            logger.warning("trial lr=%s failed (%s)", lr, exc)
            score, res = 0.0, {"error": str(exc)}
        trial = {"lr": lr, "macro_f1": round(float(score), 4), "version": res.get("version")}
        trials.append(trial)
        if best is None or trial["macro_f1"] > best["macro_f1"]:
            best = trial
    out = {"trials": trials, "best": best}
    d = run_dir() / "tune"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tune.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["tune_classifier"]
