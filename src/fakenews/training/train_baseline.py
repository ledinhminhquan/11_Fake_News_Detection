"""Train the TF-IDF + LogReg baseline classifier (sklearn, no GPU).

The dependency-light floor the transformer must beat — and the offline fallback
classifier. Trains in seconds on CPU and is saved to
``models/classifier/tfidf_logreg.joblib``.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models.classifier import TfidfLogRegClassifier
from ..data.dataset import load_news, seed_split
from . import metrics as M

logger = get_logger(__name__)


def train_baseline(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    train_items = load_news(cfg, split="train", limit=limit)
    try:
        eval_items = load_news(cfg, split="test", limit=cfg.data.max_eval_samples)
    except Exception:
        eval_items = None
    if not eval_items or len(train_items) <= 50:
        train_items, eval_items = seed_split(cfg.data.seed)

    clf = TfidfLogRegClassifier(cfg.classifier).fit([it.content for it in train_items],
                                                    [it.label for it in train_items])
    y_true = [it.label for it in eval_items]
    probs = [clf.predict_proba(it.content) for it in eval_items]
    y_pred = [1 if p >= 0.5 else 0 for p in probs]
    m = M.classification_metrics(y_true, y_pred, probs)

    if save:
        path = cfg.classifier.baseline_path
        clf.save(path)
        (cfg.classifier.output_dir / "baseline_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        logger.info("Baseline saved -> %s (macro_f1=%s)", path, m["macro_f1"])
    return {"model": "tfidf-logreg", "n_train": len(train_items), "n_eval": len(eval_items), "metrics": m}


__all__ = ["train_baseline"]
