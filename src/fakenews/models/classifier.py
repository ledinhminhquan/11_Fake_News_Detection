"""The fake-news classifier (the TRAINED core) + a TF-IDF + LogReg baseline.

* ``TfidfLogRegClassifier`` — the dependency-light baseline (sklearn): TF-IDF over
  title+text → Logistic Regression. Trains in seconds on CPU, no torch; the floor
  the transformer must beat and the offline fallback.
* ``TransformerClassifier`` — wraps a fine-tuned HF sequence classifier
  (`distilbert-base-uncased` / `microsoft/deberta-v3-base`), predicting P(fake).

Both expose ``predict_proba(text) -> P(fake)`` and ``predict(text) -> (label,
prob)`` with the internal convention **1 = fake, 0 = real**.
``load_classifier`` picks the best available (fine-tuned > trained baseline >
untrained baseline trained on the seed).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import ClassifierConfig
from ..logging_utils import get_logger
from .model_registry import resolve_latest

logger = get_logger(__name__)

LABELS = {0: "real", 1: "fake"}


class TfidfLogRegClassifier:
    name = "tfidf-logreg"
    version = "tfidf-1.0"

    def __init__(self, cfg: Optional[ClassifierConfig] = None):
        self.cfg = cfg
        self.vec = None
        self.clf = None

    def fit(self, texts: List[str], labels: List[int]) -> "TfidfLogRegClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   max_features=50000, sublinear_tf=True, min_df=1)
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, text: str) -> float:
        """Return P(fake)."""
        if self.vec is None or self.clf is None:
            return 0.5
        X = self.vec.transform([text])
        classes = list(self.clf.classes_)
        p = self.clf.predict_proba(X)[0]
        return float(p[classes.index(1)]) if 1 in classes else float(p[-1])

    def predict(self, text: str) -> Tuple[int, float]:
        pf = self.predict_proba(text)
        return (1, pf) if pf >= 0.5 else (0, 1.0 - pf)

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vec": self.vec, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str | Path, cfg: Optional[ClassifierConfig] = None) -> "TfidfLogRegClassifier":
        import joblib
        obj = cls(cfg)
        data = joblib.load(Path(path))
        obj.vec, obj.clf = data["vec"], data["clf"]
        return obj


class TransformerClassifier:
    name = "transformer"

    def __init__(self, pipe, cfg: ClassifierConfig, version: str = "clf-1.0"):
        self.pipe = pipe
        self.cfg = cfg
        self.version = version

    @classmethod
    def from_pretrained(cls, model_path: str, cfg: ClassifierConfig, device: Optional[int] = None) -> "TransformerClassifier":
        from transformers import pipeline  # lazy
        dev = device if device is not None else -1
        try:
            import torch
            if torch.cuda.is_available():
                dev = 0
        except Exception:
            pass
        pipe = pipeline("text-classification", model=model_path, tokenizer=model_path,
                        truncation=True, max_length=cfg.max_length, top_k=None, device=dev)
        return cls(pipe, cfg, version=_read_version(model_path))

    def predict_proba(self, text: str) -> float:
        out = self.pipe(text[: self.cfg.max_length * 6])
        rows = out[0] if (out and isinstance(out[0], list)) else out
        for r in rows:
            lab = str(r.get("label", "")).lower()
            if lab in ("label_1", "fake", "1") or lab.endswith("_1"):
                return float(r.get("score", 0.5))
        # fall back: 1 - P(real)
        for r in rows:
            lab = str(r.get("label", "")).lower()
            if lab in ("label_0", "real", "0"):
                return float(1.0 - r.get("score", 0.5))
        return 0.5

    def predict(self, text: str) -> Tuple[int, float]:
        pf = self.predict_proba(text)
        return (1, pf) if pf >= 0.5 else (0, 1.0 - pf)


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "model_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "clf-1.0")
        except Exception:
            pass
    return "clf-base"


def _seed_baseline(cfg: ClassifierConfig) -> TfidfLogRegClassifier:
    from ..data.dataset import load_seed_news
    items = load_seed_news()
    return TfidfLogRegClassifier(cfg).fit([it.content for it in items], [it.label for it in items])


def load_classifier(cfg: ClassifierConfig, *, prefer: str = "transformer", device: Optional[int] = None):
    """Fine-tuned transformer > saved TF-IDF baseline > a baseline trained on the seed."""
    if prefer == "transformer":
        latest = resolve_latest(cfg.output_dir)
        if latest is not None:
            try:
                return TransformerClassifier.from_pretrained(str(latest), cfg, device=device)
            except Exception as exc:
                logger.info("Transformer classifier unavailable (%s); using TF-IDF baseline.", exc)
    if cfg.baseline_path.exists():
        try:
            return TfidfLogRegClassifier.load(cfg.baseline_path, cfg)
        except Exception as exc:
            logger.info("Saved baseline unload failed (%s); training a seed baseline.", exc)
    return _seed_baseline(cfg)


__all__ = ["TfidfLogRegClassifier", "TransformerClassifier", "load_classifier", "LABELS"]
