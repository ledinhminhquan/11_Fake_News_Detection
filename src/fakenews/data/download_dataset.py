"""Prefetch + sanity-check the datasets (no large files committed).

Streaming probes confirm the classification + fact-check datasets are reachable +
report their schema WITHOUT downloading them in full. Everything degrades
gracefully: the built-in seed always works, so this is a convenience.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _probe(loader) -> Dict[str, Any]:
    try:
        return {"ok": True, **loader()}
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ok": False, "error": str(exc)}


def download_all(cfg: AppConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {"classification": {}, "liar": {}, "seed": {}}

    def clf_probe():
        from datasets import load_dataset
        dc = cfg.data
        ds = load_dataset(dc.clf_dataset, dc.clf_dataset_config or None, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": dc.clf_dataset, "reachable": True, "columns": list(first.keys())}

    def liar_probe():
        from datasets import load_dataset
        ds = load_dataset(cfg.data.liar_dataset, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": cfg.data.liar_dataset, "reachable": True, "columns": list(first.keys())}

    out["classification"] = _probe(clf_probe)
    out["liar"] = _probe(liar_probe)

    from . import samples
    out["seed"] = {"ok": True, "news": len(samples.news()), "evidence": len(samples.evidence()),
                   "claims": len(samples.claims())}
    logger.info("download_all: clf=%s liar=%s seed=%d news",
                out["classification"].get("ok"), out["liar"].get("ok"), out["seed"]["news"])
    return out


__all__ = ["download_all"]
