"""News/claim dataset loading + label normalization + the evidence corpus.

Loads a fake-news classification dataset (`GonzaloA/fake_news` default; LIAR2 for
6-way) from HF, **normalizing the label polarity to the internal convention
1 = fake, 0 = real** (dataset mirrors disagree — GonzaloA is 0=fake/1=real, so it
is flipped; LittleFish is 0=real/1=fake; see config). Falls back to the built-in
seed (``samples.py``) when the dataset / network is unavailable. Also exposes the
evidence corpus + gold-verdict claims for the agentic fact-check. ``datasets`` is
imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, DataConfig
from ..logging_utils import get_logger
from . import samples

logger = get_logger(__name__)


@dataclass
class NewsItem:
    id: str
    title: str
    text: str
    label: int                      # internal: 1 = fake, 0 = real
    source: str = ""

    @property
    def content(self) -> str:
        return (f"{self.title}. {self.text}".strip() if self.title else self.text).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text,
                "label": self.label, "source": self.source}


def _normalize_label(raw, dc: DataConfig) -> int:
    """Map a raw dataset label to internal 1=fake / 0=real."""
    try:
        v = int(raw)
    except Exception:
        s = str(raw).lower()
        return 1 if ("fake" in s or "false" in s or "pants" in s) else 0
    return 1 if v == dc.fake_label_value else 0


def load_news(cfg: AppConfig, split: str = "train", limit: Optional[int] = None) -> List[NewsItem]:
    dc = cfg.data
    if dc.use_hf:
        try:
            from datasets import load_dataset  # lazy
            ds = load_dataset(dc.clf_dataset, dc.clf_dataset_config or None, split=split)
            cap = limit or (dc.max_train_samples if split == "train" else dc.max_eval_samples)
            if cap and len(ds) > cap:
                ds = ds.select(range(cap))
            cols = set(ds.column_names)
            tcol = dc.title_col if dc.title_col in cols else ""
            xcol = dc.text_col if dc.text_col in cols else ("statement" if "statement" in cols else dc.text_col)
            out: List[NewsItem] = []
            for i, r in enumerate(ds):
                text = str(r.get(xcol, "") or "").strip()
                if not text:
                    continue
                out.append(NewsItem(id=f"{split}{i:06d}", title=str(r.get(tcol, "") or "").strip(),
                                    text=text[:4000], label=_normalize_label(r.get(dc.label_col), dc)))
            if out:
                logger.info("Loaded %d %s news items from %s", len(out), split, dc.clf_dataset)
                return out
        except Exception as exc:
            logger.warning("Could not load %s (%s); using seed.", dc.clf_dataset, exc)
    return load_seed_news()


def load_seed_news() -> List[NewsItem]:
    return [NewsItem(id=r["id"], title=r.get("title", ""), text=r["text"], label=int(r["label"]))
            for r in samples.news()]


def seed_split(seed: int = 42, eval_frac: float = 0.3) -> Tuple[List[NewsItem], List[NewsItem]]:
    """Deterministic train/eval split of the seed corpus (balanced)."""
    import random
    items = load_seed_news()
    rng = random.Random(seed)
    rng.shuffle(items)
    n_eval = max(2, int(len(items) * eval_frac))
    return items[n_eval:], items[:n_eval]


def load_evidence(cfg: AppConfig) -> Dict[str, Dict[str, str]]:
    """Evidence corpus {id: {text, source}} — seed offline; on Colab build from FEVER/wiki."""
    return {e["id"]: {"text": e["text"], "source": e.get("source", "")} for e in samples.evidence()}


def load_claims(cfg: AppConfig) -> List[Dict[str, str]]:
    """Gold-verdict claims for fact-check eval (seed offline)."""
    return samples.claims()


__all__ = ["NewsItem", "load_news", "load_seed_news", "seed_split", "load_evidence", "load_claims"]
