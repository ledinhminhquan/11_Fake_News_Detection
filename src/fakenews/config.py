"""Typed configuration + YAML loader for the Fake News Detection system.

Single source of truth for the news/claim datasets, the trainable fake-news
classifier (+ TF-IDF baseline), the stance/NLI model, the evidence retriever, the
agent decision thresholds and serving. Paths come from environment variables so
nothing is hard-coded. **Internal label convention: 1 = fake, 0 = real** (fake is
the positive/detected class); dataset loaders normalize to this.

Environment overrides
---------------------
* ``FAKENEWS_ARTIFACTS_DIR`` – base for data/models/runs (Drive on Colab)
* ``FAKENEWS_DATA_DIR``      – dataset cache / processed
* ``FAKENEWS_MODEL_DIR``     – trained models (classifier + stance)
* ``FAKENEWS_INDEX_DIR``     – evidence index
* ``FAKENEWS_RUN_DIR``       – eval/benchmark/analysis JSON
* ``HF_HOME``                – HuggingFace cache
* ``FAKENEWS_LLM_API_KEY``   – optional key for the LLM fact-check brain

Verified ids (confirmed on the HF Hub during research — keep exact):
  clf-data  GonzaloA/fake_news (title/text/label, 0=fake/1=real — FLIP on load) ·
            chengxuphd/liar2 (Apache, 6-way) · ErfanMoosaviMonazzah/fake-news-detection-dataset-English (openrail)
  factcheck fever/fever (cc-by-sa+gpl) · BeIR/fever (retrieval eval)
  clf-base  distilbert-base-uncased (Apache, default) · microsoft/deberta-v3-base (MIT) · answerdotai/ModernBERT-base
  stance    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (MIT, NLI; entail→support/contra→refute/neutral→abstain)
  retrieval sentence-transformers/all-MiniLM-L6-v2 (Apache) + cross-encoder/ms-marco-MiniLM-L6-v2 (Apache) + BM25
  (⚠️ label polarity differs across mirrors — normalize. AVOID ucsbai/liar broken viewer.)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("FAKENEWS_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("FAKENEWS_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("FAKENEWS_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def index_dir() -> Path:
    return Path(_env("FAKENEWS_INDEX_DIR", str(artifacts_dir() / "index"))).expanduser()


def run_dir() -> Path:
    return Path(_env("FAKENEWS_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Classification dataset + the evidence corpus for the fact-check."""
    # PRIMARY (VERIFIED): GonzaloA/fake_news (title/text/label, pre-split). 0=fake,1=real → FLIP to internal 1=fake.
    clf_dataset: str = "GonzaloA/fake_news"
    clf_dataset_config: str = ""
    title_col: str = "title"
    text_col: str = "text"
    label_col: str = "label"
    # the raw dataset value that means FAKE (GonzaloA: 0). loaders map: internal = 1 if raw==fake_label_value else 0.
    fake_label_value: int = 0
    use_hf: bool = True
    max_train_samples: int = 24000
    max_eval_samples: int = 4000
    # secondary 6-way credibility (Apache): collapse 0-2=fake, 3-5=real
    liar_dataset: str = "chengxuphd/liar2"
    # fact-check / stance source (cc-by-sa + gpl — flag): fever/fever
    fever_dataset: str = "fever/fever"
    seed: int = 42


@dataclass
class ClassifierConfig:
    """Trainable fake-news classifier (transformer) + TF-IDF baseline."""
    base_model: str = "distilbert-base-uncased"   # Apache, T4-light. alt microsoft/deberta-v3-base (MIT) / answerdotai/ModernBERT-base
    max_length: int = 384
    num_labels: int = 2                            # 0=real, 1=fake
    # training (HF Trainer)
    num_train_epochs: int = 3
    learning_rate: float = 2.0e-5
    per_device_train_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_class_weights: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    eval_steps: int = 200
    save_steps: int = 200
    seed: int = 42
    output_subdir: str = "classifier"
    baseline_filename: str = "tfidf_logreg.joblib"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / self.baseline_filename


@dataclass
class StanceConfig:
    """Stance / NLI model for evidence→claim judging (pretrained zero-shot)."""
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # MIT; entail→support, contra→refute, neutral→abstain
    nli_fallback: str = "facebook/bart-large-mnli"                   # MIT, zero-shot
    max_length: int = 256
    support_threshold: float = 0.5     # entailment prob above this => support
    refute_threshold: float = 0.5      # contradiction prob above this => refute
    enabled: bool = True
    output_subdir: str = "stance"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class RetrievalConfig:
    """Evidence retrieval over the corpus (dense + BM25)."""
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"     # Apache, 384-dim
    reranker: str = "cross-encoder/ms-marco-MiniLM-L6-v2"        # Apache
    top_k: int = 5                     # evidence passages retrieved per claim
    rrf_k: int = 60
    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = True


@dataclass
class AgentConfig:
    """Agent decision thresholds (D1–D5) + optional LLM fact-check brain."""
    # D1 — claim routing (article vs short claim)
    short_claim_words: int = 40        # <= => treat as a single claim
    # D2 — check-worthiness / classifier-confidence gate
    skip_factcheck_confidence: float = 0.95   # very confident classifier + not check-worthy => skip retrieval
    factcheck_in_auto: bool = True
    # D3 — evidence-coverage gate
    min_evidence: int = 1              # below this relevant evidence => widen/abstain
    min_evidence_relevance: float = 0.05
    # D4 — stance aggregation / verdict gate
    verdict_margin: float = 0.2        # |support - refute| weight margin to call a verdict
    prior_weight: float = 0.3          # the classifier prior's soft weight (evidence dominates)
    # D5 — confidence / abstain gate
    min_verdict_confidence: float = 0.5
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "FAKENEWS_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Fake News & Misinformation Detection API"
    api_version: str = "1.0.0"
    log_requests: bool = True
    request_log_subdir: str = "request_logs"
    max_text_chars: int = 50000

    @property
    def request_log_path(self) -> Path:
        return run_dir() / self.request_log_subdir / "requests.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Fake News & Misinformation Detection System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    stance: StanceConfig = field(default_factory=StanceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "classifier": ClassifierConfig, "stance": StanceConfig,
             "retrieval": RetrievalConfig, "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(),
            "index": index_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "ClassifierConfig", "StanceConfig", "RetrievalConfig", "AgentConfig",
           "ServingConfig", "AppConfig", "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "index_dir", "run_dir"]
