"""Matplotlib charts for the fake-news report/slides.

Renders the central evidence from the held-out classification + fact-check eval:
  * a **classification** bar chart — majority-class floor vs TF-IDF+LogReg baseline
    vs the trained classifier across accuracy / macro-F1 / ROC-AUC (the headline);
  * a **per-class** P/R/F1 chart for the trained model (real vs fake), which exposes
    the asymmetric cost of a missed fake vs a wrongly-flagged real story;
  * an optional classifier **error-breakdown** chart (false positive / false negative
    / correct) from error analysis — a wrongly-flagged real article (false positive)
    is the censorship-risk error this project most wants to surface.

Returns saved PNG paths under ``run_dir()/report``. matplotlib is lazy-imported
inside a try/except; every chart degrades to ``None`` when it (or the data) is
unavailable so the report still builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)

# palette
_MAJORITY = "#cbd5e0"  # light grey (majority floor)
_BASELINE = "#9aa7b4"  # grey-blue (TF-IDF baseline)
_MODEL = "#2b6cb0"     # blue (trained classifier)
_REAL = "#2f855a"      # green (real class)
_FAKE = "#c53030"      # red (fake class)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _f(d: Dict[str, Any], key: str) -> Optional[float]:
    v = (d or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def classification_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    """Grouped bars: majority / TF-IDF baseline / trained classifier across
    accuracy / macro-F1 / ROC-AUC."""
    model = (eval_art or {}).get("model") or {}
    if not model:
        return None
    try:
        plt = _mpl()
        metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("roc_auc", "ROC-AUC")]
        majority = (eval_art or {}).get("majority") or {}
        baseline = (eval_art or {}).get("baseline") or {}
        series = []
        if majority:
            series.append(("majority", majority, _MAJORITY))
        if baseline:
            series.append(("TF-IDF+LogReg", baseline, _BASELINE))
        series.append(("classifier", model, _MODEL))

        n = len(series)
        x = list(range(len(metrics)))
        width = 0.8 / n
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for si, (label, data, color) in enumerate(series):
            vals = [(_f(data, k) or 0.0) for k, _ in metrics]
            offs = [i + (si - (n - 1) / 2) * width for i in x]
            bars = ax.bar(offs, vals, width=width, label=label, color=color)
            for rect, v in zip(bars, vals):
                if v > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title("Fake-news classifier vs baseline / majority floor")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("classification_chart skipped (%s)", exc)
        return None


def per_class_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    """Grouped bars: real vs fake precision/recall/F1 for the trained classifier."""
    model = (eval_art or {}).get("model") or {}
    per_class = model.get("per_class") or {}
    if not per_class or not (per_class.get("real") or per_class.get("fake")):
        return None
    try:
        plt = _mpl()
        metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
        series = []
        if per_class.get("real"):
            series.append(("real", per_class["real"], _REAL))
        if per_class.get("fake"):
            series.append(("fake", per_class["fake"], _FAKE))
        if not series:
            return None

        n = len(series)
        x = list(range(len(metrics)))
        width = 0.8 / n
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for si, (label, data, color) in enumerate(series):
            vals = [(_f(data, k) or 0.0) for k, _ in metrics]
            offs = [i + (si - (n - 1) / 2) * width for i in x]
            bars = ax.bar(offs, vals, width=width, label=label, color=color)
            for rect, v in zip(bars, vals):
                if v > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title("Per-class performance (real vs fake)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("per_class_chart skipped (%s)", exc)
        return None


def error_modes_chart(err_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    """Classifier error breakdown: false positive (real flagged fake — censorship risk)
    vs false negative (fake passed as real) vs correct."""
    if not err_art:
        return None
    # accept {false_positive, false_negative, correct} or {fp, fn, tp, tn}
    fp = err_art.get("false_positive", err_art.get("fp"))
    fn = err_art.get("false_negative", err_art.get("fn"))
    correct = err_art.get("correct")
    if correct is None and ("tp" in err_art or "tn" in err_art):
        correct = (err_art.get("tp") or 0) + (err_art.get("tn") or 0)
    vals = [fp, fn, correct]
    if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return None
    try:
        plt = _mpl()
        labels = ["false positive\n(real→fake)", "false negative\n(fake→real)", "correct"]
        nums = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
                for v in vals]
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.bar(labels, nums, color=["#dd6b20", "#c53030", "#2f855a"])
        ax.set_ylabel("# articles")
        ax.set_title("Classifier error modes")
        for i, v in enumerate(nums):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("error_modes_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    """Build every available chart; returns ``[(name, path), ...]`` (only the ones produced)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []
    charts: List[Tuple[str, Path]] = []
    for name, fn, key in [("classification", classification_chart, "eval"),
                          ("per_class", per_class_chart, "eval"),
                          ("errors", error_modes_chart, "error_analysis")]:
        try:
            p = fn(arts.get(key) or {}, out_dir / f"{name}.png")
        except Exception as exc:
            logger.info("chart %s skipped (%s)", name, exc)
            p = None
        if p:
            charts.append((name, p))
    return charts


__all__ = ["classification_chart", "per_class_chart", "error_modes_chart", "build_all"]
