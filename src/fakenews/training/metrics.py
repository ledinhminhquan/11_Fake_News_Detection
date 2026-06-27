"""Classification + fact-check metrics.

Classification: accuracy, **macro-F1 (headline)**, per-class precision/recall/F1,
ROC-AUC, and **ECE (Expected Calibration Error)** — over-confidence is dangerous
here. Fact-check: verdict accuracy + per-class. Uses ``sklearn`` when available,
with pure-python fallbacks so the metrics compute with no heavy deps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           y_prob_fake: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """y in {0=real, 1=fake}; y_prob_fake = P(fake) for ROC-AUC / ECE."""
    n = len(y_true)
    acc = _safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), n)
    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for c, name in ((0, "real"), (1, "fake")):
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t != c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != c and t == c)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per_class[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    out: Dict[str, Any] = {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4),
                           "per_class": per_class, "n": n}
    if y_prob_fake is not None and len(y_prob_fake) == n:
        out["roc_auc"] = round(_roc_auc(y_true, y_prob_fake), 4)
        out["ece"] = round(_ece(y_true, y_prob_fake), 4)
    return out


def _roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(list(y_true), list(scores)))
    except Exception:
        # Mann-Whitney U fallback
        pos = [s for t, s in zip(y_true, scores) if t == 1]
        neg = [s for t, s in zip(y_true, scores) if t == 0]
        if not pos or not neg:
            return 0.5
        wins = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
        return wins / (len(pos) * len(neg))


def _ece(y_true: Sequence[int], probs: Sequence[float], n_bins: int = 10) -> float:
    """Expected Calibration Error of P(fake)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for t, p in zip(y_true, probs):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((t, p))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for _, p in b) / len(b)
        acc = sum(t for t, _ in b) / len(b)        # fraction actually fake
        ece += (len(b) / n) * abs(acc - conf)
    return ece


def factcheck_metrics(pred: List[str], gold: List[str]) -> Dict[str, Any]:
    """Verdict accuracy; 'unverified' counts as a miss for accuracy but is reported separately."""
    n = len(gold)
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    abstain = sum(1 for p in pred if p == "unverified")
    decided = [(p, g) for p, g in zip(pred, gold) if p != "unverified"]
    sel_acc = _safe_div(sum(1 for p, g in decided if p == g), len(decided))
    return {"accuracy": round(_safe_div(correct, n), 4), "abstain_rate": round(_safe_div(abstain, n), 4),
            "selective_accuracy": round(sel_acc, 4), "coverage": round(_safe_div(len(decided), n), 4), "n": n}


__all__ = ["classification_metrics", "factcheck_metrics"]
