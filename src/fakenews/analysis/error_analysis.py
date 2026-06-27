"""Fake-news error analysis: where the classifier and the fact-check agent slip.

Two passes, both fully offline (TF-IDF + LogReg classifier + lexical stance over the
seed evidence; ``use_hf=False``):

1. **Classifier pass** — run the classifier on each eval news item (the held-out seed
   split) and collect every misclassification ``{id, true_label, pred_label, prob,
   text[:120]}``. We split the errors into the two costly directions:

   * ``false_positive`` (``fp``) — a *real* item flagged as *fake* (a credible story
     wrongly suppressed);
   * ``false_negative`` (``fn``) — a *fake* item passed as *real* (misinformation that
     slipped through).

   A 2x2 confusion (real/fake x pred) and a few example errors are also reported.

2. **Fact-check pass** — run the agent on the gold claims (``samples.claims()``) and
   compare the produced verdict to gold: ``correct`` (verdict matches gold),
   ``abstain`` (verdict == 'unverified'), ``wrong`` (a confident but mismatched
   verdict).

The short keys ``fp`` / ``fn`` / ``correct`` / ``wrong`` / ``abstain`` are what the
charts read directly. Degrades gracefully — any failure is logged and returns a stub
dict rather than raising.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_LABEL_NAMES = {0: "real", 1: "fake"}


def error_analysis(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    """Classifier misclassifications + fact-check verdict-vs-gold breakdown.

    Uses ``seed_split(cfg.data.seed)`` for the classifier eval set and ``samples.claims()``
    for the fact-check pass, running ``FakeNewsAgent`` offline (``load_model=False``).
    Degrades gracefully — any failure is logged and returns a stub dict rather than raising.
    """
    cfg = cfg or AppConfig()
    # Force the offline seed corpus + TF-IDF / lexical stack for determinism.
    try:
        cfg.data.use_hf = False
    except Exception:
        pass

    try:
        from ..data.dataset import seed_split          # lazy
        from ..data import samples                      # lazy
        from ..agent.fakenews_agent import FakeNewsAgent  # lazy (pulls classifier + stance)
    except Exception as exc:
        logger.warning("error_analysis: imports unavailable (%s)", exc)
        return _stub(str(exc), save)

    try:
        seed = getattr(cfg.data, "seed", 42)
        _train, eval_items = seed_split(seed)
    except Exception as exc:
        logger.warning("error_analysis: could not build eval data (%s)", exc)
        return _stub(str(exc), save)

    if limit:
        eval_items = eval_items[:limit]
    if not eval_items:
        logger.info("error_analysis: empty eval set")
        return _stub("empty eval set", save, n_eval=0)

    try:
        agent = FakeNewsAgent(cfg, load_model=False)
    except Exception as exc:
        logger.warning("error_analysis: could not build agent (%s)", exc)
        return _stub(str(exc), save, n_eval=len(eval_items))

    # ---- classifier pass -----------------------------------------------------
    classifier = agent.classifier
    fp = fn = 0
    # confusion[true][pred] of {real, fake}
    confusion = {"real": {"real": 0, "fake": 0}, "fake": {"real": 0, "fake": 0}}
    errors: List[Dict] = []
    n_eval = 0
    n_correct = 0

    for it in eval_items:
        try:
            pred_label, prob = classifier.predict(it.content)
            pred_label = int(pred_label)
        except Exception as exc:
            logger.info("error_analysis: classify failed for %s (%s)", getattr(it, "id", "?"), exc)
            continue
        true_label = int(it.label)
        n_eval += 1
        tname, pname = _LABEL_NAMES.get(true_label, "real"), _LABEL_NAMES.get(pred_label, "real")
        confusion[tname][pname] += 1
        if pred_label == true_label:
            n_correct += 1
            continue
        # misclassification
        if true_label == 0 and pred_label == 1:
            fp += 1   # real flagged fake
        elif true_label == 1 and pred_label == 0:
            fn += 1   # fake passed as real
        if len(errors) < 12:
            errors.append({
                "id": getattr(it, "id", "?"),
                "true_label": tname,
                "pred_label": pname,
                "prob": round(float(prob), 4),
                "text": (it.content or "")[:120],
            })

    clf_accuracy = round(n_correct / max(1, n_eval), 4)

    # ---- fact-check pass -----------------------------------------------------
    correct = wrong = abstain = 0
    fc_examples: List[Dict] = []
    n_claims = 0
    try:
        gold_claims = samples.claims()
    except Exception as exc:
        logger.info("error_analysis: could not load gold claims (%s)", exc)
        gold_claims = []

    for c in gold_claims:
        claim = str(c.get("claim", "")).strip()
        gold = str(c.get("verdict", "")).strip()
        if not claim:
            continue
        try:
            job = agent.run(text=claim, mode="factcheck", save=False)
            verdict = job.verdict or "unverified"
        except Exception as exc:
            logger.info("error_analysis: agent run failed for a claim (%s)", exc)
            continue
        n_claims += 1
        if verdict == "unverified":
            abstain += 1
            case = "abstain"
        elif verdict == gold:
            correct += 1
            case = "correct"
        else:
            wrong += 1
            case = "wrong"
        if len(fc_examples) < 12:
            fc_examples.append({
                "claim": claim[:120], "gold": gold, "verdict": verdict, "case": case,
                "confidence": round(float(job.confidence), 4) if job.confidence is not None else None,
            })

    fc_accuracy = round(correct / max(1, n_claims), 4)

    result = {
        "classifier": getattr(classifier, "name", "?"),
        "classifier_version": getattr(classifier, "version", "?"),
        "stance": getattr(agent.stance, "name", "?"),
        "stance_version": getattr(agent.stance, "version", "?"),
        # classifier pass
        "n_eval": n_eval,
        "clf_accuracy": clf_accuracy,
        "false_positive": fp, "false_negative": fn,
        "confusion": confusion,
        "errors": errors,
        # fact-check pass
        "n_claims": n_claims,
        "factcheck_accuracy": fc_accuracy,
        "factcheck_examples": fc_examples,
        # short keys the charts read directly
        "fp": fp, "fn": fn,
        "correct": correct, "wrong": wrong, "abstain": abstain,
    }
    if save:
        _save(result)
    logger.info("error analysis: clf fp=%d fn=%d (acc=%.3f) | factcheck correct=%d wrong=%d abstain=%d",
                fp, fn, clf_accuracy, correct, wrong, abstain)
    return result


def _stub(error: str, save: bool, **extra) -> Dict:
    result = {
        "classifier": "?", "stance": "?",
        "n_eval": 0, "clf_accuracy": 0.0,
        "false_positive": 0, "false_negative": 0,
        "confusion": {"real": {"real": 0, "fake": 0}, "fake": {"real": 0, "fake": 0}},
        "errors": [],
        "n_claims": 0, "factcheck_accuracy": 0.0, "factcheck_examples": [],
        "fp": 0, "fn": 0, "correct": 0, "wrong": 0, "abstain": 0,
        "error": error,
    }
    result.update(extra)
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("error_analysis: could not save (%s)", exc)


__all__ = ["error_analysis"]
