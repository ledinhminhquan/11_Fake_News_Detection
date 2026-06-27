"""Evaluate the classifier vs baselines + the agentic fact-check on held-out data.

Classification: the main classifier (fine-tuned transformer when present, else the
TF-IDF baseline) vs the **TF-IDF + LogReg** baseline and the **majority-class**
floor — accuracy / macro-F1 / per-class / ROC-AUC / ECE. Fact-check: run the agent
on the gold-verdict claims → verdict accuracy + abstain/coverage. Runs offline on
the seed; on Colab swap in the real test split.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..data.dataset import load_news, seed_split, load_claims
from ..models.classifier import TfidfLogRegClassifier, load_classifier
from . import metrics as M

logger = get_logger(__name__)


def _eval_clf(clf, items) -> Dict:
    y_true = [it.label for it in items]
    probs = [clf.predict_proba(it.content) for it in items]
    y_pred = [1 if p >= 0.5 else 0 for p in probs]
    return M.classification_metrics(y_true, y_pred, probs)


def evaluate(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    # ---- classification ----
    try:
        eval_items = load_news(cfg, split="test", limit=limit or cfg.data.max_eval_samples)
        train_items = load_news(cfg, split="train", limit=cfg.data.max_train_samples)
    except Exception:
        eval_items = None
    if not eval_items or len(eval_items) <= 2:
        train_items, eval_items = seed_split(cfg.data.seed)

    result: Dict = {"n_eval": len(eval_items)}
    main = load_classifier(cfg.classifier, prefer="transformer")
    result["classifier_name"] = main.name
    result["model"] = _eval_clf(main, eval_items)

    baseline = TfidfLogRegClassifier(cfg.classifier).fit([it.content for it in train_items],
                                                         [it.label for it in train_items])
    result["baseline"] = _eval_clf(baseline, eval_items)

    # majority-class floor
    maj = 1 if sum(it.label for it in train_items) * 2 >= len(train_items) else 0
    y_true = [it.label for it in eval_items]
    result["majority"] = M.classification_metrics(y_true, [maj] * len(eval_items),
                                                  [float(maj)] * len(eval_items))

    # ---- fact-check ----
    try:
        from ..agent.fakenews_agent import FakeNewsAgent
        agent = FakeNewsAgent(cfg, load_model=False)
        claims = load_claims(cfg)
        preds, golds = [], []
        for c in claims:
            job = agent.run(c["claim"], mode="factcheck", save=False)
            preds.append(job.verdict)
            golds.append(c["verdict"])
        result["factcheck"] = M.factcheck_metrics(preds, golds)
    except Exception as exc:
        logger.info("fact-check eval skipped (%s)", exc)
        result["factcheck"] = {"error": str(exc)}

    m, b = result["model"], result["baseline"]
    result["summary"] = {
        "classifier": main.name,
        "model_macro_f1": m["macro_f1"], "baseline_macro_f1": b["macro_f1"],
        "model_accuracy": m["accuracy"], "model_roc_auc": m.get("roc_auc"), "model_ece": m.get("ece"),
        "majority_macro_f1": result["majority"]["macro_f1"],
        "beats_baseline": (m["macro_f1"] >= b["macro_f1"]),
        "factcheck_accuracy": result.get("factcheck", {}).get("accuracy"),
    }
    if save:
        out = run_dir() / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"eval-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Eval saved: %s macro_f1=%s vs baseline %s | factcheck_acc=%s", main.name,
                    m["macro_f1"], b["macro_f1"], result["summary"]["factcheck_accuracy"])
    return result


__all__ = ["evaluate"]
