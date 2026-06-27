"""Fine-tune the fake-news classifier (HF Trainer, macro-F1).

Trains an ``AutoModelForSequenceClassification`` (`distilbert-base-uncased` base,
or `microsoft/deberta-v3-base` / `answerdotai/ModernBERT-base`) on (title+text →
label) from a clean fake-news dataset (`GonzaloA/fake_news` default), with class
weights for imbalance. Internal labels are normalized to 0=real / 1=fake. Resume-
safe; bf16/tf32 on H100/A100; macro-F1 as the model-selection metric. Heavy
imports are lazy. **Watch the source-leakage caveat** (docs/model_selection.md):
report the cross-domain number, not just in-domain.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models import model_registry as reg
from ..data.dataset import load_news, seed_split

logger = get_logger(__name__)


def train_classifier(cfg: AppConfig, limit: Optional[int] = None, resume: bool = True,
                     base_model: Optional[str] = None) -> Dict:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    cc = cfg.classifier
    model_id = base_model or cc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(cc.tf32)

    train_items = load_news(cfg, split="train", limit=limit)
    try:
        eval_items = load_news(cfg, split="validation", limit=cfg.data.max_eval_samples)
    except Exception:
        eval_items = None
    if not eval_items:
        train_items, eval_items = (train_items, train_items[: min(200, len(train_items))]) \
            if len(train_items) > 50 else seed_split(cfg.data.seed)
    logger.info("Training %s on %d items (eval %d)", model_id, len(train_items), len(eval_items))

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=cc.num_labels)

    def to_ds(items):
        return Dataset.from_dict({"text": [it.content for it in items],
                                  "labels": [int(it.label) for it in items]})

    def preprocess(batch):
        return tok(batch["text"], max_length=cc.max_length, truncation=True)

    train_ds = to_ds(train_items).map(preprocess, batched=True, remove_columns=["text"])
    eval_ds = to_ds(eval_items).map(preprocess, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tok)

    # class weights for imbalance
    labels = [int(it.label) for it in train_items]
    n0, n1 = labels.count(0), labels.count(1)
    w = torch.tensor([len(labels) / (2 * max(1, n0)), len(labels) / (2 * max(1, n1))], dtype=torch.float)

    from . import metrics as M

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        prob = _softmax(logits)[:, 1]
        pred = np.argmax(logits, axis=-1)
        m = M.classification_metrics(list(y), list(pred), list(prob))
        return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                "roc_auc": m.get("roc_auc", 0.0), "ece": m.get("ece", 0.0)}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=w.to(outputs.logits.device) if cc.use_class_weights else None)
            return (loss, outputs) if return_outputs else loss

    out_dir = cc.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=cc.num_train_epochs, learning_rate=cc.learning_rate,
        per_device_train_batch_size=cc.per_device_train_batch_size,
        per_device_eval_batch_size=cc.per_device_train_batch_size,
        weight_decay=cc.weight_decay, warmup_ratio=cc.warmup_ratio,
        bf16=bool(cc.bf16), fp16=bool(cc.fp16),
        eval_strategy="steps", save_strategy="steps", eval_steps=cc.eval_steps, save_steps=cc.save_steps,
        save_total_limit=2, logging_steps=50, seed=cc.seed, report_to=[],
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                              data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics)
    last = get_last_checkpoint(str(out_dir)) if resume and out_dir.exists() else None
    if last:
        logger.info("Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    metrics = {}
    try:
        metrics = {k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.info("final eval failed (%s)", exc)

    version = reg.make_version(model_id)
    final_dir = out_dir / version
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature={"train": len(train_items), "dataset": cfg.data.clf_dataset,
                                          "seed": cfg.data.seed}, metrics=metrics)
    reg.update_latest_pointer(out_dir, final_dir)
    (out_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Classifier training done -> %s", final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id,
            "n_train": len(train_items), "metrics": metrics}


def _softmax(x):
    import numpy as np
    x = np.asarray(x, dtype="float64")
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


__all__ = ["train_classifier"]
