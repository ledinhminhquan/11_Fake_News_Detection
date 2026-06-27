"""Stance / NLI scoring: does an evidence snippet SUPPORT, REFUTE or stay NEUTRAL
on a claim?

* ``NLIStance`` wraps a pretrained NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-
  fever-anli`, zero-shot): premise = evidence, hypothesis = claim →
  entailment ⇒ support, contradiction ⇒ refute, neutral ⇒ neutral.
* ``LexicalStance`` is the dependency-free fallback: token overlap gives relevance,
  and negation/falsity cues in the evidence flip support → refute — so a verdict is
  still produced offline with no torch.

``load_stance`` returns the NLI model when transformers is available, else the
lexical heuristic.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..config import StanceConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9']+")
_NEG = re.compile(r"\b(no|not|never|cannot|can't|don't|doesn't|isn't|false|fake|hoax|baseless|"
                  r"debunk\w*|myth|dangerous|impossible|fraudulent|untrue|misinformation)\b", re.I)
_SUPPORT = re.compile(r"\b(reduces?|associated with|recommend\w*|prevent\w*|confirm\w*|"
                      r"shown to|effective|true|supported)\b", re.I)


def _toks(s: str):
    return {w for w in _WORD.findall((s or "").lower()) if len(w) > 2}


class LexicalStance:
    name = "lexical"
    version = "lexical-1.0"

    def __init__(self, cfg: Optional[StanceConfig] = None):
        self.cfg = cfg or StanceConfig()

    def score(self, claim: str, evidence: str) -> Tuple[str, float]:
        ct, et = _toks(claim), _toks(evidence)
        if not ct or not et:
            return "neutral", 0.0
        overlap = len(ct & et) / len(ct)
        if overlap < 0.12:
            return "neutral", round(overlap, 3)
        neg = bool(_NEG.search(evidence))
        sup = bool(_SUPPORT.search(evidence))
        conf = min(0.95, 0.5 + overlap)
        if neg and not sup:
            return "refute", round(conf, 3)
        if sup and not neg:
            return "support", round(conf, 3)
        # overlap but ambiguous cues → lean by which cue is present
        return ("refute" if neg else "support"), round(0.5 + overlap / 2, 3)


class NLIStance:
    name = "nli"

    def __init__(self, pipe, cfg: StanceConfig, version: str = "nli-1.0"):
        self.pipe = pipe
        self.cfg = cfg
        self.version = version

    @classmethod
    def load(cls, cfg: StanceConfig, device: Optional[int] = None) -> "NLIStance":
        from transformers import pipeline  # lazy
        dev = device if device is not None else -1
        try:
            import torch
            if torch.cuda.is_available():
                dev = 0
        except Exception:
            pass
        pipe = pipeline("text-classification", model=cfg.nli_model, tokenizer=cfg.nli_model,
                        top_k=None, truncation=True, max_length=cfg.max_length, device=dev)
        return cls(pipe, cfg, version=cfg.nli_model.split("/")[-1])

    def score(self, claim: str, evidence: str) -> Tuple[str, float]:
        # NLI: premise=evidence, hypothesis=claim. Many NLI pipelines take a single
        # "premise [SEP] hypothesis" string via a dict input.
        out = self.pipe({"text": evidence, "text_pair": claim})
        rows = out[0] if (out and isinstance(out[0], list)) else out
        probs = {str(r["label"]).lower(): float(r["score"]) for r in rows}
        ent = probs.get("entailment", 0.0)
        con = probs.get("contradiction", 0.0)
        neu = probs.get("neutral", 0.0)
        if ent >= self.cfg.support_threshold and ent >= con:
            return "support", round(ent, 3)
        if con >= self.cfg.refute_threshold and con > ent:
            return "refute", round(con, 3)
        return "neutral", round(max(neu, 1.0 - max(ent, con)), 3)


def load_stance(cfg: StanceConfig, *, prefer: str = "nli", device: Optional[int] = None):
    if prefer == "nli" and cfg.enabled:
        try:
            return NLIStance.load(cfg, device=device)
        except Exception as exc:
            logger.info("NLI stance unavailable (%s); using lexical stance.", exc)
    return LexicalStance(cfg)


__all__ = ["NLIStance", "LexicalStance", "load_stance"]
