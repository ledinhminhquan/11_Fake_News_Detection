"""Data loading + label normalization + the classifier baseline + metrics +
evidence retrieval / stance."""

from __future__ import annotations

from fakenews.config import AppConfig
from fakenews.data.dataset import load_evidence, load_news, load_seed_news, seed_split
from fakenews.factcheck.retriever import EvidenceRetriever
from fakenews.factcheck.stance import LexicalStance
from fakenews.models.classifier import TfidfLogRegClassifier
from fakenews.training.metrics import classification_metrics, factcheck_metrics


def test_seed_news_balanced():
    items = load_seed_news()
    assert len(items) >= 30
    labels = [it.label for it in items]
    assert set(labels) == {0, 1}
    assert 0.3 < (sum(labels) / len(labels)) < 0.7      # roughly balanced
    tr, ev = seed_split(42)
    assert tr and ev and not (set(id(x) for x in tr) & set(id(x) for x in ev))


def test_baseline_classifier(cfg):
    items = load_seed_news()
    clf = TfidfLogRegClassifier(cfg.classifier).fit([it.content for it in items], [it.label for it in items])
    # a clearly fake claim should score high P(fake)
    pf = clf.predict_proba("Miracle pill melts thirty pounds in a single day, doctors furious.")
    assert pf > 0.5
    label, prob = clf.predict("Health agency recommends regular handwashing to reduce infections.")
    assert label in (0, 1) and 0 <= prob <= 1


def test_metrics():
    m = classification_metrics([0, 1, 0, 1], [0, 1, 1, 1], [0.1, 0.9, 0.6, 0.8])
    assert 0 <= m["macro_f1"] <= 1 and "roc_auc" in m and "ece" in m
    fm = factcheck_metrics(["fake", "real", "unverified"], ["fake", "real", "fake"])
    assert fm["accuracy"] == round(2 / 3, 4) and fm["abstain_rate"] == round(1 / 3, 4)


def test_evidence_retrieval_and_stance(cfg):
    retr = EvidenceRetriever.from_corpus(cfg.retrieval, load_evidence(cfg))
    hits = retr.retrieve("5G towers secretly control the weather", k=3)
    assert hits and any("5G" in h["text"] for h in hits)
    st, sc = LexicalStance(cfg.stance).score("5G towers control the weather",
                                             "5G is a radio communication technology; it cannot control the weather.")
    assert st == "refute" and sc > 0
