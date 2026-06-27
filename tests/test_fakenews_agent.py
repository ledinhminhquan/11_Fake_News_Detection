"""The end-to-end fake-news / fact-check agent (TF-IDF + LogReg classifier +
lexical stance + BM25 over the seed evidence, fully offline) + decision helpers."""

from __future__ import annotations

from fakenews.agent.fakenews_agent import FakeNewsAgent
from fakenews.agent.policy import abstain_gate, checkworthy_gate, coverage_gate, detect_input
from fakenews.config import AppConfig
from fakenews.factcheck.verdict import aggregate_verdict


def _cfg():
    c = AppConfig()
    c.data.use_hf = False
    return c


def test_agent_factcheck_fake():
    agent = FakeNewsAgent(_cfg(), load_model=False)
    assert agent.classifier.name == "tfidf-logreg"
    assert agent.stance.name == "lexical"
    job = agent.run("Drinking bleach cures every virus overnight.", mode="factcheck", save=False)
    sd = job.to_dict()
    assert sd["verdict"] == "fake"
    assert sd["n_refute"] >= 1
    assert any("WHO" in (e.get("source") or "") for e in sd["evidence"])
    # the five decision points must all fire
    assert {d["id"] for d in sd["decisions"]} >= {"D1", "D2", "D3", "D4", "D5"}
    assert all(t["ok"] for t in sd["trace"])


def test_agent_factcheck_real():
    agent = FakeNewsAgent(_cfg(), load_model=False)
    job = agent.run("Handwashing with soap reduces the spread of infections.", mode="factcheck", save=False)
    sd = job.to_dict()
    assert sd["verdict"] == "real"
    assert sd["n_support"] >= 1


def test_classify_path():
    agent = FakeNewsAgent(_cfg(), load_model=False)
    r = agent.classify("BREAKING: aliens bought the entire city using gold bars last night.")
    assert r["label"] in ("fake", "real")
    assert 0.0 <= r["prior_fake"] <= 1.0


def test_decision_helpers():
    ac = AppConfig().agent
    # D1
    assert detect_input("Vaccines cause autism.", "auto", ac)["is_claim"] is True
    assert detect_input("word " * 100, "auto", ac)["is_claim"] is False
    # D2
    assert checkworthy_gate(0.6, True, "auto", ac)["factcheck"] is True
    assert checkworthy_gate(0.99, False, "auto", ac)["branch"] == "confident_skip"
    assert checkworthy_gate(0.6, False, "classify", ac)["factcheck"] is False
    # D3
    assert coverage_gate([], ac)["enough"] is False
    assert coverage_gate([{"relevance": 0.9}], ac)["enough"] is True
    # D5
    assert abstain_gate({"abstained": True, "confidence": 0.9, "verdict": "fake"}, ac)["abstained"] is True
    assert abstain_gate({"abstained": False, "confidence": 0.2, "verdict": "fake"}, ac)["abstained"] is True
    assert abstain_gate({"abstained": False, "confidence": 0.9, "verdict": "fake"}, ac)["abstained"] is False


def test_verdict_aggregation():
    ac = AppConfig().agent
    ev_fake = [{"stance": "refute", "score": 0.9, "relevance": 1.0}]
    r = aggregate_verdict(0.6, ev_fake, ac)
    assert r["verdict"] == "fake" and not r["abstained"]
    ev_real = [{"stance": "support", "score": 0.9, "relevance": 1.0}]
    r2 = aggregate_verdict(0.4, ev_real, ac)
    assert r2["verdict"] == "real"
    # no evidence => abstain
    r3 = aggregate_verdict(0.5, [], ac)
    assert r3["abstained"] is True and r3["verdict"] == "unverified"
