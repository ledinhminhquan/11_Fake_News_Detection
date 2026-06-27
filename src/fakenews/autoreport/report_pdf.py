"""Generate the submission report.pdf for the Fake News & Misinformation Detection system.

A 10–15 page report covering every Section-I deliverable: problem & business value,
data (with license flags), model + baselines, the held-out eval tables (accuracy /
macro-F1 / per-class P/R/F1 / ROC-AUC / ECE for the classifier vs TF-IDF+LogReg and
the majority floor, plus fact-check verdict accuracy / coverage / abstain-rate for
the agentic evidence-grounded checker), the agent architecture (decisions D1–D5:
claim-routing, check-worthiness, evidence-coverage, verdict, abstain), deployment,
continual learning & monitoring, data privacy & robustness (source leakage,
political bias, calibration), project plan, and ethics. Live numbers come from
``run_dir()`` artifacts via :mod:`artifact_loader`; missing metrics degrade to
placeholders.

Section prose is **self-contained** (built into this module, adapted from the design
brief) so the report renders even with no ``docs/*.md`` present; if a
``docs/<file>.md`` exists it is used instead. reportlab is lazy-imported; when it is
unavailable a Markdown fallback (``report.md``) is written and returned.

Internal label convention: 0 = real, 1 = fake (fake is the detected/positive class).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import (base_model, beats_baseline, classifier_name,
                              clf_metric, clf_per_class, factcheck_metric,
                              has_eval, has_factcheck, latency, load_artifacts,
                              model_version, read_doc, stance_name)

logger = get_logger(__name__)

_SUBTITLE = ("Detecting fake news & misinformation: a trainable transformer "
             "classifier (DistilBERT, fake-vs-real) → an agentic, evidence-grounded "
             "fact-check (claim routing → check-worthiness → evidence retrieval + "
             "NLI stance → verdict → abstain, decisions D1–D5). Flags content for "
             "human review — never auto-censors. cf. KaiDMML/FakeNewsNet.")

# (heading, docs/*.md filename to prefer if present)
_SECTIONS = [
    ("1. Problem Definition & Business Value", "problem_definition.md"),
    ("2. Data Description", "data_description.md"),
    ("3. Model Selection & Optimization", "model_selection.md"),
    ("4. Evaluation Protocol & Baselines", "evaluation.md"),
    ("5. Agent Architecture (Decisions D1–D5)", "agent_architecture.md"),
    ("6. Deployment", "deployment.md"),
    ("7. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("8. Data Privacy & Model Robustness", "privacy_robustness.md"),
    ("9. Project Plan & Teamwork", "project_plan.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Built-in section prose (Markdown). Used when docs/<file>.md is absent.
# ─────────────────────────────────────────────────────────────────────────────

def _builtin_sections(cfg: AppConfig, arts: Dict[str, Any]) -> Dict[str, str]:
    mf = clf_metric(arts, "model", "macro_f1")
    bf = clf_metric(arts, "baseline", "macro_f1")
    fc = factcheck_metric(arts, "accuracy")
    if mf is not None and bf is not None:
        res_line = (f"In the latest held-out eval the classifier reaches "
                    f"macro-F1 **{mf:.3f}** vs the TF-IDF+LogReg baseline **{bf:.3f}**"
                    + (f"; the agentic fact-check reaches verdict accuracy **{fc:.3f}** "
                       f"(coverage {factcheck_metric(arts, 'coverage') or 0:.2f})."
                       if fc is not None else "."))
    else:
        res_line = "Run `fakenews evaluate` to populate the live numbers here."
    return {
        "problem_definition.md": f"""
## What it does
Given a piece of news — a **headline + article body** or a short **claim** — produce
a credibility assessment: a fake-vs-real **classification** with a calibrated
probability, and, when the input is a check-worthy claim, an **evidence-grounded
fact-check** that retrieves supporting/refuting passages, judges their stance, and
returns a **verdict** (*real* / *fake* / *unverified*) with the evidence attached.
The system **flags content for human review — it never auto-censors or deletes.**

## The job-to-be-done
- **Platform trust & safety reviewer** — "surface likely-false stories for a human
  to check, with the evidence already gathered, instead of reading every post."
- **Journalist / fact-checker** — "is this viral claim supported by sources? show me
  what refutes it."
- **Reader / media-literacy tool** — "how credible is this article, and why?"
- **Researcher** — "an auditable, reproducible misinformation-detection baseline."

## Why it is hard (and why NLP helps)
Fake news mimics the surface form of real reporting; lexical cues alone overfit to a
particular outlet or era and transfer badly (the **source-leakage** trap — see §8).
A trained **transformer classifier** captures stylistic and semantic signal, while
the **evidence-grounded fact-check** grounds a verdict in retrieved sources rather
than style — the two complement each other and the agent abstains when neither is
confident. Reference corpus: **KaiDMML/FakeNewsNet**.

## Success metrics
- **Business:** review-queue precision (fraction of flagged items that are truly
  fake), analyst time saved, and — crucially — the **false-positive rate on real
  news** (wrongly flagging legitimate reporting is the censorship-risk error).
- **Technical:** **accuracy, macro-F1 (headline), per-class precision/recall/F1,
  ROC-AUC, and ECE (calibration)** for the classifier; **verdict accuracy,
  coverage, abstain-rate, and selective accuracy** for the fact-check; p50/p95 latency.

## Contribution to demonstrate
The fine-tuned classifier beats the TF-IDF+LogReg baseline and the majority floor on
macro-F1, the fact-check adds evidence-grounded verdicts with calibrated abstention,
and the whole system is **well-calibrated** (low ECE) so confidence can gate review.
{res_line}
""",
        "data_description.md": """
## Classification data (what we *train* on)
- **`GonzaloA/fake_news`** — primary fake-vs-real article corpus (title / text /
  label, pre-split train/val/test). **License flag: unspecified / unknown on the Hub
  — treated as research-only**, not redistributed in a shipped model. Label polarity
  differs across mirrors (here raw 0 = fake); loaders **normalize to internal
  1 = fake, 0 = real**.
- **`chengxuphd/liar2`** — secondary 6-way political-statement credibility set.
  **License flag: Apache-2.0 — PERMISSIVE** (the redistribution-safe source);
  the 6 labels are collapsed (0–2 → fake, 3–5 → real) for the binary task.
- **`ErfanMoosaviMonazzah/fake-news-detection-dataset-English`** — auxiliary
  (openrail) for robustness checks across distributions.

## Fact-check / evidence supervision
- **`fever/fever`** (FEVER) — claim + evidence + label for the retrieval + NLI
  stance pipeline. **License flag: CC-BY-SA 3.0 + GPL — share-alike / copyleft**;
  used for evaluation and stance calibration, flagged for license hygiene. `BeIR/fever`
  provides a retrieval-eval split.

## Stance / retrieval models (pretrained, not trained here)
NLI stance from **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** (MIT;
entail → support, contradict → refute, neutral → abstain); dense retrieval from
**`sentence-transformers/all-MiniLM-L6-v2`** (Apache) + BM25 + a MiniLM cross-encoder
re-ranker (Apache).

## Held-out evaluation & offline seed
Held-out GonzaloA test split (and a LIAR2 slice) on Colab. For offline tests a small
**seed of ~40 labelled news items + 16 evidence passages + 8 gold-verdict claims**
(``data/samples.py``) stands in, so the whole pipeline — TF-IDF+LogReg classifier +
lexical stance — is testable with **no network and no GPU**.

## Splitting & hygiene
Split by **article**, never by sentence — no article in the test split appears in
training. We actively guard against **source leakage** (outlet/byline shortcuts that
inflate accuracy but do not transfer); see §8.
""",
        "model_selection.md": f"""
## Trainable core
A **transformer sequence classifier** is the NLP heart of the detector; an
**agentic evidence-grounded fact-check** (retrieval + NLI stance + rule aggregation)
is the second component. The stance/retrieval models are pretrained and **not**
fine-tuned here.

- **Classifier base:** `{base_model(arts)}` (default `distilbert-base-uncased`,
  Apache, 67M, T4-light; `num_labels={cfg.classifier.num_labels}`, max length
  {cfg.classifier.max_length}). Upgrades: `microsoft/deberta-v3-base` (MIT),
  `answerdotai/ModernBERT-base`.
- **Stance model:** `{stance_name(arts)}` — zero-shot NLI mapping entail → support,
  contradict → refute, neutral → abstain.

## Optimization
- **Classifier:** HF Trainer, {cfg.classifier.num_train_epochs} epochs, lr
  {cfg.classifier.learning_rate:g}, batch {cfg.classifier.per_device_train_batch_size},
  weight-decay {cfg.classifier.weight_decay}, warmup {cfg.classifier.warmup_ratio},
  **class weights** for imbalance; bf16+tf32 on Ampere+, fp16 on T4. Selected on
  **macro-F1**, never raw accuracy (the classes are not always balanced).
- **Calibration:** ECE is tracked alongside accuracy; over-confidence is dangerous
  when a probability gates a review queue, so a mis-calibrated model is penalised.
- **Threshold:** the keep/flag threshold on P(fake) is tunable; the agent reads the
  probability as a soft prior, not a hard label.

## Baselines
1. **Majority class** — the trivial floor (predict the dominant label everywhere).
2. **TF-IDF + Logistic Regression** — the classic strong lexical baseline (also the
   offline / no-GPU fallback classifier).
3. **Zero-shot NLI** — using the stance model directly as a weak veracity signal.

{res_line}
""",
        "evaluation.md": f"""
## Protocol (held-out articles + gold claims)
**Classification:** the system label vs the gold label on held-out articles.
**Fact-check:** run the agent on the gold-verdict claim set and compare its verdict
(*real* / *fake* / *unverified*) to the gold verdict.

## Metrics
- **Classification — accuracy, macro-F1 (headline), per-class precision/recall/F1,
  ROC-AUC, and ECE** (Expected Calibration Error). Macro-F1 is the headline so the
  minority class is not ignored; **per-class recall on *real*** is watched because a
  low value means legitimate news is being wrongly flagged.
- **Fact-check — verdict accuracy, coverage** (fraction not abstained),
  **abstain-rate**, and **selective accuracy** (accuracy on the decided subset).
- **Latency** — p50/p95 of classify / retrieve / stance / total.

## Baselines
- Classification: **majority-class floor**, **TF-IDF + LogReg**, zero-shot NLI.
- Fact-check: the classifier prior alone (no evidence) is the floor the
  evidence-grounded verdict must improve on or sensibly abstain over.

{res_line} The classifier must clear the TF-IDF baseline and the majority floor on
macro-F1, and the fact-check must add value through calibrated, evidence-backed
verdicts — the central claims of this project.
""",
        "agent_architecture.md": f"""
## FSM
A deterministic finite-state machine; every tool returns a uniform
`{{ok, data, meta, error}}` dict and every transition is logged to a trace. States:
`INGEST → CLASSIFY → ROUTE_CLAIM → CHECKWORTHY_GATE → RETRIEVE_EVIDENCE →
STANCE → VERDICT → ABSTAIN_GATE → ASSEMBLE`. An optional LLM **brain**
(`{cfg.agent.llm_model}`, OFF by default) may only *propose a value from a closed
legal set*; rules clamp to bounds and win on disagreement. The agent runs fully on
rules with **zero paid API calls** by default.

## Five decisions (each acts on the model's own intermediate output)
- **D1 — claim routing.** Decide whether the input is a long **article** (classify
  path) or a short **claim** (≤ {cfg.agent.short_claim_words} words → fact-check
  path). Records `is_claim`, the extracted claim, and the requested mode.
- **D2 — check-worthiness gate.** If the classifier is very confident
  (≥ {cfg.agent.skip_factcheck_confidence}) and the text is not check-worthy, skip
  retrieval and return the classification — don't burn evidence lookups on the obvious.
  Otherwise proceed to retrieval.
- **D3 — evidence-coverage gate.** Require at least {cfg.agent.min_evidence} relevant
  passage(s) (relevance ≥ {cfg.agent.min_evidence_relevance}); below that, widen the
  search or **abstain** rather than rule on thin evidence.
- **D4 — verdict gate.** Aggregate support vs refute stance (with the classifier prior
  at soft weight {cfg.agent.prior_weight}); call a verdict only when the support–refute
  margin ≥ {cfg.agent.verdict_margin}. Evidence dominates the prior.
- **D5 — abstain gate.** If verdict confidence < {cfg.agent.min_verdict_confidence},
  return **unverified** (abstain) instead of a low-confidence *fake*/*real* call —
  abstention is a feature, not a failure.

The agent emits `{{clf_label, clf_prob, verdict, confidence, abstained, evidence[],
n_support, n_refute, rationale, decisions[], trace[], metrics}}` with a full audit
trail; every verdict cites the evidence it rests on.
""",
        "deployment.md": f"""
## Serving
A single FastAPI process (`{cfg.serving.api_title}`, {cfg.serving.api_version})
exposes the pipeline. Heavy deps (torch, transformers, sentence-transformers,
matplotlib) are **lazy-imported** so the package imports and runs on
numpy/pandas/sklearn alone — the TF-IDF+LogReg classifier + lexical stance answer
(in degraded mode) even without GPU libraries or a network.

## Endpoints
- `POST /analyze` — `{{text, title?, mode?}}` → `{{clf_label, clf_prob, verdict,
  confidence, abstained, evidence[], decisions{{}}, trace[], metrics, model_versions}}`.
  `mode` ∈ {{auto, classify, factcheck}}; `verdict` ∈ {{real, fake, unverified}}.
- `GET /healthz` — readiness (model load state). `GET /version` — pinned
  `model_version` ({cfg.serving.model_version}), base models, git SHA.

## UI / CLI / packaging
A Gradio UI with a **paste-article** tab and a **check-a-claim** tab, a "decisions
trace" expander (D1–D5 + timings) that shows the retrieved evidence and stance, and a
privacy banner when the brain is enabled. CLI `fakenews run / evaluate / serve`.
Multi-stage Docker (CPU default) and an HF Space (Docker SDK) that bakes the seed
data and auto-disables the brain when the API key is unset.

## Latency, scalability, versioning
Classification is light; evidence retrieval + stance dominate wall-clock on the
fact-check path. The check-worthiness gate (D2) avoids retrieval on confident,
non-check-worthy inputs. Requests are logged to
`{cfg.serving.request_log_subdir}/requests.jsonl` (metadata only). Models are pinned
by `model_version`; scores are comparable only within a version.
""",
        "continual_learning_monitoring.md": """
## Continual learning
- **Classifier refresh:** misinformation drifts (new narratives, new framings), so
  the classifier is periodically re-fine-tuned on freshly labelled articles under a
  new `model_version`; promoted only if held-out macro-F1 is non-regressing **and ECE
  does not worsen** (a more-accurate but worse-calibrated model is rejected).
- **Evidence-corpus refresh:** the retrieval corpus is updated as new fact-checks and
  primary sources appear, so the agent can ground verdicts in current evidence rather
  than stale snapshots.

## Monitoring
- **Quality drift:** track accuracy, macro-F1, per-class **real-recall** (the
  false-flag signal), ROC-AUC, ECE, and fact-check verdict accuracy / coverage on a
  rolling eval slice; alert on regression beyond a tolerance band.
- **Operational:** p50/p95 classify / retrieve / stance / total latency,
  check-worthiness skip-rate, abstain-rate, and review-queue volume.
- **Distribution drift:** input length, topic / source mix, and class balance vs the
  training distribution (a concept-drift signal for misinformation). Logs are
  metadata-only by default.
""",
        "privacy_robustness.md": """
## Data privacy
- **Public text, but care still needed.** News articles and claims are largely
  public, but submitted text may quote private individuals or sensitive content. The
  core path is **local / no-network**; the optional LLM brain is **opt-in, OFF by
  default**, auto-disabled without an API key. Request logs store metadata, not raw
  article bodies.
- **License hygiene.** The redistributable path leans on the **Apache** source
  (LIAR2) + seed data; **GonzaloA (unspecified licence)** and **FEVER (CC-BY-SA +
  GPL, copyleft)** are flagged research-only / license-encumbered and kept out of any
  shipped, redistributed model.

## Model robustness
- **Source leakage** — the single biggest robustness trap: a model can hit high
  accuracy by memorising outlet style or boilerplate (URLs, datelines) instead of
  veracity, then collapse on unseen sources. We strip obvious source artefacts,
  split by article, and report transfer to a second distribution.
- **Political bias / calibration** — fake-news labels carry political and cultural
  bias; the model can inherit it. We watch **per-class real-recall** (are
  legitimate-but-unpopular-viewpoint articles disproportionately flagged?) and track
  **ECE** so a confident-but-wrong flag cannot silently gate review.
- **False positives (flagging real news)** — the censorship-risk error. The agent
  **flags for human review, never auto-removes**, and **abstains** (D5) when not
  confident rather than guessing.
- **Thin evidence** — the coverage gate (D3) abstains instead of ruling on one weak
  passage; verdicts always cite their evidence so a human can check the chain.
- **Graceful degradation** — every heavy dep has a numpy/sklearn fallback (TF-IDF +
  LogReg classifier + lexical stance), so the service answers even when GPU libraries
  or the network are absent.
""",
        "project_plan.md": """
## Build order
1. Config, logging, model registry, offline seed (`data/samples.py`).
2. Datasets layer — GonzaloA / LIAR2 loaders (label normalization) + FEVER evidence.
3. TF-IDF + LogReg baseline (also the offline fallback classifier).
4. Fine-tune the transformer classifier (HF Trainer; class weights; macro-F1 select).
5. Evidence retrieval (BM25 + dense + re-ranker) and NLI stance.
6. Implement the `FakeNewsAgent` FSM, tools, and decisions D1–D5.
7. Ship `/analyze` API, Gradio UI, CLI, Docker / HF Space.
8. Run the evaluation protocol (accuracy / macro-F1 / per-class / ROC-AUC / ECE +
   fact-check accuracy / coverage) and fill the tables.
9. Reports, monitoring, grading checklist, end-to-end autopilot.

## Teamwork & reproducibility
Single-author project (Le Dinh Minh Quan, 23127460). All paths come from config/env
(nothing hard-coded); every run writes versioned JSON under `run_dir()`; the whole
pipeline is testable **offline** on the seed with the TF-IDF classifier + lexical
stance, so CI needs no GPU, no network, and no license-encumbered data.
""",
        "ethics_statement.md": """
## Intended use & dual-use
The system is a **misinformation-triage and fact-check aid** — it produces a
credibility signal and an evidence-backed verdict for a human to **review and act
on**, and it **flags content for review; it never auto-censors, hides, or deletes**.
Treating an automated *fake* label as ground truth would be a misuse: the output is a
prompt for verification, not a verdict of record.

## Key risks & mitigations
- **Censorship / over-blocking** → flags for human review only; never auto-removes;
  abstains (D5) when not confident; every verdict cites its evidence so a human can
  overturn it.
- **False positives on real news** → macro-F1 and **per-class real-recall** are
  watched explicitly; the false-flag error is treated as the costly one.
- **Political & cultural bias** → labels are politically loaded; we audit per-class
  behaviour, avoid one-source training, and report transfer across distributions
  rather than a single in-domain number.
- **Source leakage** → strip outlet/byline artefacts, split by article, and report
  out-of-distribution transfer so headline accuracy is not a memorisation mirage.
- **Mis-calibrated confidence** → ECE is a first-class metric; a probability that
  gates a review queue must mean what it says, so over-confidence is penalised.
- **License misuse** → GonzaloA (unspecified) and FEVER (CC-BY-SA + GPL) flagged
  research-only / copyleft; the redistributable path uses permissive (Apache/LIAR2) +
  seed data.

The default configuration is offline-capable, abstains under uncertainty, attaches
evidence to every verdict, and **only ever recommends content for human review.**
""",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → reportlab flowables
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _md_to_flowables(md: str, styles, max_lines: int = 260):
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    flow, lines, in_code, code, bullet = [], md.splitlines()[:max_lines], False, [], []

    def flush():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("• " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"])); code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln); continue
        s = ln.rstrip()
        if not s:
            flush(); flow.append(Spacer(1, 5)); continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            flow.append(Paragraph(_esc(s.lstrip("#").strip()), styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        elif s.lstrip().startswith("|") and "|" in s[1:]:
            flush()
            if not re.match(r"^\s*\|[\s:|-]+\|\s*$", s):
                cells = [c.strip() for c in s.strip().strip("|").split("|")]
                flow.append(Paragraph(_esc(" — ".join(cells)), styles["Body"]))
        else:
            flush(); flow.append(Paragraph(_esc(s), styles["Body"]))
    flush()
    return flow


def _results_tables(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    # ---- classification table ----
    flow = [Paragraph("Results — classification (held-out eval)", styles["H3"])]
    rows = [["Metric", "majority", "TF-IDF+LogReg", "classifier"]]
    if has_eval(arts):
        for key, label in [("accuracy", "Accuracy ↑"), ("macro_f1", "Macro-F1 ↑"),
                           ("roc_auc", "ROC-AUC ↑"), ("ece", "ECE ↓")]:
            jv = clf_metric(arts, "majority", key)
            bv = clf_metric(arts, "baseline", key)
            mv = clf_metric(arts, "model", key)
            rows.append([label,
                         f"{jv:.3f}" if jv is not None else "—",
                         f"{bv:.3f}" if bv is not None else "—",
                         f"{mv:.3f}" if mv is not None else "—"])
    else:
        rows.append(["—", "run `evaluate`", "—", "—"])
    t = Table(rows, hAlign="LEFT", colWidths=[120, 95, 110, 95])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    flow += [t, Spacer(1, 6),
             Paragraph(f"Classifier: <b>{classifier_name(arts)}</b>"
                       + ("  (beats baseline)" if beats_baseline(arts) else ""), styles["Body"]),
             Spacer(1, 8)]

    # ---- per-class table ----
    flow.append(Paragraph("Results — per-class precision / recall / F1 (classifier)", styles["H3"]))
    prows = [["Class", "Precision", "Recall", "F1"]]
    if has_eval(arts):
        for cls, label in [("real", "real (0)"), ("fake", "fake (1)")]:
            p = clf_per_class(arts, "model", cls, "precision")
            r = clf_per_class(arts, "model", cls, "recall")
            f = clf_per_class(arts, "model", cls, "f1")
            prows.append([label,
                          f"{p:.3f}" if p is not None else "—",
                          f"{r:.3f}" if r is not None else "—",
                          f"{f:.3f}" if f is not None else "—"])
    else:
        prows.append(["—", "run `evaluate`", "—", "—"])
    pt = Table(prows, hAlign="LEFT", colWidths=[110, 105, 105, 105])
    pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#553c9a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0ecf8")])]))
    flow += [pt, Spacer(1, 4),
             Paragraph("Low <b>real recall</b> = legitimate news wrongly flagged "
                       "(the censorship-risk error this project watches).", styles["Body"]),
             Spacer(1, 8)]

    # ---- fact-check table ----
    flow.append(Paragraph("Results — agentic fact-check (gold-verdict claims)", styles["H3"]))
    frows = [["Metric", "value"]]
    if has_factcheck(arts):
        for key, label in [("accuracy", "Verdict accuracy ↑"),
                           ("coverage", "Coverage ↑"),
                           ("selective_accuracy", "Selective accuracy ↑"),
                           ("abstain_rate", "Abstain rate")]:
            v = factcheck_metric(arts, key)
            frows.append([label, f"{v:.3f}" if v is not None else "—"])
    else:
        frows.append(["run `evaluate`", "—"])
    ft = Table(frows, hAlign="LEFT", colWidths=[200, 120])
    ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f855a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf5ee")])]))
    flow += [ft, Spacer(1, 6),
             Paragraph("Abstention (<b>unverified</b>) under thin evidence or low "
                       "confidence is a feature: the agent flags for review, never "
                       "guesses a verdict.", styles["Body"]), Spacer(1, 6)]

    lat = latency(arts, "total", "p50")
    if lat is not None:
        flow.append(Paragraph(f"Latency: total p50 ≈ {lat:.0f} ms "
                              f"(retrieve p95 ≈ {latency(arts, 'retrieve', 'p95') or 0:.0f} ms).",
                              styles["Body"]))
    flow.append(Spacer(1, 8))
    return flow


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out = Path(out_path) if out_path else run_dir() / "report" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    builtins = _builtin_sections(cfg, arts)

    def section_md(fname: str) -> str:
        # Prefer the repo's docs/<file>.md when present; trim each to a report-sized
        # budget to keep the PDF in the 10–15 page target. Built-in report prose is
        # used when the doc is absent.
        doc = read_doc(fname)
        if doc.strip():
            lines = doc.splitlines()
            return "\n".join(lines[:40]) if len(lines) > 40 else doc
        return builtins.get(fname, "")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph,
                                        SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        md += f"_Generated {utc_now_iso()} · model {model_version(arts)}_\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        logger.info("Report (markdown fallback) -> %s", alt)
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=22, leading=26),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    try:
        built = dict(charts_mod.build_all(arts, out.parent / "charts"))
    except Exception as exc:
        logger.info("charts skipped (%s)", exc)
        built = {}

    story: List[Any] = [
        Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
        Paragraph(f"<b>{author}</b> — Student {cfg.student_id}", styles["Meta"]),
        Paragraph("NLP in Industry — Final Assignment (P11)", styles["Meta"]),
        Paragraph(_SUBTITLE, styles["Meta"]),
        Paragraph(f"Generated {utc_now_iso()}", styles["Body"]),
        Paragraph(f"Trained classifier: <b>{model_version(arts)}</b> (base {base_model(arts)})", styles["Body"]),
    ]
    story.append(PageBreak())
    story += _results_tables(arts, styles)
    for name in ("classification", "per_class", "errors"):
        if name in built:
            story += [Image(str(built[name]), width=13 * cm, height=7.3 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    # Sections flow continuously (a spacer between them) rather than one forced
    # page-break each, keeping the report in the 10–15 page target.
    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(section_md(fname), styles)
        story.append(Spacer(1, 10))

    try:
        SimpleDocTemplate(str(out), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=title, author=author).build(story)
    except Exception as exc:
        logger.warning("reportlab build failed (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)
    logger.info("Report -> %s", out)
    return str(out)


__all__ = ["generate_report"]
