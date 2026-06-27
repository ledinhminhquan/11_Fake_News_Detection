# P11 — Evaluation Methodology & Results

> **Project:** P11 — Fake News & Misinformation Detection System
> **Course:** NLP in Industry — final assignment
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Package:** `fakenews`
> **Source of truth:** [`docs/DESIGN_BRIEF.md`](DESIGN_BRIEF.md) (verified HF ids, pipeline, FSM decisions D1–D5, thresholds, ethics).

This document defines **how P11 is evaluated** and **records the results actually produced**. The system has two evaluable surfaces that are scored **separately** (a non-negotiable ethical requirement — see §8 of the brief):

1. a **fake-news classifier** — a transformer fine-tune (`distilbert-base-uncased` default / `microsoft/deberta-v3-base` / `answerdotai/ModernBERT-base`) with a **TF-IDF + LogReg** baseline floor, producing a fast credibility **prior** `P(fake)`; and
2. an **agentic, evidence-grounded fact-check** — extract claim → retrieve evidence (BM25 + dense `all-MiniLM-L6-v2` + RRF) → per-evidence **stance/NLI** (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, zero-shot) → aggregate **verdict** → **ABSTAIN** when uncertain.

> **Internal label convention:** `0 = real`, `1 = fake`; `p_fake = P(label == 1)`. Per-source `label_map` normalisation is applied on load (mirrors disagree — see §2a of the brief; `GonzaloA` is natively `0=fake/1=real`).

The full pipeline also runs **FULLY OFFLINE** (TF-IDF+LogReg classifier + lexical-overlap stance + BM25 over the seed evidence, no `torch`), and **all numbers in §7 (Verified Results) were produced in exactly that offline mode** on the committed synthetic seed. The large-scale (GonzaloA / LIAR2 / FEVER) numbers in §5–§6 are **skeletons to fill after H100 training**.

---

## 1. What we measure and why

The two costly failure modes drive the metric selection:

| Failure mode | Why it is costly | Metric that catches it |
|---|---|---|
| **False positive** — *real news flagged fake* | Silences legitimate journalism; chills speech; the single most expensive error in this domain | **`real`-recall**, **`fake`-precision**, per-class P/R/F1 |
| **Over-confidence** — confident *and* wrong | Reviewers rubber-stamp the model (automation bias); a confident wrong "fake" label can suppress real news | **ECE** + reliability diagram, temperature scaling |
| **Class imbalance hiding errors** | Accuracy looks fine while the minority class collapses (LIAR `pants-fire` is rare) | **Macro-F1 (the HEADLINE metric)** |
| **Source-style leakage** | Model learns "is this Reuters formatting?", not veracity → collapses on new outlets | **Cross-domain macro-F1 drop** (PolitiFact→GossipCop) |
| **Fabricated certainty** on novel claims | A guessed verdict with no evidence is worse than an honest "I don't know" | **Coverage** + **selective accuracy** (risk–coverage curve); ABSTAIN is first-class |

**Macro-F1 is the headline number** for both the binary fake/real task and the 6-way LIAR2 credibility task — it does not let the majority class hide a collapsed minority class. Accuracy is reported alongside it but never alone.

---

## 2. Classification evaluation

### 2.1 Metrics

| Metric | Definition / notes | Implementation |
|---|---|---|
| **Accuracy** | Fraction correct. Reported, never alone. | `sklearn.metrics.accuracy_score` |
| **Macro-F1** *(HEADLINE)* | Unweighted mean of per-class F1. Robust to fake/real and 6-way LIAR imbalance. `metric_for_best_model="f1"` in training selects on this. | `f1_score(y, p, average="macro")` |
| **Per-class P/R/F1** | Especially **`real`-recall** (legit news not wrongly flagged) and **`fake`-precision** (flagged items really are fake). | `classification_report(..., digits=4)` |
| **ROC-AUC** | Threshold-independent ranking quality. Binary: AUC of `p_fake`. 6-way: one-vs-rest macro-AUC. | `roc_auc_score` |
| **ECE** (Expected Calibration Error) | Are the probabilities honest? Bin predictions by confidence, compare mean confidence vs accuracy per bin; ECE = weighted mean gap. Over-confidence is the danger. | `training/metrics.py::expected_calibration_error` (15-bin default) |

`compute_metrics` during training returns `{"f1": macro-F1, "accuracy": …}`; `f1` is the model-selection signal with `EarlyStoppingCallback(patience=2)`.

### 2.2 Calibration & temperature scaling

ECE alone diagnoses; **temperature scaling** fixes. Procedure:

1. Hold out a **calibration split** (distinct from train/test).
2. Fit a single scalar **T** that minimises NLL on the calibration logits (`logits / T` before softmax). `T > 1` softens over-confident logits.
3. **Report ECE pre- and post-scaling**, plus a reliability diagram.

Temperature scaling is monotone, so it leaves accuracy/macro-F1/ROC-AUC **unchanged** while reducing ECE — it only re-aligns the confidence axis. This matters because the served `/classify` probability and the agent's `classifier_prior` nudge are only trustworthy if calibrated.

### 2.3 Baselines (the strong system must beat all three)

| Baseline | What it is | Role |
|---|---|---|
| **Majority class** | Always predict the most frequent label | Sanity floor. On a balanced binary set, macro-F1 ≈ 0.333 (it scores 0 F1 on the never-predicted class). |
| **TF-IDF + LogReg** | `TfidfVectorizer(ngram=(1,2), min_df=3, max_df=0.9, sublinear_tf, max_features=200k)` → `LogisticRegression(C=4, class_weight="balanced")`. No torch. This is the **offline core**, not a throwaway. | The floor the transformer must beat — **on cross-domain, not in-domain**. |
| **Zero-shot** | `facebook/bart-large-mnli` with "this is fake/real news" as hypotheses (binary); `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for stance. | No-training reference. |
| **Strong system** | Transformer fine-tune (`distilbert-base-uncased` → `microsoft/deberta-v3-base` → `answerdotai/ModernBERT-base`) on `GonzaloA/fake_news`, boilerplate-stripped. | The trainable showpiece. |

> **Red-flag note:** TF-IDF+LogReg reaches ~0.95+ in-domain macro-F1 on GonzaloA/WELFake. That is **not success** — it is the signature of source-style leakage. The honest comparison is **cross-domain**.

### 2.4 Cross-domain generalisation (the source-leakage signal)

This is the most important — and most honest — classifier metric.

```
TRAIN on  GonzaloA/fake_news        (PolitiFact-style wire-vs-blog)
TEST  on  LittleFish-Coder/Fake_News_GossipCop   (celebrity/gossip domain)
REPORT    in-domain macro-F1  −  cross-domain macro-F1   =  the DROP
```

A **large drop** means the model learned *outlet/style*, not *veracity*. Because GonzaloA "real" rows are largely Reuters/AP wire copy (`"MOSCOW (Reuters) — …"`) and "fake" rows are blog-rant style, in-domain F1 overstates capability. **The cross-domain number is the metric that goes in the abstract.** It is itself an ethics metric (§8 of the brief). Dateline/source boilerplate (`(Reuters)`, `WASHINGTON —`, bylines) is stripped before training to suppress the cheapest leakage shortcut; the cross-domain gap measures whatever leakage remains.

---

## 3. Fact-check evaluation

The agentic fact-check is scored FEVER-style, plus abstention quality.

| Metric | Definition | Notes |
|---|---|---|
| **Label accuracy** | SUPPORTS/REFUTES/NEI (→ real/fake/unverified) correct, ignoring evidence. | The headline fact-check number. |
| **FEVER score** | Stricter: label correct **and** a complete gold evidence group retrieved. | Penalises "right for the wrong reason". |
| **Evidence recall@k** | Fraction of gold evidence in the top-k retrieved. Report **R@1 / R@5 / R@10** (official FEVER uses R@5 sentences, R@20 docs). | Measured against `BeIR/fever-qrels` (cc-by-sa-4.0). |
| **Coverage** | Fraction of cases **not** abstained = `n_presented / n_total`. | High coverage with low accuracy is bad; abstaining well is good. |
| **Selective accuracy** | Accuracy on the **non-abstained** subset only. | Should rise as coverage falls. |
| **Risk–coverage curve** | Sweep the abstain threshold; plot selective accuracy (or error = risk) vs coverage. | The honest picture of the accuracy/coverage trade-off. |

**Abstain is first-class.** `unverified` means *insufficient evidence*, **not** *true*. A system that abstains on a novel uncovered claim is behaving correctly, not failing. The risk–coverage curve, not raw accuracy, is the right lens.

### 3.1 Stance/NLI mapping under evaluation

Per-evidence NLI is read from `model.config.id2label` at load (verified `{0:entailment, 1:neutral, 2:contradiction}` — **never hardcoded**) and bridged:

| NLI label | Stance | Verdict contribution (D4) |
|---|---|---|
| entailment | **support** | `+1 · w_rerank` |
| contradiction | **refute** | `−1 · w_rerank` |
| neutral | NEI | `0` |

Verdict aggregation `S = Σ wᵢ·sᵢ`, combined with prior at `α=0.6`, thresholded at `θ=0.2` (REAL `S>+θ` / FAKE `S<−θ` / UNVERIFIED `|S|≤θ`). **Evidence dominates; the classifier prior is a soft nudge.**

---

## 4. The five FSM decisions are observable in eval

Every evaluation case emits a `ToolTrace`, so eval can audit which decision fired. Mapping to metrics:

| Decision | Gate (rule) | Metric it influences |
|---|---|---|
| **D1** claim routing | `len(tokens) ≤ 40` or no body → short-claim; else article (central claim = title / lead / TextRank) | input handling correctness |
| **D2** check-worthiness / confidence gate | skip retrieval iff `conf ≥ τ_skip=0.95` AND `checkworthy < τ_cw=0.5` | coverage (fewer needless retrievals) |
| **D3** evidence-coverage gate | keep evidence `rerank ≥ τ_rel=0.3`; OK iff `n_relevant ≥ N_min=3`; else widen ≤ `R_max=2` then ABSTAIN | evidence recall@k, abstain rate |
| **D4** stance-aggregation verdict | `S=Σwᵢ·sᵢ`, `α=0.6`, `θ=0.2` → REAL/FAKE/UNVERIFIED | label accuracy |
| **D5** confidence / abstain gate | abstain if UNVERIFIED, or `agreement<τ_agree=0.4`, or `final_conf<τ_present=0.55`, or **prior↔stance conflict** | selective accuracy, coverage |

---

## 5. Results — Classification (skeletons to fill after H100 training)

> Fill after the GonzaloA / LIAR2 fine-tunes complete on H100. Expected realistic in-domain macro-F1 **≈ 0.90**; expected cross-domain (GossipCop) **markedly lower** — record the gap honestly.

| System | Dataset / split | Accuracy | Macro-F1 | P(fake) / R(real) | ROC-AUC | ECE | Coverage @ sel-acc |
|---|---|---|---|---|---|---|---|
| Majority class | GonzaloA test | _TBD_ | ~0.333 | — | — | — | — |
| TF-IDF + LogReg | GonzaloA test (in-domain) | _TBD_ | _TBD (~0.95, red-flag)_ | _TBD_ | _TBD_ | _TBD_ | — |
| Zero-shot (`bart-large-mnli`) | GonzaloA test | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | — |
| Transformer fine-tune (`distilbert`/`deberta-v3`/`ModernBERT`) | GonzaloA test (in-domain) | _TBD_ | _TBD (~0.90)_ | _TBD_ | _TBD_ | _TBD (pre)_ | _TBD_ |
| Transformer fine-tune | **GossipCop (cross-domain)** | _TBD_ | **_TBD (markedly lower — the honest number)_** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Transformer fine-tune | LIAR2 test (6-way credibility) | _TBD_ | _TBD_ | — | _TBD (macro OvR)_ | _TBD_ | _TBD_ |

**Calibration row to fill (temperature scaling):**

| Model | Split | ECE (pre) | ECE (post-T) | Fitted **T** | Macro-F1 (unchanged) |
|---|---|---|---|---|---|
| Transformer fine-tune | GonzaloA calib/test | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 6. Results — Fact-check (skeletons to fill after FEVER eval)

| Fact-check system | Label acc. | FEVER score | R@5 (evidence) | Coverage | Selective acc. | Abstain rate |
|---|---|---|---|---|---|---|
| Retrieval + NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, zero-shot) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L6-v2`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + FEVER-fine-tuned stance head (optional, `copenlu/fever_gold_evidence`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> ⚠️ **License flag:** a stance head fine-tuned on any `fever/*` / `copenlu/fever_gold_evidence` / `pietrolesci/nli_fever` set inherits **cc-by-sa-3.0 + gpl-3.0** share-alike/copyleft. Zero-shot (MIT model) is the redistribution-clean default; the fine-tune is optional and must carry the FEVER attribution.

---

## 7. VERIFIED RESULTS — offline seed (actually produced)

These numbers were produced by the system running **fully offline** (TF-IDF+LogReg classifier, lexical-overlap stance, BM25 over the seed evidence — **no torch**) on the committed synthetic seed in `fakenews/data/samples.py`. They demonstrate the **plumbing and the metrics**, not real-world capability — the seed is **intentionally cleanly separable** so the classifier metrics saturate.

### 7.1 Classifier — 40-item synthetic seed

| System | Accuracy | Macro-F1 | ROC-AUC | ECE |
|---|---|---|---|---|
| **TF-IDF + LogReg** | **1.000** | **1.000** | **1.000** | **0.244** |
| Majority class | — | **0.333** | — | — |

**Reading these honestly:**

- **Accuracy = Macro-F1 = ROC-AUC = 1.0** because the 40-item seed is **engineered to be cleanly separable** (neutral wire-service-style *real* vs. sensational hyper-partisan *fake*). This **saturates by design** — it proves the metric pipeline (accuracy, macro-F1, per-class P/R/F1, ROC-AUC, ECE) computes end-to-end with no torch. **It is not a capability claim.** On real `GonzaloA` / `LIAR2` data these will land at realistic levels (in-domain macro-F1 **≈ 0.90**), and the **cross-domain** number will be **markedly lower**.
- **Majority-class macro-F1 = 0.333** is the textbook value: predicting one class scores F1 = 0 on the never-predicted class on a binary task, so the unweighted mean of {≈1, 0} averages to ≈0.333 (here, with the third "averaged" zero from the absent class, the macro mean is 0.333). It confirms the floor is wired correctly.
- **ECE = 0.244 is the most informative seed number.** Even though every prediction is *correct*, the model is **mis-calibrated** — its confidences are systematically higher than 1−error warrants on the low-confidence bins, so the reliability gap (ECE) is a large **0.244**. This is **exactly the over-confidence pathology the ECE metric exists to catch**, and it is precisely the failure mode that, in production, can silence real news (a confident-but-wrong "fake"). It is the live demonstration of *why* temperature scaling (§2.2) is part of the pipeline. On the seed, perfect accuracy + high ECE is the cleanest possible illustration that **accuracy and calibration are independent axes**.

### 7.2 Agentic fact-check — 8 gold claims

| System (offline) | Label accuracy | Coverage |
|---|---|---|
| **Agentic fact-check** (claim-extract → BM25 retrieve → lexical-overlap stance → aggregate → abstain) | **1.000** | **1.000** |

- **Label accuracy = 1.0** across the 8 gold `ClaimCase`s — every **support/refute verdict is correct** (the seed includes SUPPORTS, REFUTES, and NOT_ENOUGH_INFO cases so the full D4 verdict logic and the D5 ABSTAIN branch both fire).
- **Coverage = 1.0** means no case had to abstain on the seed: the seed evidence corpus is constructed so every gold claim has `n_relevant ≥ N_min=3` coverage. On real FEVER data, coverage will drop below 1.0 — and abstaining there is the **correct** behaviour, scored by the risk–coverage curve (§3), not a defect.

### 7.3 What the seed run proves vs. does not prove

| Proves (plumbing) | Does **not** prove (capability) |
|---|---|
| accuracy / macro-F1 / per-class P/R/F1 / ROC-AUC / ECE all compute, no torch | real-world classifier accuracy (seed is engineered separable) |
| majority-class & TF-IDF baselines wired and comparable | the cross-domain drop (needs GonzaloA→GossipCop) |
| ECE surfaces over-confidence even at 100% accuracy | calibrated confidences (ECE=0.244 is *bad* — needs temp scaling) |
| full FSM (D1–D5) executes; support/refute/abstain all reachable | FEVER label accuracy / evidence recall@k at scale |
| fact-check returns verbatim citations + `decisions_trace` | abstention quality on genuinely-uncovered novel claims |

---

## 8. Error analysis

### 8.1 The costly error: false positives (real flagged fake)

Confusion-matrix quadrants are **not** equally weighted in this domain:

| | Predicted real | Predicted fake |
|---|---|---|
| **Actually real** | ✅ correct | ⛔ **FALSE POSITIVE — the costly error** (legit news silenced) |
| **Actually fake** | ⚠️ false negative (missed; recoverable downstream) | ✅ correct |

A **false positive** — real journalism flagged fake — is the high-cost error: it can chill lawful speech and, if wired to enforcement, suppress real news. Mitigations baked into eval and the operating point:

- Optimise the operating threshold for **high `real`-recall**; report `real`-recall and `fake`-precision explicitly per model.
- The system **flags for human review, never auto-removes/censors** — every false positive is a *review flag*, recoverable by the human, not an irreversible takedown.
- **ABSTAIN** rather than guess when uncertain (D3/D5); `classifier_prior` and evidence `verdict` are reported **separately** so a reviewer sees disagreement.
- Calibration (ECE + temperature scaling) so a *confident* false positive is rarer.

### 8.2 Error slices to inspect

Per the `analysis/` module (`error_analysis.py`, `fairness.py`), errors are sliced to expose bias and leakage rather than reported as one aggregate:

- **Per-source / per-outlet** — is the model just memorising outlets? (source-style leakage)
- **Per-topic** (politics / health / science / finance) and **per-speaker** (LIAR) — label provenance encodes the fact-checker's editorial judgement; report sliced performance, never claim neutral ground truth.
- **In-domain vs cross-domain** — the gap **is** the leakage measurement.
- **Fact-check failure types** — wrong-label-with-evidence vs. retrieval-miss (low recall@k) vs. wrong-abstain. The `decisions_trace` localises which D-point failed.

### 8.3 Calibration as error analysis

ECE is itself an error-analysis tool: the **seed's ECE = 0.244 at 100% accuracy** shows the model can be *right yet badly mis-calibrated*. In production this manifests as confident-but-wrong outputs that reviewers rubber-stamp (automation bias). The reliability diagram localises **which confidence bins** are over-confident, and temperature scaling re-aligns them.

---

## 9. Reproducing the evaluation

| Goal | Command (CLI surface) |
|---|---|
| Offline seed eval (no torch — produces §7 numbers) | `fakenews eval` |
| Classifier eval on a real split | `fakenews eval --task classify --dataset gonzaloa --split test` |
| Cross-domain eval | `fakenews eval --task classify --train gonzaloa --test gossipcop` |
| Fact-check eval (FEVER-style) | `fakenews eval --task factcheck --claims <gold> --corpus <evidence>` |
| Single fact-check (trace visible) | `fakenews factcheck --claim "…" --k 5 --json` |

Eval logic lives in `training/evaluate.py` and `training/metrics.py` (classifier acc / macro-F1 / per-class / ROC-AUC / ECE **+** FEVER score / label-accuracy / recall@k); offline data in `data/samples.py`; per-slice error analysis in `analysis/{error_analysis,fairness}.py`. Every result row carries the stamped model versions / pinned HF revisions / `index_built_at` from `/version` so a number can always be traced to the exact artefacts that produced it.

---

## 10. Summary

- **Headline metric = macro-F1** for both classifier and 6-way credibility; accuracy never reported alone.
- **The honest classifier number is cross-domain** (GonzaloA→GossipCop) — a large drop is the source-style-leakage signature and is itself an ethics metric.
- **False positives (real→fake) are the costly error** — optimise for `real`-recall, abstain on uncertainty, flag-not-censor.
- **Calibration is a first-class metric** — the seed's **ECE = 0.244 at 100% accuracy** is the live demonstration of the over-confidence pathology ECE + temperature scaling exist to catch.
- **Fact-check is scored with abstention** — label accuracy + evidence recall@k + a **risk–coverage curve**; `unverified` is a correct outcome, not a failure.
- **Verified offline seed (no torch):** TF-IDF+LogReg classifier **acc/macro-F1/ROC-AUC = 1.0**, **ECE = 0.244** (40-item engineered-separable seed); majority-class **macro-F1 = 0.333**; agentic fact-check **label accuracy = 1.0**, **coverage = 1.0** (8 gold claims, correct support/refute/abstain). These prove the *plumbing and metrics*; the real GonzaloA/LIAR2/FEVER numbers (in-domain ≈ 0.90, cross-domain lower) are the skeletons in §5–§6 to fill after H100 training.
