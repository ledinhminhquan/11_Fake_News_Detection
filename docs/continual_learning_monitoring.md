# P11 — Continual Learning & Monitoring (Section I.8)

> **Project:** Fake News & Misinformation Detection System — NLP-in-Industry final assignment.
> **Author:** Le Dinh Minh Quan (student 23127460). **Package:** `fakenews`.
> **Scope of this document:** how the deployed system *learns from production* (new gold from reviewer feedback, new fact-checks, evidence-corpus refresh), *retrains safely* (periodic classifier fine-tune, evidence re-index, blue/green `model_version`, temperature recalibration), *detects degradation* (rolling macro-F1 / ECE, online abstain-rate and reviewer-override drift, confidence drift), *what it monitors* (volume, verdict distribution, abstain rate, latency, calibration drift, input drift), and the *drift risks specific to misinformation* (new topics, new outlets that defeat source-style features, stale evidence) with their mitigations.
>
> **One sentence framing for this section:** because the tool **flags content for human review and never auto-removes or auto-censors** (see `docs/risks_limitations_ethics`), the human review queue is not just a UI — it is the **continual-learning data source**: every reviewer decision on a flagged item becomes a gold label that closes the loop.

This document reflects the system **as actually built**. The live monitoring implementation is `src/fakenews/monitoring/drift_report.py` (`monitoring_report(cfg, log_path=None, save=True)`), reading the JSONL request log written by `fakenews/agent/fakenews_agent.py` via `JsonlLogger` at `cfg.serving.request_log_path` (`<run_dir>/request_logs/requests.jsonl`). Versioning is `fakenews/models/model_registry.py`. Offline-first invariant holds throughout: the monitor is **stdlib-only** (no torch / numpy / pandas) so it runs on a CPU-only box and never raises past its entrypoint — a missing/empty/corrupt log returns `{"status": "no_data", ...}`.

---

## 1. Why this project needs continual learning more than most

A fake-news classifier learns *style and source patterns correlated with fakeness*, **not** truth. That correlation is exactly the thing that **rots fastest in production**:

| Force of decay | Concrete failure mode |
|---|---|
| **New misinformation topics** | A novel narrative (new disease, new election, new financial scam) appears that the classifier never saw and the evidence corpus does not cover → the classifier guesses on style, retrieval returns nothing, and the system should *abstain*, not invent a verdict. |
| **New outlets / paraphrase** | Source-style features (the model learning "Reuters dateline ⇒ real") collapse the moment a new outlet appears, or a bad actor paraphrases fake content into wire-service tone. In-domain F1 stays high while real-world accuracy falls. |
| **Stale evidence** | The evidence index is a snapshot. A claim that was *refuted* last year may be *supported* now (or vice-versa); an un-indexed event makes a checkable claim look uncheckable. |
| **Reviewer-distribution shift** | The mix of what humans send for review changes (more health, less politics), shifting the operating point the thresholds were tuned for. |

The two signals the system reports **separately** — `classifier_prior` (P(fake), fast, style-based) and the evidence-grounded `verdict` (`real` / `fake` / `unverified`) — also *drift independently*: the classifier drifts on **input/style**, the evidence layer drifts on **corpus staleness**. The monitoring design tracks both.

---

## 2. New-data sources (the feedback loop)

All three streams normalise to the repo's canonical data model (`fakenews/data` — `NewsItem`, `Evidence`, `ClaimCase`) and the **internal label convention `0 = real, 1 = fake`** (`p_fake = P(label==1)`; see `config.py` docstring and `training/metrics.py`). Every loader applies an explicit per-source `label_map` because mirror polarities disagree (`GonzaloA` 0=fake; `LittleFish-Coder/*` 0=real; `mrm8488` 1=fake) — the same discipline applies to feedback ingestion.

### 2.1 Reviewer feedback on flagged items = **gold** (primary source)

This is the highest-value stream and it is *free*: it is the by-product of the product working as designed.

```
/factcheck or /classify  →  flag + evidence + decisions_trace  →  human reviewer
        │                                                              │
        │  request log (requests.jsonl: decision events)               │ reviewer verdict
        ▼                                                              ▼
   monitoring/drift_report.py                              reviewer_feedback.jsonl  (NEW gold)
        │                                                              │
        └──────────────────  retraining trigger  ◄─────────────────────┘
```

Each reviewed item yields a record (proposed schema for `reviewer_feedback.jsonl`, keyed by the same `request_id`/claim hash the agent logs):

```json
{
  "ts": "2026-06-26T09:00:00Z",
  "request_id": "...",                      // join key back to the decision event
  "claim": "The WHO declared ...",
  "title": null,
  "system_verdict": "fake",                 // what the agent presented
  "classifier_prior": 0.88,                 // P(fake) at decision time
  "system_confidence": 0.91,
  "abstained": false,
  "reviewer_label": "fake",                 // GOLD: real | fake | unverified
  "reviewer_agreed": true,                  // == override flag (false = override)
  "reviewer_evidence_ids": ["ev-204", ...], // evidence the human actually used
  "reviewer_notes": "WHO never recommended bleach; 3 fact-checks refute.",
  "model_version": "distilbert-base-uncased-20260626T0900Z"
}
```

What each field feeds:
- `reviewer_label` → a **confirmed binary label** for classifier fine-tune (collapsed `0=real/1=fake`) and, when the reviewer used evidence, a **claim+evidence verdict pair** (SUPPORTS / REFUTES / NOT_ENOUGH_INFO) for an optional stance fine-tune.
- `reviewer_agreed == false` → an **override**, the single most important degradation signal (§5.2).
- `reviewer_evidence_ids` → which evidence passages were decisive → **evidence-corpus quality signal** (frequently-cited passages are kept hot; never-retrieved-but-relevant passages reveal a retrieval gap).
- `abstained` + a later reviewer label → tells us whether an abstention was *correct caution* (reviewer also could not decide → `unverified`) or a *miss* (reviewer found clear evidence → corpus/retrieval gap).

> **Label-quality guardrails.** Reviewer labels are themselves editorial judgements (LIAR/PolitiFact-style bias, see ethics doc). We therefore (a) require ≥2 reviewers or an adjudication step before a feedback row is promoted to training gold; (b) keep `reviewer_notes` + `reviewer_evidence_ids` for provenance/audit; (c) hold out a **frozen, human-curated gold eval set** that is *never* trained on, so the feedback loop cannot silently move the goalposts.

### 2.2 New fact-checks (curated external gold)

Published fact-checks (FEVER-style claim/evidence/verdict triples, or our own analysts' write-ups) extend the `ClaimCase` set and the evidence corpus. These slot directly into the FEVER-style eval (`training/metrics.py::factcheck_metrics`: accuracy, abstain_rate, selective_accuracy, coverage) and the optional stance fine-tune (`copenlu/fever_gold_evidence` / `pietrolesci/nli_fever` formats). **License note:** FEVER-family sources are **cc-by-sa-3.0 + gpl-3.0 (copyleft)** — any stance model fine-tuned on them inherits share-alike; this must be flagged on redistribution. Reviewer-generated fact-checks (§2.1) are first-party and license-clean.

### 2.3 Evidence-corpus refresh (the retrieval substrate)

The evidence index (BM25 + dense `all-MiniLM-L6-v2`, fused by RRF, optional `ms-marco-MiniLM-L6-v2` rerank) is a **timestamped snapshot** (`/version` exposes `index_built_at`). It is refreshed from: new fact-check articles and reviewer-cited passages (§2.1–2.2), LIAR2 `justification` rows, and periodic re-crawls of trusted source domains. Refresh is an **append + re-embed + re-index**, kept *separate* from classifier retraining so a corpus update can ship on its own cadence (faster than model retraining) and is independently versioned.

---

## 3. Retraining strategy

Two artefacts retrain on **decoupled cadences** because they decay for different reasons:

| Artefact | What retrains | Cadence (default) | Trigger |
|---|---|---|---|
| **Classifier** (`models/classifier.py` transformer fine-tune; `models/baseline_tfidf.py` floor) | Binary `0=real/1=fake` head on confirmed labels | Periodic (e.g. monthly) **or** on degradation alert | rolling macro-F1 drop, confidence/ECE drift, accumulated reviewer overrides |
| **Evidence index** (`factcheck/retriever.py`, BM25 + dense) | Re-embed + re-index appended evidence | More frequent (e.g. weekly) | new fact-checks, retrieval-gap signal, rising `unverified` rate |
| **Stance/NLI** (`factcheck/stance.py`, zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) | **Optional**; default is zero-shot (no retrain) | Rare | only if a reviewer-gold stance set shows a consistent gap |
| **Temperature** (calibration scalar) | Refit on a held-out calibration split | Every retrain + on `confidence_drop` alert | ECE regression (§5.1) |

### 3.1 Periodic classifier fine-tune on confirmed labels

The retraining job reuses the existing training stack unchanged (`training/train_classifier.py`, `training/train_baseline.py`, `training/evaluate.py`) with the same `ClassifierConfig` (`base_model="distilbert-base-uncased"` default / `microsoft/deberta-v3-base` / `answerdotai/ModernBERT-base`; `num_labels=2`; `num_train_epochs=3`; `learning_rate=2e-5`; `use_class_weights=True`; auto bf16/fp16 by GPU).

Pipeline:
1. **Assemble training pool** = original training data **+** promoted reviewer-gold (§2.1, ≥2-reviewer agreed) **+** new external fact-checks (§2.2). Normalise every source to `0=real/1=fake` via its `label_map`.
2. **Dedup** (exact + near-dup / normalized-hash) within and across the pool *and* against the frozen eval set, to prevent the feedback loop from leaking test rows into train.
3. **Re-apply the boilerplate strip** (datelines `(Reuters)`, `WASHINGTON —`, bylines) — the §"anti-leakage" discipline from the design brief — so the refreshed model does not re-learn source style.
4. **Fine-tune** (≤3 epochs, `EarlyStoppingCallback(patience=2)`; `metric_for_best_model="f1"` = macro-F1).
5. **Always retrain the TF-IDF + LogReg baseline too** — it is the no-torch floor and the offline-fallback path; a candidate that cannot beat the refreshed baseline on cross-domain is rejected.
6. **Evaluate the candidate** on the frozen in-domain eval **and** the cross-domain eval (train-domain → GossipCop-style) — the cross-domain macro-F1 gap is the honest number and an *acceptance gate*.

### 3.2 Evidence re-index

`factcheck/retriever.py` rebuilds BM25 + dense embeddings over the appended corpus, writes a new index dir, and stamps a fresh `index_built_at`. Because retrieval is gated behind `/factcheck` only and the index is read-only/memory-mapped, a new index is swapped in atomically (build beside, flip the pointer) with the same blue/green discipline as the model (§3.4). **Recall@k is re-measured against `BeIR/fever-qrels`** before promotion so a re-index can never silently reduce coverage.

### 3.3 Temperature recalibration

Over-confidence is the dangerous error here (it can silence real news). After every classifier retrain — and whenever monitoring flags `confidence_drop` — we **refit a single temperature scalar** on a held-out calibration split (logits ÷ T, T chosen to minimise NLL), then **report pre/post ECE** with a reliability diagram (`training/metrics.py::classification_metrics` returns `ece` and `roc_auc` from `P(fake)`; the helper `_ece` is a 10-bin Expected Calibration Error of P(fake)). Calibration is cheap, decoupled from the heavy fine-tune, and ships independently. The recalibrated probability is what `/classify` returns and what feeds the agent's `classifier_prior` (D2 skip-gate `skip_factcheck_confidence=0.95`, D4 `prior_weight=0.3`), so miscalibration would otherwise corrupt the agent's gates.

### 3.4 Blue/green `model_version`

Versioning is `models/model_registry.py`. A candidate is built as a **new version**, evaluated, and only then promoted by flipping the `latest` pointer — the previous version stays on disk for instant rollback.

- `make_version(base_model)` → e.g. `distilbert-base-uncased-20260626T0900Z` (artefact id, monotonic by UTC stamp).
- `write_metadata(model_path, version=, base_model=, dataset_signature=, metrics=, extra=)` → `model_meta.json` (records the training-pool signature + eval metrics so every version is auditable and reproducible).
- `update_latest_pointer(output_dir, model_path)` → atomic **blue→green flip** (symlink `latest`, or a `LATEST` marker file where symlinks are unavailable — the Windows/Colab-safe fallback).
- `resolve_latest(output_dir)` → what serving loads; **rollback = re-point `latest` at the prior version** (no rebuild).

**Promotion gate (candidate must pass *all* before the flip):**
1. macro-F1 ≥ current production on the **frozen in-domain eval** (no regression).
2. **Cross-domain** macro-F1 ≥ current − ε (style-leakage did not worsen) — this is the honest gate.
3. post-calibration **ECE ≤ current** (calibration did not regress).
4. `real`-recall ≥ current (we did not start silencing more legitimate news — the high-cost error).
5. FEVER-style label accuracy + recall@k not worse (if the stance/index changed).

Serving stamps the active `classifier_version`, pinned `nli_model`, `retriever`, and `index_built_at` into **every `/classify` and `/factcheck` response** and at `/version`, with HF **revisions/SHAs pinned** so a Hub update cannot silently change a verdict (`serving.model_version` in `config.py`, surfaced by `agent.classify(...) → {"model_version": ...}`). Pre-promotion, a **shadow / canary** phase runs the green model on a traffic slice and compares its verdict distribution and abstain rate against blue (using the same `drift_report.py` windows) — a divergence blocks the flip.

---

## 4. Monitoring metrics (what `drift_report.py` actually computes)

`monitoring_report(cfg, log_path=None, save=True)` reads the append-only **request log** (`cfg.serving.request_log_path` = `<run_dir>/request_logs/requests.jsonl`). The agent (`fakenews_agent.py`) logs one `decision` event per served request:

```json
{"ts": "...", "event": "decision", "mode": "auto|classify|factcheck",
 "verdict": "real|fake|unverified", "clf_label": "fake|real",
 "abstained": true|false,
 "metrics": {"latency_ms": ..., "confidence": ..., "brain_used": ...}}
```

The report aggregates the full log (`overall` via `_window_stats`) and splits it into an earlier **baseline** window vs a **recent** window (`_drift`, needs ≥6 events). It writes `<run_dir>/monitoring/latest.json` + `monitor-<stamp>.json` and returns a dict (never raises; empty log → `status:"no_data"`).

| Monitored signal | Field(s) in the report | Meaning / why it matters |
|---|---|---|
| **Request volume** | `request_volume` / `n_requests` / `n_events` | Throughput; a volume spike often precedes drift. |
| **Verdict distribution** | `overall.verdict_distribution`, `fake_rate`, `real_rate`, `unverified_rate` | The product's actual output mix. A jump in `unverified` = evidence coverage thinning. |
| **Abstain rate** | `overall.abstain_rate` (from `abstained`) | First-class outcome; the D5 confidence/coverage gate firing. Rising abstain = the system is *correctly* refusing to guess on shifting input — investigate the cause. |
| **Classifier-label distribution (input proxy)** | `clf_label_distribution`, `clf_fake_rate`, `clf_real_rate` | The classifier's fake-vs-real call is the proxy for the **incoming claim mix** — input drift. |
| **Latency** | `mean_latency_ms`, `p95_latency_ms` | Health/cost. p95 dominated by the gated retrieval + k stance forward passes. |
| **Calibration / confidence drift** | `mean_confidence`, `n_confidence` (from `metrics.confidence`) | Mean verdict confidence; a *drop* vs baseline is a calibration-drift tell (`_confidence_of` reads `metrics.confidence` or top-level `confidence`; degrades to unavailable if neither is logged). |
| **Mode mix** | `mode_distribution` (auto / classify / factcheck) | How traffic splits across the fast classifier-only path vs the heavier fact-check path. |

The report also returns plain-English `recommendations` (`_recommendations`) — e.g. high abstain → lower `agent.min_verdict_confidence`, raise `retrieval.top_k`, or expand the corpus; high `unverified` → check the index and `agent.min_evidence` / `min_evidence_relevance`; low mean confidence → re-check `stance.support_threshold`/`refute_threshold` and `agent.prior_weight`.

---

## 5. Degradation detection

### 5.1 Offline rolling macro-F1 / ECE (on held-out / frozen eval)

On a schedule (and pre-promotion), re-run `training/evaluate.py` against the **frozen gold eval set** (never trained on) and a fresh slice of recently-confirmed reviewer gold:
- **macro-F1** (headline; `classification_metrics.macro_f1`) — a rolling drop vs the version's at-promotion baseline = real degradation.
- **per-class** `real`-recall and `fake`-precision — guards the high-cost "legit news flagged fake" error specifically.
- **ECE** (`classification_metrics.ece`) — a rising ECE = over-confidence creeping back → trigger temperature recalibration (§3.3) even without a full retrain.
- **Cross-domain macro-F1** on a held-out outlet/domain — a widening in-domain↔cross-domain gap is the **source-style-leakage** alarm (§6).
- **Fact-check** (`factcheck_metrics`): label accuracy, `selective_accuracy`, `coverage`, `abstain_rate`, and recall@k vs `BeIR/fever-qrels`.

### 5.2 Online abstain-rate + reviewer-override drift

From production, *without* needing fresh gold:
- **Abstain-rate drift** — `drift.delta_abstain_rate`; flag **`rising_abstain_rate`** when it climbs > **0.10** between windows. The system loudly refusing to decide more often is an early, label-free degradation signal.
- **Reviewer-override drift** — the rate of `reviewer_agreed == false` in `reviewer_feedback.jsonl` (§2.1). A rising override rate is the **ground-truth** degradation signal: humans are disagreeing with the system more. (Join key = `request_id`; this is the one signal that needs the feedback file, not just the request log.) A sustained override-rate climb is the strongest single trigger for an off-cycle retrain.
- **Verdict-distribution shift** — flag **`verdict_distribution_shift`** when `|delta_unverified_rate| > 0.15`; **`fake_verdict_shift`** when `|delta_fake_rate| > 0.20`.

### 5.3 Confidence / calibration drift (online)

`drift.delta_mean_confidence` < **−0.10** → flag **`confidence_drop`**. The recommendation explicitly says: *"a possible calibration drift — re-run training/evaluate and inspect the model ECE,"* tying the cheap online signal (mean confidence falling) to the offline check (ECE) and the fix (recalibrate, §3.3). This catches the dangerous failure — the model getting quietly *less* sure — before it shows up as wrong-but-confident outputs.

### Drift flags emitted by `_drift` (exact thresholds, as implemented)

| Flag | Condition (recent − baseline) | Reading |
|---|---|---|
| `rising_abstain_rate` | `delta_abstain_rate > 0.10` | More refusals → coverage/input shift |
| `verdict_distribution_shift` | `|delta_unverified_rate| > 0.15` | Output mix moving |
| `input_label_shift` | `|delta_clf_fake_rate| > 0.20` | **Input drift** — incoming claim mix changed |
| `fake_verdict_shift` | `|delta_fake_rate| > 0.20` | Fake-verdict share moving |
| `latency_regression` | `delta_mean_latency_ms / base > 0.50` | >50% slower → infra/cost regression |
| `confidence_drop` | `delta_mean_confidence < −0.10` | **Calibration drift** — model less sure |

`drift.alert = bool(flags)`; any flag fires the operator playbook (§7).

---

## 6. Drift risks specific to misinformation + mitigations

| Drift risk | How it shows in monitoring | Mitigation (where in the system) |
|---|---|---|
| **New misinformation topics** (novel narrative the classifier never saw, corpus does not cover) | `rising_abstain_rate`, rising `unverified_rate`, retrieval recall@k drop offline | **Abstain, never guess** (D3 coverage gate `min_evidence`/`min_evidence_relevance`, D5 `min_verdict_confidence`). Fast-track the topic into the evidence corpus (§2.3 weekly refresh) and into reviewer-gold; the corpus refreshes faster than the model. |
| **New outlets / paraphrase defeating source-style features** | High *in-domain* F1 but widening **cross-domain** gap (§5.1); reviewer-override rate up while abstain stays flat (confidently wrong) | **Lean on the evidence-grounded verdict, treat `classifier_prior` as a weak prior only** (`agent.prior_weight=0.3`). Strip source/style boilerplate on every retrain (§3.1). Track the cross-domain gap as a first-class metric; keep an adversarial/paraphrase eval slice. |
| **Stale evidence** (index snapshot out of date; a claim's truth value moved, or an event is un-indexed) | rising `unverified_rate`, retrieval recall@k drop, abstentions that reviewers *could* resolve with new evidence | **Version + timestamp the index** (`index_built_at` in `/version`); decouple index refresh (weekly) from model retrain; abstain when coverage is insufficient rather than serve a stale verdict. Re-measure recall@k vs `BeIR/fever-qrels` before promoting a re-index. |
| **Reviewer-distribution shift** (the human queue's topic mix changes) | `input_label_shift`, `verdict_distribution_shift`, `mode_distribution` move | Re-evaluate on a fresh slice; re-tune thresholds against the new operating point; refresh fine-tuning data toward the new mix. |
| **Feedback-loop bias / self-reinforcement** (training only on items the system flagged) | Slow, silent: eval metrics drift apart from production behaviour | Frozen human-curated eval set that is **never** trained on; require ≥2-reviewer agreement before promotion; sample some non-flagged traffic for review so the loop sees what the system *missed*, not only what it caught. |
| **Latency / cost regression** (heavier index, larger stance model) | `latency_regression`, p95 climbing | Gate the heavy path behind `/factcheck`, cache by claim hash, batch the k stance passes; degrade to `"unverified — capacity"` under load rather than time out. |

> **Cross-cutting ethics tie-in.** Every mitigation above defaults to **abstain + flag for human review**, never auto-action. Drift is handled by *surfacing uncertainty to humans and refreshing data*, not by quietly lowering thresholds to keep the verdict rate up — lowering a gate to suppress a rising abstain rate would trade an honest "I don't know" for a confident guess, which is precisely the failure this project is built to avoid.

---

## 7. Operator playbook (closing the loop)

```
serve  ──►  request_logs/requests.jsonl   (decision events)
              │
              ▼
   monitoring/drift_report.py  →  runs/monitoring/latest.json   (volume, verdicts, abstain,
              │                                                    latency, confidence, drift flags)
              ├── alert? ──► investigate flag ──► §6 mitigation
              │                                     │
   reviewer_feedback.jsonl (gold + overrides)       ├─ corpus gap   → evidence re-index (§3.2, weekly)
              │                                     ├─ calibration  → temperature recalibration (§3.3)
              └────────────► retraining pool ───────┴─ model decay  → blue/green classifier retrain (§3.1, §3.4)
                                                          │
                                                          ▼
                                          candidate passes promotion gate? ──► flip `latest` (green)
                                                          │  no → keep blue, iterate
                                                          ▼  yes
                                          new model_version stamped into /version + every response
```

1. **Watch** — schedule `monitoring_report` (CLI/autoreport); inspect `latest.json` `drift.flags` + `recommendations`.
2. **Diagnose** — map each flag to §6 (input vs output vs latency vs calibration drift).
3. **Cheapest fix first** — recalibrate temperature (hours) or re-index evidence (decoupled, weekly) before a full fine-tune.
4. **Retrain when warranted** — degradation alert or accumulated reviewer overrides → assemble the pool (§3.1), build a **new version**, run the **promotion gate** (§3.4).
5. **Promote blue→green** — flip `latest` only on a clean gate; keep the prior version for one-flip rollback.
6. **Verify** — confirm `/version` reflects the new `classifier_version` / `index_built_at`, and watch the next monitoring window (canary) for a regression before fully committing.

**Key file/function references:** `fakenews/monitoring/drift_report.py::monitoring_report` · `fakenews/models/model_registry.py::{make_version, write_metadata, update_latest_pointer, resolve_latest}` · `fakenews/training/{train_classifier,train_baseline,evaluate}.py` · `fakenews/training/metrics.py::{classification_metrics (macro_f1, ece, roc_auc), factcheck_metrics}` · `fakenews/factcheck/retriever.py` (re-index) · `fakenews/config.py` (`ServingConfig.request_log_path`, `AgentConfig` D1–D5 thresholds, `StanceConfig.support_threshold/refute_threshold`).
