# Model Card — P11 Fake-News Classifier (+ Stance/NLI verifier)

> **Project:** P11 — Fake News & Misinformation Detection System
> **Course:** NLP in Industry — final assignment
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Package:** `fakenews` (`src/fakenews/`)
> **Card scope:** the **fine-tuned binary fake-news classifier** (the fast *prior*) and the **pretrained stance/NLI model** it is paired with inside the agentic fact-check. This card documents two models that play different roles; read both halves.

---

## 0. TL;DR

This card covers **two** models that together form the P11 credibility system:

1. **Fake-news classifier** — a transformer fine-tuned for **binary** `real` vs `fake` text classification. It produces a fast **prior** `P(fake)` over an article or short claim. It detects *style/source patterns correlated with fakeness*, **not** falsehood.
2. **Stance/NLI verifier** — the **pretrained, zero-shot** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (not fine-tuned by default), used inside the agentic fact-check to score each retrieved evidence passage as `support` / `refute` / `neutral`.

The system **flags content for human review and shows its evidence; it never auto-removes, auto-blocks, or auto-censors.** The classifier `prior` and the evidence-grounded `verdict` are reported **separately** so a reviewer can see when they disagree, and `unverified` (abstain) is a **first-class outcome**, not a failure mode.

**Internal label convention (repo-wide): `0 = real`, `1 = fake`, so `p_fake = P(label == 1)`.**

---

## 1. Model details

### 1a. Fake-news classifier (the trainable component)

| Field | Value |
|---|---|
| **Task** | Single-label text classification (`problem_type="single_label_classification"`) |
| **Classes** | `num_labels = 2`; `id2label = {0: "real", 1: "fake"}`, `label2id = {"real": 0, "fake": 1}` |
| **Output** | Calibrated `P(fake)` = softmax probability of class `1`; returned as `p_fake` |
| **Primary base** | `answerdotai/ModernBERT-base` — 149.7M params, Apache-2.0, 8192-token context (handles full articles) |
| **T4 fallback base** | `distilbert-base-uncased` — 67M params, Apache-2.0, max 512 tokens |
| **Middle option** | `FacebookAI/roberta-base` — 124.7M params, MIT, max 512 tokens |
| **No-torch floor** | `models/baseline_tfidf.py` — TF-IDF (1–2 grams) + LogisticRegression (`class_weight="balanced"`), scikit-learn only, no PyTorch |
| **Input** | `title [SEP] text` (concatenated; `text_fields = ["title", "text"]`) |
| **`max_length`** | 512 default (safe/cheap); ModernBERT may go 1024–8192 on A100/H100 for long docs |
| **Secondary config** | 6-way LIAR-style credibility head (`num_labels = 6`) trained on `chengxuphd/liar2` `statement` field for the "credibility scale" deliverable |
| **Code module** | `fakenews/models/classifier.py` (transformer, lazy `torch`); `fakenews/models/baseline_tfidf.py` (floor) |
| **Code license** | MIT (this repository) |

> **Label-polarity warning.** The repo standardizes on **`0 = real, 1 = fake`**. Several source datasets use the *opposite* convention (e.g. `GonzaloA/fake_news` is natively `0 = fake, 1 = real`). Every loader in `data/news_loaders.py` MUST apply an explicit per-source `label_map` to the repo convention and verify on a known row. Never assume polarity. See §3.

### 1b. Stance / NLI verifier (pretrained, not fine-tuned by default)

| Field | Value |
|---|---|
| **HF id** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| **Status** | VERIFIED via `hub_repo_details` (user `ledinhminhquan`) |
| **Params / license** | 184.4M / **MIT** |
| **Training mix** | MNLI + **FEVER** + ANLI (the FEVER component maps cleanly onto fact-verification) |
| **Native labels** | `entailment` / `neutral` / `contradiction`; verified `id2label = {0: entailment, 1: neutral, 2: contradiction}` |
| **Usage** | Zero-shot NLI: **premise = retrieved evidence passage, hypothesis = the claim** |
| **Label bridge** | `entailment → SUPPORTS (support)`, `contradiction → REFUTES (refute)`, `neutral → NOT_ENOUGH_INFO (neutral)` |
| **Code module** | `fakenews/factcheck/stance.py` (lazy `transformers`; lexical mock-stance fallback when absent) |
| **Scale-up** | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435.1M, MIT) on A100/H100 |
| **Baseline / fallback NLI** | `facebook/bart-large-mnli` (407.3M, MIT) |

> **Critical engineering invariant.** The NLI label order is read from `model.config.id2label` **at load time** — never hardcoded. The verified order for this model is `{0: entailment, 1: neutral, 2: contradiction}`, but a swapped model must not silently invert verdicts.

> **Default = zero-shot.** Net gain from fine-tuning the stance head is usually small, so zero-shot is the default. An **optional** fine-tune (`training/train_stance.py`) trains `microsoft/deberta-v3-base` on `copenlu/fever_gold_evidence` (228K S/R/NEI pairs) or `pietrolesci/nli_fever` (drop-in `premise/hypothesis/label`). **Any released fine-tuned stance model inherits FEVER copyleft** (cc-by-sa-3.0 + gpl-3.0) — flag for redistribution.

### How the two models combine

```
article/claim ─► CLASSIFIER (prior p_fake) ─────────────────┐
                                                            ├─► AGGREGATE (D4): score = α·stance_vote + (1−α)·prior, α=0.6
claim ─► RETRIEVE evidence ─► STANCE/NLI (per evidence) ─────┘     ─► verdict {REAL | FAKE | UNVERIFIED}  ─► DECIDE/ABSTAIN (D5)
```

**Evidence dominates; the classifier prior is a soft nudge** (`α = 0.6` on the stance vote). The two signals are surfaced separately in the API response as `classifier_prior` vs `verdict`.

---

## 2. Intended use & out-of-scope use

### Intended use (decision support, flag-for-review)

| User | Use |
|---|---|
| Journalists / professional fact-checkers | Get a `real`/`fake`/`unverified` verdict **with cited evidence and per-evidence stance**, plus an auditable `decisions_trace` to verify line-by-line. |
| Platform trust-&-safety / moderators | Use the fast classifier `prior` to **triage** a firehose and route only suspicious items to human review. Never an auto-takedown. |
| Newsroom editors / researchers | Use the optional 6-way credibility score (LIAR2) for nuance beyond binary. |
| End readers (secondary) | See a transparent credibility indicator **with the evidence shown**, to encourage verification rather than blind trust. |

**The system ASSISTS humans. Output is always a *review flag + evidence*, never an enforcement action.**

### Out-of-scope use (explicitly prohibited / unsupported)

- **NOT an automated arbiter of truth.** A `fake` label means "style/source patterns correlated with fakeness" + (for the verdict) "the indexed evidence refutes it" — **not** "this statement is false in the world."
- **NOT for auto-censorship, auto-removal, auto-blocking, or auto-takedown.** There is no enforcement action anywhere in the design. Verdicts are advisory and require human sign-off before any action.
- **NOT a neutral ground truth.** LIAR/PolitiFact labels encode an editorial judgement; outputs must never be presented as objective fact.
- **NOT a generalizable detector across languages or domains.** Training data is English-centric and US-politics / celebrity-skewed (PolitiFact / GossipCop). Cross-domain performance drops sharply (§5).
- **NOT a standalone classifier verdict.** The classifier alone must not be used to label content; the evidence-grounded layer + abstain exist precisely to compensate for style/source leakage.
- **NOT an evasion / adversarial tool.** No "detection-evading score" is shipped; adversarial tooling is gated, rate-limited, and logged.

---

## 3. Training data

### Classifier training datasets (binary `real`/`fake`)

| HF id | Status | License | Size / splits | Native polarity | Role |
|---|---|---|---|---|---|
| **`GonzaloA/fake_news`** | VERIFIED | **unknown** ⚠️ flag | 40.6K (train 24.4K / val 8.1K / test 8.1K) | **0 = fake, 1 = real** → remap | **PRIMARY** — clean `title`+`text`, pre-split, parquet (loads torch-free) |
| `ErfanMoosaviMonazzah/fake-news-detection-dataset-English` | VERIFIED | **openrail** | 44.3K (30.0K/6.0K/8.3K) | 0 = fake, 1 = real → remap | ISOT-style; most license-clean large binary mirror |
| `mohammadjavadpirhadi/fake-news-detection-dataset-english` | VERIFIED | **MIT** ✅ | 10K–100K | binary | **MIT-clean alt** — prefer if license hygiene is paramount |
| `davanstrien/WELFake` | VERIFIED | **unknown** ⚠️ flag (source ≈ CC-BY 4.0, verify) | 72.1K (35K real / 37K fake) | 0 = fake, 1 = real → remap | Larger alt (4-corpus merge); make own split + dedup |
| **`chengxuphd/liar2`** | VERIFIED | **apache-2.0** ✅ | 23.0K (18.4K/2.3K/2.3K) | 6-way `label ∈ 0–5` | **SECONDARY** 6-way credibility; `justification` = free evidence |
| `LittleFish-Coder/Fake_News_GossipCop` | VERIFIED | **apache-2.0** ✅ | 12.7K (9,988/2,672) | **0 = real, 1 = fake** | **Cross-domain eval** target (do not train on) |
| `LittleFish-Coder/Fake_News_PolitiFact` | VERIFIED | **apache-2.0** ✅ | 483 (381/102) | **0 = real, 1 = fake** | FakeNewsNet-PolitiFact content mirror |
| `mrm8488/fake-news` | VERIFIED | not declared ⚠️ | 44.9K single train | **1 = fake, 0 = real** (already repo polarity) | aux/eval only; no title, no splits |

### Stance / verdict datasets (optional fine-tune + evidence-recall eval)

| HF id | Status | License | Role |
|---|---|---|---|
| `copenlu/fever_gold_evidence` | VERIFIED | cc-by-sa-3.0 + gpl-3.0 ⚠️ copyleft | Cleanest single claim+gold-evidence S/R/NEI set for an optional verdict-head fine-tune |
| `pietrolesci/nli_fever` | VERIFIED | (FEVER cc-by-sa-3.0 + gpl-3.0) ⚠️ | FEVER reframed as NLI (`premise/hypothesis/label`) — drop-in stance fine-tune |
| `tals/vitaminc` | VERIFIED | cc-by-sa-3.0 ⚠️ | Robustness to subtle edits (optional add) |
| `BeIR/fever` + `BeIR/fever-qrels` | VERIFIED | cc-by-sa-4.0 ⚠️ | Measure **evidence recall@k** |
| `fever/fever` | VERIFIED | cc-by-sa-3.0 + gpl-3.0 ⚠️ | SUPPORTS/REFUTES/NEI verification + evidence corpus |
| `ImperialCollegeLondon/health_fact` (PUBHEALTH) | VERIFIED | **mit** ✅ | Health-domain 4-way fact-checking |

### ⚠️ License flags (read before redistribution)

- **Unknown / unstated — research/eval only:** `GonzaloA/fake_news` (the **primary** classifier set), `davanstrien/WELFake`, `mrm8488/fake-news`, plus generic-`cc` mirrors. For commercial/redistribution-clean training, prefer `mohammadjavadpirhadi/...` (MIT) or `ErfanMoosaviMonazzah/...` (openrail).
- **Copyleft / share-alike (attribution + share-alike; GPL on code portions):** all `fever/*`, `copenlu/fever_gold_evidence`, `pietrolesci/nli_fever`, `tals/vitaminc`, `BeIR/*`. **A stance head fine-tuned on these inherits share-alike** — flag any released artifact.
- **Non-commercial — do NOT ship commercially:** `allenai/scifact` (cc-by-nc-2.0). Use `BeIR/scifact` (cc-by-sa-4.0) instead.
- **FakeNewsNet raw** (`KaiDMML/FakeNewsNet`): a crawler, not a packaged dataset — tweet IDs + URLs only; article bodies need re-crawling (Twitter ToS + publisher copyright; ASU academic-use). **Network-gated, excluded from CI.** Use the Apache-2.0 `LittleFish-Coder/*` mirrors.
- **Cleanly permissive (prefer):** `chengxuphd/liar2`, `mohammadjavadpirhadi/...`, `ImperialCollegeLondon/health_fact`, `LittleFish-Coder/*`, and **every recommended model** (all MoritzLaurer DeBERTa NLI, `bart-large-mnli`, `microsoft/deberta-v3-*`, `BAAI/bge-small-en-v1.5`, MiniLM) — all MIT/Apache.

### Preprocessing (anti-leakage — critical for fake news)

1. **Strip source/style boilerplate** before training: datelines and source markers (`(Reuters)`, `WASHINGTON —`), bylines. In ISOT/WELFake/GonzaloA, all "real" rows are wire-service copy and all "fake" rows are blog-rant style — the model otherwise learns *"is this Reuters formatting"*, not veracity.
2. **Per-source `label_map` normalization** to repo convention `0 = real, 1 = fake`, verified on a known row.
3. **Dedup** exact + near-duplicate (MinHash / normalized-hash) within and across splits (WELFake and GonzaloA overlap in sources).
4. **LIAR/LIAR2 metadata is dropped** from features: `speaker`, `state_info`, `subject`, and per-speaker `*_counts` columns encode the label distribution. Train on `statement` (+ optionally `context`) only.
5. **Binary collapse for the LIAR family:** `{pants-fire, false, barely-true} → FAKE(1)`, `{half-true, mostly-true, true} → REAL(0)`.

### Offline fallback corpus

`fakenews/data/samples.py` ships a fully synthetic, license-safe corpus so the **whole pipeline runs with only numpy/pandas/scikit-learn, no network and no torch**: `SAMPLE_NEWS` (~36 balanced labeled items), `SAMPLE_EVIDENCE` (~16 mixed snippets), `SAMPLE_CLAIMS` (~6 `ClaimCase` with gold FEVER verdicts incl. ≥1 SUPPORTS / REFUTES / NEI). Plus `tests/fixtures/fake_news_tiny.csv` (~10 rows, repo convention `0 = real, 1 = fake`).

---

## 4. Training procedure & hyperparameters

### Classifier — HF `Trainer` configuration

```python
model_config = {
    "model_name": "answerdotai/ModernBERT-base",   # T4 -> "distilbert-base-uncased"
    "num_labels": 2,                                # 6 for liar2 credibility
    "problem_type": "single_label_classification",
    "id2label": {0: "real", 1: "fake"},            # REPO convention (0=real, 1=fake)
    "label2id": {"real": 0, "fake": 1},            # GonzaloA native is 0=fake/1=real -> remap on load!
    "max_length": 512,                              # ModernBERT can go 1024-8192
    "text_fields": ["title", "text"],              # concat title [SEP] text
}

training_args = {            # transformers.TrainingArguments
    "output_dir": "outputs/fakenews_cls",
    "num_train_epochs": 3,                 # few epochs — these sets overfit fast
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,     # see GPU table
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "bf16": True,                          # A100/H100/L4; set fp16=True on T4 instead
    "fp16": False,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",         # macro-F1 (compute_metrics returns "f1")
    "greater_is_better": True,
    "save_total_limit": 2,
    "seed": 42,
    "report_to": "none",
    "dataloader_num_workers": 4,
    "group_by_length": True,
}
# EarlyStoppingCallback(early_stopping_patience=2) on metric_for_best_model
```

`compute_metrics` returns `{"f1": macro-F1, "accuracy": ...}`. For the LIAR2 6-way config, macro-F1 is computed over labels 0–5.

### Class weights for imbalance

WELFake / LIAR2 are imbalanced (GonzaloA is near-balanced). A `WeightedTrainer` overrides `compute_loss` with `nn.CrossEntropyLoss(weight=class_weights)` where `class_weights = n / (k · bincount)`, normalized:

```python
class_weights = torch.tensor([w_real, w_fake])  # = n/(k*bincount), normalized
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights.to(out.logits.device))(out.logits, labels)
        return (loss, out) if return_outputs else loss
```

### GPU profiles (auto-adapt; ModernBERT-base @ seq 512)

| GPU | Precision | train batch | eval batch | grad_accum | max_length | Notes |
|---|---|---|---|---|---|---|
| **H100 80GB** | bf16 | 64 | 128 | 1 | 1024–2048 | push long context |
| **A100 40/80GB** | bf16 | 32 | 64 | 1 | 512–1024 | primary target |
| **L4 24GB** | bf16 | 16 | 32 | 2 | 512 | bf16 ok on Ada |
| **T4 16GB** | **fp16** | 8 | 16 | 4 | 384–512 | base → `distilbert-base-uncased`; `fp16=True, bf16=False` |

Detection: `torch.cuda.get_device_capability()` (≥ 8.0 → bf16; T4 is 7.5 → fp16) and `torch.cuda.get_device_properties(0).total_memory` for batch size. Effective batch ≈ 32–64 across all profiles (`batch × grad_accum`).

### Anti-overfitting discipline

≤ 3 epochs + `EarlyStoppingCallback(patience=2)` (these sets memorize in 1–2 epochs); boilerplate stripping; dedup; mandatory **cross-domain** eval; report in-domain vs cross-domain gap. **A near-perfect in-domain macro-F1 (~0.95+) is a red flag for source-style leakage, not a success.**

### Stance / NLI — training procedure

**Default: none (zero-shot).** Optional fine-tune (`training/train_stance.py`): base `microsoft/deberta-v3-base`, `num_labels = 3`, same `TrainingArguments` shape, `lr = 2e-5`, 2–3 epochs, `metric_for_best_model = "f1"` (macro), trained on `copenlu/fever_gold_evidence` or `pietrolesci/nli_fever`, optionally + `tals/vitaminc` for robustness.

---

## 5. Evaluation

### Classifier metrics

- **Macro-F1 — the headline metric** (handles binary fake/real imbalance and the 6-way LIAR2 imbalance where `pants-fire` is rare).
- **Accuracy** + **per-class P/R/F1.** False positives (legitimate news flagged fake) are the high-cost error → `real`-recall and `fake`-precision are reported explicitly.
- **ROC-AUC** (binary) / one-vs-rest macro-AUC (6-way) — threshold-independent ranking.
- **ECE (Expected Calibration Error)** + reliability diagram — over-confidence can silence real news. Temperature scaling on a held-out calibration split; report **pre/post ECE**.
- **Selective accuracy at fixed coverage** (accuracy on the non-abstained set) + a risk–coverage curve.

### Baselines (the system must beat all)

| Baseline | Description |
|---|---|
| **Majority class** | Trivial floor |
| **TF-IDF + LogReg** | The no-torch core (`models/baseline_tfidf.py`); expect ~0.95+ in-domain macro-F1 — a **red flag**, not success |
| **Zero-shot** | `facebook/bart-large-mnli` (fake/real as hypothesis); `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (stance) |

**Strong system:** `answerdotai/ModernBERT-base` / `distilbert-base-uncased` fine-tune. **The transformer must beat the baseline on the *cross-domain* eval**, not in-domain.

### Cross-domain generalization (the honest number)

Train on `GonzaloA` (PolitiFact-style) → test on `LittleFish-Coder/Fake_News_GossipCop` (and/or WELFake/LIAR2); **report the macro-F1 drop**. A large drop is the signature of **source-style leakage** and is itself an ethics metric.

### Fact-check (FEVER-style) metrics

- **Label accuracy** — SUPPORTS/REFUTES/NEI, ignoring evidence.
- **FEVER score** — stricter: label correct **and** a complete evidence group retrieved.
- **Evidence recall@k** — R@1/5/10 (FEVER uses recall@5 sentences / @20 documents); measured against `BeIR/fever-qrels`.
- **Abstain quality** — coverage (fraction not abstained) vs selective accuracy.

### Results table skeletons (to be filled at training time)

| System | Dataset / split | Accuracy | Macro-F1 | P(fake) / R(real) | ROC-AUC | ECE | Coverage @ sel-acc |
|---|---|---|---|---|---|---|---|
| Majority class | GonzaloA test | | | | — | — | — |
| TF-IDF + LogReg | GonzaloA test | | | | | | — |
| Zero-shot (bart-large-mnli) | GonzaloA test | | | | | | — |
| ModernBERT fine-tune | GonzaloA test | | | | | | |
| ModernBERT fine-tune | **GossipCop (cross-domain)** | | | | | | |
| ModernBERT fine-tune | LIAR2 test (6-way) | | | — | | | |

| Fact-check system | Label acc. | FEVER score | R@5 (evidence) | Abstain rate |
|---|---|---|---|---|
| Retrieval + NLI (MoritzLaurer DeBERTa, zero-shot) | | | | |
| + cross-encoder rerank | | | | |
| + FEVER-fine-tuned stance head (optional) | | | | |

---

## 6. Limitations & biases

| Limitation / bias | Why it matters |
|---|---|
| **Source-style leakage (the #1 trap)** | The classifier learns *"this outlet/formatting ⇒ fake"*, not truth. Near-perfect in-domain F1 hides this; performance collapses on new outlets and entrenches bias against named sources. Measured via the cross-domain drop; mitigated by boilerplate stripping and by treating `classifier_prior` as a weak prior the evidence layer can override. |
| **Style ≠ truth** | A well-written falsehood or a clumsily-written truth fools the style classifier. The evidence-grounded verdict exists to compensate; the two signals are reported separately. |
| **Political bias in labels/data** | LIAR/PolitiFact labels encode the fact-checker's editorial judgement + topic/speaker skew. Performance must be sliced by topic/speaker/source; output is never neutral ground truth. |
| **English-centric** | Training data is English only; no claim is made about other languages. |
| **Domain skew** | PolitiFact (US politics) + GossipCop (celebrity) dominate; coverage of science/health/finance is thin outside the synthetic corpus. |
| **Coarse binary proxy** | `fake/real` is a coarse cut over a spectrum; the LIAR2 6-way head captures more nuance. |
| **Fixed evidence corpus** | The fact-check verifies against a **versioned, fixed corpus** and cannot know facts outside it. **`unverified` means *insufficient evidence*, NOT *true*.** Novel claims with no coverage → abstain. |
| **Adversarial paraphrase / evasion** | Style classifiers are brittle to paraphrase; the evidence layer is more robust but not immune. An adversarial/paraphrase eval set is maintained. |
| **Calibration drift** | ECE can degrade off-distribution; over-confidence is dangerous, hence mandatory ECE reporting + temperature scaling + abstain. |

---

## 7. Ethical considerations

**This is the most ethically loaded project in the set.** Non-negotiable framing: **the system flags content for human review; it never auto-removes, auto-blocks, or auto-censors, and it always shows its evidence.** It assists human fact-checkers/moderators; it is not an automated arbiter of truth.

| Risk | Mitigation baked into the deliverable |
|---|---|
| **Censorship / free-speech chilling** | Output a **review flag + evidence**, never an enforcement action; **no auto-takedown anywhere**; verdicts are advisory |
| **False positives silencing real news** | Operating point tuned for high `real`-recall; per-class costs reported; **abstain rather than guess**; human sign-off before any action |
| **Political bias** | Document label provenance; report sliced performance; never present output as neutral truth |
| **Source-style leakage** | Measure cross-domain drop; strip/ablate source/style features; prefer the **evidence-grounded** verdict over the style prior |
| **Automation bias / over-trust** | Calibration (ECE) + mandatory confidence display; **abstain on uncertainty**; UI states it is a decision-support signal and evidence must be read |
| **Hallucinated / misattributed citations** | Citations are **extracted verbatim** from the retrieved corpus with source IDs, never generated; `decisions_trace` lets a reviewer verify each one |
| **Stale / out-of-scope evidence** | Index is versioned + timestamped (`/version`); abstain when coverage is insufficient rather than fabricate |
| **Dual-use** | No "evasion score" shipped; adversarial tooling rate-limited, logged, gated |

**The evidence layer + abstain + human review are the core ethical safeguards.** The agentic fact-check exists so that a verdict is grounded in cited, verbatim evidence; the abstain branch (`unverified`) is a first-class outcome reachable from any state; and the classifier `prior` and evidence `verdict` are reported **separately** so a reviewer sees disagreement. A prior↔stance conflict (classifier says FAKE but evidence strongly SUPPORTS) is surfaced as `conflict — needs human review`, never silently resolved.

---

## 8. License

| Component | License |
|---|---|
| **This repository / code** | **MIT** |
| Classifier base — `answerdotai/ModernBERT-base` | Apache-2.0 ✅ |
| Classifier base — `distilbert-base-uncased` | Apache-2.0 ✅ |
| Classifier base — `FacebookAI/roberta-base` | MIT ✅ |
| Stance/NLI — `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | MIT ✅ |
| Stance/NLI scale-up — `...-large-mnli-fever-anli-ling-wanli` | MIT ✅ |
| NLI baseline — `facebook/bart-large-mnli` | MIT ✅ |
| Retriever — `BAAI/bge-small-en-v1.5` / `sentence-transformers/all-MiniLM-L6-v2` | MIT ✅ / Apache-2.0 ✅ |
| Optional fine-tune base — `microsoft/deberta-v3-base` | MIT ✅ |
| **Classifier training data** | ⚠️ **`GonzaloA/fake_news` license unknown** — research/eval only. License-clean training alternatives: `mohammadjavadpirhadi/...` (MIT), `ErfanMoosaviMonazzah/...` (openrail), `chengxuphd/liar2` (Apache-2.0). |
| **Optional stance fine-tune data** | ⚠️ FEVER family is **cc-by-sa-3.0 + gpl-3.0 copyleft** — a released fine-tuned stance model inherits share-alike. |

**Net:** the **code is MIT** and **all base models are MIT/Apache (cleanly permissive)**. The license risk is concentrated in **training data**: the primary classifier set (`GonzaloA/fake_news`) is license-unknown (use a permissive alternative for any redistributed weights), and any FEVER-fine-tuned stance head inherits copyleft. **Do not redistribute trained weights without resolving the training-data license.**

---

## 9. Versioning & reproducibility

- **`model_meta.json`** is written alongside each trained artifact (tracked by `fakenews/models/model_registry.py`) recording: base model id, dataset id(s) + split, label map applied, hyperparameters, seed (`42`), metrics, and `index_built_at`.
- **Pin HF revisions, not just repo names.** Every model id (classifier base, NLI, retriever, reranker) is pinned to a **commit SHA / revision** so a Hub update can never silently change a verdict.
- **Version stamps in every response** and at `GET /version`: `classifier_version` (`classifier_vX.Y`), `nli_model` (+ revision), `retriever`, `index_built_at`, `git_sha`. The `/classify` response carries `model_version`; the `/factcheck` response carries `classifier_prior`, `verdict`, the full `evidence` list, the `decisions_trace`, and `abstained` — even on abstain.
- **Deterministic FSM thresholds** (auditable, fixed in `agent/policy.py`): `τ_skip = 0.95`, `τ_cw = 0.5` (D2); `τ_rel = 0.3`, `N_min = 3`, `R_max = 2` (D3); `α = 0.6`, `θ = 0.2` (D4); `τ_agree = 0.4`, `τ_present = 0.55` (D5). The optional LLM "brain" may only *propose* a value from the legal set at each decision point; on parse-fail / out-of-set / timeout the deterministic rule fires. The brain **never decides flow**.
- **Reproducibility:** fixed `seed = 42`; offline `data/samples.py` + `tests/fixtures/fake_news_tiny.csv` make the full pipeline runnable with no network and no torch (TF-IDF classifier, lexical-overlap stance, BM25 over the seed evidence).

---

*Model card for P11 — Fake News & Misinformation Detection System. Reflects the system as built in package `fakenews`. Every Hugging Face id and license above was verified against the project design brief (the single source of truth); no unverified ids are introduced. License flags are advisory — resolve training-data licensing before redistributing trained weights.*
