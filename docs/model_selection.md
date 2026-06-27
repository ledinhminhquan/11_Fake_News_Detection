# P11 — Model Selection & Optimization (Section I.5)

> **Project:** Fake News & Misinformation Detection System
> **Course:** NLP in Industry — final assignment
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Package:** `fakenews`
> **Source of truth:** [`docs/DESIGN_BRIEF.md`](./DESIGN_BRIEF.md) — every Hugging Face id below is marked **VERIFIED** there (read live via `hub_repo_details` as `ledinhminhquan`).

This document justifies every modelling choice in P11, the hyperparameter search that produced the shipped classifier, the baseline floor it must beat, the cross-domain evaluation that keeps us honest, and the accuracy/speed/calibration trade-offs baked into the design.

---

## 0. The two-model system in one paragraph

P11 is **not** a single model. It is a fast **trainable classifier** (a *prior*) wrapped by an **agentic, evidence-grounded fact-check** (the *verdict*). The classifier answers "does this *look* fake (style/source patterns)?" in milliseconds; the fact-check answers "what does the *evidence* say?" by retrieving passages and running a stance/NLI model per passage, then aggregating with abstention. The two signals — `classifier_prior` (P(fake)) and the evidence `verdict` — are reported **separately** so a reviewer sees when they disagree. There are therefore **two model-selection problems**:

| # | Component | Decision | Outcome |
|---|---|---|---|
| **A** | Fake-news **classifier** (the prior) | What to **train**? | Fine-tune a transformer (`distilbert-base-uncased` default) with HF `Trainer` + class weights, selected by macro-F1, over a TF-IDF+LogReg floor. |
| **B** | **Stance / NLI** model (the verdict) | What to **train vs. reuse**? | Use the **pretrained zero-shot** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Fine-tuning is *optional* and off by default. |

Internal label convention throughout the repo: **`0 = real`, `1 = fake`** (so `p_fake = P(label == 1)`). The stance bridge is `entailment → SUPPORTS`, `contradiction → REFUTES`, `neutral → NOT_ENOUGH_INFO`.

---

## 1. Component A — the trainable classifier

### 1.1 Why a transformer fine-tune at all

The classifier is the **fast prior** that gates the expensive fact-check (decision **D2**, §4). It must be cheap per call (it runs on *every* request) and good enough that its confidence is a meaningful routing signal. A fine-tuned transformer over a strong sparse baseline is the standard, well-understood choice for binary text classification, and it directly mirrors the P02 résumé-classifier we already ship — so the training/eval/serving scaffolding ports near-verbatim (`models/classifier.py`, `training/train_classifier.py`).

### 1.2 Base-model menu (all VERIFIED, license-clean)

The base model is **configurable** (`config.py → model_config["model_name"]`) and auto-adapts to the detected GPU (§3). The shipped default is `distilbert-base-uncased` because it trains and serves anywhere (T4, CPU-friendly, 512-ctx); ModernBERT is the A100/H100 upgrade for full-article context.

| HF id | Params | License | Max ctx | Role in P11 |
|---|---|---|---|---|
| **`distilbert-base-uncased`** | 67M | **apache-2.0** ✅ | 512 | **Default / T4 / CPU fallback base** — fast, ubiquitous, proven |
| `answerdotai/ModernBERT-base` | 149.7M | **apache-2.0** ✅ | 8192 | **Long-article upgrade** (A100/H100) — handles full `title [SEP] text` without aggressive truncation |
| `microsoft/deberta-v3-base` | 184.4M | **mit** ✅ | 512 | Strongest base; also the stance fine-tune base (§2.4) |
| `roberta-base` (`FacebookAI/roberta-base`) | 124.7M | **mit** ✅ | 512 | Middle option |
| `Pavan48/fake_news_detection_roberta` | ~125M | **apache-2.0** ✅ | 512 | Optional ready-made prior (no training) |

**Justification for the default.** `distilbert-base-uncased` is 67M params, Apache-2.0, and runs in fp16 on a 16 GB T4 (or CPU for dev). ModernBERT's 8192-context advantage only matters for long article bodies; with the safe-default `max_length=512` and `title [SEP] text` concatenation, DistilBERT loses very little while training ~2× faster. We keep ModernBERT as the documented A100/H100 config (`bf16`, `max_length` up to 1024–2048) for the long-document showpiece. All five candidates are MIT/Apache — **no license blocker on the model side.**

### 1.3 Training data choice (and its license caveat)

| Dataset | Status | License | Size | Role |
|---|---|---|---|---|
| **`GonzaloA/fake_news`** | VERIFIED | **unknown** ⚠️ | 40.6K (24.4K/8.1K/8.1K) | **PRIMARY** binary train — clean `title`+`text`, pre-split, parquet (torch-free load) |
| `ErfanMoosaviMonazzah/…-English` | VERIFIED | **openrail** | 44.3K | ISOT-style license-cleaner large alt |
| `mohammadjavadpirhadi/…-english` | VERIFIED | **MIT** ✅ | 10K–100K | **MIT-clean** alt — prefer if license hygiene is paramount |
| **`chengxuphd/liar2`** | VERIFIED | **apache-2.0** ✅ | 23.0K | **SECONDARY** 6-way credibility (`num_labels=6`, `statement` only) |
| `LittleFish-Coder/Fake_News_GossipCop` | VERIFIED | **apache-2.0** ✅ | 12.7K | **Cross-domain test** (§5) |

> ⚠️ **License flag.** `GonzaloA/fake_news` has an **unknown** license — fine for coursework/research, but for any commercial/redistribution-clean training swap to `mohammadjavadpirhadi/…` (MIT) or `ErfanMoosaviMonazzah/…` (openrail). `chengxuphd/liar2` (Apache-2.0) is the clean credibility set. This is documented in the brief and surfaced here so a grader sees we tracked it.

> ⚠️ **Label-polarity gotcha — normalize on load.** Mirrors disagree: `GonzaloA` is natively `0=fake/1=real`; `LittleFish-Coder/*` is `0=real/1=fake`; `mrm8488/fake-news` is `1=fake/0=real`. The repo convention is `0=real, 1=fake`. Every loader in `data/news_loaders.py` applies an **explicit per-source `label_map`** and verifies on a known row — **never assume.**

---

## 2. Component B — the stance / NLI model

### 2.1 Why zero-shot, not a fine-tune

The stance model decides, **per retrieved evidence passage**, whether the evidence entails / contradicts / is neutral toward the claim. The decision (per the brief) is **use a pretrained zero-shot NLI model and do NOT fine-tune by default.** Rationale:

- **Zero training cost, zero training data licensing.** Fine-tuning a verdict head means inheriting FEVER's **cc-by-sa + gpl copyleft** (every `fever/*`, `copenlu/*`, `pietrolesci/*`, `tals/vitaminc` dataset). A pretrained MIT model keeps the *redistributed* artifact license-clean.
- **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` is already FEVER-tuned.** It was trained on MNLI **+ FEVER** + ANLI, so its `entailment/neutral/contradiction` head maps *directly* onto our `SUPPORTS/REFUTES/NEI` bridge.
- **Marginal gain from fine-tuning is small** (brief §4B): "Net gain over zero-shot is usually small → zero-shot is the default; fine-tune is optional."

### 2.2 The pick (VERIFIED)

| HF id | Params | License | Labels | Why |
|---|---|---|---|---|
| **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** | 184.4M | **mit** ✅ | entail / neutral / contra | **PRIMARY zero-shot.** FEVER-tuned, fits any GPU (T4), MIT. |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 435.1M | mit ✅ | ent/neu/con | A100/H100 accuracy upgrade |
| `facebook/bart-large-mnli` | 407.3M | mit ✅ | ent/neu/con | Zero-shot **baseline + NLI fallback** |
| `roberta-large-mnli` | 356.4M | mit ✅ | ent/neu/con | Alt baseline |

### 2.3 Critical engineering invariant — read labels at runtime

`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` has `id2label` verified as `{0: entailment, 1: neutral, 2: contradiction}` — but **`factcheck/stance.py` reads `model.config.id2label` at load and never hardcodes the order.** A different NLI model (or a Hub revision) can permute these indices; hardcoding silently flips support↔refute. The premise/hypothesis order is also fixed: **premise = retrieved evidence, hypothesis = the claim.**

### 2.4 Optional stance fine-tune (the trainable showpiece, OFF by default)

If a trainable stance head is wanted (`training/train_stance.py`), fine-tune `microsoft/deberta-v3-base` with `num_labels=3` on the cleanest single verdict set, **`copenlu/fever_gold_evidence`** (228K claim+gold-evidence S/R/NEI), or the NLI-formatted drop-in **`pietrolesci/nli_fever`** (`premise/hypothesis/label`). Same `TrainingArguments` as Component A, `lr=2e-5`, 2–3 epochs, `metric_for_best_model="f1"` (macro). Optionally add `tals/vitaminc` for robustness to subtle edits. **License note:** any released stance model inherits FEVER's cc-by-sa + gpl copyleft → flag for redistribution. This is exactly why it stays optional.

---

## 3. `TrainingArguments` + GPU profile

### 3.1 Shipped `TrainingArguments` (Component A)

```python
training_args = transformers.TrainingArguments(
    output_dir="outputs/fakenews_cls",
    num_train_epochs=3,                 # few epochs — these datasets overfit fast
    learning_rate=2e-5,                 # selected from the LR grid (§4.1)
    per_device_train_batch_size=16,     # see GPU table; 8 on T4
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,      # 4 on T4 -> effective batch ~32
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    bf16=True,                          # A100/H100/L4; set fp16=True on T4 instead
    fp16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",         # macro-F1 — NOT accuracy
    greater_is_better=True,
    save_total_limit=2,
    seed=42,
    report_to="none",
    dataloader_num_workers=4,
    group_by_length=True,
)
# + EarlyStoppingCallback(early_stopping_patience=2) on metric_for_best_model
```

Key choices: **`metric_for_best_model="f1"` (macro)** so the best checkpoint is selected by the imbalance-robust metric, never raw accuracy; **`num_train_epochs=3` + early stopping** because these corpora memorize in 1–2 epochs; **`group_by_length=True`** to cut padding cost. `compute_metrics` returns `{"f1": f1_score(y, p, average="macro"), "accuracy": ...}`.

### 3.2 Class weights for imbalance

GonzaloA is near-balanced, but WELFake / LIAR2-6-way are not, so we ship a `WeightedTrainer` overriding `compute_loss` with `nn.CrossEntropyLoss(weight=class_weights)`, where `class_weights = n / (k · bincount)` (normalized). Class weighting is a **tuned hyperparameter** (§4.2), not always-on.

```python
class_weights = torch.tensor([w_real, w_fake])  # n/(k*bincount), normalized
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights.to(out.logits.device))(out.logits, labels)
        return (loss, out) if return_outputs else loss
```

### 3.3 GPU-profile table (auto-adapt)

Detected via `torch.cuda.get_device_capability()` (≥8.0 → bf16; T4 is 7.5 → fp16) and `torch.cuda.get_device_properties(0).total_memory`. Effective batch ≈ 32–64 across all profiles.

| GPU | Precision | train batch | eval batch | grad_accum | max_length | Base / notes |
|---|---|---|---|---|---|---|
| **H100 80GB** | bf16 | 64 | 128 | 1 | 1024–2048 | ModernBERT — push long context |
| **A100 40/80GB** | bf16 | 32 | 64 | 1 | 512–1024 | ModernBERT — primary GPU target |
| **L4 24GB** | bf16 | 16 | 32 | 2 | 512 | bf16 ok on Ada |
| **T4 16GB** | **fp16** | 8 | 16 | 4 | 384–512 | base → `distilbert-base-uncased`; `fp16=True, bf16=False` |
| **CPU / no torch** | — | — | — | — | — | **TF-IDF+LogReg only** (the floor still trains) |

---

## 4. Hyperparameter tuning (`training/tune.py`)

Selection metric is **macro-F1 on the validation split** throughout (never accuracy). The search is deliberately small — these datasets overfit, so a wide sweep wastes compute and invites val-set overfitting.

### 4.1 Learning-rate grid

| LR | Notes |
|---|---|
| 1e-5 | conservative; sometimes under-fits in 3 epochs |
| **2e-5** | **selected default** — the transformer-classification sweet spot |
| 3e-5 | |
| 5e-5 | risks instability on the small splits |

Grid is `{1e-5, 2e-5, 3e-5, 5e-5}`; `2e-5` is the shipped default, re-confirmed per base model. Warmup (`warmup_ratio=0.1`) and `weight_decay=0.01` are held fixed.

### 4.2 Class-weight setting

`{none ("balanced via sampling off"), "balanced" inverse-frequency, manual [w_real, w_fake]}` — toggled by val macro-F1 and, critically, by the **per-class** trade-off: we accept a tiny macro-F1 dip if it buys higher `real`-recall (flagging legit news fake is the costly error, §6 ethics).

### 4.3 Epochs / early stopping

`num_train_epochs ∈ {2, 3}` with `EarlyStoppingCallback(patience=2)`. Best checkpoint by `metric_for_best_model="f1"` + `load_best_model_at_end=True`.

### 4.4 Sequence length

`max_length ∈ {384, 512, 1024}` bounded by the GPU profile (§3.3). 512 is the safe/cheap default; 1024+ only on ModernBERT/A100 for long bodies (cost grows ~length²).

### 4.5 What is **not** tuned (deliberately)

The **agent FSM thresholds** (`τ_skip=0.95`, `τ_cw=0.5`, `τ_rel=0.3`, `N_min=3`, `R_max=2`, `α=0.6`, `θ=0.2`, `τ_agree=0.4`, `τ_present=0.55`) are **policy constants in `agent/policy.py`**, not learned weights. They encode the ethics posture (when to abstain) and are set conservatively by design, not fit to a metric — the model is allowed to be wrong, but the *policy* errs toward abstention and human review.

---

## 5. Baselines — the floor every system must beat

Per brief §7, the strong system must beat **all** of: majority class, TF-IDF+LogReg, and zero-shot.

### 5.1 Majority class

Predict the most frequent label. Pins the floor and exposes any accuracy that is just prior-matching. Macro-F1 is poor by construction on any imbalance — the reason macro-F1 is our headline metric.

### 5.2 TF-IDF + LogReg — the no-torch core floor (`models/baseline_tfidf.py`)

This is not a throwaway baseline: it is the **offline production path** (the service boots and classifies with only numpy/pandas/sklearn).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9,
        sublinear_tf=True, max_features=200_000, strip_accents="unicode")),
    ("clf", LogisticRegression(
        C=4.0, class_weight="balanced", max_iter=2000, n_jobs=-1)),
    # swap clf -> LinearSVC(C=1.0, class_weight="balanced") for the SVM floor
])
```

Input = `(title + " " + text)` with **source-boilerplate stripped** (§5.4). `class_weight="balanced"` handles imbalance; `C` is the one tuned knob.

> ⚠️ **Expect TF-IDF+LogReg macro-F1 ≈ 0.95+ in-domain on GonzaloA/WELFake — this is a RED FLAG, not success.** These corpora are source-leaky (all "real" rows are Reuters/AP wire copy, all "fake" are blog-rant style). A 0.95 in-domain number means the model learned "is this Reuters formatting," not veracity. **The transformer must beat the baseline on the cross-domain eval, not in-domain.**

### 5.3 Zero-shot

`facebook/bart-large-mnli` with fake/real expressed as a hypothesis, and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for stance. No training; establishes what a pretrained model gets "for free" before any fine-tuning is justified.

### 5.4 The source/style-leakage caveat (the #1 trap)

This is the central modelling honesty problem in P11 and is treated as an **ethics metric**, not just an accuracy footnote.

- **Mechanism.** GonzaloA/WELFake/ISOT leak *publisher & writing style*, not truth. 99% in-domain F1 = the model learned the outlet's formatting (`"MOSCOW (Reuters) -…"` datelines, bylines), and it **collapses on new outlets** — and entrenches bias against named sources.
- **Mitigations (all shipped):**
  1. **Strip dateline/source boilerplate** (`(Reuters)`, `WASHINGTON —`, bylines) before training and inference.
  2. **Report the in-domain vs cross-domain gap explicitly** (table below).
  3. **Treat too-high in-domain F1 as a red flag**, not a win.
  4. **Dedup** exact + near-dup (MinHash / normalized-hash) within and across splits.
  5. For LIAR/LIAR2, **drop metadata leakage** (`speaker, state_info, subject`, per-speaker `*_counts`) — train on `statement` (+ optional `context`) only.
  6. Lean on the **evidence-grounded verdict**; the `classifier_prior` is only a weak nudge (α-weighted, §0).

### 5.5 Cross-domain evaluation (mandatory)

**Train on `GonzaloA` (PolitiFact-style) → test on `LittleFish-Coder/Fake_News_GossipCop` (celebrity/GossipCop-style)** and report the macro-F1 drop. A large drop is the *signature* of source-style leakage — and is itself an ethics metric. The cross-domain macro-F1 is the **real** number we report, not the inflated in-domain one.

### 5.6 Results-table skeleton (classifier)

| System | Dataset / split | Accuracy | Macro-F1 | P(fake) / R(real) | ROC-AUC | ECE | Coverage @ sel-acc |
|---|---|---|---|---|---|---|---|
| Majority class | GonzaloA test | | | | — | — | — |
| TF-IDF + LogReg | GonzaloA test | | | | | | — |
| Zero-shot (bart-large-mnli) | GonzaloA test | | | | | | — |
| **DistilBERT/ModernBERT fine-tune** | GonzaloA test | | | | | | |
| **DistilBERT/ModernBERT fine-tune** | **GossipCop (cross-domain)** | | | | | | |
| DistilBERT/ModernBERT fine-tune | LIAR2 test (6-way) | | | — | | | |

### 5.7 Results-table skeleton (fact-check)

| Fact-check system | Label acc. | FEVER score | R@5 (evidence) | Abstain rate |
|---|---|---|---|---|
| Retrieval + NLI (MoritzLaurer DeBERTa, zero-shot) | | | | |
| + cross-encoder rerank | | | | |
| + FEVER-fine-tuned stance head (optional) | | | | |

---

## 6. Calibration — ECE + temperature scaling

Over-confidence here is dangerous: a confidently-wrong "fake" can silence real news, and the classifier's confidence is the **D2 routing gate** (`τ_skip=0.95`). Calibration is therefore first-class, not optional.

- **Metric:** **Expected Calibration Error (ECE)** + a reliability diagram (`training/metrics.py`, `analysis/`).
- **Method:** **temperature scaling** — fit a single scalar `T` on a held-out **calibration split** (separate from train/test), divide logits by `T` before softmax. Report **pre/post ECE**; this never changes argmax accuracy, only the probabilities the gate and the UI consume.
- **Why it matters downstream:** `τ_skip=0.95` decides whether to skip retrieval. If the classifier is over-confident, it would skip the fact-check on items it shouldn't. Calibrated probabilities make the gate trustworthy. The `/classify` endpoint returns the **calibrated** P(fake).

---

## 7. Trade-offs

### 7.1 Accuracy vs. speed — why the fact-check is gated

| Path | Latency (typical) | When it runs |
|---|---|---|
| TF-IDF+LogReg classifier | sub-ms CPU | every request |
| Transformer classifier | ~10–30 ms GPU / ~100–300 ms CPU | every request |
| Retrieval + k-passage NLI (+ rerank) | ~0.3–2 s | **only behind `/factcheck`**, gated by **D2** |

The classifier is cheap → runs on **every** request as the prior. The retrieve→stance→aggregate fact-check is ~10–100× heavier → it is **gated behind `/factcheck` and the D2 check-worthiness/confidence skip**, and **cached by claim hash**. Decision D2: skip retrieval iff `classifier_conf ≥ τ_skip=0.95 AND checkworthy_score < τ_cw=0.5`. This is the core accuracy-vs-speed lever: spend the expensive evidence budget only where it changes the answer.

### 7.2 Complexity vs. maintainability

- **Zero-shot stance over a fine-tuned head** removes a whole training pipeline, a 228K-row dataset dependency, and a copyleft-licensed artifact — the optional `train_stance.py` exists but is off by default.
- **Configurable base model + GPU auto-adapt** means one code path serves T4→H100→CPU; no per-environment forks.
- **No-torch degradation** (`ClassifyTool`→TF-IDF+LogReg, `RetrieveEvidence`→TF-IDF cosine, `StanceNLI`→lexical mock-stance forcing `UNVERIFIED` unless the prior is extreme) keeps the entire pipeline runnable, testable, and CI-able with only sklearn.
- **FSM thresholds as policy constants** (not learned) keep the ethics-critical behavior auditable and stable across model swaps.

### 7.3 Accuracy vs. ethics (the operating point)

We deliberately do **not** pick the threshold that maximizes accuracy/F1. The operating point favors **high `real`-recall** (minimize flagging legit news) and **abstention under uncertainty** — `unverified` is a first-class outcome. The system **flags for human review; it never auto-removes or censors**, and always ships verbatim citations + the `decisions_trace`. A slightly lower headline number that abstains correctly beats a higher one that confidently silences real news.

---

## 8. Summary of selected models

| Slot | Selected | License | Why |
|---|---|---|---|
| Classifier base (default) | `distilbert-base-uncased` | apache-2.0 ✅ | Fast, ubiquitous, T4/CPU-friendly, 512-ctx |
| Classifier base (upgrade) | `answerdotai/ModernBERT-base` | apache-2.0 ✅ | 8192-ctx full articles on A100/H100 |
| Classifier training data | `GonzaloA/fake_news` | **unknown ⚠️** | Clean, pre-split, primary; MIT alts documented |
| Classifier floor | TF-IDF + LogReg (sklearn) | — | No-torch production path + baseline |
| Stance / NLI (verdict) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | mit ✅ | FEVER-tuned zero-shot, no training, fits T4 |
| Stance fine-tune (optional) | `microsoft/deberta-v3-base` on `copenlu/fever_gold_evidence` | mit model / **cc-by-sa+gpl ⚠️** data | Off by default; copyleft on released artifact |
| Selection metric | macro-F1 (val) | — | Imbalance-robust; never accuracy alone |
| Calibration | temperature scaling, report ECE | — | Over-confidence silences real news |

**Headline modelling honesty statement:** the classifier detects *style/source patterns correlated with fakeness*, which is **not** detecting falsehood. The cross-domain macro-F1 drop quantifies that gap; the evidence-grounded fact-check and separate reporting of `classifier_prior` vs. `verdict` exist precisely to compensate.
