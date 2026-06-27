# P11 — Fake News & Misinformation Detection System — Design Brief

> NLP-in-Industry final assignment, project P11. Reference repo: [KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
> This brief is self-contained: an implementer can build the whole `src/fakenews/` repo from it without re-researching. Every Hugging Face id is marked **VERIFIED** (read live via `hub_repo_details` as user `ledinhminhquan`) or **UNVERIFIED**, with its license. Where sections disagreed, the most license-clean + verified option is chosen and the alternative noted inline.
>
> **One-line thesis:** a *trainable fake-news text classifier* (fast prior) wrapped by an *agentic, evidence-grounded fact-check* (retrieve → stance/NLI → aggregate verdict with citations → **abstain** when uncertain). The tool **assists** human fact-checkers; it never auto-censors.

---

## 1. Problem & business value

### Users and jobs-to-be-done

| User | Job-to-be-done | What the system gives them |
|---|---|---|
| **Journalists / professional fact-checkers** | "Is this claim worth checking, and what does the evidence say?" — **claim fact-check** | A verdict (`real`/`fake`/`unverified`) with cited evidence passages and a per-evidence stance, plus an audit trace they can verify line-by-line. |
| **Platform trust-&-safety / moderators** | Triage a firehose of posts — **flag-for-review** | A fast credibility signal (classifier prior) that routes only suspicious items to human review. Never an auto-takedown. |
| **Newsroom editors / researchers** | Rate source/claim credibility on a scale — **credibility scoring** | A 6-way credibility score (LIAR-style: pants-fire → true) for nuance beyond binary fake/real. |
| **End readers (secondary)** | "Can I trust this headline?" | A transparent credibility indicator with the evidence shown, encouraging verification rather than blind trust. |

### Success metrics

**Business:**
- Reviewer throughput uplift (claims triaged per hour) and reduction of human time spent on clearly-credible items (high-confidence skips).
- Precision of the *flag-for-review* queue (fraction of flagged items a human agrees were worth reviewing) — minimizes wasted reviewer attention.
- Evidence-grounding rate: fraction of verdicts shipped with ≥1 verbatim citation (target: 100% of non-abstained verdicts).
- Trust/adoption: reviewer override rate stays low *and* reviewers report the evidence was useful.

**Technical:**
- Classifier macro-F1 (in-domain **and** cross-domain — the cross-domain number is the honest one).
- Fact-check label accuracy + evidence recall@k (FEVER-style).
- Calibration (ECE) — over-confidence here can silence real news.
- Selective accuracy at a fixed coverage (accuracy on the non-abstained set).

### Non-negotiable framing

**The tool ASSISTS humans; it never auto-censors.** Output is always a *review flag + evidence*, never an enforcement action. There is no auto-takedown anywhere in the design. `unverified`/abstain is a first-class outcome, not a failure mode. The classifier detects *style/source patterns correlated with fakeness* — which is **not** the same as detecting falsehood; the agentic evidence layer exists precisely to compensate, and the two signals (`classifier_prior` vs. evidence `verdict`) are reported **separately** so a reviewer sees when they disagree.

---

## 2. Verified stack table

All ids checked via `hub_repo_details` as `ledinhminhquan`. **VERIFIED** = repo exists and its live schema/splits/license were read. License hygiene: **prefer permissive (MIT/Apache/CC-BY/CC0); flag non-commercial/copyleft/unclear.**

### 2a. Classification datasets (the trainable classifier core)

| HF id | Status | License | Size / splits | Schema (key cols) | Role |
|---|---|---|---|---|---|
| **`GonzaloA/fake_news`** | VERIFIED (preview) | **unknown** ⚠️ flag | 40.6K — train 24.4K / val 8.1K / test 8.1K | `Unnamed:0, title, text, label(int)` — **0=fake, 1=real** | **PRIMARY binary** — clean, pre-split, title+text, parquet (loads torch-free) |
| `davanstrien/WELFake` | VERIFIED (schema) | **unknown** ⚠️ flag (source WELFake ≈ CC-BY 4.0 per IEEE DataPort — verify) | 72.1K single train (35K real / 37K fake) | `title, text, label` ClassLabel **0=fake, 1=real** | Larger binary alt; merges 4 corpora (Kaggle/McIntire/Reuters/BuzzFeed) → less source-overfit, but make your own split + dedup |
| `ErfanMoosaviMonazzah/fake-news-detection-dataset-English` | VERIFIED (preview) | **openrail** | 44.3K — train 30.0K / val 6.0K / test 8.3K | `Unnamed:0, title, text, subject, date, label(int)` — **0=fake, 1=real** | ISOT-style, pre-split — **most license-clean of the large binary mirrors** |
| `mohammadjavadpirhadi/fake-news-detection-dataset-english` | VERIFIED | **MIT** ✅ | 10K–100K (viewer renamed/broken; repo exists) | text-classification binary | **MIT-clean binary alt** — prefer if license hygiene is paramount |
| **`chengxuphd/liar2`** | VERIFIED (preview) | **apache-2.0** ✅ | 23.0K — train 18.4K / val 2.3K / test 2.3K | `id, label(int 0–5), statement, date, subject, speaker, speaker_description, state_info, *_counts(6), context, justification` (16 cols) | **SECONDARY (6-way credibility)** — preferred LIAR family (clean license, working viewer, `justification` = free evidence) |
| `ucsbai/liar` (canonical `liar`; bare `liar`/`ucsbnlp/liar` redirect here) | VERIFIED (repo); **viewer broken** (legacy `.py` loader, 500/404) | **unknown** ⚠️ flag | 12.8K — train 10.24K / val 1.28K / test 1.27K | `statement, label(0–5), subject, speaker, job_title, state_info, party_affiliation, *_counts, context` | Original 6-way — **prefer LIAR2**; load via `datasets`/`trust_remote_code=True` if needed |
| `mrm8488/fake-news` | VERIFIED (preview) | not declared ⚠️ | 44.9K single train | `text, label(int)` — **1=fake, 0=real** (⚠️ OPPOSITE polarity) | binary eval/aux only — no title, no splits |
| `rickstello/FakeNewsNet` | VERIFIED (schema) | **cc** (generic) ⚠️ | 23.2K single train | `title, news_url, source_domain, tweet_num, real(int 0/1)` — **titles only, no body** | weak — title+URL only |
| `LittleFish-Coder/Fake_News_PolitiFact` | VERIFIED (schema) | **apache-2.0** ✅ | 483 — train 381 / test 102 | `text, label(int)` **0=real, 1=fake** + precomputed bert/roberta/… embeddings (51 MB train) | FakeNewsNet-PolitiFact content mirror (use `text`+`label` only) |
| `LittleFish-Coder/Fake_News_GossipCop` | VERIFIED (schema) | **apache-2.0** ✅ | 12.7K — train 9,988 / test 2,672 | same (`text, label` **0=real, 1=fake**) + embeddings (**1.3 GB** train parquet) | FakeNewsNet-GossipCop content mirror — for **cross-domain eval** |
| `Ahren09/FakeNewsNet` | VERIFIED (repo); viewer 501 | **apache-2.0** ✅ | n/a via viewer | multimodal/social-graph (source of LittleFish mirrors) | reference only — manual loading |

> **⚠️ Polarity gotcha — normalize on load.** `GonzaloA`, `ErfanMoosaviMonazzah`, `davanstrien/WELFake` → **0=fake, 1=real**. `LittleFish-Coder/*` → **0=real, 1=fake**. `mrm8488/fake-news` → **1=fake, 0=real**. The repo's internal convention is **`label: 0=REAL, 1=FAKE`** (see §9 `NewsItem`). Every loader MUST apply an explicit per-source `label_map` — never assume.

### 2b. Fact-check / FEVER datasets (the agentic evidence layer)

| HF id | Status | License | Size / splits | Schema | Labels / role |
|---|---|---|---|---|---|
| **`fever/fever`** (bare `fever`/`fever_v2.0` do NOT resolve) | VERIFIED; viewer 501 (loader script) | **cc-by-sa-3.0 + gpl-3.0** ⚠️ copyleft | ~185K claims; configs `v1.0`/`v2.0`/`wiki_pages` | `id, label, claim, evidence_*` (+ `wiki_pages` = evidence corpus) | SUPPORTS / REFUTES / NOT ENOUGH INFO — evidence corpus + verification |
| **`copenlu/fever_gold_evidence`** | VERIFIED | cc-by-sa-3.0 + gpl-3.0 ⚠️ | 228.3K train / 15.9K val / 16.0K test | `claim, label(str), evidence(list[page,line_id,sentence]), id, verifiable, original_id` | SUPPORTS/REFUTES/NEI — **cleanest single set to fine-tune a claim+evidence verdict head** |
| **`pietrolesci/nli_fever`** | VERIFIED (viewer ok) | (parent FEVER cc-by-sa-3.0 + gpl-3.0) ⚠️ | 208K–248K train / ~20K dev / ~20K test | `premise, hypothesis, label{entailment,neutral,contradiction}` | **FEVER reframed as NLI** — drop-in for optional stance fine-tune (premise=evidence, hypothesis=claim) |
| `tals/vitaminc` | VERIFIED | cc-by-sa-3.0 ⚠️ | 370.7K / 63.1K / 55.2K | `claim, evidence, label, page, revision_type, FEVER_id` | SUPPORTS/REFUTES/NEI — contrastive, robust to subtle edits (add for robustness) |
| `mwong/fever-evidence-related` | VERIFIED | cc-by-sa-3.0 + gpl-3.0 ⚠️ | 403K / 54.6K / 27.4K | `claim, evidence, labels(int)`, pre-tokenized `input_ids` | related / not-related — evidence re-ranking |
| `ImperialCollegeLondon/health_fact` (PUBHEALTH) | VERIFIED; viewer disabled (loader) | **mit** ✅ | 10K–100K | `claim, main_text, explanation, label, sources, subjects` | true/false/unproven/mixture (4-way) — health domain |
| `tdiggelm/climate_fever` (bare `climate_fever` redirects here) | VERIFIED | **unknown** ⚠️ flag | 1.5K test only | `claim_id, claim, claim_label, evidences(list)` | SUPPORTS/REFUTES/NEI/**DISPUTED** — `DISPUTED` motivates the contradictory-evidence abstain branch |
| `allenai/scifact` | VERIFIED; loader script | **cc-by-nc-2.0** ⛔ NON-COMMERCIAL | 1K–10K | `claim, evidence, cited_doc_ids, label` + rationales | SUPPORT/CONTRADICT/NOINFO — **do not use commercially** |
| `BeIR/scifact` | VERIFIED | cc-by-sa-4.0 | 1K–10K | BEIR triple (`corpus`,`queries`,`qrels`) | retrieval relevance — share-alike alternative to scifact |
| `BeIR/fever` + `BeIR/fever-qrels` | VERIFIED | **cc-by-sa-4.0** | — | Wikipedia-abstract corpus + qrels | **measure evidence recall@k** |
| `fever/feverous` | VERIFIED; loader script | cc-by-sa-3.0 ⚠️ | 100K–1M | claim + sentence/table evidence | SUPPORTS/REFUTES/NEI — bonus (tables+text) |

### 2c. Stance datasets

| HF id | Status | License | Rows | Schema | Labels |
|---|---|---|---|---|---|
| `nid989/FNC-1` (only clean FNC-1 mirror; `emergent` NOT on Hub) | VERIFIED | not stated ⚠️ flag (FNC-1 source permissive — confirm) | 40.5K / 4.5K / 5.0K | `headline, body(articleBody), stance` | agree / disagree / discuss / unrelated |

> FNC-1's 4-way labels (unrelated/discuss) are weaker for verdicts than FEVER's 3-way — treat FNC-1 as a **secondary/aux** stance signal.

### 2d. Stance / NLI models (the agentic verdict)

| HF id | Status | Params | License | Output labels | Notes |
|---|---|---|---|---|---|
| **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** | VERIFIED | 184.4M | **mit** ✅ | entailment / neutral / contradiction | **PRIMARY zero-shot pick.** MNLI+**FEVER**+ANLI tuned → maps directly to support/refute/NEI; fits T4. `id2label` verified `{0:entailment, 1:neutral, 2:contradiction}` — **read from `model.config.id2label` at load, never hardcode.** |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | VERIFIED | 435.1M | mit ✅ | ent/neu/con | Best zero-shot accuracy; use on A100/H100 |
| `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | VERIFIED | 435.1M | mit ✅ | entail / not_entail | For `zero-shot-classification` pipeline w/ custom hypotheses |
| `facebook/bart-large-mnli` | VERIFIED | 407.3M | mit ✅ | ent/neu/con | Classic zero-shot **baseline + fallback NLI** |
| `roberta-large-mnli` → `FacebookAI/roberta-large-mnli` | VERIFIED | 356.4M | mit ✅ | ent/neu/con | Alt baseline |
| `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` | VERIFIED | 356M | mit ✅ | 3-way NLI | Stronger but heavier alt |
| `cross-encoder/nli-deberta-v3-base` | VERIFIED | 184.4M | **apache-2.0** ✅ | contra/entail/neutral | sentence-transformers CrossEncoder pair-scoring |
| `microsoft/deberta-v3-base` / `-large` | VERIFIED | 184.4M / 435.1M | mit ✅ | (fill-mask base) | **FINE-TUNE base only** (FEVER/copenlu) — not zero-shot |

### 2e. Classifier base models + retrieval/rerank + libraries

| HF id | Status | Params | License | Role |
|---|---|---|---|---|
| **`answerdotai/ModernBERT-base`** | VERIFIED | 149.7M | **apache-2.0** ✅ | **PRIMARY classifier base** — 8192-ctx (full articles), P02-proven. A100/H100. |
| `distilbert-base-uncased` (`distilbert/…`) | VERIFIED | 67M | **apache-2.0** ✅ | T4 fallback base (max 512) |
| `roberta-base` (`FacebookAI/roberta-base`) | VERIFIED | 124.7M | **mit** ✅ | Middle-option base (max 512) |
| `Pavan48/fake_news_detection_roberta` | VERIFIED | ~125M | **apache-2.0** ✅ | Optional ready-made prior (no training) |
| **`BAAI/bge-small-en-v1.5`** | VERIFIED | 33.4M | **mit** ✅ | Dense evidence retriever (CPU/T4); reuse from P08/P09 |
| `sentence-transformers/all-MiniLM-L6-v2` | VERIFIED | 22.7M | **apache-2.0** ✅ | Alt dense retriever (384-d) |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | VERIFIED | 22.7M | **apache-2.0** ✅ | Evidence reranker `(query,passage)→score` |

**Libraries:** `transformers`, `datasets`, `sentence-transformers`, `torch` (all **lazy/heavy** — imported inside `run()`); core floor = `numpy`, `pandas`, `scikit-learn`, `rank_bm25` (TF-IDF); serving = `fastapi`, `uvicorn`, `pydantic`, `gradio`; optional `faiss-cpu`.

### Avoid / unclear-license note

- ⛔ **Non-commercial — DO NOT ship in a commercial product:** `allenai/scifact` (cc-by-nc-2.0). Use `BeIR/scifact` (cc-by-sa-4.0) instead.
- ⚠️ **Unknown / unstated — research/eval only, verify before redistribution:** `GonzaloA/fake_news`, `davanstrien/WELFake`, `ucsbai/liar`, `mrm8488/fake-news` (also opposite polarity), `tdiggelm/climate_fever`, `nid989/FNC-1`, `rickstello/FakeNewsNet` (generic `cc`).
- ⚠️ **Copyleft / share-alike (attribution + share-alike; GPL on code portions):** all `fever/*`, `copenlu/fever_gold_evidence`, `pietrolesci/nli_fever`, `mwong/*`, `tals/vitaminc` (cc-by-sa-3.0 ± gpl-3.0); `BeIR/*` (cc-by-sa-4.0). Usable, but the derived stance model inherits share-alike — flag for redistribution.
- ⚠️ **FakeNewsNet raw** (`KaiDMML/FakeNewsNet`): a **crawler, not a packaged dataset**. Only tweet IDs + URLs are redistributed; article bodies/tweets require re-crawling (Twitter ToS + publisher copyright). Copyright 2019 Arizona Board of Regents (ASU), academic-use + citation. **Treat as network-gated; exclude from CI.** Use the Apache-2.0 `LittleFish-Coder/*` content mirrors instead.
- ✅ **Cleanly permissive (prefer these):** `chengxuphd/liar2` (Apache-2.0), `mohammadjavadpirhadi/…` (MIT), `ImperialCollegeLondon/health_fact` (MIT), `LittleFish-Coder/*` + `Ahren09/FakeNewsNet` (Apache-2.0), and **every recommended model** (all MoritzLaurer DeBERTa NLI, `bart-large-mnli`, `roberta-large-mnli`, `microsoft/deberta-v3-*`, `BAAI/bge-small-en-v1.5`, MiniLM — MIT/Apache).

---

## 3. System pipeline

```
                                ┌──────────────────────────────────────────────────────────┐
                                │                     P11 FACT-CHECK FSM                     │
                                └──────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌───────────────┐    ┌──────────────┐
  │ INGEST/  │ ─► │   PARSE /      │ ─► │   CLASSIFY   │  (P02 classifier → fast PRIOR)
  │ NORMALIZE│    │ CLAIM-EXTRACT  │    │  fake|real,p │   transformer OR tfidf_logreg fallback
  └──────────┘    └──────┬────────┘    └──────┬───────┘
                         │ D1                   │
                 (article vs short claim;       │
                  extract central claim)        ▼
                         │               ┌──────────────┐
                         └──────────────►│ CHECK-WORTHY?│  D2  conf ≥ τ_skip AND checkworthy<τ_cw
                                         └──────┬───────┘
                       ┌────────────────────────┴────────────────────────┐
               high-conf & NOT check-worthy                  check-worthy OR low-conf
                       │ (skip retrieval)                                │
                       ▼                                                 ▼
                ┌─────────────┐                                  ┌──────────────┐
                │  PRESENT    │ ◄───────────────────────┐        │   RETRIEVE   │ (P08/P09 BM25+dense, RRF)
                │ (prior only)│                         │        │  evidence[]  │
                └─────────────┘                         │        └──────┬───────┘
                                                         │               │ D3  evidence-coverage gate
                                                         │        ┌──────┴────────────┐
                                                         │   too few / low-rel    enough evidence
                                                         │     │ widen query (≤R_max)     │
                                                         │     │                          ▼
                                                         │     │                   ┌──────────────┐
                                                         │     │                   │  STANCE/NLI  │ (DeBERTa-mnli-fever-anli)
                                                         │     │                   │ per evidence │  entail/neutral/contra
                                                         │     │                   └──────┬───────┘
                                                         │     │                          ▼
                                                         │     │                   ┌──────────────┐
                                                         │     │                   │  AGGREGATE   │ D4 prior + weighted stance votes
                                                         │     │                   │   verdict    │  → REAL|FAKE|UNVERIFIED
                                                         │     │                   └──────┬───────┘
                                                         │     │                          ▼
                                                         │     └─────────────────► ┌──────────────┐
                                                         │   exhausted retries     │DECIDE/ABSTAIN│ D5 confidence/agreement/conflict
                                                         │                         └──────┬───────┘
                                                         │            ┌───────────────────┴───────────────────┐
                                                         │       confident                              low agreement / conflict
                                                         │            │                                       │
                                                         │            ▼                                       ▼
                                                         │     ┌──────────────────┐              ┌──────────────────────┐
                                                         └────►│     PRESENT       │              │  ABSTAIN /           │
                                                               │ label+confidence+ │              │ "unverified — needs  │
                                                               │ citations+        │              │  human review"       │
                                                               │ rationale         │              └──────────────────────┘
                                                               └──────────────────┘

 Cross-cutting: every transition → ToolTrace(state, tool, inputs_hash, outputs, latency_ms, brain_proposal?, rule_applied).
 Brain (optional LLM) NEVER decides flow; at each D-point it may only PROPOSE a value from the legal set. On parse-fail/timeout the deterministic rule fires.
```

Canonical ACFC stages (FEVER shared task; Guo/Schlichtkrull/Vlachos survey; ClaimBuster) map onto the FSM: **claim detection / check-worthiness → document retrieval → evidence selection → claim verification (stance/NLI) → verdict {SUPPORTED, REFUTED, NEI}**. SUPPORTED→REAL-leaning, REFUTED→FAKE-leaning, NEI→ABSTAIN.

---

## 4. Trainable model plan

### (A) Fake-news classifier fine-tune

**Decision & justification.** The strongest trainable component is a **binary real/fake article classifier**, trained primarily on **`GonzaloA/fake_news`** (VERIFIED; clean `title`+`text`; pre-split train/val/test; parquet → loads torch-free; the closest analog to the P02 resume classifier). Concatenate `title [SEP] text`. Base = **`answerdotai/ModernBERT-base`** (8192-ctx handles full articles without aggressive truncation), auto-downgrading to `distilbert-base-uncased` on T4. A documented **secondary config** trains **6-way credibility on `chengxuphd/liar2`** (Apache-2.0, `num_labels=6`, claim-only `statement` field) for the "credibility scale" deliverable. Always report the TF-IDF baseline first.

> License note: `GonzaloA/fake_news` is license-**unknown** (flag). For commercial/redistribution-clean training prefer `mohammadjavadpirhadi/fake-news-detection-dataset-english` (MIT) or `ErfanMoosaviMonazzah/…` (openrail). `chengxuphd/liar2` (Apache-2.0) is the license-clean credibility set.

**TF-IDF + LogReg BASELINE (the floor to beat — sklearn only, no torch):**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# input = (title + " " + text), source-boilerplate stripped (see leakage caveat)
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9,
        sublinear_tf=True, max_features=200_000, strip_accents="unicode")),
    ("clf", LogisticRegression(
        C=4.0, class_weight="balanced", max_iter=2000, n_jobs=-1)),
    # swap clf -> LinearSVC(C=1.0, class_weight="balanced") for the SVM floor
])
baseline.fit(train_texts, train_labels)
pred = baseline.predict(test_texts)
print(classification_report(test_labels, pred, digits=4))
print("macro-F1:", f1_score(test_labels, pred, average="macro"))
```

Expect TF-IDF+LogReg macro-F1 ~0.95+ in-domain on GonzaloA/WELFake — **this is a red flag, not success** (these sets are source-leaky). The transformer must beat the baseline on the **cross-domain** eval.

**HF Trainer config (core A):**

```python
model_config = {
    "model_name": "answerdotai/ModernBERT-base",   # T4 -> "distilbert-base-uncased"
    "num_labels": 2,                                # 6 for liar2 credibility
    "problem_type": "single_label_classification",
    "id2label": {0: "real", 1: "fake"},            # REPO convention (0=REAL,1=FAKE); see §9.
    "label2id": {"real": 0, "fake": 1},            # NB: GonzaloA native is 0=fake/1=real -> remap on load!
    "max_length": 512,        # ModernBERT can go 1024-8192; 512 = safe/cheap default
    "text_fields": ["title", "text"],              # concat title [SEP] text
}

training_args = {            # transformers.TrainingArguments
    "output_dir": "outputs/fakenews_cls",
    "num_train_epochs": 3,                 # few epochs — these datasets overfit fast
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

> **Label convention warning:** the repo standardizes on **`0=REAL, 1=FAKE`** (so `p_fake = P(label==1)`). `GonzaloA/fake_news` is natively `0=fake/1=real` — the loader MUST remap. Keep `id2label`/`label2id` consistent with whatever the loader produces and verify on a known row.

**Class-weight handling for imbalance** (needed for WELFake/LIAR2; GonzaloA is near-balanced):

```python
import torch, torch.nn as nn
class_weights = torch.tensor([w_real, w_fake])  # = n/(k*bincount), normalized
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights.to(out.logits.device))(out.logits, labels)
        return (loss, out) if return_outputs else loss
```

`compute_metrics` returns `{"f1": f1_score(y, p, average="macro"), "accuracy": ...}`. For LIAR2 6-way also report macro-F1 over labels 0–5.

**Long-article handling.** ModernBERT supports up to 8192 tokens — bump `max_length` to 1024/2048 on A100/H100 for long docs (cost grows ~length²). On RoBERTa/DistilBERT cap at 512 (`truncation=True`). Longformer (`allenai/longformer-base-4096`) is a slower alternative; not needed with ModernBERT.

#### Anti-overfitting / leakage checklist — CRITICAL for fake news

1. **SOURCE/STYLE LEAKAGE (the #1 trap):** GonzaloA/WELFake/ISOT leak *publisher & writing style*, not truth — all "real" rows are Reuters/AP wire copy (`"MOSCOW (Reuters) -…"`), all "fake" are blog-rant style. 99% in-domain F1 means the model learned "is this Reuters formatting," not veracity. **Mitigate:** (a) strip dateline/source boilerplate (`(Reuters)`, `WASHINGTON —`, bylines) before training; (b) report the **in-domain vs cross-domain gap** explicitly; (c) treat too-high in-domain F1 as a red flag.
2. **Cross-domain eval (mandatory):** train on `GonzaloA` → test on `LittleFish-Coder/Fake_News_GossipCop` (and/or WELFake/LIAR2). Or train PolitiFact-style → test GossipCop-style. The cross-domain macro-F1 is the **real** metric — expect a large drop.
3. **Dedup:** exact + near-dup (MinHash / normalized-hash) within and across splits before training; WELFake (4-corpus merge) and GonzaloA overlap in sources.
4. **Class balance:** GonzaloA ~balanced; WELFake 35K/37K fine; LIAR2 6-way imbalanced (class weights). Always report **macro-F1**, never accuracy alone.
5. **Few epochs / early stopping:** ≤3 epochs, `EarlyStoppingCallback(patience=2)` — these sets memorize in 1–2 epochs.
6. **No metadata leakage into LIAR/LIAR2:** drop `speaker, state_info, subject` and the per-speaker `*_counts` columns from features — they encode the label distribution. Train on `statement` (+ optionally `context`) only.

**GPU-profile table (auto-adapt; ModernBERT-base @ seq 512, core A):**

| GPU | Precision | train batch | eval batch | grad_accum | max_length | Notes |
|---|---|---|---|---|---|---|
| **H100 80GB** | bf16 | 64 | 128 | 1 | 1024–2048 | push long context |
| **A100 40/80GB** | bf16 | 32 | 64 | 1 | 512–1024 | primary target |
| **L4 24GB** | bf16 | 16 | 32 | 2 | 512 | bf16 ok on Ada |
| **T4 16GB** | **fp16** | 8 | 16 | 4 | 384–512 | base → `distilbert-base-uncased`; `fp16=True, bf16=False` |

Detect via `torch.cuda.get_device_capability()` (≥8.0 → bf16; T4 is 7.5 → fp16) and `torch.cuda.get_device_properties(0).total_memory` for batch size. Effective batch = batch × grad_accum ≈ 32–64 across all profiles.

### (B) Stance / NLI

**RECOMMENDATION: use pretrained zero-shot — do NOT fine-tune by default.** Use **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** (MIT, 184M, MNLI+FEVER+ANLI tuned, fits any GPU, zero training cost). Premise = retrieved evidence passage, hypothesis = the claim. Map `entailment→SUPPORTS`, `contradiction→REFUTES`, `neutral→NOT ENOUGH INFO`. **Read `model.config.id2label` at load** (verified `{0:entailment,1:neutral,2:contradiction}`) — never hardcode. **Scale-up:** swap to `…-large-mnli-fever-anli-ling-wanli` on A100/H100.

**Optional FINE-TUNE (the trainable stance showpiece).** Fine-tune `microsoft/deberta-v3-base` (or continue-tune the MoritzLaurer base) on **`copenlu/fever_gold_evidence`** (228K clean claim+gold-evidence S/R/NEI pairs) — the cleanest single verdict-head dataset — or on the NLI-formatted **`pietrolesci/nli_fever`** (`premise/hypothesis/label`, drop-in). Same `TrainingArguments` with `num_labels=3`, `lr=2e-5`, 2–3 epochs, `metric_for_best_model="f1"` macro. Optionally add `tals/vitaminc` for robustness to subtle edits. Net gain over zero-shot is usually small → **zero-shot is the default; fine-tune is optional.** Note the FEVER copyleft on any released stance model.

---

## 5. Agent architecture

Deterministic FSM (uniform `run(**kwargs)->dict` tools, `ToolTrace`, optional LLM **brain** that proposes only legal values with a rule fallback). States: **INGEST/NORMALIZE → PARSE/CLAIM-EXTRACT → CLASSIFY → CHECK-WORTHY? → RETRIEVE → STANCE/NLI → AGGREGATE → DECIDE/ABSTAIN → PRESENT**, with **ABSTAIN** reachable from any state. The brain **never decides flow** — at each decision point it may only PROPOSE a value from the legal set; on JSON-parse-fail / out-of-set / timeout, the deterministic rule fires.

### Decision-point table (≥5; each acts on intermediate outputs)

| ID | State | Predicate / threshold (rule) | Branches | Brain may propose (constrained) |
|---|---|---|---|---|
| **D1** | PARSE / CLAIM-EXTRACT | `len(tokens) ≤ 40` OR no body → **short-claim route**; else **article route** (central claim = title or top-sentence by TextRank/lead-3). `claim_len ≥ 5 tokens` else `INVALID_INPUT`. | short → CLASSIFY+CHECK; article → summarize-to-claim then continue | the *central claim string* from a fixed candidate set {title, lead sentence, top-TextRank}. Cannot invent text. Fallback = title. |
| **D2** | CHECK-WORTHY? | Skip gate = `(classifier_conf ≥ τ_skip=0.95) AND (checkworthy_score < τ_cw=0.5)` → present prior; else RETRIEVE. `checkworthy_score` = ClaimBuster-style heuristic (named entity / number / quantifier / verifiable predicate) or zero-shot "is this a checkable factual claim?". | skip → PRESENT(prior); proceed → RETRIEVE | `checkworthy ∈ {true,false}` + which entities/numbers make it checkable. Cannot raise its own confidence. Fallback = rule heuristic. |
| **D3** | RETRIEVE (coverage) | After **RRF** of BM25+dense top-k, keep evidence with reranker `score ≥ τ_rel=0.3`. Coverage OK if `n_relevant ≥ N_min=3`. Else **widen** (lower τ, expand query w/ entities, add corpus) up to `R_max=2` retries; still short → ABSTAIN `unverified (insufficient evidence)`. | enough → STANCE; thin → widen/retry; exhausted → ABSTAIN | query reformulation / expansion terms (entities, synonyms) + which corpus to add — from extracted entities only. Cannot change τ_rel/N_min. Fallback = entity-augmented query. |
| **D4** | AGGREGATE (verdict) | Per-evidence stance: entail→+1, contradiction→−1, neutral→0, weighted by reranker relevance: `S = Σ wᵢ·sᵢ`. Combine with prior: `score = α·(stance vote) + (1−α)·(prior signal)`, **α=0.6**. Verdict: `S>+θ → REAL/SUPPORTED`; `S<−θ → FAKE/REFUTED`; `|S|≤θ → UNVERIFIED` (**θ=0.2** of max). | REAL \| FAKE \| UNVERIFIED | per-evidence stance ∈ {support,refute,neutral} **only when NLI is near-tie** (margin<0.1); must cite the evidence span. Cannot override a confident NLI head or change α/θ. Fallback = NLI argmax + rule sum. |
| **D5** | DECIDE / ABSTAIN | Abstain if any: `verdict==UNVERIFIED`; OR `agreement = |support−refute|/n_evidence < τ_agree=0.4`; OR `final_conf < τ_present=0.55`; OR **prior↔stance conflict** (classifier FAKE but evidence SUPPORTS strongly → flag `conflict, needs human review`). Else PRESENT with label+confidence+citations+rationale. | present \| abstain("needs human review") | a NL rationale string + whether to surface a conflict note — bounded to citing retrieved evidence ids. Cannot flip abstain→present; thresholds fixed. Fallback = templated rationale. |

### Typed tool contracts

Uniform `run(**kwargs) -> {"ok": bool, "data": {...}, "meta": {...}}`; heavy deps (torch/transformers/sentence-transformers) imported **inside** `run()`; absent → fall back to TF-IDF/sklearn.

```python
IngestParse.run(raw: str, source_url: str|None) -> {
    "doc_type":"article"|"claim", "title":str, "body":str, "lang":str, "ok":bool}

ClaimExtract.run(title:str, body:str, doc_type:str) -> {           # D1
    "claim":str, "candidates":list[str], "entities":list[str],
    "method":"title|lead|textrank"}

ClassifyTool.run(text:str, mode:"binary"|"liar6") -> {             # P02 prior
    "label":"fake"|"real", "p_fake":float, "probs":dict[str,float],
    "backend":"transformer"|"tfidf_logreg"}                        # tfidf_logreg = no-torch

CheckWorthy.run(claim:str, classifier_conf:float) -> {            # D2
    "checkworthy":bool, "score":float, "has_entity":bool, "has_number":bool,
    "skip_retrieval":bool, "tau_skip":float, "tau_cw":float}

RetrieveEvidence.run(claim:str, entities:list[str], corpora:list[str],   # D3
    top_k:int=20, widen:int=0) -> {
    "evidence":list[{"id":str,"text":str,"source":str,
                     "bm25":float,"dense":float,"rrf":float,"rerank":float}],
    "n_relevant":int, "coverage_ok":bool, "tau_rel":float, "backend":"dense|tfidf"}

StanceNLI.run(claim:str, evidence:list[dict]) -> {               # premise=evidence, hypothesis=claim
    "stances":list[{"id":str,"label":"support"|"refute"|"neutral",
                    "probs":{"entail":float,"neutral":float,"contradiction":float},
                    "margin":float}],
    "backend":"deberta_mnli"|"bart_mnli"|"none", "id2label":dict}  # id2label read at runtime

AggregateVerdict.run(p_fake:float, stances:list[dict], alpha:float=0.6, theta:float=0.2) -> {  # D4
    "verdict":"REAL"|"FAKE"|"UNVERIFIED", "stance_score":float,
    "n_support":int, "n_refute":int, "n_neutral":int, "combined_conf":float}

DecideAbstain.run(verdict:str, combined_conf:float, n_evidence:int,        # D5
    n_support:int, n_refute:int, p_fake:float) -> {
    "action":"present"|"abstain", "final_label":str, "confidence":float,
    "agreement":float, "conflict":bool, "reason":str}

Present.run(final_label:str, confidence:float, evidence:list[dict],
    verdict_meta:dict, rationale:str) -> {
    "label":str, "confidence":float,
    "citations":list[{"id":str,"source":str,"snippet":str,"stance":str}],
    "rationale":str, "abstained":bool}

# Brain (optional; never sets flow):
Brain.propose(state:str, legal_values:list, context:dict) -> {"value":<legal>, "raw":str}
#   on JSON-parse fail / not-in-legal_values / timeout -> caller applies the deterministic rule.
ToolTrace.log(state, tool, inputs_hash, outputs, latency_ms, brain_proposal, rule_applied) -> None
```

**Engineering invariants.** (1) NLI label order from `model.config.id2label` at load — never hardcode. (2) Stance premise = evidence, hypothesis = claim. (3) **No-torch degradation:** `ClassifyTool`→TF-IDF+LogReg; `RetrieveEvidence`→TF-IDF cosine; `StanceNLI`→`backend:"none"` (lexical mock-stance: count support/refute terms) and `AggregateVerdict` falls back to prior-only, forcing `UNVERIFIED` unless the prior is extreme.

### Worked example (every decision fires)

> **Input (short claim):** "The WHO declared that drinking bleach cures COVID-19 in 2021."

1. **IngestParse** → `doc_type:"claim"`, body short. **CLASSIFY** → `label:"fake", p_fake:0.88, backend:"transformer"`.
2. **D1 / ClaimExtract** → 11 tokens ≤ 40 → short-claim route. `claim` = input; `entities:["WHO","COVID-19","2021","bleach"]`; method `title`.
3. **D2 / CheckWorthy** → `classifier_conf=0.88 < τ_skip=0.95` → do **not** skip; `checkworthy_score=0.92` (named org + medical predicate + date) → **proceed to RETRIEVE**.
4. **D3 / RetrieveEvidence** → RRF(BM25,dense) over evidence corpus returns 4 passages with `rerank ≥ 0.3` (WHO statements, fact-check articles). `n_relevant=4 ≥ N_min=3` → **coverage_ok**, no widen needed.
5. **StanceNLI** → premise=each passage, hypothesis=claim: 3× `contradiction→refute` (probs ~0.94), 1× `neutral`. Backend `deberta_mnli`.
6. **D4 / AggregateVerdict** → `S = −(0.94+0.91+0.88) + 0 ≈ −2.7` (weighted), prior `p_fake=0.88` agrees. `score` strongly negative, `|S| > θ` → **verdict `FAKE`/REFUTED**, `n_refute=3`, `combined_conf=0.91`.
7. **D5 / DecideAbstain** → `agreement = |0−3|/4 = 0.75 ≥ τ_agree=0.4`; `combined_conf=0.91 ≥ τ_present=0.55`; prior agrees (no conflict) → **action `present`**.
8. **Present** → `label:"fake", confidence:0.91`, **citations** = the 3 refuting WHO/fact-check passages (id+source+snippet+stance), rationale: "Multiple authoritative sources contradict this claim; no source supports it." `abstained:false`. Full `ToolTrace` attached.

> *Counter-example (abstain):* a novel claim with no corpus coverage → D3 widens twice, still `n_relevant<3` → **ABSTAIN** `unverified (insufficient evidence)`, never a guessed label.

---

## 6. Deployment

**Service surface (FastAPI).** One `fakenews-api` app exposes both the classifier and the agentic fact-checker. All heavy imports are lazy; with only numpy/pandas/sklearn installed, the service still boots and serves the TF-IDF+LogReg classifier and the TF-IDF evidence-retrieval fact-check.

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/classify` | POST | `{text}` or `{title, body}` | `{label:"fake"\|"real", probability:float, model_version:str}` |
| `/factcheck` | POST | `{claim, k?:int=5, threshold?:float}` | `{verdict:"real"\|"fake"\|"unverified", confidence:float, classifier_prior:float, evidence:[{text, source, stance:"support"\|"refute"\|"neutral", score}], rationale:str, decisions_trace:[...], abstained:bool}` |
| `/healthz` | GET | — | `{status:"ok", models_loaded:bool}` (503 until index/model warm) |
| `/version` | GET | — | `{api, classifier_version, nli_model, retriever, index_built_at, git_sha}` |

`/classify` returns the calibrated probability of the **fake** class + explicit `model_version`. `/factcheck` runs the FSM and **always** returns the evidence list and `decisions_trace`, even on abstain — transparency is non-optional (see §8).

**Example `/factcheck` response (abridged):**
```json
{
  "verdict": "fake", "confidence": 0.91, "classifier_prior": 0.88,
  "evidence": [
    {"text":"WHO has not recommended...","source":"who.int/...","stance":"refute","score":0.94}
  ],
  "rationale":"Multiple authoritative sources contradict this claim; none support it.",
  "decisions_trace":[{"state":"RETRIEVE","n_relevant":4,"coverage_ok":true}, ...],
  "abstained": false
}
```

**Gradio UI** (HF Space, Gradio SDK, under `ledinhminhquan`). Tab 1 **Classify**: paste article/headline → label + confidence bar + a visible "this is a style/probability signal, not a verdict" notice. Tab 2 **Fact-check**: paste a claim → verdict chip (`real`/`fake`/**`unverified`** with distinct neutral styling), confidence gauge, and a **highlighted evidence list** — each passage with source/citation, a stance badge (support=green / refute=red / neutral=grey), and score. Abstain renders prominently ("Not enough evidence — flagged for human review"), never a fake label by default.

**CLI.** `fakenews classify --text … | --file a.txt`; `fakenews factcheck --claim "…" --k 5 --json`; `fakenews serve`; `fakenews index build --corpus …`. JSON mode for scripting.

**Packaging & deploy.** Dockerfile (CPU base; optional CUDA layer auto-adapting H100/A100/L4/T4 via `torch.cuda` detection, falling back to CPU/TF-IDF). `pip install fakenews` = light core; `fakenews[torch]` / `[gpu]` extras pull the transformer stack. Same image runs API + Space.

**Latency & scalability.** Classifier path is fast (TF-IDF+LogReg sub-ms CPU; transformer ~10–30 ms GPU / ~100–300 ms CPU per doc) → run on every request. The retrieval+NLI fact-check is heavier (retrieval + k stance forward passes + optional rerank ≈ 0.3–2 s) → **gated behind `/factcheck` only**, cached by claim hash. Evidence index built offline (FAISS/numpy dense + sparse), versioned, memory-mapped; stance/rerank batch the k passages in one forward pass. Scale: stateless API replicas behind a load balancer, shared read-only index volume, request queue so the gated NLI path degrades to `"unverified — capacity"` rather than timing out.

**Versioning.** Stamp semantic model versions (`classifier_vX.Y`, pinned NLI/retriever ids **+ revisions/commit SHAs**, `index_built_at`) into every response and at `/version`. Pin HF **revisions**, not just repo names, so a Hub update can never silently change a verdict.

---

## 7. Metrics, baselines & evaluation

**Classification (primary).** On held-out test **and** a cross-domain split:
- **Macro-F1** — headline metric (handles fake/real and 6-way LIAR imbalance; `pants-fire` is rare).
- **Accuracy** + **per-class P/R/F1** — false positives (legit news flagged fake) are the costly error → report `real`-recall and `fake`-precision explicitly.
- **ROC-AUC** (binary) / one-vs-rest macro-AUC (6-way) — threshold-independent ranking.
- **ECE (Expected Calibration Error)** + reliability diagram — over-confidence silences real news. Apply temperature scaling on a held-out calibration split; report pre/post ECE.

**Fact-check (FEVER-style).**
- **Label accuracy** — SUPPORTS/REFUTES/NEI accuracy ignoring evidence.
- **FEVER score** — stricter: label correct *and* a complete evidence group retrieved.
- **Evidence retrieval recall@k** — report R@1/5/10 (official FEVER uses recall@5 sentences, @20 documents); measure against `BeIR/fever-qrels`.
- **Abstain quality** — coverage (fraction not abstained) vs. selective accuracy; a risk–coverage curve.

**Baselines (must beat all):** **majority class**; **TF-IDF+LogReg** (the no-torch core); **zero-shot** (`facebook/bart-large-mnli` for fake/real-as-hypothesis; `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for stance). Strong system = transformer fine-tune (`answerdotai/ModernBERT-base` / `distilbert-base-uncased` / `microsoft/deberta-v3-base`).

**Cross-domain generalization (called out explicitly):** train on `GonzaloA` (PolitiFact-style), test on `LittleFish-Coder/Fake_News_GossipCop`, and report the macro-F1 drop. A large drop is the signature of **source-style leakage** — this number is itself an ethics metric.

**Results table skeletons:**

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

## 8. Risks, limitations, ethics

The most ethically loaded project in the set. **Non-negotiable framing: the system flags content for human review; it never auto-removes, auto-blocks, or auto-censors, and it always shows its evidence.** It assists human fact-checkers/moderators; it is not an automated arbiter of truth.

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Censorship / free-speech chilling** | An automated "fake" label that triggers removal suppresses lawful speech and chills publishing | Output a **review flag + evidence**, never an enforcement action; **no auto-takedown** anywhere; verdicts are advisory |
| **False positives silencing real news** | Flagging legitimate journalism as fake is the high-cost error | Optimize the operating point for high `real`-recall; report per-class costs; **abstain** rather than guess; human sign-off before any action |
| **Political bias in labels/data** | LIAR/PolitiFact labels encode the fact-checker's editorial judgment + topic/speaker skew | Document label provenance; report performance sliced by topic/speaker/source; never present output as neutral ground truth |
| **Source-style leakage** | Model learns "this outlet ⇒ fake," not truth; collapses on new outlets, entrenches bias against named sources | Measure cross-domain drop (PolitiFact→GossipCop); strip/ablate source/style features; prefer the **evidence-grounded** verdict; treat `classifier_prior` as a weak prior only |
| **Adversarial paraphrase / evasion** | Bad actors paraphrase to flip the label; classifiers are brittle | Evidence-based fact-check is more robust than style classification; keep an adversarial/paraphrase eval set; don't rely on the classifier alone |
| **Automation bias / over-trust** | Reviewers rubber-stamp the model; confident-and-wrong outputs propagate | Calibration (ECE) + mandatory confidence display; **abstain on uncertainty**; UI states it's a decision-support signal with evidence required to be read |
| **Dual-use** | The model can be probed to craft detection-evading text | Don't ship an "evasion score"; rate-limit; log; gate adversarial tooling |
| **Stale / out-of-scope evidence** | Index goes out of date; novel claims have no coverage | Version + timestamp the index (`/version`); **abstain** when retrieval coverage is insufficient rather than fabricate |
| **Hallucinated / misattributed citations** | A fact-check is only trustworthy if its citations are real | Citations are **extracted verbatim** from the retrieved corpus with source IDs, never generated; `decisions_trace` lets a reviewer verify each one |

**Limitations to state plainly:** English-centric data; PolitiFact/GossipCop domain skew (US politics + celebrity); "fake/real" is a coarse proxy for a spectrum (LIAR's 6-way captures this better); the system verifies against a **fixed corpus** and cannot know facts outside it; **`unverified` means *insufficient evidence*, not *true*.** The classifier detects *style/source patterns correlated with fakeness*, which is **not** detecting falsehood — the agentic evidence layer compensates, and `classifier_prior` vs. evidence `verdict` are reported **separately** so a reviewer sees disagreement.

**Design commitments baked into the deliverable:** mandatory transparency + verbatim citations (every verdict ships its evidence + `decisions_trace`); abstain under uncertainty (`unverified` is first-class, not a failure); assist rather than replace human judgment; **never take an irreversible action on content automatically.**

---

## 9. Repo module map + reuse audit

Target package `src/fakenews/` mirrors the `resume_screener` / `scisearch` / `citerec` / `meetingai` layout (P11 dir currently holds only an empty `fakenews/__init__.py`).

### Canonical data model

```python
@dataclass
class NewsItem:
    id: str            # "liar2-train-0007" / "gonzaloa-train-42"
    title: str | None
    text: str          # article body OR short claim (classifier input)
    label: int         # REPO CONVENTION: 0 = REAL, 1 = FAKE
    source: str | None # "liar2" | "gonzaloa" | "fakenewsnet" | "synthetic"
    claim: str | None  # short claim to fact-check (== text for claims; headline for articles)
    credibility: int | None  # OPTIONAL 6-way LIAR label for the credibility-scale head

@dataclass
class Evidence:        # one retrievable snippet
    id: str; text: str; source: str; url: str | None; date: str | None

@dataclass
class ClaimCase:       # claim + gold verdict for FEVER-style eval
    id: str; claim: str
    gold_verdict: str  # "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO"
    gold_evidence_ids: list[str]
```

**Stance/NLI label bridge:** `entailment→SUPPORTS`, `contradiction→REFUTES`, `neutral→NOT_ENOUGH_INFO` (premise=evidence, hypothesis=claim).

**Dataset → canonical mapping (binary collapse for LIAR family):** `{pants-fire,false,barely-true}→FAKE(1)`, `{half-true,mostly-true,true}→REAL(0)`. Each loader applies an explicit per-source `label_map` to the repo convention `0=REAL,1=FAKE`. LIAR2 `justification` → `Evidence` corpus rows (free in-domain evidence).

### Module map

| Module (in `fakenews/`) | Responsibility |
|---|---|
| `config.py` | AppConfig dataclasses — news/evidence paths, model ids, `stance.model_id`, `factcheck.abstain_threshold`, thresholds (τ_skip, τ_cw, τ_rel, N_min, R_max, α, θ, τ_agree, τ_present) |
| `logging_utils.py` | `get_logger`, JSONL logger |
| `cli.py` | subcommands `classify, factcheck, train-classifier, train-stance, eval, serve, report, index build` |
| `api/` (`main.py, app_combined.py, dependencies.py, schemas.py, ui.py`) | FastAPI `/classify`+`/factcheck`+`/healthz`+`/version`; Pydantic schemas; Gradio UI |
| `agent/state.py` | `Action ∈ {FAKE, REAL, UNVERIFIED, ABSTAIN, NEEDS_EVIDENCE}`; `ToolTrace`; `AgentState{claim,label,label_conf,evidence[],stances[],verdict,citations[],confidence}` |
| `agent/policy.py` | deterministic rules (D1–D5 thresholds; abstain rule) |
| `agent/tools.py` | tool wrappers: `ClassifierTool, ClaimExtractorTool, EvidenceRetrieverTool, StanceTool, VerdictTool` |
| `agent/factcheck_agent.py` + `llm_orchestrator.py` | FSM spine CLASSIFY→EXTRACT→RETRIEVE→STANCE→VERDICT→(ABSTAIN?); singleton-load-with-fallback; optional brain |
| `models/baseline_tfidf.py` | TF-IDF+LogReg binary real/fake (CPU no-torch floor) |
| `models/classifier.py` | transformer fine-tune wrapper (lazy torch; `num_labels` 2 or 6) |
| `models/model_registry.py` | version/artifact tracking |
| `models/{retriever.py, bm25.py, vector_store.py}` | evidence retriever (lazy sentence-transformers) |
| `factcheck/hybrid.py` | RRF + minmax_norm fusion of BM25+dense |
| `factcheck/stance.py` | **NEW** — zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; lazy import; lexical mock-stance fallback; entail/contra/neutral → SUPPORTS/REFUTES/NEI |
| `factcheck/verdict.py` | **NEW** — aggregate stances → verdict + confidence + citations; ABSTAIN logic |
| `factcheck/claim_extractor.py` | **NEW (light)** — headline/first-sentence claim extraction (rule-based; optional LLM) |
| `data/news_loaders.py` | **NEW** — LIAR2/GonzaloA/WELFake loaders + per-source `label_map` + binary collapse |
| `data/evidence_corpus.py` | build evidence corpus from LIAR2 `justification` + articles |
| `data/samples.py` | **NEW** — offline tiny fallback (below) |
| `training/train_classifier.py` | binary/6-way fine-tune; HW auto-adapt |
| `training/train_stance.py` | **NEW (optional)** — FEVER fine-tune on `pietrolesci/nli_fever` / `copenlu/fever_gold_evidence` |
| `training/{evaluate.py, metrics.py}` | classifier acc/macro-F1 (port) + **NEW** FEVER score + label-accuracy + recall@k |
| `training/tune.py` | HPO loop |
| `autoreport/{report_pdf,slides_pptx,charts,artifact_loader}.py` | charts → confusion matrix + verdict distribution |
| `monitoring/drift_report.py` | input-text / label drift |
| `grading/checklist.py` | rubric: classifier beats baseline, fact-check cites evidence, abstains |
| `automation/autopilot.py` | end-to-end orchestration |
| `analysis/{error_analysis,latency,fairness}.py` | fairness → per-topic / per-source error slices |

### PORT-FROM (P02/P08/P09/P10) vs BUILD-NEW audit

"Port ~verbatim" = copy + rename imports/strings. "Adapt" = port skeleton, swap domain logic. "New" = write fresh.

| Component | Disposition | Source |
|---|---|---|
| `config.py`, `cli.py`, `api/*` | Port ~verbatim (adapt fields/schemas) | P02/P09/P10 |
| `logging_utils.py`, `models/model_registry.py`, `autoreport/*`, `monitoring/drift_report.py`, `automation/autopilot.py` | **Port verbatim** | P02/P09 |
| `agent/state.py`, `agent/policy.py`, `agent/tools.py`, `agent/factcheck_agent.py`, `llm_orchestrator.py` | Port ~verbatim spine + graceful fallback; **adapt** Action enum / tool set / spine | P02 `agent/*`, `screening_agent.py` |
| `models/baseline_tfidf.py`, `models/classifier.py` | **Port ~verbatim** (swap num_labels/base) | P02 |
| `models/{retriever,bm25,vector_store}.py`, `factcheck/hybrid.py` (RRF) | **Port ~verbatim / verbatim** (corpus = evidence) | P08 `search/hybrid.py`, P08/P09 retrieval |
| `training/{train_classifier,evaluate,metrics,tune}.py` | Port ~verbatim + **adapt** (add FEVER metrics) | P02 + P09 metrics |
| `grading/checklist.py`, `analysis/*` | Port ~verbatim (adapt items) | P02 |
| **`factcheck/stance.py`, `factcheck/verdict.py`, `factcheck/claim_extractor.py`, `data/news_loaders.py`, `data/samples.py`, `training/train_stance.py`, FEVER metrics in `metrics.py`** | **BUILD NEW** | (claim_extractor reuses P10 `actions/extractor.py` heuristics; train_stance scaffolds from P09 `train_reranker.py`) |

**Net new code:** `factcheck/{stance,verdict,claim_extractor}.py`, `data/news_loaders.py`, `data/samples.py`, optional `training/train_stance.py`, FEVER metrics. Everything else (trainable classifier core, full retrieval stack, all ops scaffolding) ports largely verbatim.

**Reuse source paths (absolute):**
- `D:\NLP Industry Projects\02_Resume_Screening_Ranking\src\resume_screener\` — agent spine (`agent\screening_agent.py`, `agent\state.py`), classifiers (`models\baseline_tfidf.py`, `models\classifier.py`), `data\samples.py`.
- `D:\NLP Industry Projects\08_Scientific_Literature_Search\src\scisearch\search\hybrid.py` (RRF) and `…\models\{retriever,bm25,vector_store}.py`.
- `D:\NLP Industry Projects\09_Citation_Recommendation\src\citerec\` — corpus/metrics, `data\download_dataset.py`.
- `D:\NLP Industry Projects\10_Meeting_Minutes\src\meetingai\` — synthetic/extractor heuristics.
- Target (currently empty): `D:\NLP Industry Projects\11_Fake_News_Detection\src\fakenews\`.

### Offline-fallback design

Commit a fully synthetic, license-safe `fakenews/data/samples.py` (analogue of P02 `samples.py` + P09 `SEED_PAPERS`) so the **whole pipeline runs with only numpy/pandas/sklearn, no network**:
- `SAMPLE_NEWS`: ~36 short labeled items, balanced `label∈{0,1}` across politics/health/science/finance, each `{id,title,text,label,source:"synthetic",claim}` — enough to fit TF-IDF+LogReg and produce non-trivial macro-F1.
- `SAMPLE_EVIDENCE`: ~16 `Evidence` snippets (some supporting, some refuting, some off-topic) so TF-IDF/RRF retrieval returns mixed hits.
- `SAMPLE_CLAIMS`: ~6 `ClaimCase` with gold FEVER verdicts (≥1 SUPPORTS, ≥1 REFUTES, ≥1 NOT_ENOUGH_INFO) + `gold_evidence_ids` → exercises stance aggregation, the **ABSTAIN** branch, and FEVER-style scoring with zero network.

Also commit `tests/fixtures/fake_news_tiny.csv` (~10 rows, `title,text,label`, repo convention **0=real, 1=fake** — remap if mirroring GonzaloA): ~5 neutral wire-service-style real + ~5 sensational hyper-partisan fake. When `transformers` is absent, `StanceNLI` returns `backend:"none"` and a lexical mock-stance (count support/refute terms) keeps the verdict aggregator + metrics runnable end-to-end.

---

*Build order:* (1) data model + `samples.py` fallback → (2) port baseline+classifier (P02) → (3) port retrieval+RRF (P08/P09) → (4) build `factcheck/{claim_extractor,stance,verdict}` → (5) port agent FSM + wire D1–D5 → (6) API/Gradio/CLI/Docker/Space → (7) metrics + cross-domain eval → (8) autoreport/monitoring/grading. Default stack: TF-IDF+LogReg floor → ModernBERT-base binary fine-tune on `GonzaloA` (boilerplate-stripped) → cross-domain eval on GossipCop + WELFake → zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` stance with abstention; LIAR2 6-way as the credibility-scale stretch config.