# P11 — Data Description (Section I.4)

> **Project:** P11 — Fake News & Misinformation Detection System
> **Course:** NLP in Industry — final assignment
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Package:** `fakenews`
> **Scope of this document:** the data that feeds the two halves of the system — (1) the trainable **fake-news classifier** (fast prior, `P(fake)`) and (2) the agentic, evidence-grounded **fact-check** (retrieve → stance/NLI → verdict → abstain). Every Hugging Face id, license, size, and label convention below is taken from the verified stack in `docs/DESIGN_BRIEF.md`. License hygiene flags (⚠️ / ⛔ / ✅) are carried through verbatim.

---

## 1. Overview — two data regimes, one label discipline

The system consumes two structurally different kinds of data, and the single most important rule that ties them together is **label discipline**: every loader normalizes to one internal convention before anything downstream runs.

| Regime | Purpose | Internal label space | Primary dataset |
|---|---|---|---|
| **Classification corpora** | Train/eval the binary real/fake classifier (the *prior*) and the optional 6-way credibility head | **`0 = real`, `1 = fake`** (so `p_fake = P(label == 1)`) | `GonzaloA/fake_news` |
| **Fact-check / FEVER corpora** | Provide the evidence corpus + claim→verdict supervision for the agentic layer | `SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO` (NLI bridge: entail→SUPPORTS, contra→REFUTES, neutral→NEI) | `fever/fever`, `BeIR/fever` |

> **Internal convention (repo-wide):** `NewsItem.label` is **`0 = REAL`, `1 = FAKE`** (see `fakenews` data model). This is the single source of truth. The classifier predicts `p_fake = P(label==1)`. **Several source mirrors use the opposite polarity** — see §4, the label-polarity gotcha, which is the central data-engineering hazard of this project.

Everything in this project is **English-language**. There is no multilingual claim — that is stated explicitly as a limitation (§9).

---

## 2. Classification datasets (the trainable classifier core)

These supply `(title, text, label)` rows for the real/fake classifier. The **primary** training set is `GonzaloA/fake_news`; the others are license-cleaner alternatives or cross-domain test sets.

| HF id | Status | License | Size / splits | Key columns | Native polarity | Role in P11 |
|---|---|---|---|---|---|---|
| **`GonzaloA/fake_news`** | VERIFIED (preview) | **unknown** ⚠️ flag | 40.6K — train 24.4K / val 8.1K / test 8.1K | `Unnamed:0, title, text, label(int)` | **0 = fake, 1 = real** | **PRIMARY binary** — clean, pre-split, `title`+`text`, parquet (loads torch-free) |
| `ErfanMoosaviMonazzah/fake-news-detection-dataset-English` | VERIFIED (preview) | **openrail** | 44.3K — train 30.0K / val 6.0K / test 8.3K | `Unnamed:0, title, text, subject, date, label(int)` | **0 = fake, 1 = real** | ISOT-style, pre-split — most license-clean of the large binary mirrors |
| `mohammadjavadpirhadi/fake-news-detection-dataset-english` | VERIFIED | **MIT** ✅ | 10K–100K (viewer renamed/broken; repo exists) | text-classification binary | binary | **MIT-clean binary alt** — prefer when license hygiene is paramount |
| **`chengxuphd/liar2`** | VERIFIED (preview) | **apache-2.0** ✅ | 23.0K — train 18.4K / val 2.3K / test 2.3K | `id, label(0–5), statement, date, subject, speaker, speaker_description, state_info, *_counts(6), context, justification` | 6-way credibility (0–5) | **SECONDARY (6-way credibility)** head; `justification` → free evidence rows |
| `davanstrien/WELFake` | VERIFIED (schema) | **unknown** ⚠️ flag (source ≈ CC-BY 4.0 per IEEE DataPort — verify) | 72.1K single train (35K real / 37K fake) | `title, text, label` | **0 = fake, 1 = real** | Larger binary alt (4-corpus merge); make own split + dedup |
| `LittleFish-Coder/Fake_News_GossipCop` | VERIFIED (schema) | **apache-2.0** ✅ | 12.7K — train 9,988 / test 2,672 | `text, label(int)` (+ embeddings, 1.3 GB train parquet) | **0 = real, 1 = fake** | FakeNewsNet-GossipCop mirror — **cross-domain eval target** |
| `LittleFish-Coder/Fake_News_PolitiFact` | VERIFIED (schema) | **apache-2.0** ✅ | 483 — train 381 / test 102 | `text, label(int)` (+ embeddings) | **0 = real, 1 = fake** | FakeNewsNet-PolitiFact content mirror (use `text`+`label`) |
| `mrm8488/fake-news` | VERIFIED (preview) | not declared ⚠️ | 44.9K single train | `text, label(int)` (no title, no splits) | **1 = fake, 0 = real** | binary eval/aux only — **opposite polarity** |
| `rickstello/FakeNewsNet` | VERIFIED (schema) | **cc** (generic) ⚠️ | 23.2K single train | `title, news_url, source_domain, tweet_num, real(0/1)` | titles only (`real` col) | weak — title+URL only, no body |
| `ucsbai/liar` | VERIFIED (repo); viewer broken (legacy `.py` loader) | **unknown** ⚠️ flag | 12.8K — train 10.24K / val 1.28K / test 1.27K | `statement, label(0–5), subject, speaker, …, context` | 6-way (0–5) | original LIAR — **prefer LIAR2**; load via `trust_remote_code=True` if needed |

**Why `GonzaloA/fake_news` is primary:** it is the closest analog to the P02 résumé classifier — clean `title`+`text`, already split train/val/test, distributed as parquet so it loads without torch, and near class-balanced. The classifier concatenates the two text fields as `title [SEP] text`.

**License caveat on the primary set:** `GonzaloA/fake_news` is license-**unknown** (⚠️). For a redistribution-clean training run, swap to `mohammadjavadpirhadi/…` (MIT ✅) or `ErfanMoosaviMonazzah/…` (openrail). `chengxuphd/liar2` (Apache-2.0 ✅) is the license-clean choice for the credibility scale.

---

## 3. Fact-check / FEVER datasets (the agentic evidence layer)

These supply the **evidence corpus** the retriever searches and the **claim→verdict** supervision for evaluation (and the optional stance fine-tune). They are *not* used to train the classifier.

| HF id | Status | License | Size / splits | Schema | Labels / role |
|---|---|---|---|---|---|
| **`fever/fever`** (bare `fever` does not resolve) | VERIFIED; viewer 501 (loader script) | **cc-by-sa-3.0 + gpl-3.0** ⚠️ copyleft | ~185K claims; configs `v1.0` / `v2.0` / `wiki_pages` | `id, label, claim, evidence_*` (+ `wiki_pages` corpus) | SUPPORTS / REFUTES / NEI — evidence corpus + verification |
| **`BeIR/fever`** + `BeIR/fever-qrels` | VERIFIED | **cc-by-sa-4.0** ⚠️ share-alike | — | Wikipedia-abstract corpus + qrels | **measure evidence recall@k** |
| `copenlu/fever_gold_evidence` | VERIFIED | cc-by-sa-3.0 + gpl-3.0 ⚠️ | 228.3K train / 15.9K val / 16.0K test | `claim, label(str), evidence(list), id, verifiable, original_id` | SUPPORTS/REFUTES/NEI — cleanest single set to fine-tune a claim+evidence verdict head |
| `pietrolesci/nli_fever` | VERIFIED (viewer ok) | (parent FEVER cc-by-sa-3.0 + gpl-3.0) ⚠️ | 208K–248K train / ~20K dev / ~20K test | `premise, hypothesis, label{entailment,neutral,contradiction}` | FEVER reframed as NLI — drop-in for optional stance fine-tune |
| `tals/vitaminc` | VERIFIED | cc-by-sa-3.0 ⚠️ | 370.7K / 63.1K / 55.2K | `claim, evidence, label, page, revision_type, FEVER_id` | SUPPORTS/REFUTES/NEI — contrastive robustness add-on |
| `ImperialCollegeLondon/health_fact` (PUBHEALTH) | VERIFIED; viewer disabled | **mit** ✅ | 10K–100K | `claim, main_text, explanation, label, sources, subjects` | true/false/unproven/mixture (4-way) — health domain |
| `tdiggelm/climate_fever` | VERIFIED | **unknown** ⚠️ flag | 1.5K test only | `claim_id, claim, claim_label, evidences(list)` | SUPPORTS/REFUTES/NEI/**DISPUTED** — `DISPUTED` motivates the contradictory-evidence abstain branch |
| `BeIR/scifact` | VERIFIED | cc-by-sa-4.0 ⚠️ | 1K–10K | BEIR triple (`corpus`, `queries`, `qrels`) | retrieval relevance — share-alike alt to `allenai/scifact` |

> The agentic layer's **stance/NLI verdict** is produced zero-shot by `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT ✅), so FEVER data is needed mainly as the **evidence corpus** and as **evaluation** ground truth (label accuracy, FEVER score, evidence recall@k via `BeIR/fever-qrels`). A FEVER fine-tune (on `copenlu/fever_gold_evidence` or `pietrolesci/nli_fever`) is *optional* and would inherit the copyleft flag.

---

## 4. ⚠️ The label-polarity gotcha — normalize on load

**This is the single most error-prone aspect of the data.** The phrase "label 1" means **opposite things** across mirrors. A silent mismatch flips every prediction and produces a model that looks trained but is exactly wrong. Therefore **every loader applies an explicit per-source `label_map`** to the internal convention `0 = REAL, 1 = FAKE`; no loader is allowed to assume polarity.

### Per-source polarity table

| Source | Native meaning of label | Mapping applied to reach internal `0=real, 1=fake` |
|---|---|---|
| `GonzaloA/fake_news` | **0 = fake, 1 = real** | **invert** → `{0→1, 1→0}` |
| `ErfanMoosaviMonazzah/…` | 0 = fake, 1 = real | invert → `{0→1, 1→0}` |
| `davanstrien/WELFake` | 0 = fake, 1 = real | invert → `{0→1, 1→0}` |
| `LittleFish-Coder/Fake_News_GossipCop` | **0 = real, 1 = fake** | identity → `{0→0, 1→1}` |
| `LittleFish-Coder/Fake_News_PolitiFact` | 0 = real, 1 = fake | identity → `{0→0, 1→1}` |
| `mrm8488/fake-news` | **1 = fake, 0 = real** | identity → `{0→0, 1→1}` |
| `rickstello/FakeNewsNet` | `real` col: 1 = real, 0 = fake | invert the `real` column → `{1→0, 0→1}` |
| `chengxuphd/liar2` / `ucsbai/liar` | 6-way credibility 0–5 (no binary polarity) | binary collapse, see below |

> **Note the trap directly:** `GonzaloA` (primary) is `0=fake`, but `LittleFish-Coder/*` (cross-domain test) is `0=real`. If you train on GonzaloA and evaluate on GossipCop **without** normalizing both, the cross-domain accuracy will read as near-zero — not because the model failed, but because the labels are flipped. The per-source map prevents this.

### 6-way → binary collapse (LIAR family)

For the LIAR / LIAR2 credibility scale, the binary collapse used when these feed the binary classifier is:

```
{pants-fire, false, barely-true} → FAKE (1)
{half-true, mostly-true, true}   → REAL (0)
```

The full 6-way label (`credibility ∈ {0..5}`) is kept separately on `NewsItem.credibility` for the optional 6-way head; it is **not** silently merged into the binary label.

### Verification discipline

Each loader is expected to **verify polarity on a known row** after mapping (e.g. a Reuters wire item must come out `label == 0` real; a sensational hyper-partisan item must come out `label == 1` fake). The `id2label` / `label2id` on the model config (`{0: "real", 1: "fake"}`) is kept consistent with whatever the loader produces.

---

## 5. Sizes, languages, and field shapes

- **Language:** English only across all classification and fact-check corpora. No non-English split is claimed.
- **Primary training volume:** `GonzaloA/fake_news` = **40.6K** rows (train 24.4K / val 8.1K / test 8.1K), `title`+`text`+`label`.
- **Cross-domain test volume:** `LittleFish-Coder/Fake_News_GossipCop` = **12.7K** (train 9,988 / test 2,672); `…_PolitiFact` = **483** (381 / 102).
- **Credibility set:** `chengxuphd/liar2` = **23.0K** (train 18.4K / val 2.3K / test 2.3K), claim-only `statement` field (+ `justification` evidence).
- **Evidence / FEVER:** `fever/fever` ≈ **185K** claims plus a `wiki_pages` evidence corpus; `BeIR/fever` is the Wikipedia-abstract corpus used to measure **evidence recall@k** against `BeIR/fever-qrels`.

**Field shapes used downstream** (canonical data model in `fakenews`):

```python
@dataclass
class NewsItem:
    id: str            # "gonzaloa-train-42" / "liar2-train-0007"
    title: str | None
    text: str          # article body OR short claim (classifier input)
    label: int         # INTERNAL CONVENTION: 0 = REAL, 1 = FAKE
    source: str | None # "gonzaloa" | "liar2" | "fakenewsnet" | "synthetic"
    claim: str | None  # short claim to fact-check (== text for claims; headline for articles)
    credibility: int | None  # OPTIONAL 6-way LIAR label for the credibility head

@dataclass
class Evidence:        # one retrievable snippet
    id: str; text: str; source: str; url: str | None; date: str | None

@dataclass
class ClaimCase:       # claim + gold verdict for FEVER-style eval
    id: str; claim: str
    gold_verdict: str  # "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO"
    gold_evidence_ids: list[str]
```

Loaders live in `fakenews/data/news_loaders.py` (classification + per-source `label_map` + binary collapse) and `fakenews/data/evidence_corpus.py` (builds the evidence corpus from LIAR2 `justification` + article bodies).

---

## 6. Preprocessing

The preprocessing pipeline is deliberately conservative — aggressive cleaning is what *creates* the source-style leakage problem (§8), so the steps below are designed to *reduce* leakage, not amplify it.

| Step | What it does | Where / why |
|---|---|---|
| **Title + text concat** | `title [SEP] text` becomes the single classifier input; for short claims `text` is the claim itself | matches the brief's `text_fields = ["title", "text"]`; a headline alone is often the strongest signal |
| **Source/dateline boilerplate strip** | Remove wire-service datelines and bylines (`(Reuters)`, `WASHINGTON —`, `MOSCOW (Reuters) -…`) **before** training | the #1 anti-leakage step — all "real" rows in ISOT/GonzaloA/WELFake are Reuters/AP formatting; stripping it forces the model toward content, not formatting |
| **Truncation** | `truncation=True`, `max_length=512` default (ModernBERT can go 1024–8192 on A100/H100; RoBERTa/DistilBERT capped at 512) | cost grows ~length²; 512 is the safe/cheap floor |
| **Dedup (exact + near-dup)** | MinHash / normalized-hash dedup **within and across splits** before training | WELFake (4-corpus merge) and GonzaloA overlap in sources; un-deduped overlap inflates in-domain scores |
| **LIAR metadata drop** | Drop `speaker, state_info, subject` and the per-speaker `*_counts` columns from features | these encode the label distribution directly — keeping them is leakage; train on `statement` (+ optionally `context`) only |
| **Evidence snippeting** | LIAR2 `justification` and FEVER `wiki_pages` are split into short `Evidence` rows with stable ids + source | gives the retriever in-domain evidence with verbatim, citable spans |

---

## 7. Splits, imbalance, and noise handling

### Splits

- **Pre-split sources are used as-is:** `GonzaloA/fake_news` (24.4K / 8.1K / 8.1K), `chengxuphd/liar2` (18.4K / 2.3K / 2.3K), `ErfanMoosaviMonazzah/…` (30.0K / 6.0K / 8.3K), `LittleFish-Coder/*` (train/test).
- **Single-train sources** (`davanstrien/WELFake`, `mrm8488/fake-news`) get a **custom stratified split + dedup** before use.
- A **held-out calibration split** is carved out for temperature scaling / ECE (the classifier's calibration is an ethics metric — over-confidence silences real news).
- **Cross-domain split is mandatory and separate:** train on `GonzaloA` (PolitiFact-style), test on `LittleFish-Coder/Fake_News_GossipCop`. This is *the* honest generalization number (see §8).

### Imbalance

- `GonzaloA` is near-balanced; `WELFake` (35K real / 37K fake) is mild; **LIAR2 6-way is imbalanced** (`pants-fire` is rare).
- Handling: **`class_weight="balanced"`** for the TF-IDF+LogReg baseline; a **`WeightedTrainer`** with `class_weights = n / (k · bincount)` for the transformer; **macro-F1 reported as the headline metric**, never accuracy alone.

### Noise

- **Few epochs + early stopping** (≤3 epochs, `EarlyStoppingCallback(patience=2)`): these datasets memorize in 1–2 epochs, so long training overfits noise.
- **Dedup** (above) removes the most damaging near-duplicate noise.
- **Abstain as noise insurance** on the fact-check side: when retrieval coverage is thin (`n_relevant < N_min = 3`) the agent returns `unverified` rather than a guess (decision D3).

---

## 8. The source-style-leakage limitation (and how the evidence layer compensates)

**This is the defining data limitation of the project, and it is called out as an ethics metric, not a footnote.**

**What it is.** On corpora like ISOT / GonzaloA / WELFake, the "real" class is wire-service copy (Reuters/AP) and the "fake" class is blog-rant style. A classifier scoring **~0.95+ macro-F1 in-domain** has very likely learned *"is this Reuters formatting?"* — the **outlet and writing style**, not the **truth** of the content. **High in-domain F1 is therefore a red flag, not a success.**

**Why it matters.** A model that keys on outlet style:
- collapses on new outlets it never saw,
- entrenches bias against named sources, and
- can be defeated by adversarial paraphrase (rewrite fake content in wire-service style → label flips).

**How the data + system design compensate:**

1. **Boilerplate stripping** (§6) removes the most obvious style tells before training.
2. **Mandatory cross-domain eval:** train on `GonzaloA` → test on `LittleFish-Coder/Fake_News_GossipCop`; **report the macro-F1 drop explicitly**. A large drop is the *signature* of source-style leakage — the cross-domain number is the one that counts.
3. **The classifier is only a weak prior.** The agentic evidence layer is the design's answer to leakage: it retrieves evidence (BM25 + dense `all-MiniLM-L6-v2` + RRF), runs **stance/NLI per evidence** (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), and lets **evidence dominate** the verdict (`α = 0.6` on the stance vote; the prior is a soft nudge, `1−α`).
4. **The two signals are reported separately.** Every fact-check response carries `classifier_prior` *and* the evidence `verdict` as distinct fields, so a reviewer sees when they disagree — and a prior↔stance conflict triggers a "needs human review" flag (decision D5) rather than a silent decision.

In short: the data is leaky by nature, the system treats that as a known hazard, measures it (cross-domain drop), and routes the trustworthy decision through **evidence**, not style.

---

## 9. Offline seed dataset (zero-network fallback)

So the **entire pipeline runs with only `numpy`, `pandas`, `scikit-learn` — no torch, no network**, the package ships a fully synthetic, license-safe seed in `fakenews/data/samples.py`. This is what lets CI, graders, and offline demos exercise the classifier, retrieval, stance aggregation, and the **ABSTAIN** branch end-to-end.

| Seed object | Count | Contents | Exercises |
|---|---|---|---|
| `SAMPLE_NEWS` | **~40** short labeled items (`{id, title, text, label, source:"synthetic", claim}`), balanced `label ∈ {0,1}` across politics / health / science / finance | enough to fit TF-IDF+LogReg and produce non-trivial macro-F1 | classifier + baseline floor |
| `SAMPLE_EVIDENCE` | **~16** `Evidence` snippets — some supporting, some refuting, some off-topic | mixed BM25 / TF-IDF / RRF retrieval hits | retrieval + coverage gate (D3) |
| `SAMPLE_CLAIMS` | **~8** `ClaimCase` with gold FEVER verdicts (≥1 SUPPORTS, ≥1 REFUTES, ≥1 NOT_ENOUGH_INFO) + `gold_evidence_ids` | stance aggregation, ABSTAIN, FEVER-style scoring with zero network | stance/verdict (D4) + abstain (D5) |

Additionally, `tests/fixtures/fake_news_tiny.csv` (~10 rows: `title, text, label`, **internal convention 0=real, 1=fake**) gives ~5 neutral wire-service-style real rows + ~5 sensational hyper-partisan fake rows — remap if mirroring a `0=fake` source like GonzaloA.

**No-torch degradation path** (so the seed alone is sufficient):
- `ClassifyTool` → TF-IDF + LogReg (no torch).
- `RetrieveEvidence` → BM25 / TF-IDF cosine over the seed evidence.
- `StanceNLI` → `backend:"none"` with a lexical-overlap mock-stance (count support/refute terms).
- `AggregateVerdict` → falls back to prior-only and forces `UNVERIFIED` unless the prior is extreme — i.e. the system **abstains by default** when it has no real evidence model, which is the correct conservative behavior.

---

## 10. Licensing summary and flags

| Class | Datasets | Action |
|---|---|---|
| ✅ **Cleanly permissive** | `chengxuphd/liar2` (Apache-2.0), `mohammadjavadpirhadi/…` (MIT), `ImperialCollegeLondon/health_fact` (MIT), `LittleFish-Coder/*` (Apache-2.0) | preferred; safe to redistribute with attribution |
| ⚠️ **Unknown / unstated** | `GonzaloA/fake_news` (PRIMARY), `davanstrien/WELFake`, `ucsbai/liar`, `mrm8488/fake-news` (also opposite polarity), `tdiggelm/climate_fever`, `rickstello/FakeNewsNet` (generic `cc`) | research/eval only; **verify before any redistribution**; prefer a clean alt for shipped artifacts |
| ⚠️ **openrail** | `ErfanMoosaviMonazzah/…` | use-restriction license — review the RAIL behavioral terms before commercial use |
| ⚠️ **Copyleft / share-alike** | all `fever/*`, `copenlu/fever_gold_evidence`, `pietrolesci/nli_fever`, `tals/vitaminc` (cc-by-sa-3.0 ± gpl-3.0); `BeIR/*` (cc-by-sa-4.0) | usable, but a stance model **fine-tuned** on these inherits share-alike — flag for redistribution |
| ⛔ **Non-commercial — do NOT ship commercially** | `allenai/scifact` (cc-by-nc-2.0) | use `BeIR/scifact` (cc-by-sa-4.0) instead |
| ⚠️ **Network-gated raw crawler (not a packaged dataset)** | `KaiDMML/FakeNewsNet` (GitHub) — tweet IDs + URLs only; bodies require re-crawling (Twitter ToS + publisher copyright); © 2019 Arizona Board of Regents, academic-use | **exclude from CI**; use the Apache-2.0 `LittleFish-Coder/*` content mirrors instead |

**Bottom line:** the *primary* training set (`GonzaloA/fake_news`) is license-**unknown** and is treated as research/eval-grade. For a redistribution-clean build, the license-clean substitution path is `mohammadjavadpirhadi/…` (MIT) or `ErfanMoosaviMonazzah/…` (openrail) for binary, `chengxuphd/liar2` (Apache-2.0) for credibility, and the **MIT** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for stance — keeping the shipped model stack free of the FEVER copyleft as long as the stance head is used zero-shot.
