# P11 — Data Card

> **Project:** Fake News & Misinformation Detection System (`fakenews` package)
> **Course:** NLP in Industry — final assignment · **Author:** Le Dinh Minh Quan (23127460)
> **Scope of this card:** every dataset the system can train on, evaluate against, or retrieve evidence from — plus the committed offline seed. Source of truth: [`docs/DESIGN_BRIEF.md`](./DESIGN_BRIEF.md) §2 (all HF ids read live via `hub_repo_details` as user `ledinhminhquan`).

This document is a structured **data card**: one card per dataset (id · link · license · size · schema · role · biases · redistribution · label polarity), grouped by the role each set plays in the two-stage pipeline:

1. **Classifier datasets** — train/evaluate the fast fake/real **prior** (transformer fine-tune + TF-IDF baseline).
2. **Fact-check / FEVER datasets** — the agentic, evidence-grounded verdict (retrieve → stance/NLI → aggregate → abstain).
3. **The offline seed** (`fakenews/data/samples.py`) — license-safe, network-free fallback so the whole pipeline runs with only numpy/pandas/sklearn.

---

## 0. How to read this card

- **Internal label convention (the whole repo standardises on this):** `0 = REAL`, `1 = FAKE`. `p_fake = P(label == 1)`. The `NewsItem.label` field (§9 of the brief) always carries this convention.
- **Label polarity is NOT consistent across mirrors.** Every loader in `fakenews/data/news_loaders.py` MUST apply an explicit per-source `label_map` to normalise to `0=REAL, 1=FAKE`. **Never assume** — verify on a known row. The "Label polarity" line in each card below is the *native* polarity as published on the Hub.
- **License hygiene flags** used throughout:
  - ✅ **permissive** (MIT / Apache-2.0 / CC-BY / CC0) — safe to use and redistribute derivatives.
  - ⚠️ **flag** — license unknown / unstated / copyleft / share-alike → research/eval OK, **verify before redistribution**; a derived model may inherit share-alike obligations.
  - ⛔ **non-commercial / network-gated** — do not ship commercially / cannot be packaged & redistributed as-is.
- **"Status"** = verification state from the brief: `VERIFIED` (repo + schema/splits/license read live) vs viewer-broken notes.

### Polarity quick-reference (the gotcha table)

| Native polarity | Datasets | `label_map` applied on load |
|---|---|---|
| **0 = fake, 1 = real** | `GonzaloA/fake_news`, `davanstrien/WELFake`, `ErfanMoosaviMonazzah/…-English` | swap → `{0:1, 1:0}` |
| **0 = real, 1 = fake** (matches repo convention) | `LittleFish-Coder/Fake_News_PolitiFact`, `LittleFish-Coder/Fake_News_GossipCop`, the offline seed | identity `{0:0, 1:1}` |
| **1 = fake, 0 = real** | `mrm8488/fake-news` | identity `{0:0, 1:1}` *(already matches; ⚠️ opposite of GonzaloA — easy to mis-handle)* |
| **6-way credibility (0–5)** | `chengxuphd/liar2`, `ucsbai/liar` | binary collapse (see LIAR cards) |

---

## 1. Classifier datasets (the trainable fake/real prior)

### 1.1 `GonzaloA/fake_news` — **PRIMARY binary training set**

| Field | Value |
|---|---|
| **HF id** | `GonzaloA/fake_news` |
| **Link** | https://huggingface.co/datasets/GonzaloA/fake_news |
| **Status** | VERIFIED (preview read) |
| **License** | **unknown** ⚠️ flag |
| **Size / splits** | 40.6K rows — train 24.4K / validation 8.1K / test 8.1K (pre-split) |
| **Schema** | `Unnamed: 0` (index), `title` (str), `text` (str, article body), `label` (int) |
| **Label polarity** | **0 = fake, 1 = real** → loader swaps to repo convention `0=REAL, 1=FAKE` |
| **Role** | **Primary** binary real/fake fine-tune target. Clean `title`+`text`, pre-split, stored as **parquet** → loads torch-free. Closest analog to the P02 resume classifier. Classifier input = `title [SEP] text`. |
| **Biases / caveats** | **Source/style leakage (#1 trap):** "real" rows are predominantly wire-service copy (Reuters/AP datelines like `"MOSCOW (Reuters) -…"`); "fake" rows are blog-rant style. ~0.95+ in-domain macro-F1 is a **red flag**, not success — the model can learn "is this Reuters formatting", not veracity. Strip datelines/source boilerplate before training; report in-domain **vs** cross-domain gap. Near-balanced classes. |
| **Redistribution** | ⚠️ License unknown → use for research/eval; **verify before redistributing** the data or a model trained on it. For redistribution-clean training prefer `mohammadjavadpirhadi/…` (MIT) or `ErfanMoosaviMonazzah/…` (openrail). |

### 1.2 `davanstrien/WELFake` — larger binary alternative

| Field | Value |
|---|---|
| **HF id** | `davanstrien/WELFake` |
| **Link** | https://huggingface.co/datasets/davanstrien/WELFake |
| **Status** | VERIFIED (schema read) |
| **License** | **unknown** ⚠️ flag (source WELFake ≈ CC-BY 4.0 per IEEE DataPort — **verify** independently) |
| **Size / splits** | 72.1K rows, **single `train` split** (≈35K real / ≈37K fake) — no official val/test |
| **Schema** | `title` (str), `text` (str), `label` (ClassLabel) |
| **Label polarity** | **0 = fake, 1 = real** → loader swaps to `0=REAL, 1=FAKE` |
| **Role** | Larger binary alternative; merges 4 corpora (Kaggle / McIntire / Reuters / BuzzFeed) → less single-source overfit. Make your **own** train/val/test split + dedup. Also usable as a cross-domain eval target. |
| **Biases / caveats** | 4-corpus merge → **overlapping sources** with GonzaloA/ISOT → run exact + near-dup (MinHash / normalized-hash) dedup *within and across* splits before training. Still style-leaky (Reuters wire vs blog). |
| **Redistribution** | ⚠️ License on the HF mirror unstated; upstream WELFake is reportedly CC-BY 4.0 but confirm before redistribution. |

### 1.3 `ErfanMoosaviMonazzah/fake-news-detection-dataset-English` — most license-clean large binary mirror

| Field | Value |
|---|---|
| **HF id** | `ErfanMoosaviMonazzah/fake-news-detection-dataset-English` |
| **Link** | https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English |
| **Status** | VERIFIED (preview read) |
| **License** | **openrail** (more permissive than the unknowns above; OpenRAIL carries use-based restrictions — read them) |
| **Size / splits** | 44.3K rows — train 30.0K / validation 6.0K / test 8.3K (pre-split) |
| **Schema** | `Unnamed: 0`, `title` (str), `text` (str), `subject` (str), `date` (str), `label` (int) |
| **Label polarity** | **0 = fake, 1 = real** → loader swaps to `0=REAL, 1=FAKE` |
| **Role** | ISOT-style, pre-split — the **most license-clean of the large binary mirrors**; recommended when license hygiene matters but you want a large pre-split corpus. |
| **Biases / caveats** | ISOT lineage → strong **dateline/source leakage** (same trap as GonzaloA). `subject` and `date` columns encode source patterns → **do not** feed them as features. |
| **Redistribution** | OpenRAIL allows redistribution with the use-based restrictions attached; propagate the license to derivatives. |

### 1.4 `mohammadjavadpirhadi/fake-news-detection-dataset-english` — MIT-clean binary

| Field | Value |
|---|---|
| **HF id** | `mohammadjavadpirhadi/fake-news-detection-dataset-english` |
| **Link** | https://huggingface.co/datasets/mohammadjavadpirhadi/fake-news-detection-dataset-english |
| **Status** | VERIFIED (repo exists; viewer renamed/broken) |
| **License** | **MIT** ✅ |
| **Size / splits** | 10K–100K range (viewer broken — confirm exact counts on load) |
| **Schema** | text-classification, binary |
| **Label polarity** | Binary (verify on load before mapping) → normalise to `0=REAL, 1=FAKE` |
| **Role** | **MIT-clean binary alternative** — prefer this base when license hygiene is paramount (commercial / redistribution-clean training). |
| **Biases / caveats** | Same English fake-news corpus family → source/style leakage applies. Viewer renamed/broken → load via `datasets` and inspect a few rows to confirm schema + polarity. |
| **Redistribution** | ✅ MIT — freely redistributable with attribution. |

### 1.5 `chengxuphd/liar2` — **SECONDARY: 6-way credibility (preferred LIAR family)**

| Field | Value |
|---|---|
| **HF id** | `chengxuphd/liar2` |
| **Link** | https://huggingface.co/datasets/chengxuphd/liar2 |
| **Status** | VERIFIED (preview read; working viewer) |
| **License** | **apache-2.0** ✅ |
| **Size / splits** | 23.0K rows — train 18.4K / validation 2.3K / test 2.3K |
| **Schema (16 cols)** | `id`, `label` (int 0–5), `statement`, `date`, `subject`, `speaker`, `speaker_description`, `state_info`, `true_counts`, `mostly_true_counts`, `half_true_counts`, `mostly_false_counts`, `false_counts`, `pants_fire_counts`, `context`, `justification` |
| **Label polarity** | **6-way credibility** `0–5` (pants-fire → true). Binary collapse for the prior: `{pants-fire, false, barely-true} → FAKE(1)`, `{half-true, mostly-true, true} → REAL(0)`. |
| **Role** | **Secondary head** — the "credibility scale" deliverable (`num_labels=6`, claim-only `statement` field). Its **`justification`** column is free in-domain **evidence** → fed into `data/evidence_corpus.py` as `Evidence` rows. |
| **Biases / caveats** | US-politics skew; labels encode **PolitiFact editorial judgment** (political-bias risk). **Metadata leakage:** the `speaker`, `state_info`, `subject`, `context`, `*_counts` columns encode the label distribution — **drop them from features**; train on `statement` (+ optionally `context`) only. 6-way imbalanced (`pants-fire` rare) → class weights + report **macro-F1**. |
| **Redistribution** | ✅ Apache-2.0 — clean to redistribute; the license-clean credibility set. |

### 1.6 `ucsbai/liar` — original 6-way LIAR (prefer LIAR2)

| Field | Value |
|---|---|
| **HF id** | `ucsbai/liar` (canonical; bare `liar` and `ucsbnlp/liar` redirect here) |
| **Link** | https://huggingface.co/datasets/ucsbai/liar |
| **Status** | VERIFIED (repo); **viewer broken** (legacy `.py` loader → 500/404) |
| **License** | **unknown** ⚠️ flag |
| **Size / splits** | 12.8K rows — train 10.24K / validation 1.28K / test 1.27K |
| **Schema** | `statement`, `label` (0–5), `subject`, `speaker`, `job_title`, `state_info`, `party_affiliation`, `barely_true_counts`, `false_counts`, `half_true_counts`, `mostly_true_counts`, `pants_on_fire_counts`, `context` |
| **Label polarity** | **6-way** `0–5`; same binary-collapse map as LIAR2. |
| **Role** | Original Datasets 6-way credibility set — **prefer LIAR2** (clean license, working viewer, `justification` evidence). Load via `datasets` with `trust_remote_code=True` only if you specifically need the original. |
| **Biases / caveats** | Same metadata-leakage and political-bias caveats as LIAR2 (`speaker`/`party_affiliation`/`*_counts` leak the label). Broken viewer → don't rely on preview. |
| **Redistribution** | ⚠️ Unknown license → research/eval only; verify before redistribution. |

### 1.7 `mrm8488/fake-news` — ⚠️ opposite-polarity binary (eval/aux only)

| Field | Value |
|---|---|
| **HF id** | `mrm8488/fake-news` |
| **Link** | https://huggingface.co/datasets/mrm8488/fake-news |
| **Status** | VERIFIED (preview read) |
| **License** | **not declared** ⚠️ flag |
| **Size / splits** | 44.9K rows, **single `train` split** (no official val/test) |
| **Schema** | `text` (str), `label` (int) — **no `title`, no splits** |
| **Label polarity** | **1 = fake, 0 = real** — i.e. already matches repo convention `0=REAL, 1=FAKE`, but **OPPOSITE to GonzaloA/WELFake**. The single most error-prone polarity in the set → identity map, but assert on a known row. |
| **Role** | Binary **eval / auxiliary only** — no title, no splits, undeclared license. Useful as an extra cross-domain probe, not a primary trainer. |
| **Biases / caveats** | Same English fake-news leakage; the polarity flip vs GonzaloA is the classic bug source → the explicit `label_map` exists precisely for this. |
| **Redistribution** | ⚠️ Undeclared → research/eval only. |

### 1.8 `LittleFish-Coder/Fake_News_PolitiFact` — FakeNewsNet-PolitiFact content mirror

| Field | Value |
|---|---|
| **HF id** | `LittleFish-Coder/Fake_News_PolitiFact` |
| **Link** | https://huggingface.co/datasets/LittleFish-Coder/Fake_News_PolitiFact |
| **Status** | VERIFIED (schema read) |
| **License** | **apache-2.0** ✅ |
| **Size / splits** | 483 rows — train 381 / test 102 |
| **Schema** | `text` (str), `label` (int) **+ precomputed `bert`/`roberta`/… embeddings** (≈51 MB train parquet). **Use only `text` + `label`.** |
| **Label polarity** | **0 = real, 1 = fake** — already matches repo convention → identity map. |
| **Role** | Apache-2.0 **content mirror of FakeNewsNet-PolitiFact** (avoids scraping the raw KaiDMML crawler). Small → eval / cross-domain reference, not a primary trainer. |
| **Biases / caveats** | Tiny (483 rows). US-politics domain skew. The bundled embeddings would leak model-specific representations → ignore them and use raw `text`. |
| **Redistribution** | ✅ Apache-2.0 — redistributable (preferred over re-crawling FakeNewsNet). |

### 1.9 `LittleFish-Coder/Fake_News_GossipCop` — FakeNewsNet-GossipCop content mirror (**cross-domain eval**)

| Field | Value |
|---|---|
| **HF id** | `LittleFish-Coder/Fake_News_GossipCop` |
| **Link** | https://huggingface.co/datasets/LittleFish-Coder/Fake_News_GossipCop |
| **Status** | VERIFIED (schema read) |
| **License** | **apache-2.0** ✅ |
| **Size / splits** | 12.7K rows — train 9,988 / test 2,672 |
| **Schema** | `text` (str), `label` (int) **+ precomputed embeddings** (**1.3 GB** train parquet — large; use `text`+`label` only) |
| **Label polarity** | **0 = real, 1 = fake** — matches repo convention → identity map. |
| **Role** | **The cross-domain eval set.** Train on `GonzaloA` (PolitiFact-style) → test here (celebrity-gossip domain) and report the **macro-F1 drop** — the signature of source-style leakage and itself an ethics metric. |
| **Biases / caveats** | Celebrity/gossip domain skew (very different from political/wire news). The 1.3 GB embedding payload → stream/select columns to avoid loading it. |
| **Redistribution** | ✅ Apache-2.0. |

> **Cross-domain protocol (from §7 of the brief):** train on `GonzaloA` → evaluate on `LittleFish-Coder/Fake_News_GossipCop`. A large macro-F1 drop is expected and **reported explicitly** — it quantifies how much the classifier learned source style vs veracity.

---

## 2. Fact-check / FEVER datasets (the agentic evidence layer)

> These power **retrieval** (evidence corpus + recall@k measurement) and **stance/verdict** training. Most carry **CC-BY-SA share-alike + GPL** (copyleft) obligations — a stance model fine-tuned on them inherits share-alike. Flag for redistribution.

### 2.1 `fever/fever` — FEVER verification + evidence corpus

| Field | Value |
|---|---|
| **HF id** | `fever/fever` (bare `fever` and `fever_v2.0` do **not** resolve) |
| **Link** | https://huggingface.co/datasets/fever/fever |
| **Status** | VERIFIED; viewer 501 (loader script) |
| **License** | **cc-by-sa-3.0 + gpl-3.0** ⚠️ copyleft / share-alike |
| **Size / configs** | ~185K claims; configs `v1.0` / `v2.0` / `wiki_pages` |
| **Schema** | `id`, `label`, `claim`, `evidence_*` (+ `wiki_pages` config = the evidence corpus) |
| **Label polarity** | 3-way verification: **SUPPORTS / REFUTES / NOT ENOUGH INFO**. Bridge to repo verdict: SUPPORTS→REAL-leaning, REFUTES→FAKE-leaning, NEI→ABSTAIN. |
| **Role** | Canonical fact-verification benchmark + Wikipedia evidence corpus. |
| **Biases / caveats** | Wikipedia-only evidence → world-knowledge skew; claims are synthetically mutated from Wikipedia sentences (not naturally occurring misinformation). Viewer needs the loader script. |
| **Redistribution** | ⚠️ Share-alike + GPL on code portions → derivatives inherit CC-BY-SA; attribute and propagate the license. |

### 2.2 `copenlu/fever_gold_evidence` — cleanest claim+evidence verdict set

| Field | Value |
|---|---|
| **HF id** | `copenlu/fever_gold_evidence` |
| **Link** | https://huggingface.co/datasets/copenlu/fever_gold_evidence |
| **Status** | VERIFIED |
| **License** | **cc-by-sa-3.0 + gpl-3.0** ⚠️ copyleft |
| **Size / splits** | 228.3K train / 15.9K validation / 16.0K test |
| **Schema** | `claim`, `label` (str), `evidence` (list of `[page, line_id, sentence]`), `id`, `verifiable`, `original_id` |
| **Label polarity** | SUPPORTS / REFUTES / NEI (string labels) |
| **Role** | **The cleanest single dataset to fine-tune a claim+gold-evidence verdict head** (optional trainable stance showpiece; `microsoft/deberta-v3-base`, `num_labels=3`). |
| **Biases / caveats** | Inherits FEVER's Wikipedia/synthetic-claim biases; gold evidence is curated (easier than retrieved evidence at inference). |
| **Redistribution** | ⚠️ Share-alike — a released stance model inherits CC-BY-SA. |

### 2.3 `pietrolesci/nli_fever` — FEVER reframed as NLI (drop-in stance fine-tune)

| Field | Value |
|---|---|
| **HF id** | `pietrolesci/nli_fever` |
| **Link** | https://huggingface.co/datasets/pietrolesci/nli_fever |
| **Status** | VERIFIED (viewer OK) |
| **License** | parent FEVER **cc-by-sa-3.0 + gpl-3.0** ⚠️ copyleft |
| **Size / splits** | 208K–248K train / ~20K dev / ~20K test |
| **Schema** | `premise`, `hypothesis`, `label` ∈ {entailment, neutral, contradiction} |
| **Label polarity** | NLI 3-way; maps `entailment→SUPPORTS`, `contradiction→REFUTES`, `neutral→NEI`. premise = evidence, hypothesis = claim. |
| **Role** | **Drop-in** for an optional stance fine-tune — already in the exact `premise/hypothesis/label` shape the `StanceNLI` tool consumes. |
| **Biases / caveats** | Same FEVER lineage; label noise from the FEVER→NLI reframing. |
| **Redistribution** | ⚠️ Inherits FEVER share-alike. |

### 2.4 `tals/vitaminc` — contrastive, edit-robust verification

| Field | Value |
|---|---|
| **HF id** | `tals/vitaminc` |
| **Link** | https://huggingface.co/datasets/tals/vitaminc |
| **Status** | VERIFIED |
| **License** | **cc-by-sa-3.0** ⚠️ share-alike |
| **Size / splits** | 370.7K train / 63.1K validation / 55.2K test |
| **Schema** | `claim`, `evidence`, `label`, `page`, `revision_type`, `FEVER_id` |
| **Label polarity** | SUPPORTS / REFUTES / NEI |
| **Role** | Optional **robustness** add-on to the stance fine-tune — contrastive pairs built from Wikipedia revisions force sensitivity to subtle factual edits. |
| **Biases / caveats** | Wikipedia-revision domain; designed to be hard → may not match real-world claim phrasing. |
| **Redistribution** | ⚠️ Share-alike. |

### 2.5 `mwong/fever-evidence-related` — evidence re-ranking

| Field | Value |
|---|---|
| **HF id** | `mwong/fever-evidence-related` |
| **Link** | https://huggingface.co/datasets/mwong/fever-evidence-related |
| **Status** | VERIFIED |
| **License** | **cc-by-sa-3.0 + gpl-3.0** ⚠️ copyleft |
| **Size / splits** | 403K train / 54.6K validation / 27.4K test |
| **Schema** | `claim`, `evidence`, `labels` (int), pre-tokenized `input_ids` |
| **Label polarity** | **related / not-related** (binary relevance, not a verdict) |
| **Role** | Evidence **re-ranking** training (does this passage relate to the claim) — feeds the `cross-encoder/ms-marco-MiniLM-L6-v2` reranker stage. |
| **Biases / caveats** | Pre-tokenized `input_ids` are tokenizer-specific → regenerate if your tokenizer differs. |
| **Redistribution** | ⚠️ Share-alike. |

### 2.6 `ImperialCollegeLondon/health_fact` (PUBHEALTH) — health-domain verification

| Field | Value |
|---|---|
| **HF id** | `ImperialCollegeLondon/health_fact` |
| **Link** | https://huggingface.co/datasets/ImperialCollegeLondon/health_fact |
| **Status** | VERIFIED; viewer disabled (loader script) |
| **License** | **mit** ✅ |
| **Size / splits** | 10K–100K range |
| **Schema** | `claim`, `main_text`, `explanation`, `label`, `sources`, `subjects` |
| **Label polarity** | 4-way: **true / false / unproven / mixture** |
| **Role** | Health-domain fact-check (license-clean). The `unproven` / `mixture` labels motivate the **ABSTAIN** branch for genuinely uncertain claims. |
| **Biases / caveats** | Health-claim domain skew; `mixture`/`unproven` don't map cleanly to binary SUPPORTS/REFUTES. |
| **Redistribution** | ✅ MIT. |

### 2.7 `tdiggelm/climate_fever` — climate-domain verification (motivates DISPUTED→abstain)

| Field | Value |
|---|---|
| **HF id** | `tdiggelm/climate_fever` (bare `climate_fever` redirects here) |
| **Link** | https://huggingface.co/datasets/tdiggelm/climate_fever |
| **Status** | VERIFIED |
| **License** | **unknown** ⚠️ flag |
| **Size / splits** | 1.5K rows, **test only** |
| **Schema** | `claim_id`, `claim`, `claim_label`, `evidences` (list) |
| **Label polarity** | SUPPORTS / REFUTES / NEI / **DISPUTED** |
| **Role** | Climate-domain eval. The **`DISPUTED`** label directly motivates the **contradictory-evidence abstain branch** (D5 conflict / low-agreement → abstain). |
| **Biases / caveats** | Tiny, test-only → eval not train. License unknown. |
| **Redistribution** | ⚠️ Unknown license → eval only. |

### 2.8 `allenai/scifact` — ⛔ NON-COMMERCIAL scientific verification

| Field | Value |
|---|---|
| **HF id** | `allenai/scifact` |
| **Link** | https://huggingface.co/datasets/allenai/scifact |
| **Status** | VERIFIED; loader script |
| **License** | **cc-by-nc-2.0** ⛔ **NON-COMMERCIAL** |
| **Size / splits** | 1K–10K range |
| **Schema** | `claim`, `evidence`, `cited_doc_ids`, `label` + rationales |
| **Label polarity** | SUPPORT / CONTRADICT / NOINFO |
| **Role** | Scientific-claim verification — **do not use in a commercial product.** Use `BeIR/scifact` (CC-BY-SA-4.0) instead. |
| **Biases / caveats** | Scientific-abstract domain; **non-commercial license is the headline constraint.** |
| **Redistribution** | ⛔ Non-commercial — research only; do not ship. |

### 2.9 `BeIR/scifact` — share-alike retrieval alternative to scifact

| Field | Value |
|---|---|
| **HF id** | `BeIR/scifact` |
| **Link** | https://huggingface.co/datasets/BeIR/scifact |
| **Status** | VERIFIED |
| **License** | **cc-by-sa-4.0** ⚠️ share-alike (commercially usable, unlike `allenai/scifact`) |
| **Size / splits** | 1K–10K range |
| **Schema** | BEIR triple: `corpus`, `queries`, `qrels` |
| **Label polarity** | Retrieval relevance (`qrels`), not a verdict |
| **Role** | Retrieval-relevance alternative to `allenai/scifact` — share-alike, so safe where non-commercial is not. |
| **Biases / caveats** | Scientific domain; relevance only (no S/R/NEI verdicts). |
| **Redistribution** | ⚠️ Share-alike (attribution + CC-BY-SA-4.0). |

### 2.10 `BeIR/fever` + `BeIR/fever-qrels` — **evidence-recall@k measurement**

| Field | Value |
|---|---|
| **HF id** | `BeIR/fever` and `BeIR/fever-qrels` |
| **Link** | https://huggingface.co/datasets/BeIR/fever · https://huggingface.co/datasets/BeIR/fever-qrels |
| **Status** | VERIFIED |
| **License** | **cc-by-sa-4.0** ⚠️ share-alike |
| **Size / splits** | Wikipedia-abstract corpus + qrels |
| **Schema** | BEIR `corpus` / `queries` + `qrels` (relevance judgments) |
| **Label polarity** | Retrieval relevance (`qrels`) |
| **Role** | **The set used to measure evidence recall@k** (R@1/5/10) for the retrieval stage — the honest "did we find the right evidence" metric. |
| **Biases / caveats** | Wikipedia-abstract corpus only. |
| **Redistribution** | ⚠️ Share-alike. |

### 2.11 `fever/feverous` — tables + text verification (bonus)

| Field | Value |
|---|---|
| **HF id** | `fever/feverous` |
| **Link** | https://huggingface.co/datasets/fever/feverous |
| **Status** | VERIFIED; loader script |
| **License** | **cc-by-sa-3.0** ⚠️ share-alike |
| **Size / splits** | 100K–1M range |
| **Schema** | `claim` + sentence/table evidence |
| **Label polarity** | SUPPORTS / REFUTES / NEI |
| **Role** | Bonus / stretch — verification over **tables + text** (structured evidence). |
| **Biases / caveats** | Table evidence needs special handling; not used in the default pipeline. |
| **Redistribution** | ⚠️ Share-alike. |

### 2.12 `nid989/FNC-1` — stance (auxiliary)

| Field | Value |
|---|---|
| **HF id** | `nid989/FNC-1` (only clean FNC-1 mirror; `emergent` is NOT on the Hub) |
| **Link** | https://huggingface.co/datasets/nid989/FNC-1 |
| **Status** | VERIFIED |
| **License** | **not stated** ⚠️ flag (FNC-1 source is reportedly permissive — confirm) |
| **Size / splits** | 40.5K train / 4.5K validation / 5.0K test |
| **Schema** | `headline`, `body` (`articleBody`), `stance` |
| **Label polarity** | 4-way stance: **agree / disagree / discuss / unrelated** |
| **Role** | **Secondary / aux** stance signal. FNC-1's 4-way labels (with `unrelated`/`discuss`) are weaker for verdicts than FEVER's 3-way → not the primary stance set. |
| **Biases / caveats** | `unrelated`/`discuss` don't map to SUPPORTS/REFUTES; news-article domain. License unstated. |
| **Redistribution** | ⚠️ License unstated → verify before redistribution. |

---

## 3. The offline seed — `fakenews/data/samples.py`

> **The committed, license-safe, fully synthetic fallback** so the entire pipeline runs with **only numpy/pandas/sklearn, no torch, no network**. Original/synthetic text (no copied dataset), deliberately avoiding real named individuals. This is the only "dataset" actually shipped inside the repo.

| Field | Value |
|---|---|
| **Location** | `D:\NLP Industry Projects\11_Fake_News_Detection\src\fakenews\data\samples.py` |
| **Accessors** | `samples.news()`, `samples.evidence()`, `samples.claims()` (each returns deep copies); constants `SEED_NEWS`, `SEED_EVIDENCE`, `SEED_CLAIMS` |
| **License** | ✅ **Original/synthetic — fully redistributable** (authored for this project; no third-party data) |
| **Label polarity** | **0 = real, 1 = fake** — **matches the repo convention exactly** → identity map, no remap needed |
| **Redistribution** | ✅ Ships in the repo and tests; safe to redistribute without restriction. |

### 3.1 `SEED_NEWS` — labeled classifier samples

| Property | Value |
|---|---|
| Rows | **40** (balanced: 20 real `r01–r20`, 20 fake `f01–f20`) |
| Schema | `id` (str), `title` (str), `text` (str), `label` (int 0/1) |
| Real (`0`) style | Sober factual statements across health / science / civic / finance / environment (e.g. `r01` handwashing guidance, `r05` central bank holds rate). |
| Fake (`1`) style | Clearly false / sensational misinformation (e.g. `f01` "drinking bleach cures every virus", `f04` "5G towers control the weather"). |
| Role | Fits TF-IDF + LogReg and produces non-trivial macro-F1 with no torch; doubles as the linear-separability sanity check. |

### 3.2 `SEED_EVIDENCE` — retrievable evidence corpus

| Property | Value |
|---|---|
| Rows | **16** (`e01–e16`) |
| Schema | `id` (str), `source` (str, e.g. `WHO`, `NASA`, `Cardiology journal`), `text` (str) |
| Mix | Refuting snippets for the fake claims (`e01` bleach, `e04` 5G, `e05` vaccine microchips) **and** supporting snippets for the real claims (`e13` exercise/heart, `e14` measles/vaccination, `e15` handwashing) → TF-IDF / BM25 / RRF returns mixed hits. |
| Role | Backs the offline `RetrieveEvidence` (TF-IDF cosine / BM25) + lexical-overlap stance fallback. |

### 3.3 `SEED_CLAIMS` — claims with gold verdicts (FEVER-style)

| Property | Value |
|---|---|
| Rows | **8** |
| Schema | `claim` (str), `verdict` (str ∈ {`fake`, `real`}) |
| Balance | 5 `fake` (bleach, 5G weather, vaccine microchips, gravity off, cotton-candy clouds) + 3 `real` (exercise↔heart, handwashing, measles vaccination). |
| Role | Exercises stance aggregation, the **ABSTAIN** branch (claims with no matching evidence get insufficient coverage), and FEVER-style label scoring — all with zero network. |

> **Companion fixture:** the brief specifies `tests/fixtures/fake_news_tiny.csv` (~10 rows, `title,text,label`, repo convention **0=real, 1=fake**: ~5 wire-service-style real + ~5 sensational fake) for unit tests. When `transformers` is absent, `StanceNLI` returns `backend:"none"` and a lexical mock-stance keeps the verdict aggregator + metrics runnable end-to-end on this seed.

---

## 4. Avoid / gated list (read before adding a dataset)

| Item | Problem | What to do instead |
|---|---|---|
| **bare `liar`** | Viewer broken (legacy `.py` loader → 500/404); redirects to `ucsbai/liar` | Use **`chengxuphd/liar2`** (Apache-2.0, working viewer, has `justification` evidence). |
| **`KaiDMML/FakeNewsNet`** (raw GitHub) | A **crawler, not a packaged dataset** — only tweet IDs + URLs are redistributed; article bodies require **re-scraping** (Twitter ToS + publisher copyright). Copyright 2019 Arizona Board of Regents (ASU), academic-use + citation. **Network-gated; exclude from CI.** | Use the Apache-2.0 content mirrors **`LittleFish-Coder/Fake_News_PolitiFact`** + **`…_GossipCop`** (or `Ahren09/FakeNewsNet`, viewer 501). |
| **`allenai/scifact`** | ⛔ **cc-by-nc-2.0 — non-commercial** | Use **`BeIR/scifact`** (cc-by-sa-4.0). |
| **`mrm8488/fake-news`** | ⚠️ Undeclared license **and** polarity (`1=fake`) flips vs GonzaloA (`0=fake`) → silent label bug | Eval/aux only; assert polarity on a known row before use. |
| **`fever/*`, `copenlu/*`, `pietrolesci/nli_fever`, `mwong/*`, `tals/vitaminc`, `BeIR/*`, `fever/feverous`** | ⚠️ **Copyleft / share-alike** (CC-BY-SA-3.0/4.0 ± GPL-3.0) — a derived stance model inherits share-alike | Usable for research; **flag for redistribution** and propagate the license to any released model. |
| **`GonzaloA/fake_news`, `davanstrien/WELFake`, `ucsbai/liar`, `tdiggelm/climate_fever`, `nid989/FNC-1`, `rickstello/FakeNewsNet`** | ⚠️ **Unknown / unstated / generic-`cc` license** | Research/eval OK; **verify before redistribution**. For redistribution-clean training prefer `mohammadjavadpirhadi/…` (MIT) / `ErfanMoosaviMonazzah/…` (openrail) / `chengxuphd/liar2` (Apache-2.0). |
| **Any binary loader without an explicit `label_map`** | Polarity differs per mirror (0=fake vs 0=real vs 1=fake) → systematic label inversion that *still trains* and silently reports inverted metrics | Every loader applies a per-source `label_map` → repo convention **`0=REAL, 1=FAKE`**; **verify on a known row** at load time. |

### License summary (at a glance)

| Tier | Datasets |
|---|---|
| ✅ **Permissive (prefer)** | `chengxuphd/liar2` (Apache), `mohammadjavadpirhadi/…` (MIT), `ImperialCollegeLondon/health_fact` (MIT), `LittleFish-Coder/*` (Apache), `Ahren09/FakeNewsNet` (Apache), **the offline seed** (original) |
| ⚠️ **OpenRAIL (use-restricted)** | `ErfanMoosaviMonazzah/…-English` |
| ⚠️ **Share-alike / copyleft** | all `fever/*`, `copenlu/fever_gold_evidence`, `pietrolesci/nli_fever`, `mwong/*`, `tals/vitaminc` (CC-BY-SA-3.0 ± GPL-3.0); `BeIR/*` (CC-BY-SA-4.0) |
| ⚠️ **Unknown / unstated** | `GonzaloA/fake_news`, `davanstrien/WELFake`, `ucsbai/liar`, `mrm8488/fake-news`, `tdiggelm/climate_fever`, `nid989/FNC-1`, `rickstello/FakeNewsNet` |
| ⛔ **Non-commercial / network-gated** | `allenai/scifact` (CC-BY-NC-2.0); `KaiDMML/FakeNewsNet` raw crawler (academic-use, re-scrape required) |

---

## 5. Ethics & provenance notes (data-specific)

- **Labels are editorial, not ground truth.** LIAR / LIAR2 / PolitiFact labels encode a fact-checker's judgment with topic + speaker skew. Document provenance; report performance sliced by topic / speaker / source; never present output as neutral truth.
- **Source-style leakage is a data property, not just a modeling bug.** GonzaloA / WELFake / ISOT mirrors leak *publisher + writing style*. The mandatory cross-domain eval (`GonzaloA` → `GossipCop`) makes the leakage visible; the agentic **evidence** verdict (`classifier_prior` reported *separately* from the evidence `verdict`) is the mitigation.
- **English-centric + US-domain skew** across the whole corpus (PolitiFact politics, GossipCop celebrity, FEVER Wikipedia). State plainly: the system verifies against a **fixed corpus** and cannot know facts outside it; **`unverified` means insufficient evidence, not true.**
- **Citations are extracted verbatim** from the retrieved corpus (with source IDs), never generated — a fact-check is only trustworthy if its evidence is real.

---

*Card maintained alongside `docs/DESIGN_BRIEF.md`. All HF ids verified live via `hub_repo_details` as `ledinhminhquan`; do not add an id here without verifying its existence, schema, license, and native label polarity.*
