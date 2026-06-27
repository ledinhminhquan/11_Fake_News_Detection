# Ethics & Responsible AI — P11 Fake News & Misinformation Detection System

> **Section I.11 — Ethics & Responsible AI.** NLP-in-Industry final assignment, project P11. Author: Le Dinh Minh Quan (student 23127460).
> Package: `fakenews`. Companion docs: [`DESIGN_BRIEF.md`](DESIGN_BRIEF.md) (single source of truth — verified Hugging Face ids, pipeline, FSM decisions D1–D5, metrics).

---

## 0. Why this is the most ethically loaded project in the set

A system that attaches the word **"fake"** to a piece of writing is, by design, a tool that can be turned into an instrument of censorship. It sits on the most dangerous intersection in applied NLP: it (a) makes a claim about *truth*, (b) about *speech*, (c) often about *political* speech, (d) at *scale*, and (e) with the perceived neutrality of "the algorithm said so." Every one of those properties multiplies harm if the design is careless.

This document is therefore not a compliance appendix — it is the **specification that constrains the product**. The non-negotiable commitments in §6 are not aspirations; they are wired into the code (the agent FSM in `agent/factcheck_agent.py`, the abstain logic in `factcheck/verdict.py`, the API contract in `api/schemas.py`, and the Gradio UI in `api/ui.py`). If a feature request conflicts with §6, the feature loses.

**The one-sentence ethical thesis:** *this tool flags content for a human to review and always shows the evidence behind the flag; it never auto-removes, auto-blocks, auto-ranks-down, or auto-censors anything, and it abstains rather than guess.*

---

## 1. Stakeholders — who benefits and who can be harmed

Responsible-AI analysis starts by naming everyone the system touches, **including people who never use it but are affected by its outputs** (the publishers it labels, the readers who see the label).

### 1a. Who benefits

| Beneficiary | Benefit | Condition under which the benefit is real |
|---|---|---|
| **Informed public / readers** | A transparent credibility signal *with the evidence shown*, encouraging verification instead of blind sharing | Only if the evidence and `unverified` outcomes are surfaced honestly — a bare label trains blind trust, the opposite of the goal |
| **Professional fact-checkers / journalists** | Triage: "is this claim worth checking, and what does the evidence already say?" — a cited evidence trace they can verify line-by-line | Only as a *first-pass assistant*; the human still does the verification and signs off |
| **Platform trust-&-safety / moderators** | A fast prior that routes only *suspicious* items into a human review queue, raising reviewer throughput | Only as a **router to humans**, never as an auto-takedown trigger |
| **Newsroom editors / researchers** | A 6-way credibility scale (LIAR2 `pants-fire`→`true`) for nuance beyond binary fake/real, plus per-topic/per-source error slices | Only with the documented label-provenance and bias caveats attached |

### 1b. Who can be harmed

| Who | How they can be harmed | Severity |
|---|---|---|
| **Legitimate publishers / journalists** | A false "fake" label that triggers removal or down-ranking suppresses lawful, accurate reporting — the **highest-cost error** | Critical |
| **Politically targeted sources / dissidents** | A tool trained on one polity's fact-checks can be repurposed to flag an opposition outlet as "fake" at scale — state-grade censorship | Critical |
| **Named outlets caught by source-style leakage** | The classifier can learn "this outlet ⇒ fake" rather than "this claim is false," entrenching bias against a source regardless of the article's truth | High |
| **Authors of true-but-unusual claims** | Novel or non-mainstream true claims with no corpus coverage can be flagged or mis-handled if the system guesses instead of abstaining | High |
| **The general reader (automation bias)** | Over-trust in a confident-and-wrong label, or in a fabricated-looking citation, propagates misinformation *under the banner of fact-checking* | High |
| **Speakers chilled out of the conversation** | Even without removal, the *threat* of being labelled "fake" chills lawful publishing (free-speech chilling effect) | Medium–High |

> **Asymmetry of harm.** A false negative (a fake item not flagged) is a missed catch a human can still find later. A false positive (legitimate news labelled fake and acted upon) **silences a true voice** and is far harder to undo. The operating point, the abstain policy, and the "flag-not-censor" rule all exist to protect the false-positive side.

---

## 2. Censorship and free-speech chilling effects

This is the central risk. An automated "fake" verdict that is wired to an enforcement action (delete, block, shadow-ban, de-monetise, down-rank) is **automated censorship of lawful speech** — and because the model is imperfect and biased (§4, §5), it will censor true and legitimate speech too.

**Design response (hard constraints):**

- **No enforcement output exists anywhere in the system.** The API (`/factcheck`, `/classify`) returns a *verdict + evidence + decision trace*. There is no `remove`, `block`, `mute`, or `downrank` field, and no code path that calls such an action. The output is **advisory**.
- **The output is a review flag, not a ruling.** Even a high-confidence `fake` verdict is framed as "flagged for human review, here is the evidence," never "this is false, removed."
- **Chilling is mitigated by transparency and recourse.** Because every verdict ships its verbatim citations and `decisions_trace`, a flagged publisher (or the reviewer) can *see exactly why* and contest it. Opaque flags chill; auditable flags invite correction.
- **`unverified` is a first-class, non-punitive outcome** (`factcheck/verdict.py`). The default for "we don't have the evidence" is **not** a soft "leaning fake" — it is an explicit, neutral "insufficient evidence — needs human review."

---

## 3. Political bias in labels and training data

The training labels are **human editorial judgements, not ground truth**. LIAR/LIAR2 and PolitiFact/GossipCop labels encode a specific fact-checker's calls, a US-politics + celebrity topic skew, and a speaker/source distribution. Treating those labels as neutral truth launders one editorial worldview into an "objective algorithm."

| Source of bias | Where it enters | Mitigation in `fakenews` |
|---|---|---|
| **Editorial label bias** | LIAR2 (`chengxuphd/liar2`, Apache-2.0) and FakeNewsNet mirrors inherit the fact-checker's calls | Document label provenance in the model card; **never** present output as neutral ground truth; report `classifier_prior` and evidence-`verdict` separately so a reviewer sees the model's prior is just a prior |
| **Topic / speaker skew** | US politics + celebrity dominate the corpora | Report metrics **sliced by topic / speaker / source** (`analysis/fairness.py`); state the domain skew as a stated limitation (§7) |
| **Metadata leakage** | LIAR2 `speaker`, `state_info`, `subject`, per-speaker `*_counts` columns encode the label distribution → model learns "this speaker lies" | **Drop those columns from features** (brief §4 anti-leakage rule 6); train on `statement` (+ optional `context`) only |
| **Single-polity worldview** | English-centric, predominantly US fact-checks | Stated limitation; never marketed as a universal arbiter; per-source slices expose where it fails |

The deliverable's stance is explicit: **the classifier detects style/source patterns correlated with fakeness — which is *not* the same as detecting falsehood.** The agentic, evidence-grounded layer exists precisely to compensate, and the two signals are reported **separately**.

---

## 4. Source-style leakage — the bias-entrenchment trap

**The #1 technical trap in fake-news detection.** Datasets like `GonzaloA/fake_news` and WELFake leak *publisher and writing style*, not truth: the "real" rows are wire-service copy (`"MOSCOW (Reuters) —…"`) and the "fake" rows are blog-rant style. A model that scores 99% in-domain macro-F1 has usually learned **"is this formatted like Reuters,"** not **"is this true.** That is an ethics failure dressed as a benchmark win: it **entrenches bias against named outlets** and collapses the moment a new (legitimate) outlet appears.

**Design response:**

1. **Boilerplate stripping** before training — remove datelines / source markers (`(Reuters)`, `WASHINGTON —`, bylines) so the model cannot key on publisher formatting (`data/news_loaders.py`).
2. **Mandatory cross-domain evaluation** — train on `GonzaloA` (PolitiFact-style), test on `LittleFish-Coder/Fake_News_GossipCop` (cross-domain). **The cross-domain macro-F1 is the honest number, and the in-domain→cross-domain *drop* is itself an ethics metric** (the size of the style-leakage problem). A *too-high* in-domain F1 is treated as a **red flag, not a success**.
3. **The evidence verdict dominates; the classifier prior is only a soft nudge.** In aggregation (`factcheck/verdict.py`, decision **D4**), evidence stance carries weight `α = 0.6` and the prior `(1−α) = 0.4`; a confident NLI head cannot be overridden by the prior. The style-classifier is deliberately demoted to a weak prior.

---

## 5. Adversarial evasion and automation bias

### 5a. Adversarial paraphrase / evasion

A bad actor can paraphrase a fake article to flip a brittle style-classifier's label. **Mitigation:** the evidence-grounded fact-check is far more robust than style classification (paraphrasing the claim does not change what the *evidence* says); keep an adversarial/paraphrase eval set; **never rely on the classifier alone** — the verdict is evidence-driven.

### 5b. Automation bias / over-trust

The subtler harm: reviewers **rubber-stamp** the model, and a confident-and-wrong output propagates *with the authority of a fact-check.* **Mitigation:**

- **Calibration is a first-class metric.** Report **ECE (Expected Calibration Error)** + a reliability diagram, with temperature scaling on a held-out split (over-confidence here can silence real news). See [`metrics.md`].
- **Mandatory confidence display** and **mandatory evidence reading** in the UI — the verdict chip is never shown without its citations.
- **Abstain on uncertainty** (D5) — the system declines rather than emit a low-confidence guess that a tired reviewer would wave through.

### 5c. Dual-use

The model can be probed to craft detection-evading text. **Mitigation:** do **not** ship an "evasion score" or adversarial-generation endpoint; rate-limit; log; gate any adversarial tooling.

---

## 6. Non-negotiable design commitments (wired into the build)

These are the load-bearing ethics requirements. Each maps to a concrete artifact.

| # | Commitment | Where it lives in the code |
|---|---|---|
| **C1** | **Flags for review — NEVER auto-censors.** No remove/block/downrank output or code path exists. Output is advisory. | No enforcement field in `api/schemas.py`; `/factcheck` returns verdict+evidence+trace only |
| **C2** | **Mandatory evidence / citations.** Every non-abstained verdict ships ≥1 **verbatim** citation with a source id. Target: 100% of non-abstained verdicts cite evidence. | `Present.run(...)` returns `citations[]`; `factcheck/verdict.py` attaches per-evidence stance + snippet |
| **C3** | **Citations are extracted, never generated.** Snippets are copied verbatim from the retrieved corpus with source ids — never produced by an LLM — so a fact-check cannot rest on a hallucinated source. | Retriever returns corpus passages; `decisions_trace` lets a reviewer verify each id |
| **C4** | **Abstain on uncertainty.** `unverified` is a first-class outcome (insufficient evidence / low agreement / low confidence / prior↔evidence conflict), not a failure mode. | D3 coverage gate + D5 abstain gate (`agent/policy.py`, `factcheck/verdict.py`) |
| **C5** | **`classifier_prior` vs. evidence `verdict` reported SEPARATELY.** A reviewer always sees both, and a conflict between them raises an explicit `conflict — needs human review` flag (D5). | `/factcheck` response carries both `classifier_prior` and `verdict`; D5 emits `conflict` |
| **C6** | **Human sign-off before any action.** The tool assists; the human decides. No irreversible action is ever taken on content automatically. | Product/operational rule; enforced by the absence of any enforcement path (C1) |
| **C7** | **Transparency is non-optional.** The evidence list and full `decisions_trace` are returned **even on abstain**. | `/factcheck` always returns `evidence` + `decisions_trace`; `ToolTrace` logs every FSM transition |

---

## 7. Limitations stated plainly

A responsible tool advertises its own blind spots:

- **English-centric** data and models.
- **Domain skew:** PolitiFact / GossipCop ≈ US politics + celebrity; performance degrades off-domain (the cross-domain drop quantifies this).
- **Coarse labels:** binary "fake/real" is a proxy for a spectrum; LIAR2's 6-way credibility captures the gradient better but is still an editorial scale.
- **Fixed-corpus knowledge:** the fact-check verifies against a **versioned, timestamped evidence index** and *cannot know facts outside it*. The index can go stale (`/version` exposes `index_built_at`).
- **`unverified` ≠ false.** It means *insufficient evidence*, full stop. It must never be read as "probably fake."
- **The classifier is a weak prior, not a truth oracle.** It detects style/source correlates of fakeness, which is not falsehood detection — hence the separate, evidence-dominant verdict.

---

## 8. Explainability for non-technical stakeholders

A flag a fact-checker cannot understand is a flag they cannot trust or contest. Every verdict is explainable to a non-technical reviewer through three artifacts, all present on `/factcheck` and rendered in the Gradio UI (`api/ui.py`):

1. **The verdict chip + confidence** — `real` / `fake` / **`unverified`** with a confidence gauge. `unverified` renders prominently with distinct neutral styling ("Not enough evidence — flagged for human review"), never a fake label by default.
2. **The evidence list** — each retrieved passage shown with its **source/citation**, a colour-coded **stance badge** (support = green / refute = red / neutral = grey), and a relevance score. The reviewer reads the actual evidence, not just a number.
3. **The decision trace** — a plain-language `decisions_trace` of which gate fired and why (e.g. `RETRIEVE: n_relevant=4, coverage_ok=true`; `AGGREGATE: 3 refute / 0 support → FAKE`; `DECIDE: prior agrees, confidence 0.91 → present`). This is the "show your work" that lets a non-technical reviewer audit and overrule the machine.

Plus the always-visible UI notice on the **Classify** tab: *"this is a style/probability signal, not a verdict."*

---

## 9. Risk → mitigation table

(Consolidated from `DESIGN_BRIEF.md` §8 — the canonical risk register.)

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Censorship / free-speech chilling** | An automated "fake" label wired to removal suppresses lawful speech and chills publishing | Output a **review flag + evidence**, never an enforcement action; **no auto-takedown** anywhere; verdicts are advisory (C1) |
| **False positives silencing real news** | Flagging legitimate journalism as fake is the highest-cost error | Optimise the operating point for high `real`-recall; report per-class costs; **abstain** rather than guess; human sign-off before any action (C4, C6) |
| **Political bias in labels / data** | LIAR/PolitiFact labels encode the fact-checker's editorial judgement + topic/speaker skew | Document label provenance; report metrics sliced by topic/speaker/source; never present output as neutral ground truth |
| **Source-style leakage** | Model learns "this outlet ⇒ fake," not truth; collapses on new outlets, entrenches bias against named sources | Measure cross-domain drop (PolitiFact→GossipCop); strip/ablate source/style features; prefer the **evidence-grounded** verdict; treat `classifier_prior` as a weak prior only |
| **Adversarial paraphrase / evasion** | Bad actors paraphrase to flip the label; style classifiers are brittle | Evidence-based fact-check is more robust than style classification; keep an adversarial/paraphrase eval set; don't rely on the classifier alone |
| **Automation bias / over-trust** | Reviewers rubber-stamp the model; confident-and-wrong outputs propagate | Calibration (ECE) + mandatory confidence display; **abstain on uncertainty**; UI states it's a decision-support signal with evidence required to be read |
| **Dual-use** | The model can be probed to craft detection-evading text | Don't ship an "evasion score"; rate-limit; log; gate adversarial tooling |
| **Stale / out-of-scope evidence** | Index goes out of date; novel claims have no coverage | Version + timestamp the index (`/version`); **abstain** when retrieval coverage is insufficient rather than fabricate |
| **Hallucinated / misattributed citations** | A fact-check is only trustworthy if its citations are real | Citations are **extracted verbatim** from the retrieved corpus with source ids, never generated; `decisions_trace` lets a reviewer verify each one (C2, C3, C7) |

---

## 10. Data-licensing ethics (provenance & redistribution)

Responsible data use is part of the ethics story. The brief flags every dataset's licence; the ethically relevant items:

- **⛔ Non-commercial — do NOT ship in a commercial product:** `allenai/scifact` (cc-by-nc-2.0). Use the share-alike `BeIR/scifact` (cc-by-sa-4.0) instead.
- **⚠️ Unknown / unstated licence — research / eval only, verify before redistribution:** `GonzaloA/fake_news` (the **PRIMARY** classifier set — licence **unknown**, flagged), `davanstrien/WELFake`, `ucsbai/liar`, `mrm8488/fake-news`, `tdiggelm/climate_fever`, `nid989/FNC-1`. For redistribution-clean training, prefer `mohammadjavadpirhadi/…` (MIT) or `ErfanMoosaviMonazzah/…` (openrail); `chengxuphd/liar2` (Apache-2.0) is the clean credibility set.
- **⚠️ Copyleft / share-alike (attribution + share-alike; GPL on code portions):** all `fever/*`, `copenlu/fever_gold_evidence`, `pietrolesci/nli_fever`, `tals/vitaminc` (cc-by-sa-3.0 ± gpl-3.0), `BeIR/*` (cc-by-sa-4.0). Usable, but **any released stance model fine-tuned on FEVER inherits the share-alike obligation** — flag on redistribution.
- **⚠️ `KaiDMML/FakeNewsNet` raw:** a crawler, not a packaged dataset — only tweet ids + URLs are redistributed; article bodies require re-crawling (Twitter ToS + publisher copyright; ASU academic-use + citation). **Treated as network-gated, excluded from CI;** use the Apache-2.0 `LittleFish-Coder/*` content mirrors instead.
- **⚠️ Label-polarity hazard:** mirrors disagree (`GonzaloA` 0=fake/1=real; `LittleFish` 0=real/1=fake; `mrm8488` 1=fake). The repo standardises on **`0=REAL, 1=FAKE`** and every loader applies an explicit per-source `label_map` — a silent polarity flip would invert every verdict, an ethics-grade bug.
- **✅ Cleanly permissive (preferred), MIT/Apache:** all recommended models — `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT), `answerdotai/ModernBERT-base` (Apache-2.0), `distilbert-base-uncased` (Apache-2.0), `microsoft/deberta-v3-base` (MIT), `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), `BAAI/bge-small-en-v1.5` (MIT).

---

## 11. Summary — the ethical contract

The system is built to be **an assistant to human judgement, never a replacement for it.** It earns trust by being **auditable** (verbatim citations + full decision trace on every call, even on abstain), **humble** (abstains on uncertainty; reports its weak style-prior separately from the evidence verdict; advertises its biases and domain limits), and **harm-aware** (no enforcement path exists; the operating point and abstain policy protect the false-positive / silence-real-news side of the asymmetry). It flags content for a human to review and shows the evidence behind the flag — and it does nothing else to the content. That is the contract, and it is enforced in code, not just in this document.
