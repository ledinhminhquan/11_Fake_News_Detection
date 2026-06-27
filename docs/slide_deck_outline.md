# P11 — Slide Deck Outline (`slides.pptx`)

> **Course:** NLP in Industry — Final Assignment, Project **P11 — Fake News & Misinformation Detection System**
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Spec:** Section II.3 presentation deck (~12 slides). This outline is the source the auto-generated `autoreport/slides_pptx.py` follows — one section below = one slide, each with a title, 3–6 talking-point bullets, and a suggested visual. All Hugging Face ids, licenses, thresholds, and metric names are taken verbatim from `docs/DESIGN_BRIEF.md` (the single source of truth). The system as built is the `fakenews` package: a trainable real/fake **classifier** (fast prior) wrapped by an **agentic, evidence-grounded fact-check** (retrieve → stance/NLI → aggregate → abstain).

---

## Slide deck map

| # | Slide | Purpose |
|---|---|---|
| 1 | Title & Team | Identify project + author |
| 2 | Business Problem & Motivation | Why this matters, who uses it |
| 3 | Proposed NLP Solution | Classifier prior + evidence fact-check thesis |
| 4 | System Architecture Diagram | The FSM pipeline end-to-end |
| 5 | Data Overview | Datasets, licenses, polarity gotcha |
| 6 | Model & Evaluation Results | Macro-F1 / ROC-AUC / ECE + fact-check tables |
| 7 | Agentic AI Component | The 5 decision points D1–D5 |
| 8 | Deployment Overview | API / Gradio / CLI / Docker / Space |
| 9 | Ethics, Privacy & Risks | Flags-for-review, never censors (centerpiece) |
| 10 | Key Takeaways & Future Work | What we learned, what's next |

> Two optional appendix/back-matter slides (11–12) carry the full results tables and the verified stack reference, keeping the 10 core slides clean for presentation.

---

## Slide 1 — Title & Team

**Title:** Fake News & Misinformation Detection — *A classifier prior wrapped by an evidence-grounded fact-check that flags, never censors*

- **Project P11**, NLP in Industry final assignment.
- **Author:** Le Dinh Minh Quan — student **23127460**.
- One-line thesis: a *trainable fake-news text classifier* (fast prior) wrapped by an *agentic, evidence-grounded fact-check* — retrieve → stance/NLI → aggregate verdict with citations → **abstain** when uncertain.
- Package name: **`fakenews`**; runs FULLY OFFLINE on a `numpy`/`pandas`/`scikit-learn` floor (TF-IDF+LogReg classifier, BM25 retrieval, lexical stance — no torch required).
- Framing in one sentence: the tool **assists** human fact-checkers and moderators; it **never auto-censors**.

**Suggested visual:** Clean title card — project name, author + student id, university/course footer, and a small "flag for human review, not takedown" badge/lockup. Subtle background motif of a magnifying glass over a news article.

---

## Slide 2 — Business Problem & Motivation

**Title:** Why automated fact-check assistance — and why "assist," not "arbitrate"

- Misinformation moves faster than human fact-checkers can verify it; reviewers need **triage + evidence**, not another opaque black-box verdict.
- Four users / jobs-to-be-done:
  - **Journalists / professional fact-checkers** — "Is this claim worth checking, and what does the evidence say?" → verdict + cited passages + auditable trace.
  - **Platform trust-&-safety / moderators** — triage a firehose → a fast credibility signal that routes only suspicious items to a human (never an auto-takedown).
  - **Newsroom editors / researchers** — rate credibility on a scale → 6-way LIAR-style score (pants-fire → true).
  - **End readers (secondary)** — a transparent credibility indicator *with the evidence shown*, encouraging verification over blind trust.
- The hard problem: a classifier detects **style/source patterns correlated with fakeness** — which is **not** the same as detecting falsehood. The evidence layer exists to compensate.
- Business metrics: reviewer-throughput uplift, **flag-queue precision** (fraction of flags a human agrees were worth reviewing), evidence-grounding rate (target 100% of non-abstained verdicts ship ≥1 verbatim citation), and a low-but-useful override rate.

**Suggested visual:** A 2×2 "user → job → output" matrix (the four personas), with a side callout box reading *"classifier ≠ truth detector; that's why we add evidence."*

---

## Slide 3 — Proposed NLP Solution

**Title:** Two signals, reported separately — a fast prior and an evidence-grounded verdict

- **Signal 1 — Classifier (fast prior):** a binary real/fake transformer fine-tune giving `P(fake)` on every request.
  - Base models: `distilbert-base-uncased` (default / T4, Apache-2.0) → `microsoft/deberta-v3-base` (MIT) → `answerdotai/ModernBERT-base` (Apache-2.0, 8192-ctx for full articles).
  - **TF-IDF + LogReg baseline** is the no-torch floor every transformer must beat (especially cross-domain).
- **Signal 2 — Agentic fact-check (evidence-grounded verdict):** extract claim → **retrieve** evidence (BM25 + dense `all-MiniLM-L6-v2` fused by **RRF**) → **stance/NLI** per passage (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, zero-shot, MIT) → **aggregate** → **abstain** when uncertain.
- Stance bridge: `entailment → SUPPORTS`, `contradiction → REFUTES`, `neutral → NOT ENOUGH INFO` (premise = evidence, hypothesis = claim). NLI label order is read from `model.config.id2label` at load — never hardcoded.
- **Evidence dominates; the classifier prior is only a soft nudge** (`α = 0.6` on the stance vote). The two are surfaced **separately** so a reviewer immediately sees when prior and evidence disagree.
- Internal label convention: **`0 = real, 1 = fake`** — every dataset loader normalizes to this on load (mirrors disagree, see Slide 5).

**Suggested visual:** Two parallel lanes — "Classifier prior `P(fake)`" (fast, dashed) and "Evidence fact-check verdict" (slower, solid) — merging into a single output card that shows BOTH numbers side by side, never blended into one.

---

## Slide 4 — System Architecture Diagram

**Title:** One deterministic FSM, eight states, abstain reachable everywhere

- Pipeline spine: **INGEST/NORMALIZE → PARSE/CLAIM-EXTRACT → CLASSIFY → CHECK-WORTHY? → RETRIEVE → STANCE/NLI → AGGREGATE → DECIDE/ABSTAIN → PRESENT**, with **ABSTAIN** reachable from any state.
- Maps onto the canonical automated-fact-checking stages (FEVER shared task): claim detection / check-worthiness → document retrieval → evidence selection → verification (stance/NLI) → verdict `{SUPPORTED, REFUTED, NEI}`.
- The five decision gates live on the transitions: **D1** claim routing, **D2** check-worthiness / confidence skip, **D3** evidence-coverage, **D4** stance-aggregation verdict, **D5** confidence / abstain (detailed on Slide 7).
- Optional **LLM "brain"** never decides flow — at each D-point it may only *propose* a value from the legal set; on parse-fail / out-of-set / timeout the deterministic rule fires.
- **Cross-cutting transparency:** every transition emits a `ToolTrace(state, tool, inputs_hash, outputs, latency_ms, brain_proposal?, rule_applied)` so a reviewer can replay each step.
- **Graceful degradation:** with no torch, `ClassifyTool → TF-IDF+LogReg`, `RetrieveEvidence → TF-IDF cosine / BM25`, `StanceNLI → lexical mock-stance` — the whole FSM still runs offline.

**Suggested visual:** The FSM block diagram from `DESIGN_BRIEF.md` §3 — boxes for each state, D1–D5 labels on the branch arrows, the "skip-retrieval" shortcut from D2 to PRESENT, and a red ABSTAIN sink with arrows from D3/D4/D5. Footer band: "every transition → ToolTrace."

---

## Slide 5 — Data Overview

**Title:** Verified datasets — and the polarity & leakage traps we normalize away

- **Classifier core:**

  | Dataset (HF id) | Role | Size | License | Native polarity |
  |---|---|---|---|---|
  | `GonzaloA/fake_news` | **PRIMARY** binary | 40.6K (24.4K/8.1K/8.1K) | **unknown ⚠️** | 0=fake, 1=real |
  | `chengxuphd/liar2` | 6-way credibility | 23.0K | **Apache-2.0 ✅** | label 0–5 |
  | `ErfanMoosaviMonazzah/…-English` | license-clean binary alt | 44.3K | **openrail** | 0=fake, 1=real |
  | `mohammadjavadpirhadi/…-english` | MIT-clean binary alt | 10K–100K | **MIT ✅** | binary |
  | `LittleFish-Coder/Fake_News_GossipCop` | **cross-domain eval** | 12.7K | **Apache-2.0 ✅** | 0=real, 1=fake |

- **Fact-check / evidence:** `fever/fever` (cc-by-sa-3.0 + gpl-3.0 ⚠️ copyleft), `copenlu/fever_gold_evidence` (verdict-head fine-tune), `BeIR/fever` + `BeIR/fever-qrels` (cc-by-sa-4.0 — used to measure evidence recall@k).
- ⚠️ **Polarity gotcha — normalize on load.** `GonzaloA`/`ErfanMoosaviMonazzah`/`WELFake` → 0=fake; `LittleFish-Coder/*` → 0=real; `mrm8488/fake-news` → 1=fake. Every loader applies an explicit per-source `label_map` to the repo convention **0=REAL, 1=FAKE** — never assume.
- ⚠️ **License flags:** `GonzaloA` license is **unknown** (research/eval only); all `fever/*` and `BeIR/*` are **copyleft / share-alike** — a released stance model inherits share-alike. `allenai/scifact` is **cc-by-nc (non-commercial) — avoided**; we use `BeIR/scifact` (cc-by-sa-4.0) instead. `KaiDMML/FakeNewsNet` is a **crawler, not a packaged dataset** (network-gated, excluded from CI) — we use the Apache-2.0 `LittleFish-Coder/*` mirrors.
- ⚠️ **Source-style leakage (the #1 trap):** "real" rows are Reuters/AP wire copy, "fake" rows are blog-rant style — 99% in-domain F1 means the model learned *"is this Reuters formatting,"* not veracity. Mitigation: strip dateline/source boilerplate, report the **in-domain vs cross-domain gap**.
- **Offline fallback** (`fakenews/data/samples.py`): ~36 synthetic labeled news items, ~16 evidence snippets, ~6 `ClaimCase` with gold FEVER verdicts → the whole pipeline runs with no network.

**Suggested visual:** Split panel — left: the dataset table with a license-color legend (green = permissive, amber = unknown/copyleft, red = non-commercial); right: a small "polarity normalizer" diagram showing three mirrors with different label orders all funneling into a single `0=real / 1=fake` box.

---

## Slide 6 — Model & Evaluation Results

**Title:** Macro-F1 is the headline — and the cross-domain drop is the honest number

- **Classification metrics:** Macro-F1 (headline — handles imbalance), accuracy + per-class P/R/F1, **ROC-AUC**, and **ECE (Expected Calibration Error)** with a reliability diagram (over-confidence here silences real news; temperature-scale on a held-out split, report pre/post ECE).
- **Costly error called out:** false positives (legit journalism flagged fake) are the high-cost error → we report **`real`-recall** and **`fake`-precision** explicitly, not just accuracy.
- **Classifier results skeleton** (must beat majority-class, TF-IDF+LogReg, and zero-shot):

  | System | Split | Acc | Macro-F1 | P(fake) / R(real) | ROC-AUC | ECE |
  |---|---|---|---|---|---|---|
  | Majority class | GonzaloA test | | | | — | — |
  | TF-IDF + LogReg | GonzaloA test | | | | | |
  | Zero-shot (`bart-large-mnli`) | GonzaloA test | | | | | |
  | ModernBERT fine-tune | GonzaloA test | | | | | |
  | ModernBERT fine-tune | **GossipCop (cross-domain)** | | | | | |
  | ModernBERT fine-tune | LIAR2 test (6-way) | | | — | | |

- **Cross-domain is the real metric:** train on `GonzaloA` (PolitiFact-style) → test on `GossipCop`; expect a large macro-F1 drop. *That drop is itself an ethics metric* — it quantifies source-style leakage. A too-high in-domain F1 (~0.95+) is a red flag, not success.
- **Fact-check results skeleton** (FEVER-style: label accuracy, FEVER score, evidence **recall@k** vs `BeIR/fever-qrels`, abstain rate / selective accuracy):

  | Fact-check system | Label acc. | FEVER score | R@5 (evidence) | Abstain rate |
  |---|---|---|---|---|
  | Retrieval + NLI (MoritzLaurer DeBERTa, zero-shot) | | | | |
  | + cross-encoder rerank | | | | |
  | + FEVER-fine-tuned stance head (optional) | | | | |

**Suggested visual:** Two stacked result tables (classifier + fact-check) with the cross-domain row highlighted, plus a small inset **reliability diagram** (predicted vs actual confidence) and a **risk–coverage curve** illustrating selective accuracy vs abstain rate.

---

## Slide 7 — Agentic AI Component (D1–D5)

**Title:** Five deterministic decision points, each acting on intermediate outputs

| ID | State | Rule (threshold) | Branches |
|---|---|---|---|
| **D1** | PARSE / CLAIM-EXTRACT | `len(tokens) ≤ 40` or no body → short-claim route; else article route (central claim = title / lead / top-TextRank); `claim_len ≥ 5` else `INVALID_INPUT` | short ↔ article |
| **D2** | CHECK-WORTHY? | skip gate = `(classifier_conf ≥ τ_skip=0.95) AND (checkworthy < τ_cw=0.5)` | skip → present prior; else → RETRIEVE |
| **D3** | RETRIEVE (coverage) | keep evidence with rerank `score ≥ τ_rel=0.3`; coverage OK if `n_relevant ≥ N_min=3`; else widen up to `R_max=2` retries | enough → STANCE; exhausted → ABSTAIN |
| **D4** | AGGREGATE (verdict) | `S = Σ wᵢ·sᵢ` (entail +1 / contra −1 / neutral 0); `score = α·stance + (1−α)·prior`, **α=0.6**; verdict by `±θ`, **θ=0.2** | REAL \| FAKE \| UNVERIFIED |
| **D5** | DECIDE / ABSTAIN | abstain if `verdict==UNVERIFIED` OR `agreement < τ_agree=0.4` OR `final_conf < τ_present=0.55` OR **prior↔stance conflict** | present \| abstain ("needs human review") |

- **Evidence dominates the verdict** (`α = 0.6` weight on the stance vote); the classifier prior is a soft nudge, not the decider.
- The optional **brain** may only PROPOSE constrained values (a central-claim string from a fixed candidate set, a check-worthy boolean, query-expansion terms, a near-tie stance, a rationale string) — it **cannot** change thresholds, override a confident NLI head, or flip abstain → present. On failure the deterministic rule fires.
- **Conflict surfacing (D5):** if the classifier says FAKE but evidence strongly SUPPORTS (or vice-versa), the output is flagged `conflict — needs human review`, not silently resolved.
- **Worked example** ("WHO declared drinking bleach cures COVID-19"): D1 short-claim → D2 proceed (checkworthy 0.92) → D3 4 relevant passages → 3× contradiction → D4 verdict **FAKE** (`combined_conf 0.91`) → D5 present with 3 refuting citations. A novel uncovered claim instead widens twice then **ABSTAINs** — never a guessed label.

**Suggested visual:** A horizontal "decision pipeline" strip — five gates D1→D5 as diamonds with their key threshold annotated on each, the worked-example claim flowing through and lighting up each gate, and a branch peeling off to a red ABSTAIN node.

---

## Slide 8 — Deployment Overview

**Title:** One service, lazy imports — boots and serves even with no GPU and no torch

- **FastAPI `fakenews-api`** exposes both signals; all heavy imports are lazy, so with only numpy/pandas/sklearn the service still serves the TF-IDF classifier + TF-IDF/BM25 fact-check.

  | Endpoint | Method | Returns |
  |---|---|---|
  | `/classify` | POST | `{label, probability (P(fake)), model_version}` |
  | `/factcheck` | POST | `{verdict, confidence, classifier_prior, evidence[], rationale, decisions_trace[], abstained}` |
  | `/healthz` | GET | `{status, models_loaded}` (503 until warm) |
  | `/version` | GET | pinned classifier / NLI / retriever ids **+ revisions** + `index_built_at` + `git_sha` |

- `/factcheck` **always** returns the evidence list and `decisions_trace`, even on abstain — transparency is non-optional. `classifier_prior` and `verdict` are returned as **separate fields**.
- **Gradio UI** (HF Space under `ledinhminhquan`): Tab 1 *Classify* (label + confidence bar + a "style/probability signal, not a verdict" notice); Tab 2 *Fact-check* (verdict chip real/fake/**unverified**, confidence gauge, highlighted evidence list with stance badges support=green / refute=red / neutral=grey + source citations). Abstain renders prominently as "Not enough evidence — flagged for human review."
- **CLI:** `fakenews classify`, `fakenews factcheck --claim … --k 5 --json`, `fakenews serve`, `fakenews index build`.
- **Packaging:** Dockerfile (CPU base; optional CUDA layer auto-adapting H100/A100/L4/T4 via `torch.cuda` detection, falling back to CPU/TF-IDF). `pip install fakenews` = light core; `fakenews[torch]` / `[gpu]` extras pull the transformer stack.
- **Versioning:** HF **revisions / commit SHAs** are pinned into every response so a Hub update can never silently change a verdict.

**Suggested visual:** Deployment topology — client → stateless API replicas behind a load balancer → shared read-only evidence index volume; the classifier path marked "fast / every request," the NLI fact-check path marked "gated, cached by claim hash." Inset: the Gradio two-tab mockup.

---

## Slide 9 — Ethics, Privacy & Risks  *(centerpiece)*

**Title:** It flags for human review — it never removes, blocks, or censors

- **Non-negotiable framing:** the system outputs a **review flag + evidence**, never an enforcement action. **No auto-takedown anywhere** in the design. `unverified` / abstain is a first-class outcome, not a failure mode.
- **Top risks and mitigations:**

  | Risk | Mitigation |
  |---|---|
  | Censorship / free-speech chilling | Advisory flag + evidence only; no auto-takedown; verdicts never trigger removal |
  | False positives silencing real news | Optimize for high `real`-recall; report per-class costs; **abstain** rather than guess; human sign-off |
  | Political bias in labels/data | Document label provenance; report performance sliced by topic/speaker/source; never claim neutral ground truth |
  | Source-style leakage | Measure PolitiFact→GossipCop drop; strip source/style features; prefer the evidence verdict; prior is weak-only |
  | Automation bias / over-trust | Calibration (ECE) + mandatory confidence display; abstain on uncertainty; UI states "decision support, read the evidence" |
  | Hallucinated / misattributed citations | Citations are **extracted verbatim** from the corpus with source IDs, never generated; `decisions_trace` lets a reviewer verify each one |

- **Reported separately, always:** `classifier_prior` (a style/source signal) vs the evidence-grounded `verdict` — so a reviewer sees disagreement instead of a blended, falsely-confident number.
- **Limitations stated plainly:** English-centric data; US-politics + celebrity domain skew; "fake/real" is a coarse proxy for a spectrum; the system verifies against a **fixed corpus** and cannot know facts outside it; **`unverified` means insufficient evidence, NOT true.**
- **Design commitments baked in:** mandatory transparency + verbatim citations on every verdict; abstain under uncertainty; assist rather than replace human judgment; **never take an irreversible action on content automatically.**

**Suggested visual:** A bold central banner — "FLAGS FOR HUMAN REVIEW · NEVER CENSORS" — flanked by the risk→mitigation table; a small "two separate gauges" motif (classifier_prior vs evidence verdict) reinforcing that the signals are never merged.

---

## Slide 10 — Key Takeaways & Future Work

**Title:** What we built, what we learned, where it goes next

- **Built:** a `fakenews` package pairing a trainable real/fake classifier (TF-IDF floor → ModernBERT/DeBERTa fine-tune) with an agentic, evidence-grounded fact-check (BM25+dense+RRF retrieval → zero-shot DeBERTa-NLI stance → aggregated verdict with citations and a 5-gate abstain policy).
- **Learned:** style classifiers detect *source/format*, not *truth* — the cross-domain (PolitiFact→GossipCop) macro-F1 drop makes that visible and is treated as an **ethics metric**; calibration (ECE) and abstention matter as much as raw accuracy.
- **Engineering wins:** runs **fully offline** on a numpy/pandas/sklearn floor; lazy heavy imports; deterministic FSM with an optional, strictly-constrained LLM brain; full `ToolTrace` auditability; pinned model revisions for reproducible verdicts.
- **Future work:**
  - Optional **FEVER-fine-tuned stance head** (`copenlu/fever_gold_evidence` / `pietrolesci/nli_fever`) and `tals/vitaminc` for robustness to subtle edits.
  - Live / refreshable evidence index with timestamping; expand beyond English and beyond the US-politics + celebrity domain skew.
  - Adversarial-paraphrase eval set; per-topic / per-speaker / per-source fairness slicing; richer 6-way LIAR credibility-scale deliverable.
  - License hardening for redistribution (replace unknown-license `GonzaloA` with MIT/openrail mirrors; track FEVER share-alike obligations on any released stance model).

**Suggested visual:** A two-column "Takeaways | Future Work" layout, with a small timeline/roadmap arrow at the bottom (zero-shot stance → fine-tuned stance → live index → multilingual). Closing footer reiterating the thesis: *flag + evidence, never censor.*

---

### Appendix A (back-matter) — Full results tables
Carries the complete classifier + fact-check result tables (Slide 6 skeletons, populated), reliability diagram, risk–coverage curve, and per-class / per-slice breakdowns for Q&A.

### Appendix B (back-matter) — Verified stack reference
The verified HF id / license table (classifier bases, NLI/stance models, retrievers, datasets) with permissive ✅ / unknown ⚠️ / copyleft ⚠️ / non-commercial ⛔ flags, for reproducibility and license-audit questions.
