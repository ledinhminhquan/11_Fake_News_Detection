# P11 — Project Plan, Timeline & Teamwork

**Project:** Fake News & Misinformation Detection System (package `fakenews`)
**Course:** NLP in Industry — final assignment
**Author:** Le Dinh Minh Quan — student `23127460`
**Doc spec:** Project Management & Teamwork (Section I.10)
**Source of truth:** [`docs/DESIGN_BRIEF.md`](./DESIGN_BRIEF.md) (verified HF ids, pipeline, FSM D1–D5, deployment, metrics, ethics)

> **One-line thesis.** A *trainable fake-news text classifier* (a fast prior, `P(fake)`) wrapped by an *agentic, evidence-grounded fact-check*: extract claim → retrieve evidence (BM25 + dense `all-MiniLM-L6-v2` + RRF) → stance/NLI per evidence → aggregate verdict (evidence dominates; the classifier prior is a soft nudge) → **abstain** when uncertain. The tool **flags content for human review; it never auto-removes or censors.** This is the most ethically loaded project in the set, and the plan below treats ethics as a first-class workstream, not a postscript.

This document is a single-author deliverable, but it is written as if delivered by a **simulated cross-functional team** (Section I.10). Each task is tagged with the role that would own it in a real org, so the plan doubles as a teamwork/ownership map and a scaling argument.

---

## 1. Scope, constraints & the no-torch contract

The plan is bounded by three hard constraints that shape every milestone:

| Constraint | Consequence for the plan |
|---|---|
| **Runs fully offline, no torch.** The core install (`pip install -e .`, deps = `numpy, pyyaml, pydantic, scikit-learn, joblib`) ships a TF-IDF + LogReg classifier, BM25 lexical evidence retrieval, a lexical-overlap stance, and the full agent FSM + metrics — all with **zero network and no `torch`** (see `pyproject.toml`). Heavy libs (`torch`, `transformers`, `datasets`, `sentence-transformers`) are the `[ml]` extra, imported **inside** `run()`. | Every milestone has a CPU/offline acceptance gate **before** the GPU path. CI never downloads. The synthetic fallback (`data/samples.py`) is built **first** (Milestone M1), so all downstream work is testable without a single Hub call. |
| **Internal label convention `1 = fake, 0 = real`** (`config.py` header; `fake` is the positive/detected class). HF mirrors disagree — `GonzaloA` `0=fake`, `LittleFish` `0=real`, `mrm8488` `1=fake`. | A per-source `label_map` normalization task is a **blocking prerequisite** for any training work, and is its own QA checkpoint (M2). `DataConfig.fake_label_value` records the raw value meaning *fake* per source. |
| **Assist, never censor.** Output is always a *review flag + evidence*; `unverified`/abstain is first-class; `classifier_prior` and the evidence `verdict` are reported **separately**. | The Ethics Reviewer role has sign-off authority at three gates (M4, M7, M9). No milestone is "done" if it could be read as an auto-takedown. |

---

## 2. Roles (simulated team)

Seven roles own the work. In the actual deliverable one person plays all seven; the tags make the division of labour, interfaces, and scaling story explicit.

| Role | Abbrev | Owns (repo modules) | Primary responsibility |
|---|---|---|---|
| **Project Manager** | **PM** | `cli.py`, `automation/`, `grading/checklist.py`, this plan | Timeline, milestone gates, scope control, rubric tracking, the human-in-the-loop review-board contract |
| **ML Engineer — Classifier** | **MLE-C** | `models/classifier.py`, `training/train_classifier.py`, `training/train_baseline.py`, `training/tune.py` | The trainable prior: TF-IDF+LogReg floor → transformer fine-tune; calibration |
| **IR / Retrieval Engineer** | **IR** | `factcheck/retriever.py`, `models/bm25.py`, evidence corpus build | BM25 + dense + RRF + rerank evidence retrieval; recall@k |
| **NLI / Stance Engineer** | **NLI** | `factcheck/stance.py`, `factcheck/verdict.py` | Zero-shot stance (`DeBERTa-v3-base-mnli-fever-anli`), verdict aggregation, abstain logic |
| **Backend / Serving Engineer** | **BE** | `api/` (`main.py`, `app_combined.py`, `dependencies.py`, `schemas.py`, `ui.py`) | FastAPI `/classify` + `/factcheck` + `/healthz` + `/version`; Gradio UI; CLI wiring |
| **MLOps / Platform** | **OPS** | `config.py`, `models/model_registry.py`, `monitoring/drift_report.py`, `logging_utils.py`, Docker/Space, CI | Versioning, model registry, index build, drift, packaging, reproducibility |
| **Ethics / Safety Reviewer** | **ETH** | `agent/policy.py` (thresholds), `analysis/error_analysis.py`, ethics sections of report/UI | Abstain semantics, cross-domain leakage audit, fairness slices, no-censorship sign-off |

The **agent spine** (`agent/state.py`, `agent/policy.py`, `agent/tools.py`, `agent/fakenews_agent.py`, `agent/llm_orchestrator.py`) is co-owned by **NLI + PM** because it encodes both the verdict logic and the decision-governance (D1–D5) that ETH must approve.

---

## 3. Phased plan (the build order)

The brief's build order is the backbone: **research → data (seed + datasets) → classifier training + baseline → evidence retriever → stance/NLI → agent FSM → API/UI → eval → docs/report/slides.** Each phase below lists its owning roles, the concrete modules touched, and an exit gate.

```
 Phase 0   Research & design          ─ DESIGN_BRIEF.md (DONE)        ─┐
 Phase 1   Data model + offline seed  ─ data/samples.py, config.py    │ everything
 Phase 2   Classifier + baseline      ─ models/, training/            │ downstream
 Phase 3   Evidence retriever         ─ factcheck/retriever.py, bm25  │ is testable
 Phase 4   Stance / NLI + verdict     ─ factcheck/{stance,verdict}    │ offline from
 Phase 5   Agent FSM (D1–D5)          ─ agent/*                       │ Phase 1 on.
 Phase 6   API / UI / CLI / Docker    ─ api/, cli.py, Dockerfile      │
 Phase 7   Eval (incl. cross-domain)  ─ training/{evaluate,metrics}   │
 Phase 8   Ops: report/slides/grade   ─ autoreport/, grading/         ┘
```

### Phase 0 — Research & design  *(PM, ETH)* — **complete**
- Verify every HF id live via `hub_repo_details`; record license + polarity per source.
- Lock the FSM (5 decision points), the ethics framing, and the metric set.
- **Exit gate:** `DESIGN_BRIEF.md` self-contained and signed off. ✅

### Phase 1 — Data model + offline seed  *(MLE-C, OPS)*
- Canonical dataclasses `NewsItem` / `Evidence` / `ClaimCase` (label convention `1=fake, 0=real`).
- `data/samples.py`: ~36 balanced synthetic `SAMPLE_NEWS`, ~16 `SAMPLE_EVIDENCE` (support / refute / off-topic), ~6 `SAMPLE_CLAIMS` with gold FEVER verdicts (≥1 SUPPORTS, ≥1 REFUTES, ≥1 NEI).
- `config.py`: typed `AppConfig` with all thresholds; env-var paths; YAML loader.
- **Exit gate:** the *entire* pipeline runs on `samples.py` with **only** numpy/pandas/sklearn — no torch, no network.

### Phase 2 — Classifier + baseline  *(MLE-C; ETH advises on leakage)*
- `models/baseline_tfidf.py` → TF-IDF(1–2-gram) + LogReg(`class_weight="balanced"`) floor (the number to beat).
- `data/download_dataset.py` + loaders: `GonzaloA/fake_news` (PRIMARY, 40.6K, polarity-flipped), `chengxuphd/liar2` (Apache, 6-way credibility collapse `0–2=fake / 3–5=real`).
- `models/classifier.py` + `training/train_classifier.py`: transformer fine-tune, base = `distilbert-base-uncased` (default, T4) / `microsoft/deberta-v3-base` (MIT) / `answerdotai/ModernBERT-base` (8192-ctx); HW auto-adapt (bf16/fp16 by `cuda.get_device_capability`); ≤3 epochs + early stopping; class weights; temperature-scaling calibration.
- **Anti-leakage tasks (mandatory):** strip dateline/source boilerplate (`(Reuters)`, `WASHINGTON —`, bylines); exact + near-dup dedup; drop LIAR metadata columns (`speaker`, `state_info`, `*_counts`).
- **Exit gate:** transformer beats TF-IDF baseline on the **cross-domain** split (not just in-domain); ECE reported pre/post temperature scaling.

### Phase 3 — Evidence retriever  *(IR)*
- `models/bm25.py` (lexical floor, no torch) + dense `sentence-transformers/all-MiniLM-L6-v2` (384-d, Apache) + RRF fusion (`rrf_k=60`) + cross-encoder rerank `cross-encoder/ms-marco-MiniLM-L6-v2`.
- `data/evidence_corpus.py`: build the corpus from LIAR2 `justification` rows + articles + the synthetic seed.
- Versioned, memory-mapped index; `index build` CLI path.
- **Exit gate:** evidence recall@k measured against `BeIR/fever-qrels`; BM25-only path returns sane hits with torch absent.

### Phase 4 — Stance / NLI + verdict  *(NLI)*
- `factcheck/stance.py`: zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT, 184M); **read `model.config.id2label` at load** (`{0:entail,1:neutral,2:contradiction}` — never hardcode); premise = evidence, hypothesis = claim; map entail→support, contra→refute, neutral→NEI. Lexical mock-stance fallback (`backend:"none"`) when transformers absent.
- `factcheck/verdict.py`: weighted stance sum `S = Σ wᵢ·sᵢ`, combine with prior `score = α·stance + (1−α)·prior` (`prior_weight=0.3`, i.e. evidence dominates); verdict REAL/FAKE/UNVERIFIED by `verdict_margin=0.2`.
- **Exit gate:** worked example from §5 of the brief reproduces a `FAKE`/REFUTED verdict with 3 refuting citations; the abstain counter-example yields `UNVERIFIED`.

### Phase 5 — Agent FSM (D1–D5)  *(NLI + PM; ETH sign-off)*
- `agent/state.py` (`AgentState`, `ToolTrace`, `Action` enum), `agent/policy.py` (pure D1–D5 rules), `agent/tools.py` (tool wrappers), `agent/fakenews_agent.py` (FSM spine), `agent/llm_orchestrator.py` (optional brain that only *proposes* legal values; rule fires on parse-fail/timeout).
- Wire the five decisions to the live config thresholds:

| ID | State | Rule (config field) | Branch |
|---|---|---|---|
| **D1** | claim routing | `n_words ≤ short_claim_words (40)` → short claim; else article + central-claim extract | claim / article |
| **D2** | check-worthiness gate | confident article (`clf_prob ≥ skip_factcheck_confidence (0.95)`) & not a claim → skip retrieval | skip / factcheck |
| **D3** | evidence-coverage gate | `n_relevant ≥ min_evidence` with `relevance ≥ min_evidence_relevance (0.05)`; else widen/abstain | ok / insufficient |
| **D4** | verdict gate | weighted stance + prior; `|S| > verdict_margin (0.2)` → REAL/FAKE else UNVERIFIED | real / fake / unverified |
| **D5** | confidence/abstain gate | emit only if `confidence ≥ min_verdict_confidence (0.5)`; flag prior↔evidence conflict | present / abstain |

- **Exit gate:** every transition logs a `ToolTrace`; every decision is exercised by a test; the agent runs end-to-end on `samples.py` with no torch.

### Phase 6 — API / UI / CLI / Docker  *(BE, OPS)*
- `api/`: FastAPI `/classify`, `/factcheck` (always returns `evidence` + `decisions_trace`, even on abstain), `/healthz`, `/version` (stamps `classifier_version`, NLI/retriever ids **+ revisions**, `index_built_at`, `git_sha`).
- `api/ui.py`: Gradio Space (under `ledinhminhquan`) — Tab 1 Classify (with a visible "style/probability signal, not a verdict" notice), Tab 2 Fact-check (verdict chip real/fake/**unverified**, stance badges support=green / refute=red / neutral=grey, verbatim citations).
- `cli.py` subcommands (already scaffolded): `data, train-classifier, train-baseline, tune, evaluate, classify, factcheck, demo-agent, serve, benchmark, error-analysis, monitor, generate-report, generate-slides, autopilot, grade`.
- Dockerfile (CPU base + optional CUDA layer); `fakenews` / `fakenews[ml]` / `[api]` extras.
- **Exit gate:** service boots and serves classify + fact-check with **only** the core install (TF-IDF + BM25 path); `/factcheck` never omits evidence.

### Phase 7 — Evaluation incl. cross-domain  *(MLE-C, IR, NLI; ETH)*
- `training/metrics.py` + `training/evaluate.py`: accuracy, **macro-F1 (headline)**, per-class P/R/F1, ROC-AUC, **ECE** for the classifier; **label accuracy + FEVER score + evidence recall@k + abstain/coverage** for the fact-check.
- **Cross-domain (mandatory, ethics metric):** train on `GonzaloA` (PolitiFact-style) → test on `LittleFish-Coder/Fake_News_GossipCop`; report the macro-F1 drop as the source-leakage signal.
- Baselines to beat: majority class, TF-IDF+LogReg, zero-shot (`bart-large-mnli`).
- **Exit gate:** results tables populated for both in-domain and cross-domain; risk–coverage curve for abstain.

### Phase 8 — Ops: report / slides / grade  *(OPS, PM, ETH)*
- `autoreport/` charts (confusion matrix, verdict distribution), PDF report, PPTX slides.
- `monitoring/drift_report.py` (input-text / label drift), `analysis/error_analysis.py` + fairness slices (per-topic / per-source).
- `grading/checklist.py`: rubric (classifier beats baseline; fact-check cites evidence; abstains under uncertainty).
- **Exit gate:** `fakenews autopilot` runs the whole chain; `fakenews grade` passes the rubric.

---

## 4. Milestones & timeline

A simulated **9-week** schedule (the realistic effort for a team; compressed for a single author). Weeks are relative.

| # | Milestone | Phases | Lead role(s) | Exit / acceptance gate | Wk |
|---|---|---|---|---|---|
| **M0** | Design locked | 0 | PM, ETH | `DESIGN_BRIEF.md` signed off; ids verified | 1 |
| **M1** | Offline skeleton runs | 1 | MLE-C, OPS | Full pipeline on `samples.py`, no torch/network | 2 |
| **M2** | Data normalized + baseline | 2 | MLE-C | All loaders apply per-source `label_map`; TF-IDF baseline macro-F1 logged | 3 |
| **M3** | Classifier + calibration | 2 | MLE-C, ETH | Transformer trained; ECE pre/post; boilerplate stripped | 4 |
| **M4** | Evidence retriever | 3 | IR | recall@k vs `BeIR/fever-qrels`; BM25 fallback works · **ETH gate: corpus provenance + license** | 5 |
| **M5** | Stance + verdict + FSM | 4, 5 | NLI, PM | Worked example + abstain example reproduce; D1–D5 traced; **ETH gate: abstain is first-class** | 6 |
| **M6** | API / UI / CLI / Docker | 6 | BE, OPS | Service boots on core-only install; `/factcheck` always returns evidence + trace | 7 |
| **M7** | Evaluation + cross-domain | 7 | MLE-C, NLI, ETH | In-domain **and** cross-domain tables; **ETH gate: cross-domain drop reported honestly** | 8 |
| **M8** | Report, slides, grade | 8 | OPS, PM, ETH | `autopilot` green; `grade` passes; **ETH final no-censorship sign-off** | 9 |

**Critical path:** M1 → M2 → M3 → M5 → M7 → M8. M4 (retriever) runs in parallel with M3 (classifier) because they share no code; both must land before M5 (the FSM needs both a prior and evidence). The single longest pole is **data normalization + leakage control (M2/M3)** — the fake-news-specific trap, not the modelling.

---

## 5. Task breakdown (by role)

### PM
- Maintain this plan + the milestone gates; run M0/M5/M8 sign-offs.
- Own `cli.py` command surface and `automation/autopilot.py` orchestration.
- Track the rubric via `grading/checklist.py`.
- Define and document the **human-in-the-loop review-board contract** (see §8).

### MLE-C (classifier)
- `models/baseline_tfidf.py` TF-IDF+LogReg floor (no torch).
- `models/classifier.py` transformer wrapper (lazy torch, `num_labels` 2 or 6).
- `training/{train_classifier, train_baseline, tune}.py`; HW auto-adapt; class weights; early stopping.
- Temperature-scaling calibration + ECE.
- Boilerplate stripping + dedup + LIAR metadata drop (with ETH).

### IR (retrieval)
- `models/bm25.py` lexical retriever (no torch).
- `factcheck/retriever.py` dense + RRF + cross-encoder rerank (lazy sentence-transformers).
- `data/evidence_corpus.py` corpus build (LIAR2 `justification` + articles + seed).
- Index build/versioning; recall@k harness vs `BeIR/fever-qrels`.

### NLI (stance + verdict)
- `factcheck/stance.py` zero-shot DeBERTa-mnli-fever-anli + lexical fallback; `id2label` read at runtime.
- `factcheck/verdict.py` weighted aggregation + prior nudge + abstain.
- Co-own the agent spine: tool contracts, D4/D5 logic.

### BE (serving)
- `api/{main, app_combined, dependencies, schemas, ui}.py`.
- `/classify`, `/factcheck`, `/healthz`, `/version`; Gradio two-tab UI; CLI `serve`.
- Lazy-load with TF-IDF/BM25 fallback so the service boots core-only.

### OPS (platform)
- `config.py`, `logging_utils.py`, `models/model_registry.py`.
- `monitoring/drift_report.py`; Dockerfile; HF Space; CI (download-free, no torch).
- **Versioning discipline:** pin HF **revisions/commit SHAs**, not just repo names, so a Hub update can never silently change a verdict; stamp versions into every response and `/version`.

### ETH (ethics / safety)
- Own abstain semantics and threshold review in `agent/policy.py`.
- `analysis/error_analysis.py` + fairness slices (per-topic / per-source / per-speaker).
- Cross-domain leakage audit (M7); ethics sections of report + UI notices.
- Final sign-off that nothing in the deliverable performs or implies an auto-takedown.

---

## 6. Dependencies & interfaces between roles

The clean interfaces are what make the team parallelizable. Every tool exposes the uniform `run(**kwargs) -> {"ok", "data", "meta"}` contract, so roles integrate against signatures, not implementations.

```
 MLE-C  ──(p_fake float)──────────────►┐
                                        ├──► NLI: AggregateVerdict (prior nudge, α=0.3 prior weight)
 IR     ──(evidence[] + relevance)────►┘            │
                                                     ▼
 NLI    ──(stances[], verdict, citations)───► PM/NLI: agent FSM (D1–D5) ──► BE: /factcheck JSON
                                                     │                            │
 OPS    ──(model_registry versions, index)───────────┴────────────────────────────┤
                                                                                   ▼
 ETH    ──(thresholds, abstain rule, fairness gate)───────────────────► report + UI + grade
```

- **MLE-C → NLI:** the classifier exposes `p_fake` only; NLI must treat it as a *weak prior* (`prior_weight=0.3`). This boundary is deliberate — it keeps `classifier_prior` and `verdict` reported **separately** (ethics requirement).
- **IR → NLI:** retriever returns evidence with `relevance`; the coverage gate (D3) and stance both read it. Contract: verbatim passages with source ids — never generated text.
- **OPS ↔ everyone:** the model registry + `/version` is the integration ledger; no model ships without a pinned revision.
- **ETH ↔ PM:** ETH holds veto at M4/M5/M7/M8; PM cannot mark those milestones done without it.

---

## 7. Risks & mitigations

The three signature risks of *fake-news* work — label noise, source/style leakage, and the ethics surface — get the most attention. Each has an owner and a concrete, code-level mitigation already wired into the design.

| Risk | Severity | Owner | Why it bites | Mitigation (concrete) |
|---|---|---|---|---|
| **Label-polarity noise across mirrors** | High | MLE-C, OPS | `GonzaloA` `0=fake`, `LittleFish` `0=real`, `mrm8488` `1=fake` — silently flips the model | Per-source `label_map` on load (`DataConfig.fake_label_value`); verify on a known row; unit test per loader; internal convention `1=fake` everywhere |
| **Source / style leakage** (#1 trap) | High | MLE-C, ETH | "real" rows are Reuters/AP wire copy, "fake" are blog-rant style → 99% in-domain F1 measures *outlet*, not *truth* | Strip dateline/source boilerplate before training; **mandatory cross-domain eval** (GonzaloA→GossipCop) with the macro-F1 drop reported as an ethics metric; treat too-high in-domain F1 as a red flag; the **evidence verdict**, not the prior, is the trusted output |
| **Ethics: censorship / silencing real news** | High | ETH | An automated "fake" label that triggers removal chills lawful speech; false positives on legit journalism are the high-cost error | Output = **review flag + evidence**, never enforcement; **no auto-takedown anywhere**; optimize operating point for high `real`-recall; **abstain** rather than guess; human sign-off; ECE + confidence display to fight automation bias |
| **Over-confidence / miscalibration** | Med-High | MLE-C | Confident-and-wrong outputs propagate; reviewers rubber-stamp | Temperature scaling on a calibration split; report ECE + reliability diagram; `min_verdict_confidence=0.5` floor; abstain on low confidence |
| **Train/test contamination (dup)** | Med | MLE-C | WELFake/GonzaloA share sources → leaked test rows inflate scores | Exact + near-dup (MinHash) dedup within and across splits before training |
| **Prior↔evidence conflict** | Med | NLI, ETH | Classifier says FAKE but evidence SUPPORTS (or vice versa) → which to trust? | Report both **separately**; D5 raises a `conflict` flag → "needs human review"; evidence dominates (`α` favours stance) |
| **Stale / out-of-scope evidence** | Med | IR, OPS | Index ages; novel claims have no coverage → tempting to fabricate | Version + timestamp the index (`/version`); D3 coverage gate → **abstain** when retrieval is thin; never invent evidence |
| **Hallucinated / misattributed citations** | Med | NLI, BE | A fact-check is only trustworthy if its citations are real | Citations extracted **verbatim** from the corpus with source ids; `decisions_trace` lets a reviewer verify each one |
| **Adversarial paraphrase / evasion** | Med | NLI, ETH | Bad actors paraphrase to flip the style classifier | Evidence-grounded fact-check is more robust than style; keep a paraphrase eval set; don't ship an "evasion score" (dual-use) |
| **License contamination** ⚠️ | Med | OPS | `GonzaloA` license **unknown**; all FEVER mirrors **cc-by-sa + gpl copyleft**; `allenai/scifact` **cc-by-nc** ⛔; `KaiDMML/FakeNewsNet` is a crawler, not a dataset (Twitter ToS) | Use clean alternatives where possible (`chengxuphd/liar2` Apache, `mohammadjavadpirhadi/…` MIT, `LittleFish-Coder/*` Apache, `BeIR/*` for retrieval eval); flag the unknown/copyleft sources in docs; **a stance model fine-tuned on FEVER inherits share-alike** — flag on redistribution; exclude FakeNewsNet raw from CI (network-gated) |
| **Heavy-dep / GPU unavailability** | Low | OPS, BE | Grader may have no GPU / no torch | The no-torch core is the safety net: TF-IDF classifier + BM25 retrieval + lexical stance keep the *whole* pipeline runnable; CI proves it download-free |

---

## 8. Human-in-the-loop & the review board

The non-negotiable framing — **assist, never censor** — is operationalized as a review-board contract, owned by ETH + PM:

- **The model never takes an irreversible action.** `/factcheck` emits a *flag + evidence + `decisions_trace`*; a human moderator/fact-checker decides. There is no auto-remove endpoint anywhere in `api/`.
- **Abstain is a first-class outcome.** `unverified` means *insufficient evidence*, not *true*; it renders prominently in the UI ("Not enough evidence — flagged for human review"), never as a fake label.
- **Two signals, shown separately.** `classifier_prior` (style/source pattern) and the evidence `verdict` are never merged into one number in the output; when they conflict, D5 surfaces it.
- **Every verdict ships its evidence.** Verbatim citations + `decisions_trace` make each step auditable line-by-line — the precondition for a human to override.
- **Reviewer feedback loop (scaling hook).** Override decisions and "was the evidence useful?" signals feed back as labelled data and as a precision metric for the flag-for-review queue.

---

## 9. Reflection: scaling from one author to a team

This plan is built so that the seven simulated roles become seven real people with minimal rework. Three design choices make that scaling clean:

**1. Module ownership maps 1:1 to roles.** The package is already partitioned along role boundaries: `models/` + `training/` (MLE-C), `factcheck/retriever.py` + `models/bm25.py` (IR), `factcheck/{stance,verdict}.py` (NLI), `api/` (BE), `config.py` + `model_registry.py` + `monitoring/` (OPS), `agent/policy.py` thresholds + `analysis/` (ETH), `cli.py` + `automation/` + `grading/` (PM). The uniform `run(**kwargs) -> {"ok","data","meta"}` tool contract and the typed `AppConfig` are the only shared interfaces, so two engineers can work on the retriever and the stance head simultaneously without merge conflicts. New hires onboard against a signature, not a codebase.

**2. A model registry is the integration ledger.** `models/model_registry.py` + the `/version` endpoint pin every artifact (classifier version, NLI/retriever **revision SHAs**, `index_built_at`, `git_sha`) into every response. As the team grows, this is what prevents the classic failure mode of a shared system — one engineer's Hub update silently changing another's verdict. Versions are stamped, never assumed; a verdict is always reproducible from its `/version` block. The same registry is where a future "champion/challenger" rollout and A/B of classifier or stance models would plug in.

**3. A human-in-the-loop review board is the safety scaling mechanism.** As volume grows, the temptation is to automate enforcement — which this design forbids. Instead, scaling means *more reviewers fed by a better-triaged queue*: the classifier prior triages the firehose, the evidence fact-check prioritizes what a human sees, and abstain routes uncertainty to people rather than guessing. The ETH role becomes a standing review board owning abstain thresholds, fairness-slice monitoring (per-topic/source/speaker via `analysis/error_analysis.py`), and the periodic cross-domain leakage audit. Crucially, the reviewer-override signal is the feedback data that keeps the system honest — and the metric (flag-queue precision) that justifies the team's headcount.

**What would need to change at 10× scale.** (a) The offline `samples.py` seed graduates to a governed, versioned dataset with a data-contract test in CI. (b) The evidence index moves from memory-mapped local to a shared read-only volume behind stateless API replicas, with the gated NLI path degrading to `"unverified — capacity"` rather than timing out. (c) The single ETH reviewer becomes a rotating review board with documented label provenance and an appeals path — because at scale, *who decides what is fake* is the central governance question, and this system is explicitly designed to keep that decision human.

---

## 10. Definition of done (rubric crosswalk)

`grading/checklist.py` enforces the deliverable rubric; the plan's exit gates map onto it:

| Rubric item | Plan gate | Verified by |
|---|---|---|
| Classifier beats the baseline | M3 (and M7 cross-domain) | `training/evaluate.py` macro-F1 table |
| Fact-check cites evidence | M5, M6 | `/factcheck` always returns `evidence` + `decisions_trace` |
| Abstains under uncertainty | M5 | abstain example → `UNVERIFIED`; D3/D5 gates |
| Cross-domain honesty | M7 | GonzaloA→GossipCop macro-F1 drop reported |
| Calibration reported | M3 | ECE pre/post temperature scaling |
| Runs offline, no torch | M1 (all phases) | CI download-free; core-only service boot |
| Ethics: no auto-censorship | M8 | ETH final sign-off; review-board contract |
| Reproducible versions | M6, M8 | `/version` pins revisions + `git_sha` |

---

*Plan grounded in the live repo: `src/fakenews/` (`config.py`, `cli.py`, `agent/policy.py`, `factcheck/{stance,verdict,retriever}.py`, `models/`, `training/`, `api/`, `autoreport/`, `monitoring/`, `grading/`) and `pyproject.toml` (core deps torch-free; `[ml]`/`[api]`/`[report]` extras). All HF ids, licenses, and thresholds match `docs/DESIGN_BRIEF.md`.*
