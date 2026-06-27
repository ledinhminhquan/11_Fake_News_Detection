# P11 — System Architecture

**Project:** Fake News & Misinformation Detection System
**Course:** NLP in Industry — final assignment
**Author:** Le Dinh Minh Quan (student 23127460)
**Package:** `fakenews` (`src/fakenews/`)

> This document describes the system **as actually built**. Where it differs from
> `docs/DESIGN_BRIEF.md` (the planning source of truth), the difference is noted
> inline. The one-line thesis is unchanged: a **trainable fake-news classifier**
> gives a fast *prior* `P(fake)`, and an **agentic, evidence-grounded fact-check**
> (retrieve → stance/NLI → aggregate verdict with citations → **abstain** when
> uncertain) produces the verdict the user actually trusts. The tool **flags
> content for human review and always shows its evidence; it never auto-removes,
> auto-blocks, or auto-censors.**

---

## 1. Design principles (what the architecture is optimized for)

| Principle | How the architecture enforces it |
|---|---|
| **Assist, never censor** | Every surface returns a *review flag + evidence*, never an enforcement action. `abstained`/`unverified` is a first-class output. `FactCheckResponse.disclaimer` is stamped on every response. |
| **Two signals, reported separately** | The classifier `prior_fake` (style/source signal) and the evidence-grounded `verdict` are distinct fields end-to-end. A reviewer always sees when they disagree; the prior is only a *soft nudge* (`prior_weight = 0.3`). |
| **Transparency by construction** | `/factcheck` always returns the verbatim `evidence` list **and** the `decisions` trace (D1–D5) **and** the per-tool `trace`, even on abstain. Citations are extracted from the corpus, never generated. |
| **Runs fully offline** | The whole pipeline (classify → retrieve → stance → verdict) runs with only `numpy`/`pandas`/`scikit-learn`/`rank_bm25` — **no torch**. Every heavy import (`transformers`, `sentence-transformers`, `torch`, `datasets`) is lazy and degrades gracefully. |
| **Deterministic FSM** | Same input + same models + brain disabled ⇒ byte-identical output. The optional LLM brain is *advisory only* and never decides flow or invents a verdict/citation. |
| **Internal label convention** | **`1 = fake`, `0 = real`** (fake is the positive/detected class). `LABELS = {0: "real", 1: "fake"}` in `models/classifier.py`. All dataset loaders normalize polarity to this on load (mirrors disagree — see §6). |

---

## 2. End-to-end component diagram

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  CLIENTS                                                                        │
 │    CLI  `fakenews classify|factcheck|serve …`   curl/HTTP   Gradio UI (HF Space)│
 └───────────────┬───────────────────────────────────────┬────────────────────────┘
                 │  POST /classify                        │  POST /factcheck
                 ▼                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  FastAPI  (fakenews/api/main.py  ·  app_combined.py mounts the Gradio UI)      │
 │    /classify  /factcheck  /healthz  /readyz  /version                          │
 │    lazy singletons:  get_config() · get_agent()   (api/dependencies.py)        │
 └───────────────────────────────┬────────────────────────────────────────────────┘
                                 │  FakeNewsAgent.classify() / .run()
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  FakeNewsAgent  — deterministic FSM   (agent/fakenews_agent.py)                │
 │                                                                                │
 │   parse ──D1──► classify ──D2──► factcheck ──D3,D4──► present ──D5──► result    │
 │  (route)       (prior +        (retrieve+stance        (abstain                │
 │                 checkworthy)    +aggregate)             gate)                   │
 │                                                                                │
 │   each step wrapped by _step(): time + ToolTrace; tools never raise past it    │
 │   optional LLMBrain.explain() — advisory one-line rationale (off by default)   │
 └───┬───────────────────┬──────────────────────────┬──────────────────┬──────────┘
     │ tool_classify      │ tool_factcheck (retrieve)│ tool_factcheck   │ tool_present
     ▼                    ▼                          │ (stance+verdict) ▼
 ┌─────────────┐   ┌──────────────────────────┐      ▼            ┌──────────────┐
 │ CLASSIFIER  │   │  EVIDENCE RETRIEVER       │  ┌───────────┐   │  POLICY      │
 │ models/     │   │  factcheck/retriever.py   │  │ STANCE/NLI│   │ agent/       │
 │ classifier  │   │                           │  │ factcheck/│   │ policy.py    │
 │             │   │  BM25 (models/bm25.py)    │  │ stance.py │   │ D1–D5 rules  │
 │ Transformer │   │     + dense MiniLM        │  │           │   └──────┬───────┘
 │   (lazy HF) │   │     ──RRF──► fused        │  │ NLIStance │          │
 │   OR        │   │  models/model_registry   │  │  (DeBERTa │          ▼
 │ TF-IDF+     │   │                           │  │   MNLI-   │   ┌──────────────┐
 │ LogReg      │   │  corpus = evidence seed   │  │   FEVER)  │   │  VERDICT     │
 │ (sklearn)   │   │  (data/dataset.py        │  │   OR      │   │ factcheck/   │
 │             │   │   load_evidence)          │  │ Lexical   │   │ verdict.py   │
 └─────────────┘   └──────────────────────────┘  └───────────┘   │ aggregate_   │
        ▲                                              ▲          │  verdict()   │
        │  load_classifier(prefer=…)                   │          └──────────────┘
        │  fine-tuned > saved TF-IDF > seed-trained    │  load_stance(prefer=…)
        │                                              │  NLI if transformers else lexical
        └───────────── ARTIFACTS (loaded once at agent construction) ───────────┘
              models/classifier/  ·  models/<baseline>.joblib  ·  evidence corpus

 Cross-cutting:  JobState carries text→claim→prior→evidence→verdict + decisions[] + trace[].
 Optional LLM brain (agent/llm_orchestrator.py): rephrases the rule rationale only; never flips a verdict.
```

### How this maps onto the design-brief FSM

The brief specifies a 9-state FSM (`INGEST → PARSE → CLASSIFY → CHECK-WORTHY → RETRIEVE → STANCE → AGGREGATE → DECIDE/ABSTAIN → PRESENT`). The implementation **collapses those states into four sequential tool calls** that carry all five decision points:

| Brief states | Implemented tool | Decisions emitted |
|---|---|---|
| INGEST/NORMALIZE + PARSE/CLAIM-EXTRACT | `tool_parse` | **D1** claim routing |
| CLASSIFY + CHECK-WORTHY | `tool_classify` | **D2** check-worthiness / confidence gate |
| RETRIEVE + STANCE + AGGREGATE | `tool_factcheck` | **D3** evidence-coverage gate, **D4** verdict gate |
| DECIDE/ABSTAIN + PRESENT | `tool_present` | **D5** abstain gate |

The spine is linear (no looped query-widening retries in code today — D3 abstains immediately on thin coverage instead of widening), which keeps the FSM trivially deterministic and testable. ABSTAIN is reachable from D3 (insufficient evidence), D4 (mixed/inconclusive), and D5 (low confidence).

---

## 3. Data flow

### 3a. `POST /classify` — fast style/probability signal

```
client {text, title}
  → FastAPI classify()              api/main.py
    → agent.classify(text, title)   agent/fakenews_agent.py
      → run(mode="classify", save=False)
        → tool_parse      (D1)  route to claim/article, extract central claim
        → tool_classify   (D2)  classifier.predict(content) → (label, prob)
                                 classifier.predict_proba(content) → P(fake)
        → tool_factcheck  (D2 branch = classify_only → check_worthy=False)
                                 verdict follows the prior; NO retrieval, NO stance
        → tool_present    (D5)  finalize
      → {label, probability, prior_fake, model_version}
  → ClassifyResponse {label, probability, prior_fake, model_version, note:"…not a verdict…"}
```

`content = "<title>. <text>"` (title prepended when present). The response carries a permanent `note`: *"This is a style/probability signal, not a verdict. Verify with the fact-check."* Latency: TF-IDF+LogReg is sub-millisecond on CPU; the transformer pipeline is ~10–30 ms GPU / ~100–300 ms CPU per doc.

### 3b. `POST /factcheck` — agentic, evidence-grounded verdict

```
client {claim, title, mode="auto"}
  → FastAPI factcheck()             api/main.py
    → agent.run(claim, title, mode, save=True)
      │
      ├─ tool_parse        D1  policy.detect_input(content, mode)
      │                        n_words ≤ short_claim_words(40) → claim route (claim = text)
      │                        else article route → central claim = first ≥4-word sentence
      │
      ├─ tool_classify     D2  classifier.predict / predict_proba → clf_label, clf_prob, clf_prior_fake
      │                        policy.checkworthy_gate(clf_prob, is_claim, mode)
      │                          mode=="classify" → skip;  mode=="factcheck" → force
      │                          auto: a claim is always checked; an article is skipped only
      │                                when clf_prob ≥ skip_factcheck_confidence(0.95)
      │
      ├─ tool_factcheck    D3  retriever.retrieve(claim, top_k=5)
      │                          BM25 (always) + dense MiniLM (if available) ──RRF──► fused
      │                          relevance normalized to [0,1]
      │                        policy.coverage_gate(evidence)
      │                          relevant = items with relevance ≥ min_evidence_relevance(0.05)
      │                          enough = len(relevant) ≥ min_evidence(1)
      │                        per-evidence: stance.score(claim, evidence.text) → (stance, score)
      │                        if NOT enough → verdict="unverified", abstained=True   (ABSTAIN)
      │                   D4  aggregate_verdict(clf_prior_fake, scored, cfg)   factcheck/verdict.py
      │                          weight wᵢ = stance_score · relevance
      │                          ev_fake = (Σ refute_w − Σ support_w) / total  ∈ [−1,1]
      │                          prior_signal = (P(fake) − 0.5)·2
      │                          net_fake = (1−prior_weight)·ev_fake + prior_weight·prior_signal
      │                          |net_fake| < verdict_margin(0.2) → "unverified" (ABSTAIN)
      │                          else net_fake>0 → "fake"; net_fake<0 → "real"
      │
      ├─ tool_present      D5  policy.abstain_gate(verdict, confidence)
      │                          abstained OR confidence < min_verdict_confidence(0.5) → "unverified"
      │                        status = UNVERIFIED if abstained else COMPLETED
      │
      └─ (optional) LLMBrain.explain(claim, verdict, evidence)  — advisory rationale only
  → FactCheckResponse {verdict, confidence, classifier_prior_fake, classifier_label,
                       abstained, rationale, claim, evidence[], n_support, n_refute,
                       decisions[], trace[], metrics, model_versions, disclaimer}
```

The verdict semantics are explicit: the *claim* is what is checked; evidence that **supports** the claim ⇒ the news is **real**; evidence that **refutes** it ⇒ **fake**. Evidence dominates; the classifier prior is a soft nudge (`prior_weight = 0.3`, i.e. evidence weight `0.7`). Insufficient coverage or a sub-`verdict_margin` evidence margin ⇒ **`unverified`** — never a guessed label.

### Worked example (matches the brief's, every decision fires)

Input claim *"The WHO declared that drinking bleach cures COVID-19 in 2021."*

| Step | Decision | Outcome |
|---|---|---|
| `tool_parse` | **D1** | 11 words ≤ 40 → `claim` route; `claim` = input |
| `tool_classify` | **D2** | `clf_label="fake"`, `clf_prob≈0.88`; `auto` + claim → check-worthy = True → retrieve |
| `tool_factcheck` | **D3** | RRF over the evidence corpus returns relevant WHO / fact-check passages; coverage OK |
| `tool_factcheck` | **D4** | stances mostly `refute`; `net_fake > verdict_margin` and prior agrees → verdict `fake` |
| `tool_present` | **D5** | `confidence ≥ 0.5`, prior agrees → `present`; citations = the refuting passages |

A novel claim with no corpus coverage abstains at **D3** (`unverified — insufficient evidence`).

---

## 4. Repository module map

Package root: `D:\NLP Industry Projects\11_Fake_News_Detection\src\fakenews\`

| Module | Responsibility | Key symbols |
|---|---|---|
| **`config.py`** | Single typed config (dataclasses) + YAML loader/saver; env-driven paths; all model ids, thresholds (D1–D5), serving knobs. **Internal convention `1=fake, 0=real`.** | `AppConfig`, `DataConfig`, `ClassifierConfig`, `StanceConfig`, `RetrievalConfig`, `AgentConfig`, `ServingConfig`; `load_config`, `ensure_dirs`, `model_dir`, `index_dir`, `run_dir` |
| **`logging_utils.py`** | Console logger + JSONL request logger. | `get_logger`, `JsonlLogger` |
| **`cli.py`** | Single entrypoint `fakenews <command>`. Subcommands: `data, train-classifier, train-baseline, tune, evaluate, classify, factcheck, demo-agent, serve, benchmark, error-analysis, monitor, generate-report, generate-slides, autopilot, grade`. | `build_parser`, `main`, `cmd_*` |
| **`data/dataset.py`** | News/claim dataset loading + **label-polarity normalization** to `1=fake/0=real`; seed fallback; evidence corpus + gold-verdict claims. `datasets` imported lazily. | `NewsItem`, `load_news`, `load_seed_news`, `seed_split`, `load_evidence`, `load_claims`, `_normalize_label` |
| **`data/samples.py`** | Built-in synthetic, license-safe seed: labeled news, evidence snippets, gold-verdict claims → the whole pipeline runs offline with zero network. | `news()`, `evidence()`, `claims()` |
| **`data/download_dataset.py`** | Prefetch / sanity-check the HF datasets (network-gated). | `download_all` |
| **`models/classifier.py`** | The **trainable classifier core** + sklearn baseline. `load_classifier` picks fine-tuned transformer > saved baseline > seed-trained baseline. Both expose `predict_proba(text)→P(fake)` and `predict(text)→(label,prob)`. | `TfidfLogRegClassifier`, `TransformerClassifier`, `load_classifier`, `LABELS={0:"real",1:"fake"}` |
| **`models/bm25.py`** | Dependency-free BM25 lexical retriever (the offline retrieval floor). | `BM25Retriever` |
| **`models/model_registry.py`** | Version/artifact tracking; resolve the latest trained model dir. | `resolve_latest` |
| **`factcheck/retriever.py`** | Evidence retriever: BM25 (always) + dense MiniLM bi-encoder (lazy `sentence-transformers`) fused with **RRF**; relevance normalized to [0,1]. Degrades to BM25-only offline. | `EvidenceRetriever`, `_rrf`, `_DenseEvidence` |
| **`factcheck/stance.py`** | Stance/NLI: `NLIStance` (zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, premise=evidence/hypothesis=claim, entail→support / contra→refute / neutral→neutral) + `LexicalStance` token-overlap fallback. | `NLIStance`, `LexicalStance`, `load_stance` |
| **`factcheck/verdict.py`** | Aggregate per-evidence stances + classifier prior → verdict/confidence/rationale; the ABSTAIN logic (D4). | `aggregate_verdict` |
| **`agent/state.py`** | FSM context object `JobState` (input → claim → prior → evidence → verdict) + the full audit trail. | `JobState`, `JobStatus`, `Decision`, `ToolTrace`, `Evidence` |
| **`agent/policy.py`** | Pure, testable D1–D5 rules. | `detect_input` (D1), `checkworthy_gate` (D2), `coverage_gate` (D3), `abstain_gate` (D5) |
| **`agent/tools.py`** | The four FSM tools operating on `JobState`. | `tool_parse`, `tool_classify`, `tool_factcheck`, `tool_present` |
| **`agent/fakenews_agent.py`** | The FSM spine: loads classifier/retriever/stance once, runs `parse→classify→factcheck→present`, times+traces each step, exposes `run()` and `classify()`. Module-level singleton. | `FakeNewsAgent`, `get_agent` |
| **`agent/llm_orchestrator.py`** | Optional Anthropic **brain** — advisory one-line rationale over already-selected evidence; off by default; never flips a verdict or invents a citation. | `LLMBrain` |
| **`api/main.py`** | FastAPI app: `/classify`, `/factcheck`, `/healthz`, `/readyz`, `/version`. | `app`, `classify`, `factcheck`, `healthz`, `version` |
| **`api/dependencies.py`** | Lazy LRU-cached singletons. | `get_config`, `get_agent` |
| **`api/schemas.py`** | Pydantic request/response models (carry the disclaimer + "not a verdict" note). | `ClassifyRequest/Response`, `FactCheckRequest/Response`, `EvidenceOut`, `HealthResponse` |
| **`api/ui.py`** | Gradio UI builder (Classify tab + Fact-check tab with stance-coloured evidence). | `build_ui` |
| **`api/app_combined.py`** | Combined ASGI app mounting the Gradio UI on the FastAPI app (`fakenews serve --ui`). | `app` |
| **`training/train_baseline.py`** | Train + persist the TF-IDF+LogReg baseline (sklearn, no GPU). | `train_baseline` |
| **`training/train_classifier.py`** | Fine-tune the transformer classifier (HF `Trainer`, macro-F1, hardware auto-adapt, class weights). | `train_classifier` |
| **`training/tune.py`** | Lightweight LR/hyperparameter search. | `tune_classifier` |
| **`training/evaluate.py`** | Classifier vs baselines + fact-check eval (accuracy, macro-F1, ROC-AUC, ECE, FEVER-style). | `evaluate` |
| **`training/metrics.py`** | Metric helpers (macro-F1, per-class P/R/F1, ROC-AUC, ECE; FEVER label-accuracy / recall@k). | metric functions |
| **`analysis/error_analysis.py`** | Per-item misclassification analysis (error slices). | `error_analysis` |
| **`analysis/latency.py`** | Latency benchmark harness. | `benchmark` |
| **`autoreport/{artifact_loader,charts}.py`** (+ `report_pdf`, `slides_pptx`) | Load run artifacts; render charts (confusion matrix, verdict distribution); build the PDF report and PPTX slides. | `artifact_loader`, chart builders |
| **`monitoring/drift_report.py`** | Input-text / label drift report from request logs. | `monitoring_report` |
| **`automation/autopilot.py`** | One-button end-to-end: train → eval → analysis → report+slides. | `run_autopilot` |
| **`grading/checklist.py`** | Rubric self-check (classifier beats baseline, fact-check cites evidence, abstains). | `build_checklist` |

> **Deviations from the brief's planned map (§9):** the retriever lives in
> `factcheck/retriever.py` (with `models/bm25.py`), not `models/retriever.py` +
> `factcheck/hybrid.py`; dataset loading is unified in `data/dataset.py` (not a
> separate `news_loaders.py`/`evidence_corpus.py`); there is no standalone
> `claim_extractor.py` (D1 claim extraction is inlined in `policy.detect_input`)
> and no `train_stance.py` (stance is zero-shot by default, as the brief
> recommends). The agent files are named `fakenews_agent.py` / `tools.py` /
> `policy.py` / `state.py` / `llm_orchestrator.py`.

---

## 5. Configuration & thresholds (the FSM's tunable surface)

All thresholds live in `config.py` and are env/YAML-overridable. Defaults (as built):

| Group | Field | Default | Role |
|---|---|---|---|
| **Classifier** | `base_model` | `distilbert-base-uncased` | T4-light default (Apache). Alts: `microsoft/deberta-v3-base` (MIT), `answerdotai/ModernBERT-base` (Apache) |
| | `max_length` / `num_labels` | `384` / `2` | seq length / binary head |
| **Stance** | `nli_model` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | MIT, zero-shot; `nli_fallback=facebook/bart-large-mnli` |
| | `support_threshold` / `refute_threshold` | `0.5` / `0.5` | entail/contra prob to call support/refute |
| **Retrieval** | `embedder` | `sentence-transformers/all-MiniLM-L6-v2` | Apache, 384-d dense |
| | `reranker` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache (config-present) |
| | `top_k` / `rrf_k` | `5` / `60` | evidence passages / RRF constant |
| **Agent D1** | `short_claim_words` | `40` | ≤ ⇒ short-claim route |
| **Agent D2** | `skip_factcheck_confidence` | `0.95` | confident-article skip gate |
| **Agent D3** | `min_evidence` / `min_evidence_relevance` | `1` / `0.05` | coverage gate |
| **Agent D4** | `verdict_margin` / `prior_weight` | `0.2` / `0.3` | verdict margin / prior nudge weight |
| **Agent D5** | `min_verdict_confidence` | `0.5` | abstain floor |
| **Brain** | `llm_fallback_enabled` | `False` | optional advisory rationale (Anthropic) |

---

## 6. Artifacts loaded at startup

`FakeNewsAgent.__init__` loads three components **once** (and the API caches the
agent via `@lru_cache` in `api/dependencies.py`, so they load on first request):

1. **Classifier** — `load_classifier(cfg.classifier, prefer="transformer" if load_model else "tfidf")`.
   Resolution order: a fine-tuned transformer under `models/classifier/` →
   the saved TF-IDF baseline (`<output_dir>/tfidf_logreg.joblib`) → a TF-IDF
   baseline **trained on the seed at construction** (so there is always a working
   classifier, even with no trained artifact and no network).
2. **Evidence retriever** — `EvidenceRetriever.from_corpus(cfg.retrieval, load_evidence(cfg))`.
   Builds a BM25 index (always) and, if `sentence-transformers` imports, a dense
   MiniLM index. The corpus is the seed evidence offline; on Colab it can be built
   from FEVER/wiki.
3. **Stance model** — `load_stance(cfg.stance, prefer="nli" if load_model else "lexical")`.
   Loads the DeBERTa MNLI-FEVER-ANLI NLI pipeline if `transformers` is present,
   else the lexical token-overlap fallback.

Heavy artifacts (HF revisions) should be **pinned by revision** in production so a
Hub update can never silently change a verdict (`/version` stamps the loaded
classifier/stance versions and `model_version`). `GET /healthz` reports the loaded
classifier + stance backend names; `GET /readyz` warms the agent.

---

## 7. Offline-fallback design (no torch, no network)

The system is engineered so that with **only `numpy`/`pandas`/`scikit-learn`/`rank_bm25`**
installed (no `torch`, `transformers`, `sentence-transformers`, or `datasets`), the
*entire* `classify → retrieve → stance → verdict` pipeline still runs end-to-end —
this is what CI and the graded offline demo exercise.

| Component | Online (heavy deps present) | Offline fallback | Mechanism |
|---|---|---|---|
| **Classifier** | `TransformerClassifier` (fine-tuned HF, P(fake)) | `TfidfLogRegClassifier` (sklearn) trained on the seed | `load_classifier` try/except; lazy `transformers` import inside `from_pretrained` |
| **Dataset** | HF `load_dataset` (polarity-normalized) | `data/samples.py` seed news/evidence/claims | `load_news` try/except → `load_seed_news` |
| **Retriever** | BM25 + dense MiniLM, **RRF**-fused | BM25-only over the seed corpus | `EvidenceRetriever.build` try/except on `SentenceTransformer` |
| **Stance** | `NLIStance` (DeBERTa MNLI-FEVER-ANLI) | `LexicalStance` (token overlap + negation/support cue lexicons) | `load_stance` try/except → lexical |
| **Verdict** | `aggregate_verdict` (unchanged) | `aggregate_verdict` (unchanged) | stance scores feed the same aggregator |
| **Brain** | `LLMBrain.explain` (Anthropic) | rule-templated `rationale` | `available()` False by default → rules |

The fallbacks are not stubs — `LexicalStance` still emits `support`/`refute`/`neutral`
with a confidence, BM25 still ranks the seed evidence, and the seed-trained TF-IDF
classifier still produces a non-trivial `P(fake)`, so the verdict aggregator and the
FEVER-style metrics run with zero network. When stance degrades, evidence is weaker,
so the system **abstains more readily** rather than guessing — the safe direction.

---

## 8. Cross-cutting concerns

- **Tracing / audit.** Every tool call is wrapped by `FakeNewsAgent._step`, which
  records latency and a `ToolTrace` (and never lets a tool raise past it — a failed
  tool degrades the job, it does not 500 the request). Every decision point appends
  a `Decision(id=D1..D5, branch, score, detail)` to `JobState.decisions`. Both the
  `decisions` and `trace` lists are returned on `/factcheck` (including on abstain).
- **Request logging.** When `serving.log_requests` is on, decisions are appended as
  JSONL to `runs/request_logs/requests.jsonl` (consumed by `monitoring/drift_report.py`).
- **Determinism.** No randomness on the inference path; the LLM brain is off by
  default and, when on, is temperature-0 and advisory. Same input + same model
  artifacts ⇒ identical output.
- **Label-polarity normalization (data integrity).** Dataset mirrors disagree on
  polarity — `GonzaloA/fake_news` is `0=fake/1=real`, `LittleFish-Coder/*` is
  `0=real/1=fake`, `mrm8488/fake-news` is `1=fake/0=real`. `data/dataset._normalize_label`
  maps every source to the internal `1=fake/0=real` via `DataConfig.fake_label_value`
  (the raw value that means *fake*), so a mirror swap can never silently invert labels.

---

## 9. Deployment surface (summary)

| Endpoint | Method | Request | Response (key fields) |
|---|---|---|---|
| `/classify` | POST | `{text, title}` | `{label, probability, prior_fake, model_version, note}` |
| `/factcheck` | POST | `{claim, title, mode}` | `{verdict, confidence, classifier_prior_fake, classifier_label, abstained, rationale, claim, evidence[], n_support, n_refute, decisions[], trace[], metrics, model_versions, disclaimer}` |
| `/healthz` | GET | — | `{status, classifier, stance, version}` |
| `/readyz` | GET | — | `{status:"ready"}` (warms the agent) |
| `/version` | GET | — | `{app, classifier, stance, retriever:"bm25+dense", model_version}` |

Clients: **CLI** (`fakenews classify|factcheck|serve|…`), **curl/HTTP**, and the
**Gradio UI** (mounted via `app_combined.py`, `fakenews serve --ui`; deployable as
an HF Space under `ledinhminhquan`). The classifier path runs on every request; the
heavier retrieval+NLI fact-check is gated behind `/factcheck`. All heavy imports are
lazy, so the service boots and serves on a numpy/sklearn-only install.

---

## 10. License flags (carried from the design brief)

The architecture itself is permissive (all **models** are MIT/Apache:
DeBERTa MNLI-FEVER-ANLI, bart-large-mnli, MiniLM, ms-marco reranker, the classifier
bases). Dataset licenses to flag for redistribution:

- ⚠️ **`GonzaloA/fake_news`** (default classifier dataset) — **license unknown**;
  also *opposite polarity* (handled by normalization). Prefer
  `mohammadjavadpirhadi/…` (MIT) or `ErfanMoosaviMonazzah/…` (openrail) for a
  license-clean classifier; `chengxuphd/liar2` (Apache) for the 6-way credibility config.
- ⚠️ **`fever/fever`** + all FEVER-derived sets — **cc-by-sa-3.0 + gpl-3.0**
  (copyleft / share-alike); any released stance head fine-tuned on FEVER inherits
  share-alike. Zero-shot NLI (the default) avoids producing such a derivative.
- ⛔ **`allenai/scifact`** is cc-by-nc-2.0 (non-commercial) — not used; `BeIR/*`
  (cc-by-sa-4.0) is the share-alike alternative for retrieval eval.
- ⚠️ **FakeNewsNet raw** (`KaiDMML/FakeNewsNet`) is a crawler, network-gated,
  ASU academic-use — excluded from CI; the Apache `LittleFish-Coder/*` content
  mirrors are used for cross-domain eval instead.

---

*See `docs/DESIGN_BRIEF.md` for the full verified stack table, training plan,
metrics/baselines, and the ethics analysis. This document covers the system
architecture and data flow only.*
