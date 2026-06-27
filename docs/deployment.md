# P11 — Deployment

> **Project:** P11 — Fake News & Misinformation Detection System
> **Course:** NLP in Industry — final assignment
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Doc spec:** Section I.6 — Deployment
> **Package:** `fakenews` (`src/fakenews/`)
> **Source of truth:** [`DESIGN_BRIEF.md`](./DESIGN_BRIEF.md) §6 (deployment), §5 (agent FSM / decisions D1–D5), §8 (ethics)

This document specifies how the system is served, packaged, scaled, versioned, and operated. It covers four delivery surfaces that all sit on top of **one** Python package: a **FastAPI** service, a **Gradio** UI, a **CLI** (`fakenews`), and a **Docker image** that doubles as a **Hugging Face Space**.

---

## 0. The one non-negotiable: this tool *flags*, it never *removes*

Everything below serves a single ethical constraint (§8 of the brief):

> **The system FLAGS content for human review. It never auto-removes, auto-blocks, or auto-censors, and it always shows its evidence. `unverified`/abstain is a first-class outcome, not a failure.**

Concretely, every deployment surface enforces this:

- **No endpoint, CLI command, or UI button takes an enforcement action.** Output is always a *review flag + evidence*, never a takedown.
- The fast classifier signal (`prior_fake`) and the evidence-grounded `verdict` are returned as **two separate fields** so a reviewer can see when they disagree (`conflict` flag, decision **D5**).
- Every `/factcheck` response carries a **`disclaimer`** string and the full **`decisions`** trace, even when the system abstains.
- The classifier output is labelled in the UI as a **"style / probability signal, not a verdict"** — it detects style/source patterns correlated with fakeness, which is *not* the same as detecting falsehood.

---

## 1. Architecture overview

```
                         ┌──────────────────────────────────────────────┐
   clients               │                fakenews package               │
 ───────────             │                                               │
  HTTP  ─────►  ┌──────────────────┐    ┌───────────────────────────┐    │
  Gradio ────►  │  api/main.py     │───►│  agent/factcheck_agent.py │    │
  CLI   ─────►  │  FastAPI app     │    │  (FSM: D1..D5 + ABSTAIN)  │    │
                │  /classify       │    └────────────┬──────────────┘    │
                │  /factcheck      │                 │ uniform run()      │
                │  /healthz        │     ┌───────────┴───────────────┐   │
                │  /version        │     │  tools (lazy heavy deps)  │   │
                └────────┬─────────┘     │ ClassifyTool  Retrieve    │   │
                         │               │ CheckWorthy   StanceNLI   │   │
                  api/ui.py (Gradio)     │ ClaimExtract  Aggregate   │   │
                         │               └───────────┬───────────────┘   │
                         │                           │                   │
                         │              ┌────────────┴────────────┐      │
                         │              │  shared read-only state │      │
                         │              │  • classifier artifact  │      │
                         │              │  • evidence index       │      │
                         │              │    (BM25 + dense + RRF)  │      │
                         │              │  • NLI model (pinned)    │      │
                         │              └─────────────────────────┘      │
                         └───────────────────────────────────────────────┘

 Cold-start floor: with ONLY numpy/pandas/scikit-learn/rank_bm25 installed (no torch),
 the service still boots and serves: TF-IDF+LogReg classifier + BM25 lexical retrieval
 + lexical-overlap stance. Heavy imports (torch/transformers/sentence-transformers)
 happen INSIDE each tool's run() and degrade gracefully when absent.
```

**Single app, two signals.** The FastAPI app (`fakenews/api/main.py`, combined app in `app_combined.py`) exposes both the classifier (fast prior) and the agentic fact-checker. The classifier path is cheap enough to run on every request; the retrieval + NLI fact-check is heavier and **gated behind `/factcheck` only** (decisions **D2/D3** can skip retrieval entirely).

---

## 2. FastAPI service

### 2.1 Endpoint summary

| Endpoint | Method | Purpose | Cost profile |
|---|---|---|---|
| `/classify` | POST | Fast style/credibility **prior** (classifier only) | cheap — run per request |
| `/factcheck` | POST | Full agentic FSM: claim → evidence → stance → verdict → abstain | heavy — gated, cached |
| `/healthz` | GET | Liveness/readiness probe | trivial |
| `/version` | GET | Pinned model revisions + index build time + git SHA | trivial |

All request/response bodies are **Pydantic** schemas (`fakenews/api/schemas.py`). All heavy imports are lazy: the service boots with only `numpy/pandas/scikit-learn/rank_bm25` and serves the no-torch path.

---

### 2.2 `POST /classify` — fast style signal

A fast, calibrated probability of the **fake** class. This is the *prior*, explicitly **not** a verdict.

**Request** (either form accepted):

```json
{ "text": "Breaking: officials confirm the new policy doubles all pensions overnight." }
```
or
```json
{ "title": "Officials confirm pension policy", "body": "..." }
```

When both `title` and `body` are present, the loader concatenates `title [SEP] body` (the training-time convention, §4A). When only `text` is supplied it is used as-is.

**Response:**

```json
{
  "label": "fake",
  "probability": 0.88,
  "prior_fake": 0.88,
  "model_version": "classifier_v1.0+tfidf_logreg",
  "backend": "tfidf_logreg"
}
```

| Field | Type | Meaning |
|---|---|---|
| `label` | `"fake" \| "real"` | argmax of the classifier (internal labels **0=real, 1=fake**) |
| `probability` | float | confidence of the predicted `label` |
| `prior_fake` | float | calibrated **P(fake)** = `P(label==1)`; this is the value the FSM consumes as the prior |
| `model_version` | str | semantic classifier version + active backend |
| `backend` | `"transformer" \| "tfidf_logreg"` | `tfidf_logreg` when torch is unavailable |

> `probability` and `prior_fake` are equal when the predicted label is `fake`; when the prediction is `real`, `prior_fake = 1 − probability`. Both are returned so callers never have to recompute polarity. Internal label convention is **0=real, 1=fake** everywhere (`NewsItem.label`, §9).

**Probabilities are calibrated.** Temperature scaling is applied on a held-out calibration split (§7, ECE) before `prior_fake` is emitted — over-confidence here can silence legitimate news, so calibration is part of the contract, not an afterthought.

---

### 2.3 `POST /factcheck` — agentic, evidence-grounded verdict

Runs the full FSM (`agent/factcheck_agent.py`): **INGEST → CLAIM-EXTRACT (D1) → CLASSIFY → CHECK-WORTHY (D2) → RETRIEVE (D3) → STANCE/NLI → AGGREGATE (D4) → DECIDE/ABSTAIN (D5) → PRESENT**. It **always** returns the evidence list and the decision trace, even on abstain — transparency is non-optional.

**Request:**

```json
{ "claim": "The WHO declared that drinking bleach cures COVID-19 in 2021.", "mode": "auto", "k": 5, "threshold": 0.3 }
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `claim` | str | — | the claim or article text to fact-check |
| `mode` | `"auto" \| "claim" \| "article"` | `"auto"` | routing hint for **D1**; `auto` lets the FSM decide (article vs short claim) |
| `k` | int | `5` | top-k evidence passages to retrieve / score |
| `threshold` | float | `0.3` | relevance gate `τ_rel` for **D3** evidence coverage |

**Response:**

```json
{
  "verdict": "fake",
  "confidence": 0.91,
  "classifier_prior": 0.88,
  "evidence": [
    {
      "text": "WHO has not recommended drinking any disinfectant; doing so is dangerous.",
      "source": "who.int/emergencies/...",
      "stance": "refute",
      "score": 0.94,
      "relevance": 0.81
    }
  ],
  "rationale": "Multiple authoritative sources contradict this claim; none support it.",
  "decisions": [
    {"id": "D1", "state": "CLAIM-EXTRACT", "route": "short-claim", "claim_len": 11},
    {"id": "D2", "state": "CHECK-WORTHY", "skip_retrieval": false, "checkworthy_score": 0.92},
    {"id": "D3", "state": "RETRIEVE", "n_relevant": 4, "coverage_ok": true, "tau_rel": 0.3},
    {"id": "D4", "state": "AGGREGATE", "verdict": "FAKE", "stance_score": -2.7, "n_refute": 3},
    {"id": "D5", "state": "DECIDE", "action": "present", "agreement": 0.75, "conflict": false}
  ],
  "abstained": false,
  "disclaimer": "This is a decision-support flag for human review, not an automated verdict. The system never removes or blocks content. 'unverified' means insufficient evidence, not 'true'."
}
```

| Field | Type | Meaning |
|---|---|---|
| `verdict` | `"real" \| "fake" \| "unverified"` | the evidence-grounded outcome (**D4/D5**); `unverified` is first-class |
| `confidence` | float | combined confidence after aggregation (**D4**) and the abstain gate (**D5**) |
| `classifier_prior` | float | the **separate** classifier `prior_fake` — reported *alongside* the verdict so a reviewer sees disagreement |
| `evidence[]` | list | each retrieved passage with verbatim `text`, `source` id, `stance`, NLI `score`, and retrieval `relevance` |
| `evidence[].stance` | `"support" \| "refute" \| "neutral"` | per-passage NLI stance (entail→support, contradiction→refute, neutral→NEI) |
| `evidence[].score` | float | NLI head confidence for the assigned stance |
| `evidence[].relevance` | float | retrieval/rerank relevance (RRF + cross-encoder), used to weight the verdict |
| `rationale` | str | NL explanation citing evidence ids (templated fallback if brain absent) |
| `decisions[]` | list | per-decision trace D1–D5 (the `ToolTrace` projection) — always present |
| `abstained` | bool | `true` when **D5** routed to ABSTAIN (`unverified — needs human review`) |
| `disclaimer` | str | the fixed "flags for review, never auto-removes" notice — always present |

**Why `classifier_prior` is a separate field.** The classifier learns *style/source patterns correlated with fakeness*, not truth (§8, source-style leakage). The evidence verdict dominates; the prior is only a **soft nudge** (aggregation weight `α=0.6` on stance, **D4**). When the classifier says `fake` but the evidence strongly **supports** the claim, **D5** raises `conflict: true` and routes to human review rather than trusting the prior.

**Abstain is normal.** If retrieval coverage is too thin (`n_relevant < N_min=3` after up to `R_max=2` query-widening retries, **D3**), or stance agreement is below `τ_agree=0.4`, or `confidence < τ_present=0.55`, or there is a prior↔evidence conflict (**D5**), the response sets `verdict: "unverified"`, `abstained: true`, and still returns whatever evidence was found plus the full trace.

---

### 2.4 `GET /healthz`

Liveness + readiness. Returns **503** until the classifier artifact and the evidence index are warm, so a load balancer does not route traffic to a cold replica.

```json
{ "status": "ok", "models_loaded": true }
```

---

### 2.5 `GET /version`

Pins every component that can change a verdict. **Revisions/commit SHAs are pinned, not just repo names**, so a Hub update can never silently change an output.

```json
{
  "api": "fakenews-0.1.0",
  "classifier_version": "classifier_v1.0",
  "classifier_base": "answerdotai/ModernBERT-base@<sha>",
  "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@<sha>",
  "retriever": "sentence-transformers/all-MiniLM-L6-v2@<sha> + rank_bm25 (RRF)",
  "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2@<sha>",
  "index_built_at": "2026-06-20T11:42:00Z",
  "git_sha": "a1b2c3d"
}
```

> **License flags surfaced at `/version`.** `index_built_at` references the evidence corpus, which is built from FEVER mirrors (`fever/fever`, `BeIR/fever`) under **cc-by-sa** (copyleft / share-alike) and, for the classifier, `GonzaloA/fake_news` whose **license is UNKNOWN (flag)**. A derived/redistributed artifact inherits these terms — `/version` is the audit anchor for that. The default NLI model `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` is **MIT** ✅ and the retriever/reranker (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L6-v2`) are **Apache-2.0** ✅; only the *data* lineage carries copyleft/unknown flags.

---

## 3. Gradio UI

A two-tab Gradio app (`fakenews/api/ui.py`), shipped on a Hugging Face **Space** (Gradio SDK) under `ledinhminhquan`. It calls the same package internals as the API.

### Tab 1 — **Classify** (style signal)

- Input: paste an article or headline.
- Output: `label` + a confidence bar.
- A **persistent banner**: *"This is a style/probability signal, not a verdict. The classifier detects writing-style and source patterns, not truth."*

### Tab 2 — **Fact-check** (the main surface)

- Input: paste a claim (or short article).
- Output:
  - a **verdict chip** — `real` (green) / `fake` (red) / **`unverified`** (neutral grey, distinct styling so abstain never looks like a fake label);
  - a **confidence gauge**;
  - the **classifier prior** shown *separately* from the verdict (with a conflict note if they disagree);
  - a **highlighted evidence table** — one row per passage, each with `source`/citation, a **stance badge** (support = green, refute = red, neutral = grey), the NLI `score`, and the retrieval `relevance`;
  - the **disclaimer** line and a collapsible **decisions trace** (D1–D5).
- **Abstain rendering is prominent:** *"Not enough evidence — flagged for human review."* The UI never shows a fabricated `fake`/`real` label when the system abstained.

---

## 4. CLI (`fakenews`)

Defined in `fakenews/cli.py`. JSON modes make it scriptable.

```bash
# Fast classifier prior
fakenews classify --text "Officials confirm the policy doubles all pensions overnight."
fakenews classify --file article.txt --json

# Full agentic fact-check
fakenews factcheck --claim "The WHO declared bleach cures COVID-19 in 2021." --k 5 --json

# Serve the API (uvicorn) / launch the Gradio UI
fakenews serve --host 0.0.0.0 --port 8000

# Build / rebuild the evidence index (offline, versioned)
fakenews index build --corpus data/evidence/ --out outputs/index/
```

| Command | What it does |
|---|---|
| `classify` | one-shot classifier prior (`--text` or `--file`), `--json` for machine output |
| `factcheck` | full FSM verdict with `--k` evidence and `--json` (includes `decisions`, `evidence`, `disclaimer`) |
| `serve` | boot FastAPI (uvicorn) and/or Gradio |
| `index build` | build the BM25 + dense evidence index offline and stamp `index_built_at` |

`factcheck --json` emits exactly the `/factcheck` schema (verdict, confidence, classifier_prior, evidence[], rationale, decisions, abstained, disclaimer), so the CLI and API are interchangeable in pipelines.

---

## 5. Packaging & deployment

### 5.1 Install extras

```bash
pip install fakenews            # LIGHT CORE: numpy/pandas/scikit-learn/rank_bm25/fastapi/gradio
                                #   -> boots & serves the no-torch path end-to-end
pip install "fakenews[torch]"   # transformer classifier + DeBERTa NLI + dense retriever
pip install "fakenews[gpu]"     # CUDA wheels for A100/H100/L4/T4
```

The **light core** is the offline floor: TF-IDF+LogReg classifier, BM25 retrieval over the seed evidence, lexical-overlap stance. No network, no torch. The `[torch]` / `[gpu]` extras pull the transformer stack and switch the backends in-place (the `backend` fields in the responses flip from `tfidf_logreg`/`none` to `transformer`/`deberta_mnli`).

### 5.2 Dockerfile

- **CPU base image** by default (boots the light/no-torch path).
- An **optional CUDA layer** that, at runtime, detects the GPU via `torch.cuda.get_device_capability()` / `get_device_properties()` and auto-adapts precision and batch sizes (H100/A100 → bf16; **T4 7.5 → fp16**), falling back to **CPU/TF-IDF** when no GPU is present.
- The **same image** runs the API and the Hugging Face Space.

### 5.3 Hugging Face Space

Gradio SDK Space under `ledinhminhquan`, built from the same package/image. Pins the **same model revisions** as `/version`. The evidence index is built offline and shipped (or mounted) read-only; the Space is stateless apart from that shared index.

---

## 6. Latency & scalability

### 6.1 Latency budget (per request)

| Path | Component | Typical latency |
|---|---|---|
| `/classify` | TF-IDF + LogReg (CPU) | sub-millisecond |
| `/classify` | transformer fine-tune (GPU) | ~10–30 ms / doc |
| `/classify` | transformer fine-tune (CPU) | ~100–300 ms / doc |
| `/factcheck` | retrieval (BM25 + dense + RRF) | tens of ms |
| `/factcheck` | NLI stance (k passages, **batched in one forward pass**) + optional rerank | dominant cost |
| `/factcheck` | **total** (retrieval + k stance + rerank) | ~0.3–2 s |

**The classifier is fast → run it on every request. The retrieval+NLI fact-check is heavy → it is gated.** Two decision points keep cost down:

- **D2 (check-worthiness / classifier-confidence gate):** if the classifier is very confident (`classifier_conf ≥ τ_skip = 0.95`) **and** the input is not a checkable factual claim (`checkworthy_score < τ_cw = 0.5`), the FSM **skips retrieval entirely** and presents the prior. No NLI cost is paid.
- **D3 (evidence-coverage gate):** retrieval widens the query at most `R_max = 2` times; if coverage is still thin it **abstains** rather than spending more compute.

The k stance forward passes are **batched into a single NLI call**, and the cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L6-v2`) scores `(claim, passage)` pairs in one batch.

### 6.2 Scaling

- **Stateless API replicas** behind a load balancer. No per-request server state.
- **Shared read-only evidence index** (dense vectors via numpy/FAISS + sparse BM25), built offline, **memory-mapped**, mounted into every replica.
- **Caching by claim hash:** `/factcheck` results are keyed on a hash of the normalized claim (+ `k`, `threshold`, model revisions), so repeated/viral claims hit cache instead of re-running the FSM.
- **Graceful overload:** the gated NLI path degrades to `verdict: "unverified"` with a *"capacity"* reason rather than timing out — consistent with the abstain-is-first-class principle. A request queue protects the heavy path.

---

## 7. Versioning & reproducibility

| What is pinned | Where | Why |
|---|---|---|
| Classifier semantic version (`classifier_vX.Y`) | every `/classify` + `/factcheck` response (`model_version`) | trace which model produced a flag |
| **Model revisions / commit SHAs** (classifier base, NLI, retriever, reranker) | `/version` | a Hub update can never silently change a verdict |
| `index_built_at` | `/version`, stamped by `fakenews index build` | tie a verdict to the evidence snapshot it used |
| `git_sha` | `/version` | tie behaviour to source revision |

Pinning **revisions, not just repo names**, is a correctness requirement here: the same claim must produce the same verdict from the same `/version`. The default pinned components are:

- Classifier base: `answerdotai/ModernBERT-base` (Apache-2.0 ✅; T4 fallback `distilbert-base-uncased`, Apache-2.0 ✅) — trained on `GonzaloA/fake_news` (**license UNKNOWN — flag** ⚠️).
- NLI / stance: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT ✅). NLI label order is **read from `model.config.id2label` at load, never hardcoded** (verified `{0:entailment, 1:neutral, 2:contradiction}`).
- Dense retriever: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0 ✅), fused with `rank_bm25` via RRF.
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0 ✅).
- Evidence corpus: FEVER family (`fever/fever`, `BeIR/fever`) — **cc-by-sa copyleft / share-alike — flag** ⚠️ for redistribution.

---

## 8. Deployment-time license & ethics flags

| Item | Flag | Deployment consequence |
|---|---|---|
| `GonzaloA/fake_news` (classifier training data) | **license UNKNOWN** ⚠️ | research/eval use; for a redistributable model prefer `mohammadjavadpirhadi/...` (MIT) or `ErfanMoosaviMonazzah/...` (openrail); surfaced at `/version` |
| FEVER mirrors (`fever/fever`, `BeIR/fever`) | **cc-by-sa copyleft** ⚠️ | a redistributed evidence index / fine-tuned stance head inherits share-alike + attribution |
| FakeNewsNet raw (`KaiDMML/FakeNewsNet`) | crawler, ToS-gated | **excluded from the served index and CI**; use Apache-2.0 `LittleFish-Coder/*` mirrors for cross-domain eval only |
| NLI / retriever / reranker models | **MIT / Apache-2.0** ✅ | clean to ship |
| Any verdict | ethics | response always carries `disclaimer`, separate `classifier_prior`, verbatim `evidence`, full `decisions` trace; **no auto-removal anywhere** |

**Restated commitment:** the deployed service is a *decision-support tool*. It flags content for human review and shows its evidence. It never auto-removes, auto-blocks, or auto-censors; `unverified` means *insufficient evidence*, not *true*; and the classifier prior and the evidence verdict are always reported separately so a human can see when they disagree.
