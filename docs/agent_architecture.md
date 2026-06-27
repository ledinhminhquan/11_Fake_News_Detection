# P11 — Agentic AI Component (Section I.7)

> **Project:** P11 — Fake News & Misinformation Detection System
> **Course:** NLP in Industry — final assignment
> **Author:** Le Dinh Minh Quan (student 23127460)
> **Package:** `fakenews` · **Single source of truth:** [`docs/DESIGN_BRIEF.md`](./DESIGN_BRIEF.md)

This document specifies the **centerpiece** of P11: a deterministic, evidence-grounded **fact-check agent** built as a finite-state machine (FSM). A fast trainable **classifier** produces a cheap *prior* (P(fake)); the agent then runs a multi-step, tool-using reasoning loop — **extract claim → retrieve evidence → stance/NLI per passage → aggregate verdict → decide/abstain → present** — making each branch decision on the *models' own intermediate outputs*, never on free-text vibes.

The agent is **deterministic by construction**: control flow is decided exclusively by typed predicates over numeric intermediate outputs (classifier confidence, retrieval coverage, stance margins, agreement). An optional **LLM brain** is *opt-in and validated*; at each decision point it may only **propose a value from a fixed legal set**, and on any parse-failure, out-of-set value, or timeout the **deterministic rule fires**. The brain **never decides flow, never invents a verdict, and never invents a citation.**

> **Ethical anchor (non-negotiable).** This agent **flags content for human review; it never auto-removes, auto-blocks, or auto-censors.** Every non-abstained verdict ships **verbatim citations** plus a full `ToolTrace`. `unverified` / `ABSTAIN` is a **first-class outcome**, not a failure. The `classifier_prior` (style/source signal) and the evidence `verdict` are reported **separately** so a reviewer always sees when they disagree.

---

## 1. Why an agent (and not a single forward pass)

A pure classifier on fake-news corpora learns **source/writing-style patterns correlated with fakeness** (e.g. "is this Reuters wire copy or a blog rant"), **not falsehood**. In-domain macro-F1 of ~0.95+ on `GonzaloA/fake_news` / WELFake is a *red flag*, not success — it collapses cross-domain. The agentic evidence layer exists precisely to compensate: it grounds a verdict in **retrieved, citable evidence** and **abstains** when evidence is thin.

That requires **multi-step reasoning with tool use and decisions on intermediate outputs**:

| Capability | How this agent does it |
|---|---|
| **Multi-step reasoning** | 7 FSM states chained: PARSE → CLASSIFY → CHECK-WORTHY → RETRIEVE → STANCE → AGGREGATE → DECIDE → PRESENT, with an ABSTAIN sink reachable from RETRIEVE / AGGREGATE / DECIDE. |
| **Tool use** | Two heavy ML tools — **retrieval** (BM25 + dense `all-MiniLM-L6-v2` fused by RRF) and **NLI/stance** (zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) — plus classifier, claim-extractor, verdict aggregator. All share one uniform `run()` contract. |
| **Decisions on intermediate outputs** | Five decision points (D1–D5) read *numeric* intermediate outputs: `classifier_conf`, `checkworthy_score`, `n_relevant` / `rerank`, per-evidence stance `probs`/`margin`, `agreement`, `combined_conf`, and the **prior↔stance conflict** flag. |
| **Graceful degradation** | With only `numpy`/`pandas`/`scikit-learn`/`rank_bm25` installed (**no torch**), the *same FSM* runs end-to-end: TF-IDF+LogReg classifier, TF-IDF/BM25 retrieval, lexical mock-stance — abstaining more often, never crashing. |

Maps onto the canonical **Automated Claim Fact-Checking (ACFC)** pipeline (FEVER shared task; Guo/Schlichtkrull/Vlachos survey; ClaimBuster): *claim detection / check-worthiness → document retrieval → evidence selection → claim verification (stance/NLI) → verdict {SUPPORTED, REFUTED, NEI}*. We map `SUPPORTED→REAL-leaning`, `REFUTED→FAKE-leaning`, `NEI→ABSTAIN`. Internal label convention: **`0 = real`, `1 = fake`** (`p_fake = P(label==1)`).

---

## 2. State machine — ASCII flow diagram

States are upper-case boxes; **D1–D5** are the five decision points; `⊘ ABSTAIN` is the sink. Every transition emits a `ToolTrace` record.

```
                          ┌───────────────────────────────────────────────────────────┐
                          │           P11 FACT-CHECK AGENT  (deterministic FSM)         │
                          └───────────────────────────────────────────────────────────┘

  raw text ─►┌───────────────┐
             │    PARSE       │  IngestParse.run → {doc_type, title, body, lang}
             │ (ingest/normal)│
             └───────┬────────┘
                     │
                     ▼
             ┌───────────────┐
             │  CLAIM-EXTRACT │  ClaimExtract.run  ───────────── D1 ─────────────┐
             │   (D1 router)  │  short-claim route vs article route;            │
             └───────┬────────┘  central claim = {title | lead | TextRank}       │
                     │           claim_len < 5 tokens ──────────────────────────►⊘ INVALID_INPUT
                     ▼
             ┌───────────────┐
             │   CLASSIFY     │  ClassifyTool.run → {label, p_fake, probs, backend}
             │  (fast PRIOR)  │  transformer  OR  tfidf_logreg (no-torch)
             └───────┬────────┘
                     ▼
             ┌───────────────┐
             │ CHECK-WORTHY?  │  CheckWorthy.run ───────────── D2 ──────────────┐
             │   (D2 gate)    │  skip = (conf ≥ τ_skip=0.95) ∧ (cw < τ_cw=0.5)  │
             └───────┬────────┘                                                 │
        skip retrieval│                              proceed to fact-check      │
   (confident non-claim article)                                               │
                     │                                                          ▼
                     │                                                  ┌───────────────┐
                     │                          ┌──────────────────────│   RETRIEVE     │  RetrieveEvidence.run
                     │                          │   widen query        │  (D3 coverage) │  BM25 + dense → RRF → rerank
                     │                          │   (≤ R_max=2)         └───────┬────────┘
                     │                          │                  ┌───────────┴──────────── D3 ────────────┐
                     │                          │           n_relevant < N_min=3              n_relevant ≥ 3
                     │                          └──────────────────┤  (thin / low-rel)        (coverage_ok) │
                     │                                             │                                        ▼
                     │                              retries exhausted, still thin           ┌───────────────┐
                     │                                             │                        │   STANCE/NLI   │  StanceNLI.run
                     │                                             ▼                        │  per evidence  │  entail/neutral/contra
                     │                                     ⊘ ABSTAIN                         └───────┬────────┘  →support/refute/NEI
                     │                              "unverified (insufficient evidence)"            ▼
                     │                                                                      ┌───────────────┐
                     │                                                                      │  AGGREGATE     │  AggregateVerdict.run ─ D4
                     │                                                                      │ (D4 verdict)   │  S = Σ wᵢ·sᵢ ; blend prior α
                     │                                                                      └───────┬────────┘  → REAL | FAKE | UNVERIFIED
                     │                                                                              ▼
                     │                                                                      ┌───────────────┐
                     │                                                                      │ DECIDE/ABSTAIN │  DecideAbstain.run ─── D5
                     │                                                                      │   (D5 gate)    │  conf / agreement / conflict
                     │                                                                      └───────┬────────┘
                     │                                                       ┌──────────────────────┴───────────────────┐
                     │                                                  present                                  abstain
                     │                                                       │                                        │
                     ▼                                                       ▼                                        ▼
             ┌───────────────┐                                      ┌───────────────┐                       ┌──────────────────────┐
             │   PRESENT      │ ◄────────────────────────────────── │   PRESENT      │                       │  ⊘ ABSTAIN           │
             │ (prior only)   │   Present.run                       │ label+conf+    │                       │ "unverified — needs  │
             │ + style notice │                                     │ citations+     │                       │  human review"       │
             └───────────────┘                                     │ rationale      │                       └──────────────────────┘
                                                                    └───────────────┘

 Cross-cutting:  every transition → ToolTrace(state, tool, inputs_hash, outputs, latency_ms, brain_proposal?, rule_applied).
 Brain (optional LLM): at each D-point it may only PROPOSE a value ∈ legal set. parse-fail / out-of-set / timeout → deterministic rule fires.
                 The brain NEVER chooses the next state, NEVER invents a verdict, NEVER invents/edits a citation.
```

### State inventory

| State | Tool (uniform `run`) | Produces (key intermediate outputs) | Exits to |
|---|---|---|---|
| **PARSE** | `IngestParse` | `doc_type ∈ {article, claim}`, `title`, `body`, `lang` | CLAIM-EXTRACT |
| **CLAIM-EXTRACT** | `ClaimExtract` | `claim`, `candidates[]`, `entities[]`, `method` — **D1** | CLASSIFY · `⊘ INVALID_INPUT` |
| **CLASSIFY** | `ClassifyTool` | `label`, `p_fake`, `probs`, `backend` | CHECK-WORTHY |
| **CHECK-WORTHY** | `CheckWorthy` | `checkworthy`, `score`, `skip_retrieval` — **D2** | PRESENT(prior) · RETRIEVE |
| **RETRIEVE** | `RetrieveEvidence` | `evidence[]`, `n_relevant`, `coverage_ok` — **D3** | STANCE · RETRIEVE(widen) · `⊘ ABSTAIN` |
| **STANCE/NLI** | `StanceNLI` | `stances[]` (`label`, `probs`, `margin`) | AGGREGATE |
| **AGGREGATE** | `AggregateVerdict` | `verdict`, `stance_score`, `combined_conf`, vote counts — **D4** | DECIDE |
| **DECIDE/ABSTAIN** | `DecideAbstain` | `action`, `final_label`, `confidence`, `agreement`, `conflict` — **D5** | PRESENT · `⊘ ABSTAIN` |
| **PRESENT** | `Present` | `label`, `confidence`, `citations[]`, `rationale`, `abstained` | *(terminal)* |
| **⊘ ABSTAIN** | *(sink)* | `unverified` + reason + `decisions_trace` | *(terminal)* |

> **ABSTAIN is reachable from three states** (RETRIEVE / AGGREGATE / DECIDE) and is always a *clean, citable* terminal — it returns the evidence gathered so far and the full trace, never a guessed label.

### Threshold register (single source — `config.py`)

All decision predicates read from one config block so they are auditable and tunable without touching agent code.

| Symbol | Name | Default | Used by | Meaning |
|---|---|---|---|---|
| `τ_skip` | classifier-skip confidence | `0.95` | D2 | min classifier confidence to *skip* retrieval |
| `τ_cw` | check-worthiness floor | `0.50` | D2 | claim must score ≥ this to be worth checking |
| `τ_rel` | evidence relevance | `0.30` | D3 | min reranker score to keep a passage |
| `N_min` | min relevant evidence | `3` | D3 | coverage requires ≥ this many kept passages |
| `R_max` | max widen retries | `2` | D3 | retrieval re-query budget before abstain |
| `α` | prior↔evidence blend | `0.60` | D4 | weight on stance vote vs. classifier prior |
| `θ` | verdict deadband | `0.20` | D4 | `|S| ≤ θ·max ⇒ UNVERIFIED` |
| `τ_agree` | stance agreement | `0.40` | D5 | `|support−refute|/n` must clear this |
| `τ_present` | present confidence | `0.55` | D5 | min final confidence to present (else abstain) |

---

## 3. The five decision points

Each decision point is: **a typed predicate over intermediate outputs → a rule branch → (optional) what the brain may propose, constrained.** The rule is *always authoritative*; the brain is *advisory and validated*.

### D1 — Claim routing & central-claim extraction  (state CLAIM-EXTRACT)

**Predicate (rule).**
```
n = len(tokenize(text))
if doc_type == "claim" OR body is empty OR n ≤ 40:   route = SHORT_CLAIM   ; claim = text
else:                                                route = ARTICLE       ; claim = central_claim(title, body)
central_claim(title, body) = first non-empty of: title  →  lead sentence  →  top-TextRank sentence
if len(tokenize(claim)) < 5:  → ⊘ INVALID_INPUT      # too short to fact-check
entities = NER(claim)                                # named entities, numbers, dates → feed D3
```
**Branches.** `SHORT_CLAIM` → CLASSIFY+CHECK directly · `ARTICLE` → summarize-to-claim then continue · `< 5 tokens` → `⊘ INVALID_INPUT`.

**Brain may propose:** the **central claim string**, chosen *only* from the fixed candidate set `{title, lead sentence, top-TextRank sentence}`. It **cannot invent text**. Validation: proposed string must be a member of `candidates`. Fallback = `title` (else lead).
**Tool:** `ClaimExtract` (`fakenews/factcheck/claim_extractor.py`).

### D2 — Check-worthiness / classifier-confidence gate  (state CHECK-WORTHY)

**Predicate (rule).**
```
classifier_conf = max(probs)                          # from CLASSIFY
checkworthy_score = claimbuster_heuristic(claim)      # named-entity ∧ number/quantifier ∧ verifiable predicate
                                                       # (or zero-shot "is this a checkable factual claim?")
skip_retrieval = (classifier_conf ≥ τ_skip=0.95) AND (checkworthy_score < τ_cw=0.50)
```
**Branches.** `skip_retrieval == True` → **PRESENT (prior only)** with a visible *"style/probability signal, not a verdict"* notice (no fact-check spent on a confidently-non-checkable article) · else → **RETRIEVE**.

**Brain may propose:** `checkworthy ∈ {true, false}` and *which* entities/numbers make the claim checkable. It **cannot raise its own confidence** and cannot set `skip_retrieval` when the rule says proceed. Validation: boolean in-set; entity list ⊆ `entities` from D1. Fallback = rule heuristic.
**Tool:** `CheckWorthy`.

> D2 is a **cost gate**, deliberately conservative: it only skips the expensive retrieval+NLI path when the classifier is *very* confident **and** the text isn't a checkable factual claim. A check-worthy claim is *always* fact-checked regardless of classifier confidence.

### D3 — Evidence-coverage gate  (state RETRIEVE)

**Predicate (rule).**
```
hits   = RRF( BM25.topk(claim), dense.topk(claim) )           # reciprocal-rank fusion
kept   = [h for h in rerank(claim, hits) if h.rerank ≥ τ_rel=0.30]
n_relevant = len(kept)
coverage_ok = n_relevant ≥ N_min=3
```
**Branches.**
- `coverage_ok` → **STANCE/NLI**.
- thin / low-relevance **and** `widen < R_max=2` → **widen**: lower the effective τ, expand the query with `entities` (synonyms, aliases), add another corpus; retry RETRIEVE (`widen += 1`).
- retries **exhausted** and still `n_relevant < N_min` → `⊘ ABSTAIN` → `"unverified (insufficient evidence)"`.

**Brain may propose:** a **query reformulation / expansion term set** (entities, synonyms) and *which corpus to add* — drawn **only** from the `entities` extracted in D1. It **cannot change `τ_rel` or `N_min`**, and cannot declare coverage OK. Validation: terms ⊆ allowed expansion vocabulary. Fallback = entity-augmented query.
**Tool:** `RetrieveEvidence` (`fakenews/models/{retriever,bm25,vector_store}.py` + `fakenews/factcheck/hybrid.py` for RRF).

### D4 — Stance-aggregation verdict gate  (state AGGREGATE)

**Predicate (rule).** Per-evidence stance comes from the NLI head (premise = evidence, hypothesis = claim):

```
stance_value(e):  entail → +1   ·   contradiction → −1   ·   neutral → 0
wᵢ = e.rerank                                  # relevance weight (reranker / RRF score)
S  = Σ wᵢ · stance_value(eᵢ)                   # signed, relevance-weighted stance sum (normalized to [−1, +1])

prior_signal = (1 − 2·p_fake)                  # +1 ⇒ prior says REAL, −1 ⇒ prior says FAKE
score = α·S + (1−α)·prior_signal               # α = 0.60  → EVIDENCE DOMINATES, prior is a soft nudge

verdict =  REAL/SUPPORTED  if  S >  +θ·max
           FAKE/REFUTED    if  S <  −θ·max     # θ = 0.20
           UNVERIFIED      if |S| ≤  θ·max      # deadband → not enough signal either way
```
**Branches.** `REAL` · `FAKE` · `UNVERIFIED`. Note the **verdict is decided by the evidence sum `S`**; the classifier prior only shifts the reported `combined_conf` (a *soft nudge*), it cannot by itself create a REAL/FAKE verdict out of an evidence deadband.

**Brain may propose:** a **per-evidence stance ∈ {support, refute, neutral}** — but **only when the NLI head is near-tie** (`margin < 0.10`), and it **must cite the evidence span** it is judging. It **cannot override a confident NLI head** and **cannot change `α` or `θ`**. Validation: label in-set; only applied where `margin < 0.10`; cited span must exist in the evidence. Fallback = NLI `argmax` + the rule sum.
**Tool:** `AggregateVerdict` (`fakenews/factcheck/verdict.py`).

### D5 — Confidence / abstain gate  (state DECIDE)

**Predicate (rule).** Abstain if **any** of:
```
verdict == UNVERIFIED                                            # D4 deadband
agreement = |n_support − n_refute| / n_evidence  <  τ_agree=0.40 # evidence is split
final_conf  <  τ_present=0.55                                    # not confident enough to present
conflict:  classifier says FAKE (p_fake high)  AND  evidence SUPPORTS strongly   → flag "conflict, needs human review"
        (and symmetrically: classifier REAL but evidence REFUTES strongly)
```
**Branches.** none-fire → **PRESENT** (label + confidence + citations + rationale) · any-fire → `⊘ ABSTAIN("needs human review")`. A **conflict** is surfaced *explicitly* — the reviewer is told the style-prior and the evidence disagree, which is exactly the case where automation bias is most dangerous.

**Brain may propose:** a **natural-language rationale string** and **whether to surface a conflict note** — bounded to citing retrieved evidence ids only. It **cannot flip abstain → present** and **cannot change thresholds**. Validation: rationale may only reference `evidence[].id` that exist; conflict note can only *add* caution, never remove an abstain. Fallback = templated rationale.
**Tool:** `DecideAbstain` (`fakenews/factcheck/verdict.py` / `fakenews/agent/policy.py`).

### Decision-point summary

| ID | State | Reads (intermediate output) | Rule predicate | Branches | Brain proposes (constrained) | Fallback |
|---|---|---|---|---|---|---|
| **D1** | CLAIM-EXTRACT | `doc_type`, `len(tokens)`, `title/body` | `≤40 tok ∨ no body ⇒ short`; `<5 tok ⇒ invalid` | short · article · invalid | central claim ∈ {title, lead, TextRank} | title |
| **D2** | CHECK-WORTHY | `classifier_conf`, `checkworthy_score` | `conf≥0.95 ∧ cw<0.5 ⇒ skip` | skip(prior) · retrieve | `checkworthy ∈ {T,F}` + entities | rule heuristic |
| **D3** | RETRIEVE | `n_relevant`, `rerank` scores, `widen` | `n_relevant≥3 ⇒ ok`; else widen ≤2 else abstain | stance · widen · abstain | query/expansion terms ⊆ entities | entity-augmented query |
| **D4** | AGGREGATE | per-evidence `probs`, `margin`, `wᵢ`, `p_fake` | `S>+θ⇒REAL · S<−θ⇒FAKE · |S|≤θ⇒UNVERIFIED` | REAL · FAKE · UNVERIFIED | stance only if `margin<0.1`, must cite span | NLI argmax + rule sum |
| **D5** | DECIDE | `verdict`, `agreement`, `final_conf`, conflict | abstain if UNVERIFIED ∨ agree<0.4 ∨ conf<0.55 ∨ conflict | present · abstain | rationale + conflict note (cite ids) | templated rationale |

---

## 4. Typed tool contracts

Every tool exposes the **same uniform signature** so the FSM spine, the trace logger, and the no-torch fallback can treat them identically:

```python
Tool.run(**kwargs) -> {"ok": bool, "data": {...}, "meta": {...}}
```

Heavy dependencies (`torch`, `transformers`, `sentence-transformers`) are imported **inside** `run()`; if absent, the tool falls back to an sklearn/lexical path and sets `meta["backend"]` accordingly. The FSM only ever inspects `data` fields named in these contracts.

```python
# ── PARSE ────────────────────────────────────────────────────────────────────
IngestParse.run(raw: str, source_url: str | None) -> {
    "doc_type": "article" | "claim", "title": str, "body": str,
    "lang": str, "ok": bool}

# ── D1: CLAIM-EXTRACT ────────────────────────────────────────────────────────
ClaimExtract.run(title: str, body: str, doc_type: str) -> {
    "claim": str, "candidates": list[str], "entities": list[str],
    "method": "title" | "lead" | "textrank"}

# ── CLASSIFY  (fast PRIOR) ───────────────────────────────────────────────────
ClassifyTool.run(text: str, mode: "binary" | "liar6") -> {
    "label": "fake" | "real", "p_fake": float, "probs": dict[str, float],
    "backend": "transformer" | "tfidf_logreg"}            # tfidf_logreg = no-torch path

# ── D2: CHECK-WORTHY ─────────────────────────────────────────────────────────
CheckWorthy.run(claim: str, classifier_conf: float) -> {
    "checkworthy": bool, "score": float, "has_entity": bool, "has_number": bool,
    "skip_retrieval": bool, "tau_skip": float, "tau_cw": float}

# ── D3: RETRIEVE ─────────────────────────────────────────────────────────────
RetrieveEvidence.run(claim: str, entities: list[str], corpora: list[str],
                     top_k: int = 20, widen: int = 0) -> {
    "evidence": list[{"id": str, "text": str, "source": str,
                      "bm25": float, "dense": float, "rrf": float, "rerank": float}],
    "n_relevant": int, "coverage_ok": bool, "tau_rel": float,
    "backend": "dense" | "tfidf"}

# ── STANCE/NLI   (premise = evidence, hypothesis = claim) ────────────────────
StanceNLI.run(claim: str, evidence: list[dict]) -> {
    "stances": list[{"id": str, "label": "support" | "refute" | "neutral",
                     "probs": {"entail": float, "neutral": float, "contradiction": float},
                     "margin": float}],
    "backend": "deberta_mnli" | "bart_mnli" | "none",     # "none" = lexical mock-stance
    "id2label": dict}                                      # read from model.config at load — NEVER hardcode

# ── D4: AGGREGATE ────────────────────────────────────────────────────────────
AggregateVerdict.run(p_fake: float, stances: list[dict],
                     alpha: float = 0.6, theta: float = 0.2) -> {
    "verdict": "REAL" | "FAKE" | "UNVERIFIED", "stance_score": float,
    "n_support": int, "n_refute": int, "n_neutral": int, "combined_conf": float}

# ── D5: DECIDE/ABSTAIN ───────────────────────────────────────────────────────
DecideAbstain.run(verdict: str, combined_conf: float, n_evidence: int,
                  n_support: int, n_refute: int, p_fake: float) -> {
    "action": "present" | "abstain", "final_label": str, "confidence": float,
    "agreement": float, "conflict": bool, "reason": str}

# ── PRESENT ──────────────────────────────────────────────────────────────────
Present.run(final_label: str, confidence: float, evidence: list[dict],
            verdict_meta: dict, rationale: str) -> {
    "label": str, "confidence": float,
    "citations": list[{"id": str, "source": str, "snippet": str, "stance": str}],
    "rationale": str, "abstained": bool}

# ── Optional brain (never sets flow) ─────────────────────────────────────────
Brain.propose(state: str, legal_values: list, context: dict) -> {"value": <legal>, "raw": str}
#   on JSON-parse-fail / value ∉ legal_values / timeout → caller applies the deterministic rule.

# ── Cross-cutting audit ──────────────────────────────────────────────────────
ToolTrace.log(state, tool, inputs_hash, outputs, latency_ms,
              brain_proposal, rule_applied) -> None
```

### Engineering invariants

1. **NLI label order is read from `model.config.id2label` at load — never hardcoded.** (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` is verified `{0:entailment, 1:neutral, 2:contradiction}`, but the code does not assume it.)
2. **Stance direction:** premise = retrieved evidence, hypothesis = the claim. `entail→support→+1`, `contradiction→refute→−1`, `neutral→0`.
3. **No-torch degradation:** `ClassifyTool → tfidf_logreg`; `RetrieveEvidence → tfidf/BM25 cosine`; `StanceNLI → backend:"none"` (lexical mock-stance: count support/refute lexicon terms); `AggregateVerdict` then leans prior-only and yields `UNVERIFIED` unless the prior is extreme — the FSM still completes end-to-end with zero network.
4. **Determinism:** given identical tool outputs and identical config thresholds, the FSM path and verdict are reproducible bit-for-bit; the brain is the only stochastic element and it is fully overridable.
5. **Citations are verbatim.** Every entry in `citations[]` is an `id + source + snippet` copied from the retrieved corpus — never generated, never paraphrased by the brain.

### ToolTrace audit record

```python
ToolTrace(
  state          = "RETRIEVE",
  tool           = "RetrieveEvidence",
  inputs_hash    = "sha1:…",                 # claim+entities+corpora+widen, for caching & reproduction
  outputs        = {"n_relevant": 4, "coverage_ok": True, "tau_rel": 0.30},
  latency_ms     = 412,
  brain_proposal = {"value": None} | {"value": "<entity-expansion>", "raw": "…"},
  rule_applied   = "coverage_ok: n_relevant(4) ≥ N_min(3) → STANCE",
)
```

The full `decisions_trace` is **always** returned by `/factcheck` — even on abstain — so a reviewer can verify each decision *and each citation* line-by-line. Transparency is non-optional (Section 8 of the brief).

---

## 5. FSM pseudocode (the agent spine)

`fakenews/agent/factcheck_agent.py` — singleton-load-with-fallback; optional brain wired at each D-point.

```python
def fact_check(raw: str, source_url: str | None = None,
               cfg: AppConfig = CFG, brain: Brain | None = None) -> dict:
    trace = ToolTrace()

    # ── PARSE ────────────────────────────────────────────────────────────────
    p = IngestParse.run(raw, source_url)
    trace.log("PARSE", "IngestParse", outputs=p)

    # ── D1: CLAIM-EXTRACT / ROUTE ────────────────────────────────────────────
    ce = ClaimExtract.run(p["title"], p["body"], p["doc_type"])
    claim = propose_or_rule(brain, "D1", legal=ce["candidates"],
                            rule=ce["claim"], context=ce)              # brain ⊆ candidates
    if len(tokenize(claim)) < 5:
        return present_abstain("INVALID_INPUT", "claim too short to fact-check", trace)
    entities = ce["entities"]
    trace.log("CLAIM-EXTRACT", "ClaimExtract", outputs={**ce, "claim": claim})

    # ── CLASSIFY (fast prior) ────────────────────────────────────────────────
    cls = ClassifyTool.run(text=claim, mode="binary")
    p_fake, conf = cls["p_fake"], max(cls["probs"].values())
    trace.log("CLASSIFY", "ClassifyTool", outputs=cls)

    # ── D2: CHECK-WORTHINESS GATE ────────────────────────────────────────────
    cw = CheckWorthy.run(claim, classifier_conf=conf)
    skip = (conf >= cfg.tau_skip) and (cw["score"] < cfg.tau_cw)       # rule is authoritative
    skip = brain_may_confirm(brain, "D2", skip, legal={True, False})   # brain cannot force proceed→skip
    trace.log("CHECK-WORTHY", "CheckWorthy", outputs={**cw, "skip_retrieval": skip},
              rule_applied=f"skip={skip} (conf={conf:.2f}, cw={cw['score']:.2f})")
    if skip:
        return Present.run(cls["label"], conf, evidence=[], verdict_meta=cls,
                           rationale="High-confidence non-checkable item; classifier prior only "
                                     "(style/probability signal, NOT a verified verdict).")

    # ── D3: RETRIEVE + COVERAGE GATE (widen loop) ────────────────────────────
    ev, widen = None, 0
    while True:
        ev = RetrieveEvidence.run(claim, entities, cfg.corpora, top_k=20, widen=widen)
        trace.log("RETRIEVE", "RetrieveEvidence", outputs=ev,
                  rule_applied=f"n_relevant={ev['n_relevant']}, widen={widen}")
        if ev["coverage_ok"]:                                          # n_relevant ≥ N_min
            break
        if widen >= cfg.R_max:                                         # exhausted → abstain
            return present_abstain("UNVERIFIED", "insufficient evidence", trace, evidence=ev["evidence"])
        entities = expand_query(brain, "D3", entities, context=ev)     # brain ⊆ entities only
        widen += 1

    # ── STANCE / NLI (premise = evidence, hypothesis = claim) ────────────────
    st = StanceNLI.run(claim, ev["evidence"])
    trace.log("STANCE", "StanceNLI", outputs=st)

    # ── D4: AGGREGATE → VERDICT ──────────────────────────────────────────────
    stances = brain_break_ties(brain, "D4", st["stances"])            # only where margin<0.1, must cite
    agg = AggregateVerdict.run(p_fake, stances, alpha=cfg.alpha, theta=cfg.theta)
    trace.log("AGGREGATE", "AggregateVerdict", outputs=agg,
              rule_applied=f"S={agg['stance_score']:.2f} → {agg['verdict']}")

    # ── D5: DECIDE / ABSTAIN ─────────────────────────────────────────────────
    dec = DecideAbstain.run(agg["verdict"], agg["combined_conf"], len(ev["evidence"]),
                            agg["n_support"], agg["n_refute"], p_fake)
    trace.log("DECIDE", "DecideAbstain", outputs=dec, rule_applied=dec["reason"])
    if dec["action"] == "abstain":
        return present_abstain(dec["final_label"], dec["reason"], trace,
                               evidence=ev["evidence"], conflict=dec["conflict"])

    # ── PRESENT ──────────────────────────────────────────────────────────────
    rationale = brain_rationale(brain, "D5", dec, st, ev) or template_rationale(dec, st)
    return Present.run(dec["final_label"], dec["confidence"], ev["evidence"],
                       verdict_meta=agg, rationale=rationale)          # ships citations + trace
```

`propose_or_rule` / `brain_may_confirm` / `expand_query` / `brain_break_ties` / `brain_rationale` are thin wrappers: each calls `Brain.propose(...)`, **validates the result against the legal set**, and on any failure returns the deterministic value. This is the entire mechanism by which "the brain proposes, the rule decides."

---

## 6. Worked example — every decision fires

> **Input (short claim):** *"The WHO declared that drinking bleach cures COVID-19 in 2021."*

| Step | State / Decision | What happens | Key intermediate output |
|---|---|---|---|
| 1 | **PARSE** | Short text, no article body. | `doc_type="claim"`, `body=""` |
| 2 | **D1 · CLAIM-EXTRACT** | 11 tokens ≤ 40 → **short-claim route**. Claim = the input. NER pulls entities. | `claim="The WHO declared…2021."`, `entities=["WHO","COVID-19","2021","bleach"]`, `method="title"`. `len ≥ 5` ✓ (not invalid). |
| 3 | **CLASSIFY** | Transformer prior. | `label="fake"`, `p_fake=0.88`, `backend="transformer"`, `conf=0.88` |
| 4 | **D2 · CHECK-WORTHY** | `conf=0.88 < τ_skip=0.95` → cannot skip. `checkworthy_score=0.92` (named org + medical predicate + date). | `skip_retrieval=False` → **proceed to RETRIEVE** |
| 5 | **D3 · RETRIEVE** | RRF(BM25, dense) over the evidence corpus; rerank. 4 passages clear `τ_rel=0.30` (WHO statements, fact-check articles). | `n_relevant=4 ≥ N_min=3` → `coverage_ok=True`, **no widen** |
| 6 | **STANCE/NLI** | premise = each passage, hypothesis = claim. | 3× `contradiction→refute` (`probs.contradiction≈0.94/0.91/0.88`), 1× `neutral`. `backend="deberta_mnli"` |
| 7 | **D4 · AGGREGATE** | `S = −(0.94+0.91+0.88)+0 ≈ −2.73` (relevance-weighted, normalized). Prior `p_fake=0.88` agrees (`prior_signal≈−0.76`). `|S| > θ` and `S<0`. | `verdict="FAKE"`, `n_refute=3`, `n_support=0`, `combined_conf=0.91` |
| 8 | **D5 · DECIDE** | `agreement=|0−3|/4=0.75 ≥ τ_agree=0.40` ✓; `conf=0.91 ≥ τ_present=0.55` ✓; prior agrees → **no conflict**. | `action="present"`, `conflict=False` |
| 9 | **PRESENT** | Returns the verdict with verbatim citations. | `label="fake"`, `confidence=0.91`, `citations`=the 3 refuting WHO/fact-check passages (`id+source+snippet+stance`), `rationale="Multiple authoritative sources contradict this claim; no source supports it."`, `abstained=False` |

**Full `ToolTrace` attached** to the response. Reported separately: `classifier_prior = 0.88` (fake) **and** evidence `verdict = FAKE` — here they agree, so no conflict note is surfaced.

### Counter-examples (the abstain branches that make this honest)

| Scenario | Firing decision | Outcome |
|---|---|---|
| **Novel claim, no corpus coverage** | D3: widen ×2, still `n_relevant<3` | `⊘ ABSTAIN` → `"unverified (insufficient evidence)"` — never a guessed label |
| **Evidence split 2 support / 2 refute** | D5: `agreement=0/4=0.0 < τ_agree=0.40` | `⊘ ABSTAIN` → `"unverified (conflicting evidence)"` |
| **Classifier says FAKE (p_fake=0.9) but evidence strongly SUPPORTS** | D5: `conflict=True` | `⊘ ABSTAIN` → flagged `"conflict — needs human review"`, prior and evidence shown side-by-side |
| **Confident non-checkable article** ("Opinion: I love autumn") | D2: `conf≥0.95 ∧ cw<0.5` | PRESENT prior only, with the explicit *"style signal, not a verdict"* notice — no retrieval spent |

---

## 7. Module map (agent component)

| Module (`fakenews/`) | Role in the agent |
|---|---|
| `agent/state.py` | `Action ∈ {FAKE, REAL, UNVERIFIED, ABSTAIN, NEEDS_EVIDENCE}`; `ToolTrace`; `AgentState{claim, label, label_conf, evidence[], stances[], verdict, citations[], confidence}` |
| `agent/policy.py` | the **deterministic D1–D5 rules** + abstain rule (reads the threshold register) |
| `agent/tools.py` | uniform `run()` wrappers: `ClassifierTool, ClaimExtractorTool, EvidenceRetrieverTool, StanceTool, VerdictTool` |
| `agent/factcheck_agent.py` | the **FSM spine** (Section 5 pseudocode); singleton-load-with-fallback |
| `agent/llm_orchestrator.py` | optional **brain**: `Brain.propose`, validation against legal sets, rule fallback |
| `factcheck/claim_extractor.py` | D1 — headline / lead / TextRank claim extraction (rule-based; optional LLM) |
| `factcheck/hybrid.py` | RRF + min-max fusion of BM25 + dense (feeds D3) |
| `factcheck/stance.py` | zero-shot `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; lazy import; lexical mock-stance fallback |
| `factcheck/verdict.py` | D4 aggregation + D5 abstain logic → verdict, confidence, citations |
| `models/{classifier,baseline_tfidf}.py` | the fast prior (transformer / TF-IDF+LogReg no-torch) |
| `models/{retriever,bm25,vector_store}.py` | evidence retrieval backends (lazy sentence-transformers) |
| `config.py` | the **threshold register** (`τ_skip, τ_cw, τ_rel, N_min, R_max, α, θ, τ_agree, τ_present`) + model ids |

---

## 8. Model & license notes relevant to the agent

| Component | Verified id | License | Note |
|---|---|---|---|
| Stance / NLI head (default) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | **MIT** ✅ | `id2label` read at load; MNLI+**FEVER**+ANLI tuned → maps to support/refute/NEI |
| NLI fallback / baseline | `facebook/bart-large-mnli` | **MIT** ✅ | classic zero-shot fallback |
| Dense retriever | `sentence-transformers/all-MiniLM-L6-v2` | **Apache-2.0** ✅ | 384-d; CPU/T4-friendly |
| Dense retriever (alt) | `BAAI/bge-small-en-v1.5` | **MIT** ✅ | reused from P08/P09 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | **Apache-2.0** ✅ | `(query,passage)→score` for D3 `rerank` |
| Classifier base (prior) | `answerdotai/ModernBERT-base` / `distilbert-base-uncased` | **Apache-2.0** ✅ | trainable prior; auto-downgrades on T4 |

> ⚠️ **Copyleft flag:** any stance head **fine-tuned** on FEVER-family data (`copenlu/fever_gold_evidence`, `pietrolesci/nli_fever`, `tals/vitaminc`) inherits **cc-by-sa-3.0 + gpl-3.0** share-alike — flag for redistribution. The **default agent uses the zero-shot MIT NLI model** and therefore avoids this; fine-tuning is optional.

---

## 9. How this satisfies the rubric (Section I.7)

- **Multi-step reasoning** — a 7-state FSM (PARSE→CLASSIFY→CHECK-WORTHY→RETRIEVE→STANCE→AGGREGATE→DECIDE→PRESENT) with an ABSTAIN sink reachable from three states.
- **Tool use** — two heavy ML tools (hybrid **retrieval** with RRF; zero-shot **NLI/stance**) plus classifier / claim-extractor / verdict aggregator, all under one uniform `run()` contract.
- **Decisions on intermediate outputs** — **five** typed decision points (D1–D5), each branching on the models' own numeric outputs (`classifier_conf`, `checkworthy_score`, `n_relevant`/`rerank`, stance `probs`/`margin`, `agreement`, `combined_conf`, conflict).
- **Deterministic & auditable** — rules are authoritative; the optional LLM brain only *proposes legal values* and is overridden on any failure; every transition emits a `ToolTrace`; the full `decisions_trace` ships with every response.
- **Evidence-grounded & honest** — verbatim citations on every non-abstained verdict; `unverified` is first-class; `classifier_prior` and evidence `verdict` are reported separately and **conflicts are surfaced for human review**. **The agent flags; it never auto-censors.**
