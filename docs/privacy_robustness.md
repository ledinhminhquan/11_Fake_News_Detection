# P11 — Data Privacy & Model Robustness

> Section I.9 of the *NLP in Industry* final assignment, project **P11 — Fake News & Misinformation Detection System**.
> Author: Le Dinh Minh Quan (student 23127460). Package: `fakenews` (`src/fakenews/`).
> Companion to `docs/DESIGN_BRIEF.md` (the single source of truth for ids, licenses, thresholds, and the FSM). This document covers (1) what personal data the system can touch and how it is minimized, and (2) how the pipeline behaves under adversarial, out-of-domain, and degraded conditions — with the **abstain gate** and **offline graceful degradation** treated as first-class robustness properties.

---

## 0. Why this project's privacy/robustness posture is distinctive

P11 is the **most ethically loaded** system in the assignment set. Two structural facts shape everything below:

1. **It is an assist tool, never an arbiter.** The system emits a *review flag + evidence*, never an auto-takedown. A robustness failure here does not silently delete content; at worst it produces a wrong *advisory* that a human reviewer reads alongside the cited evidence and `decisions_trace`. The human-in-the-loop is itself a privacy and robustness control.
2. **Two signals, reported separately.** A fast **classifier prior** (`p_fake`, a style/source pattern signal) and an **evidence-grounded verdict** (retrieve → stance/NLI → aggregate) are returned as distinct fields. The classifier is the *brittle* component (vulnerable to paraphrase, source-style leakage); the evidence layer is the *robust* one. Keeping them separate means a single attack rarely flips both, and a reviewer always sees when they disagree.

Compared with the meeting-minutes project (P10), P11 handles **less PII** — it ingests claims and news text, not multi-speaker transcripts with names, calendars, and action-item owners. But it is not PII-free: **user-submitted text can contain personal information**, and the optional LLM "brain" can send that text to a third party. Those two facts drive the privacy design.

---

## 1. Data privacy

### 1.1 Data inventory — what flows through the system

| Data class | Where it enters | Sensitivity | Retention default |
|---|---|---|---|
| **User-submitted query text** (a claim, headline, or pasted article at `/classify` or `/factcheck`) | API / Gradio / CLI | **May contain PII** — a claim can name a private individual, quote a private message, include a phone number, address, or health detail | **Not retained raw.** Processed in-memory; only a hash + derived metrics are logged (§1.3) |
| **`source_url`** (optional, passed to `IngestParse.run`) | API request | Low–medium; a URL can reveal what a user is investigating | Not fetched server-side by default; stored only as a field on the trace if provided |
| **Evidence corpus** (FEVER/BeIR Wikipedia abstracts, LIAR2 `justification`, synthetic `SAMPLE_EVIDENCE`) | Built offline at index time | Public / published; license-tracked (see Brief §2b) | Persistent, versioned, read-only at serve time |
| **Training data** (`GonzaloA/fake_news`, `chengxuphd/liar2`, FEVER mirrors) | Offline training only | Public datasets; LIAR/PolitiFact name public figures (speakers) | Never loaded by the serving path |
| **`ToolTrace` records** (`state, tool, inputs_hash, outputs, latency_ms, brain_proposal?, rule_applied`) | Every FSM transition | Low — designed to be PII-free (hashes, not raw text) | Operational logs; rotated |
| **Citations returned to the user** | `Present.run` output | Public (verbatim from the corpus) | Returned, not stored per-user |

**Key asymmetry:** the *evidence* the system reasons over is public and license-tracked; the *only* potentially-private data is the **user's own submitted text**. So privacy work concentrates entirely on the request path.

### 1.2 Core principle — don't retain raw queries

The serving path is designed to be **stateless with respect to query content**. The `/factcheck` FSM holds the claim text in memory for the duration of one request, returns the verdict + evidence + trace, and **does not persist the raw claim**. Concretely:

- **`inputs_hash`, not `inputs`.** `ToolTrace.log(state, tool, inputs_hash, outputs, latency_ms, …)` is deliberately specified with `inputs_hash` (a SHA-256 of the canonicalized input), never the raw text. This is what lets the audit trail exist without becoming a PII store. The hash also doubles as the **claim-cache key** (Brief §6: "cached by claim hash"), so caching needs no plaintext either.
- **Outputs in the trace are bounded.** What we log per state is the *decision* (`coverage_ok`, `n_relevant`, `verdict`, `action`, `reason`), not the user's words. Evidence snippets in the trace come from the **public corpus**, not the user.
- **No per-user profile.** The system does not build a history keyed to an end user. There is no "this user previously checked claim X" store. Each request is independent.
- **Logs are metrics-shaped.** Drift monitoring (`monitoring/drift_report.py`) operates on *input-text statistics and label/verdict distributions*, not on retained raw queries — it watches aggregate drift, not individual submissions.

> Implementation note: the request schemas (`api/schemas.py`) should mark query fields as transient and the logger (`logging_utils.py` JSONL logger) should be the *only* sink, configured to receive hashes + metrics. If a deployment genuinely needs raw-query capture (e.g., to build an internal eval set), that must be an **explicit, documented opt-in** with its own retention window and access controls — never the default.

### 1.3 PII handling for the text that does arrive

Because a submitted claim *can* embed personal data, the request path applies defense-in-depth:

1. **Minimize at the edge.** Only the fields needed for the verdict (`title`, `body`/`claim`, optional `source_url`) are accepted; nothing else (no user id, no auth-derived identity) is attached to the claim record.
2. **Hash before log.** As above — the raw claim never reaches a durable sink.
3. **Optional PII scrub before any external call.** Before the *optional* LLM brain sees text (§1.4), a light regex/entity pass can redact obvious direct identifiers (emails, phone numbers, long digit runs). This is a mitigation for the external-call path specifically; the local pipeline does not need it because it never leaves the process.
4. **Citations are corpus-only.** The evidence shown back to the user is extracted **verbatim from the public corpus** with source IDs (Brief §8: "Citations are extracted verbatim … never generated"). The system never echoes the user's potentially-private text back inside a "citation," so the output channel cannot leak the input.

### 1.4 The optional LLM "brain" is the only external egress — opt-in / off by default

The agent FSM is **deterministic**. The optional LLM brain (`agent/llm_orchestrator.py`) does **not decide flow** — at each decision point D1–D5 it may only *propose* a value from a fixed legal set, and on parse-fail/timeout the deterministic rule fires (Brief §5). This design has a direct privacy payoff:

| Property | Consequence for privacy |
|---|---|
| Brain is **optional** | The entire system runs end-to-end with **no external model call** (the default, offline-capable mode in §2.5). |
| Brain only **proposes constrained values** | It receives a bounded context, and its output is validated against a legal set — it cannot exfiltrate by returning arbitrary text into the control flow. |
| Brain is the **only component that can send text off-box** | A single, auditable egress point. Turn it off and there is **zero external data flow**. |

**Commitment:** the LLM brain is **off by default** and gated behind an explicit `enable_brain` / `llm_orchestrator` config flag. Enabling it is an informed choice that "this deployment will send (potentially user-submitted) claim text to an external provider." When enabled:

- The user/operator is told, in the UI and docs, that text leaves the box.
- The optional PII scrub (§1.3.3) runs before the call.
- Every brain invocation is recorded in the trace as `brain_proposal` with `rule_applied` showing whether the proposal was accepted or overridden — so external influence is fully auditable.
- A timeout/parse-failure on the brain is **safe**: the deterministic rule takes over, so an unreachable or misbehaving provider degrades to the offline behavior rather than failing the request.

### 1.5 Data-handling summary

- **Default mode is fully local and stateless-by-content:** no raw-query retention, no external egress, hashes + metrics only.
- **External egress exists only through the opt-in LLM brain**, with PII scrub and full trace logging.
- **All persistent data is public + license-tracked** (evidence corpus, training sets) and never includes user submissions.
- **License hygiene as a redistribution-privacy concern:** several corpora are share-alike/copyleft (all `fever/*`, `copenlu/*`, `pietrolesci/nli_fever`, `BeIR/*` → cc-by-sa-3.0/4.0 ± gpl-3.0) and `GonzaloA/fake_news` is **license-unknown**; a stance model fine-tuned on FEVER **inherits share-alike**. This is flagged for any redistribution (Brief §2 "Avoid / unclear-license note").

---

## 2. Model robustness

The thesis of P11's robustness story: **the brittle component (style classifier) is deliberately demoted to a weak prior, and the robust component (evidence-grounded fact-check) dominates the verdict — with abstention as the backstop when even evidence is insufficient.**

### 2.1 Threat & failure surface

| # | Condition | Which component is exposed | Robust mitigation in P11 |
|---|---|---|---|
| R1 | **Adversarial paraphrase / style evasion** | Style classifier (brittle) | Verdict is evidence-dominated (α=0.6 on stance vs. prior); paraphrase rarely changes the *facts* the evidence speaks to |
| R2 | **Out-of-domain / new outlet** (source-style leakage) | Style classifier | Cross-domain eval makes the drop visible; evidence layer is source-agnostic; prior treated as weak only |
| R3 | **Noisy text** (typos, OCR, casing, boilerplate) | Retrieval + classifier | Hybrid BM25+dense+RRF retrieval; boilerplate stripping; lexical fallbacks |
| R4 | **Insufficient / stale evidence coverage** | Whole verdict | D3 coverage gate + D5 abstain → `unverified`, never a fabricated label |
| R5 | **Conflicting / disputed evidence** | Aggregation | D4/D5 low-agreement → abstain; `DISPUTED`-style claims surface as conflict, not a forced verdict |
| R6 | **Over-confidence (miscalibration)** | Classifier probability | ECE measurement + temperature scaling; confident-and-wrong is the dangerous case |
| R7 | **Heavy-dependency / network outage** | Transformer + dense + brain | Offline degradation: TF-IDF+LogReg + lexical stance + BM25, no torch |
| R8 | **Prior↔evidence conflict** | Cross-component | D5 explicitly flags `conflict, needs human review` instead of picking a side |

### 2.2 Adversarial paraphrase attacks — evidence beats style

**The attack.** A bad actor rewrites a false claim to read like neutral wire copy ("MOSCOW (Reuters) — …" style), or rewrites a true statement in hyper-partisan blog style, specifically to flip a **style classifier**. This works against style classifiers because they learned *formatting and outlet voice*, not veracity (Brief §4 leakage caveat: "99% in-domain F1 means the model learned 'is this Reuters formatting,' not veracity").

**Why P11 resists it.**

- **The verdict is evidence-grounded, not style-grounded.** Paraphrasing the wrapper does not change the underlying proposition, so retrieval still finds the same evidence and the NLI head (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) still entails/contradicts it. The Brief states this directly (§8): "Evidence-based fact-check is more robust than style classification; … don't rely on the classifier alone."
- **The classifier is a *prior*, not the decision.** In `AggregateVerdict` (D4): `score = α·(stance vote) + (1−α)·(prior signal)` with **α=0.6** — stance evidence is weighted *above* the prior. A flipped prior is a minority vote.
- **A flipped prior triggers a conflict flag, not a silent flip.** If the paraphrase succeeds in flipping `p_fake` while the evidence still refutes/supports, D5 fires the **prior↔stance conflict** branch ("classifier FAKE but evidence SUPPORTS strongly → flag `conflict, needs human review`"). The attack converts into a *human review request*, which is the safe outcome.
- **NLI robustness can be hardened.** `tals/vitaminc` (contrastive, "robust to subtle edits") is listed as an optional add for the stance fine-tune precisely to harden against minimal adversarial edits to claim/evidence.

**Eval commitment.** Keep an **adversarial/paraphrase eval set** (Brief §8) and report the classifier-only vs. full-pipeline accuracy on it. The gap is the robustness dividend of the evidence layer.

### 2.3 Out-of-domain / new-outlet inputs — source-style leakage

**The problem.** Training sets like `GonzaloA/fake_news` and WELFake leak *publisher and writing style*: all "real" rows are Reuters/AP wire copy, all "fake" rows are blog-rant style. A model trained on them can reach ~0.95+ in-domain macro-F1 **and still be useless on a new outlet** — it learned "is this Reuters formatting," not truth. This is the **#1 trap** (Brief §4 anti-overfitting checklist) and it is simultaneously a robustness failure and an ethics failure (it "entrenches bias against named sources").

**Mitigations baked into the build:**

1. **Boilerplate stripping** before training (`(Reuters)`, `WASHINGTON —`, datelines, bylines) so the classifier cannot anchor on outlet signatures (Brief §4.1a).
2. **Mandatory cross-domain eval.** Train on `GonzaloA` (PolitiFact-style) → test on `LittleFish-Coder/Fake_News_GossipCop` (cross-domain), and **report the macro-F1 drop explicitly**. The cross-domain number is the honest metric; a large drop is the *signature* of source-style leakage and "is itself an ethics metric" (Brief §7).
3. **Treat too-high in-domain F1 as a red flag, not success** (Brief §4.1c).
4. **Evidence layer is source-agnostic.** Retrieval + NLI reason about the *claim's proposition* against a fixed public corpus; they do not know or care which outlet phrased it. A brand-new outlet that the classifier has never seen still gets a fair evidence-based verdict.
5. **Metadata leakage guard for LIAR/LIAR2:** drop `speaker, state_info, subject` and the per-speaker `*_counts` columns — they encode the label distribution and would make the model memorize *who said it* rather than *what was said* (Brief §4.6).

> Net: the classifier's OOD brittleness is **measured** (cross-domain table) and **contained** (weak prior, source-agnostic evidence verdict), not hidden.

### 2.4 Noisy text robustness

Real inputs are messy: typos, inconsistent casing, OCR artifacts, HTML cruft, dateline boilerplate, truncated articles.

- **Hybrid retrieval tolerates lexical noise.** `RetrieveEvidence` fuses **BM25 (sparse) + dense (`sentence-transformers/all-MiniLM-L6-v2` / `BAAI/bge-small-en-v1.5`) via RRF**. Dense retrieval handles paraphrase/synonym noise; BM25 handles exact-term and rare-entity matches the dense model may miss. RRF means a passage only needs to rank well under *one* retriever to surface — a noise-robustness property of the fusion itself.
- **Optional reranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`) re-scores `(query, passage)` pairs, recovering relevance ranking when the first-stage scores are noisy.
- **Boilerplate normalization** in `IngestParse`/training removes datelines and source signatures so they are not mistaken for content.
- **Long-article handling.** ModernBERT's 8192-ctx (vs. 512 on DistilBERT/RoBERTa) reduces truncation-induced information loss on long docs; `max_length` auto-adapts by GPU profile.
- **Tokenizer-level resilience.** Subword tokenization (DeBERTa-v3 / ModernBERT / DistilBERT) degrades gracefully on typos rather than dropping out-of-vocabulary terms entirely.

### 2.5 The ABSTAIN gate as a robustness property

Abstention is not an error path — it is a **designed robustness guarantee**: *when the system is not in a regime where it can be trusted, it says so instead of guessing.* `unverified` is first-class (Brief §8: "abstain is a first-class outcome, not a failure mode").

Three gates can trigger abstention, each guarding a different failure mode:

| Gate | State | Rule (from Brief §5 decision table) | Robustness guarantee |
|---|---|---|---|
| **D3 — evidence-coverage** | RETRIEVE | After RRF, keep evidence with reranker `score ≥ τ_rel = 0.3`; coverage OK iff `n_relevant ≥ N_min = 3`. Too thin → widen query (entities/synonyms) up to `R_max = 2` retries; still short → **ABSTAIN `unverified (insufficient evidence)`** | No verdict on out-of-corpus / novel / stale claims. Stops fabrication on R4. |
| **D4 — verdict** | AGGREGATE | `S = Σ wᵢ·sᵢ` (entail +1, contra −1, neutral 0, reranker-weighted); `score = α·S + (1−α)·prior`, **α=0.6**; `\|S\| ≤ θ` (**θ=0.2** of max) → **UNVERIFIED** | When evidence is balanced/weak, no forced REAL/FAKE. Handles R5. |
| **D5 — confidence / agreement / conflict** | DECIDE/ABSTAIN | Abstain if any of: `verdict == UNVERIFIED`; `agreement = \|support − refute\| / n_evidence < τ_agree = 0.4`; `final_conf < τ_present = 0.55`; **prior↔stance conflict** | Catches low-agreement, low-confidence, and cross-component disagreement (R5, R6, R8) |

**Worked abstain case (Brief §5 counter-example):** a novel claim with no corpus coverage → D3 widens the query twice → still `n_relevant < 3` → **ABSTAIN `unverified (insufficient evidence)`**, never a guessed label. The user gets "Not enough evidence — flagged for human review," rendered prominently in the Gradio UI with distinct neutral styling, **never a fake/real label by default**.

This is why abstention is a *robustness* mechanism and not a cop-out: it converts every "the model is outside its competent regime" situation into an explicit, honest, human-routable signal.

### 2.6 Offline graceful degradation — the system never hard-fails

Every heavy dependency (`torch`, `transformers`, `sentence-transformers`, the LLM brain) is **lazy-imported inside `run()`**; if absent, the tool falls back. The whole pipeline runs with **only `numpy` / `pandas` / `scikit-learn` / `rank_bm25`** and **no network** (Brief §5 invariants, §9 offline-fallback design):

| Stage | Full (heavy deps) | Degraded (no torch / offline) |
|---|---|---|
| **Classify** (`ClassifyTool`) | Transformer fine-tune (`ModernBERT`/`DistilBERT`) | **TF-IDF + LogReg** (`models/baseline_tfidf.py`), `backend:"tfidf_logreg"`, sub-ms CPU |
| **Retrieve** (`RetrieveEvidence`) | BM25 + dense (MiniLM/BGE) + RRF | **BM25 / TF-IDF cosine** over the seed evidence, `backend:"tfidf"` |
| **Stance** (`StanceNLI`) | Zero-shot DeBERTa-MNLI-FEVER-ANLI | **Lexical mock-stance** (count support/refute terms), `backend:"none"` |
| **Aggregate** (`AggregateVerdict`) | Stance-dominated + prior nudge | **Prior-only**, forced `UNVERIFIED` unless prior is extreme |
| **Brain** (optional) | LLM proposes constrained values | Deterministic rule fires (always available) |

Offline fixtures ship in-repo so this is testable with zero network: `SAMPLE_NEWS` (~36 labeled items), `SAMPLE_EVIDENCE` (~16 snippets, mixed support/refute/off-topic), `SAMPLE_CLAIMS` (~6 `ClaimCase` with gold FEVER verdicts incl. ≥1 SUPPORTS / ≥1 REFUTES / ≥1 NOT_ENOUGH_INFO), plus `tests/fixtures/fake_news_tiny.csv`. These exercise the classifier, RRF retrieval, stance aggregation, **the abstain branch**, and FEVER-style scoring end-to-end without torch.

> Robustness consequence: a missing model file, a torch import error, or a network outage **degrades quality, not availability**. The service still boots (`/healthz`), still classifies, still fact-checks (more conservatively, abstaining more often). There is no single point of total failure. Under capacity pressure, the gated NLI path degrades to `"unverified — capacity"` rather than timing out (Brief §6).

### 2.7 Calibration — over-confidence is the dangerous failure

In a misinformation tool, a **confident-and-wrong** output is worse than an uncertain one: an over-confident "fake" can silence real news, and reviewers exhibit **automation bias** (rubber-stamping confident model output). So calibration is a named robustness requirement, not a nicety.

- **Measure ECE** (Expected Calibration Error) + a reliability diagram on a held-out calibration split (Brief §7). Report **pre- and post-calibration** ECE.
- **Temperature scaling** is applied on the held-out split to deflate over-confident logits before the probability is exposed at `/classify` (which returns the *calibrated* `p_fake`).
- **Calibration interacts with the abstain gate.** D5's `final_conf < τ_present = 0.55` threshold only does its job if confidence is *calibrated* — otherwise a systematically over-confident model would clear the bar while being wrong. Calibration is therefore a prerequisite for the abstain gate to be trustworthy.
- **Confidence is always displayed**, never hidden, so reviewers can weight it (Brief §8 automation-bias mitigation). The classifier prior and the evidence confidence are shown **separately**, so a high prior cannot masquerade as a high evidence-confidence.

### 2.8 Robustness eval plan (what to actually report)

| Robustness axis | Eval | Metric / artifact |
|---|---|---|
| Adversarial paraphrase | classifier-only vs. full pipeline on a paraphrase eval set | accuracy gap (robustness dividend) |
| Out-of-domain | train `GonzaloA` → test `LittleFish-Coder/Fake_News_GossipCop` | **cross-domain macro-F1 drop** |
| Abstain quality | risk–coverage curve | selective accuracy at fixed coverage; abstain rate |
| Calibration | held-out calibration split | ECE pre/post temperature scaling; reliability diagram |
| Coverage / staleness | claims with no corpus support | abstain-on-insufficient-evidence rate (should be ~100%) |
| Offline degradation | run pipeline with torch absent | end-to-end pass on `SAMPLE_*` fixtures; more-abstain behavior verified |

---

## 3. Failure cases and mitigations (consolidated)

| Failure | Symptom | Why it happens | Mitigation in P11 |
|---|---|---|---|
| **Paraphrase evades classifier** | Style classifier flips on reworded text | Classifier learned outlet voice, not facts | Evidence-dominated verdict (α=0.6); D5 conflict flag; vitaminc-hardened stance (optional); paraphrase eval set |
| **New outlet collapse (OOD)** | High in-domain F1, poor real-world | Source-style leakage | Boilerplate stripping; mandatory cross-domain eval + reported drop; source-agnostic evidence layer; prior is weak only |
| **Novel/uncovered claim** | No evidence retrieved | Claim outside the fixed corpus | D3 widen ×2 → **ABSTAIN `unverified (insufficient evidence)`**, never a guess |
| **Stale index** | Confidently wrong on recent events | Corpus out of date | Version + timestamp index (`/version`, `index_built_at`); abstain on thin coverage |
| **Conflicting evidence** | Sources disagree | Genuinely disputed claim | D4 `\|S\| ≤ θ` → UNVERIFIED; D5 low-agreement → abstain; surfaced as conflict, not forced verdict |
| **Over-confident wrong label** | Calibrated trust violated; real news silenced | Miscalibration + automation bias | ECE + temperature scaling; D5 `τ_present`; separate prior/evidence confidence; mandatory confidence display |
| **Prior↔evidence disagreement** | Classifier says FAKE, evidence SUPPORTS | Style signal ≠ truth | D5 explicit `conflict, needs human review` flag — does not auto-pick a side |
| **Torch / network unavailable** | Heavy backends absent | Deployment / outage | Lazy imports + TF-IDF/LogReg + lexical stance + BM25 fallback; service still boots and abstains more |
| **LLM brain unreachable / malformed** | External provider down or returns junk | Timeout / parse-fail / out-of-set value | Deterministic rule fires; brain can only *propose* constrained values; fully auditable in trace |
| **Hallucinated citation** | A "cited" passage doesn't exist | Generated rather than retrieved evidence | Citations **extracted verbatim** from the corpus with source IDs; `decisions_trace` makes each verifiable |
| **PII in a submitted claim** | Personal data in a query | Users paste arbitrary text | No raw-query retention (hash only); optional PII scrub before any external call; corpus-only citations |

---

## 4. Summary

**Privacy.** P11 handles less PII than transcript-based projects, but user-submitted text can still carry personal data. The design responds by (a) **not retaining raw queries** — only `inputs_hash` + metrics reach durable storage; (b) keeping the default pipeline **fully local with zero external egress**; (c) making the **LLM brain the single, opt-in, off-by-default egress point**, with PII scrub and full trace auditing; and (d) ensuring all persistent data is **public and license-tracked**, with copyleft/unknown-license corpora explicitly flagged for redistribution.

**Robustness.** The system's defining move is to **demote the brittle style classifier to a weak prior (α=0.6 favors evidence)** and let an **evidence-grounded fact-check dominate** — which is why adversarial paraphrase and new-outlet (out-of-domain) inputs, the classic style-classifier failures, do not control the verdict. **Abstention is a first-class robustness guarantee** (D3 coverage, D4 verdict, D5 confidence/agreement/conflict gates) that turns every "outside competent regime" situation into an honest `unverified — needs human review`. **Offline graceful degradation** (TF-IDF+LogReg + lexical stance + BM25, no torch, no network) means the system degrades in *quality*, never *availability*. And **calibration (ECE + temperature scaling)** directly attacks the most dangerous failure mode — confident, wrong outputs that could silence legitimate news — while keeping the classifier prior and evidence confidence visibly separate so a human reviewer always sees the full picture.
