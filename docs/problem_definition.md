# P11 — Problem Definition

> **Course:** NLP in Industry — Final Assignment · **Section I.2 (Problem Definition)**
> **Author:** Le Dinh Minh Quan — Student ID 23127460
> **Project:** P11 — Fake News & Misinformation Detection System (package `fakenews`)
> **Source of truth:** [`docs/DESIGN_BRIEF.md`](./DESIGN_BRIEF.md)

---

## 1. Business context

Misinformation is a systemic problem, not a niche one. False and misleading content spreads
faster and farther than corrections, and the damage is concrete: health misinformation
(e.g. "drinking bleach cures COVID-19") drives real medical harm; fabricated political claims
distort elections; manufactured financial rumours move markets. Once a false claim circulates,
a retraction rarely reaches the same audience.

The human defence against this — professional fact-checkers and platform trust-&-safety teams —
is **structurally overwhelmed**. Fact-checking is slow, manual, and evidence-intensive: a single
claim can take hours to verify against primary sources, while suspect content arrives as a
firehose. Reviewers cannot read everything, so two failure modes dominate:

1. **Under-coverage** — genuinely harmful claims are never reviewed because the queue is too long.
2. **Wasted attention** — reviewers burn time on clearly-credible items that never needed a check.

P11 attacks this throughput gap. It is a **decision-support** system that triages incoming
content and, for claims worth checking, produces an **evidence-grounded verdict with citations**
that a human can verify line-by-line — turning hours of manual evidence-gathering into a
reviewable draft.

> **Non-negotiable framing (repeated throughout this project).** The tool **assists** humans; it
> **never auto-removes, auto-blocks, or auto-censors** content. Every output is a *review flag plus
> evidence*, never an enforcement action. `unverified` / abstain is a first-class, correct outcome —
> not a failure. This is the most ethically loaded project in the assignment set, and that framing
> is baked into the design, not bolted on (see §8 of the design brief).

---

## 2. Target users and jobs-to-be-done

| User | Primary job-to-be-done | What the system gives them |
|---|---|---|
| **Journalists / professional fact-checkers** | **Claim fact-check** — "Is this claim worth checking, and what does the evidence say?" | A verdict (`real` / `fake` / `unverified`) with cited evidence passages, a per-evidence stance badge, and an audit `decisions_trace` they can verify line-by-line. |
| **Platform trust-&-safety / moderators** | **Flag-for-review** — triage a firehose of posts | A fast credibility signal (the classifier prior) that routes only suspicious items to a human queue. Never an auto-takedown. |
| **Newsroom editors / researchers** | **Credibility scoring** — rate a claim/source on a scale | A 6-way LIAR-style credibility score (`pants-fire → true`) for nuance beyond binary fake/real. |
| **End readers (secondary)** | "Can I trust this headline?" | A transparent credibility indicator with the evidence shown, encouraging verification over blind trust. |

The three core jobs-to-be-done map directly onto the surfaced capabilities:

- **Flag-for-review** → fast classifier prior `P(fake)` on every item (`/classify`).
- **Claim fact-check** → the full agentic, evidence-grounded pipeline (`/factcheck`).
- **Credibility scoring** → the optional 6-way LIAR2 credibility head (`mode="liar6"`).

---

## 3. Precise problem statement

**Input.** A piece of text — either a full **article** (title + body) or a short **claim**.

**Output.** Two clearly separated signals:

1. **Classifier prior** — a fast, calibrated `P(fake)` from a trainable text classifier
   (internal label convention: **`0 = real`, `1 = fake`**). This is a *style/source signal*, not a
   truth judgement.
2. **Evidence-grounded verdict** — for check-worthy claims, an agentic fact-check that retrieves
   evidence, runs stance/NLI per passage, and aggregates a verdict ∈ {`real`, `fake`,
   **`unverified`**} **with verbatim citations**, or **abstains** when evidence is insufficient or
   conflicting.

**The pipeline (as actually built in package `fakenews`):**

```
INGEST/NORMALIZE → PARSE/CLAIM-EXTRACT (D1) → CLASSIFY (prior P(fake))
   → CHECK-WORTHY? (D2) → RETRIEVE evidence (BM25 + dense MiniLM + RRF) (D3)
   → STANCE/NLI per evidence (D4) → AGGREGATE verdict → DECIDE/ABSTAIN (D5) → PRESENT
```

- The **classifier** (transformer fine-tune — `distilbert-base-uncased` default /
  `microsoft/deberta-v3-base` / `answerdotai/ModernBERT-base`; plus a TF-IDF + LogReg baseline floor)
  gives the fast **prior**.
- The **agentic fact-check** then does the real work: extract the central claim → retrieve evidence
  (BM25 + dense `sentence-transformers/all-MiniLM-L6-v2`, fused with Reciprocal Rank Fusion) →
  run **stance/NLI** per evidence passage
  (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, zero-shot: `entail → support`,
  `contradiction → refute`, `neutral → NEI`) → **aggregate** a verdict where the evidence dominates
  and the classifier prior is only a soft nudge → **abstain** (`unverified`) on uncertainty.
- **The two signals are reported separately** (`classifier_prior` vs. evidence `verdict`) so a
  reviewer immediately sees when they disagree — which is itself a useful "needs human review" signal.

**Runs fully offline.** With only `numpy` / `pandas` / `scikit-learn` / `rank_bm25` installed (no
`torch`), the whole pipeline degrades gracefully: TF-IDF + LogReg classifier, lexical-overlap
stance, and BM25 over the committed seed evidence corpus. The service still boots and serves.

**Agent decision points (FSM, ≥5 — see brief §5).** D1 claim routing (article vs. short claim;
extract the central claim); D2 check-worthiness / classifier-confidence gate (skip retrieval only if
very confident *and* not a checkable claim); D3 evidence-coverage gate (too little evidence →
abstain); D4 stance-aggregation verdict gate (`real` / `fake` / `unverified`); D5 confidence /
abstain gate (abstain on low agreement, low confidence, or prior↔evidence conflict).

---

## 4. Why NLP is required

This problem cannot be solved with rules, keyword lists, or metadata alone — every step is an
open-vocabulary natural-language understanding task:

- **Claim extraction & check-worthiness** — identifying the central verifiable assertion in free
  text, and judging whether it is even a checkable factual claim, requires syntactic/semantic
  parsing, not pattern matching.
- **Evidence retrieval** — matching a paraphrased claim to relevant evidence requires *semantic*
  retrieval (dense embeddings) on top of lexical BM25; keyword overlap alone misses paraphrase and
  synonymy.
- **Stance / NLI** — deciding whether a passage *supports*, *refutes*, or is *neutral* toward a
  claim is exactly natural-language inference; there is no non-NLP proxy for "does this evidence
  contradict this claim?"
- **Robustness to adversarial paraphrase** — bad actors rewrite content to flip naive
  classifiers; an evidence-grounded NLP verdict is far harder to evade than surface-style detection.

A keyword/blocklist approach would catch neither novel misinformation nor benign uses of flagged
terms, and could not *justify* any decision. The evidence-grounded NLP design is what makes the
output **transparent and contestable** rather than an opaque label.

---

## 5. Success metrics

### 5.1 Business metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **Reviewer throughput uplift** | Claims triaged per reviewer-hour vs. the manual baseline | The core value proposition: more harmful claims reviewed, less time on credible ones. |
| **Flag-queue precision** | Fraction of *flagged-for-review* items a human agrees were worth reviewing | High precision means reviewer attention is not wasted; low precision erodes trust and adoption. |
| **Evidence-grounding rate** | Fraction of non-abstained verdicts shipped with ≥ 1 verbatim citation (**target: 100%**) | A verdict without evidence is not actionable or auditable; grounding is the product. |
| **Reviewer override rate / usefulness** | How often reviewers reverse the system, and whether they report the evidence as useful | Trust signal: low override *and* useful evidence = the tool is genuinely assisting. |

### 5.2 Technical metrics

**Classifier (in-domain *and* cross-domain — the cross-domain number is the honest one):**

| Metric | Role |
|---|---|
| **Macro-F1** | **Headline metric** — robust to the fake/real and 6-way LIAR class imbalance. |
| **Accuracy + per-class P/R/F1** | Surface the costly error explicitly: false positives (legit news flagged fake) → report `real`-recall and `fake`-precision. |
| **ROC-AUC** (binary) / macro-AUC (6-way) | Threshold-independent ranking quality. |
| **ECE (Expected Calibration Error)** + reliability diagram | **Over-confidence is dangerous** — a confidently-wrong "fake" can silence real news. Temperature-scale on a held-out split; report pre/post ECE. |
| **Cross-domain macro-F1 gap** (train `GonzaloA` → test GossipCop) | A large drop is the signature of **source-style leakage**; this number is itself an ethics metric. |

**Fact-check (FEVER-style):**

| Metric | Role |
|---|---|
| **Label accuracy** | SUPPORTS / REFUTES / NEI accuracy ignoring evidence. |
| **Evidence retrieval recall@k** (R@1/5/10) | Did retrieval surface the right evidence? Measured against `BeIR/fever-qrels`. |
| **Selective accuracy @ coverage** | Accuracy on the **non-abstained** set vs. abstain rate (risk–coverage curve) — rewards abstaining instead of guessing. |
| **Abstain / coverage quality** | Fraction abstained vs. selective accuracy — a good abstainer trades coverage for correctness. |

**Baselines the system must beat:** majority-class, TF-IDF + LogReg (the no-torch floor), and
zero-shot. The transformer fine-tune must beat the baseline **on the cross-domain split**, not just
in-domain (in-domain macro-F1 ≈ 0.95+ on these source-leaky sets is a red flag, not success).

---

## 6. What this system is *not* (ethical boundary)

To keep the framing unambiguous, the success criteria above are constrained by hard boundaries:

- It **assists** human fact-checkers and moderators — it is **not** an automated arbiter of truth.
- It **flags content for human review**; it **never auto-removes, auto-blocks, or auto-censors**.
  There is **no auto-takedown** anywhere in the design.
- It **always shows its evidence** — citations are extracted **verbatim** from the retrieved corpus
  with source IDs, never generated, and every verdict ships its `decisions_trace`.
- It **abstains under uncertainty** — `unverified` means *insufficient evidence*, **not** *true*.
- It reports the **classifier prior** and the **evidence verdict separately**, because the classifier
  detects *style/source patterns correlated with fakeness* — which is **not** the same as detecting
  falsehood.

A successful P11 is therefore one that demonstrably improves reviewer throughput and flag-queue
precision **while never taking an irreversible action on content automatically** — high recall on
legitimate news, transparent evidence on every verdict, and a confident, well-calibrated *abstain*
whenever the evidence does not support a call.
