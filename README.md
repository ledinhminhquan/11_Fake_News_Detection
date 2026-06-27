# 🕵️ Fake News & Misinformation Detection System

> Classify whether a news article / claim is **fake** vs **real** (with a calibrated
> confidence), then **fact-check** it: retrieve evidence → judge stance (support /
> refute / neutral) → aggregate a verdict with **citations** → **abstain** when
> uncertain. A trainable transformer classifier (with a TF-IDF + LogReg baseline)
> provides a fast prior; an agentic FSM wraps the classify → retrieve → stance →
> verdict → abstain pipeline.

> ⚖️ **The system flags content for human review and always shows its evidence — it
> never auto-removes, auto-blocks or auto-censors.** `unverified` (abstain) is a
> first-class outcome. It is an *assistant to human fact-checkers*, not an
> automated arbiter of truth.

**NLP in Industry — Final Assignment.** Author: **Le Dinh Minh Quan** (Student `23127460`).
Reference: [KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).

---

## ✅ How this repo meets every assignment requirement

| Requirement | Where it is delivered |
|---|---|
| **Business problem** | [`docs/problem_definition.md`](docs/problem_definition.md) |
| **Dev infra & tooling** | `src/fakenews/` package, `pyproject.toml`, `requirements*.txt`, `Makefile`, Docker, CI |
| **Data management** | dataset loaders + label normalization + synthetic seed ([`data/dataset.py`](src/fakenews/data/dataset.py)); [`docs/data_description.md`](docs/data_description.md), [`docs/data_card.md`](docs/data_card.md) |
| **Model selection & optimization** | fine-tuned classifier + **majority/TF-IDF/zero-shot baselines**; macro-F1, ROC-AUC, ECE; [`docs/model_selection.md`](docs/model_selection.md) |
| **Deployment** | FastAPI `/classify` + `/factcheck` + Gradio + CLI + Docker + HF Space; [`docs/deployment.md`](docs/deployment.md) |
| **Agentic AI** | deterministic FSM with **5 decision points** (classify → retrieve → stance → verdict → abstain) + optional LLM brain; [`docs/agent_architecture.md`](docs/agent_architecture.md) |
| **Continual learning & monitoring** | [`docs/continual_learning_monitoring.md`](docs/continual_learning_monitoring.md) + [`monitoring/drift_report.py`](src/fakenews/monitoring/drift_report.py) |
| **Privacy & robustness** | [`docs/privacy_robustness.md`](docs/privacy_robustness.md) |
| **Project management** | [`docs/project_plan.md`](docs/project_plan.md) |
| **Ethics** | [`docs/ethics_statement.md`](docs/ethics_statement.md) — the centrepiece |
| **Report + slides** | auto-generated `report.pdf` + `slides.pptx` (`fakenews autopilot`) |

---

## 🏗️ Pipeline

```
article / claim
  │  parse: short claim vs article → extract central claim     ── D1 claim routing
  ▼
CLASSIFIER (transformer / TF-IDF) → P(fake) prior              ── D2 check-worthiness gate
  │  (very confident non-claim article → skip retrieval)
  ▼
RETRIEVE evidence (BM25 + dense → RRF) over the corpus         ── D3 evidence-coverage gate (→ abstain)
  │  STANCE / NLI per evidence (support / refute / neutral)
  ▼
AGGREGATE verdict (evidence dominates, prior is a soft nudge)  ── D4 verdict gate
  ▼
real / fake / UNVERIFIED + confidence + citations + rationale  ── D5 confidence / abstain gate
```

## 📦 Models & data (ids VERIFIED on the HF Hub)

| Role | Id | License |
|---|---|---|
| **Classifier (trained core)** | `distilbert-base-uncased` (default) · `microsoft/deberta-v3-base` · `answerdotai/ModernBERT-base` | Apache / MIT / Apache |
| Stance / NLI (pretrained) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (zero-shot) | MIT |
| Evidence retrieval | `sentence-transformers/all-MiniLM-L6-v2` + `cross-encoder/ms-marco-MiniLM-L6-v2` + BM25 | Apache |
| **Baselines** | majority-class · TF-IDF + LogReg · zero-shot | — |
| Classifier data | `GonzaloA/fake_news` (⚠️ license unknown) · `chengxuphd/liar2` (Apache, 6-way) · `mohammadjavadpirhadi/…-english` (MIT) | mixed |
| Fact-check data | `fever/fever` (cc-by-sa-3.0 + gpl-3.0) · `BeIR/fever` (cc-by-sa-4.0) | copyleft |

> **⚠️ Label-polarity gotcha:** GonzaloA = 0 fake / 1 real; LittleFish = 0 real / 1 fake; mrm8488 = 1 fake.
> The loaders **normalize to the internal convention 0=real, 1=fake** (set `fake_label_value` per dataset).
> A bundled **40-sample seed + 16 evidence snippets + 8 gold claims** powers the fully-offline demo/tests/eval.

## 🗂️ Repository layout

```
src/fakenews/
├── config.py  cli.py  logging_utils.py
├── data/         samples.py (seed) · dataset.py · download_dataset.py
├── models/       classifier.py (transformer + TF-IDF baseline) · bm25.py · model_registry.py
├── factcheck/    retriever.py (BM25+dense+RRF) · stance.py (NLI + lexical) · verdict.py
├── training/     train_classifier.py · train_baseline.py · evaluate.py · tune.py · metrics.py
├── agent/        state.py · policy.py (D1–D5) · tools.py · llm_orchestrator.py · fakenews_agent.py
├── api/          schemas.py · dependencies.py · main.py · ui.py · app_combined.py
├── analysis/ autoreport/ monitoring/ automation/ grading/
configs/ · data/ · models/ · tests/ · docs/ · notebooks/ · app/ · deploy/ · sample_data/
```

---

## 🚀 Quickstart

```bash
pip install -e ".[ml,api,report]"

fakenews demo-agent --fast              # fact-check the seed claims (offline)
fakenews factcheck --claim "Drinking bleach cures every virus overnight." --fast
fakenews classify --text "Scientists confirm clouds are made of cotton candy." --fast
```

### Train
```bash
fakenews --config configs/train.yaml train-baseline      # TF-IDF + LogReg (sklearn, no GPU)
fakenews --config configs/train.yaml train-classifier    # fine-tune the transformer (macro-F1)
fakenews evaluate                                        # vs majority/TF-IDF + fact-check (macro-F1, ROC-AUC, ECE)
```
On Colab/GPU use the notebook (below) — it auto-profiles H100/A100/L4/T4.

### Serve
```bash
fakenews serve --ui --port 7860         # FastAPI /classify + /factcheck + Gradio UI at /ui
```

### One-button train → report + slides + self-grade
```bash
fakenews autopilot --no-train           # eval → analysis → report.pdf + slides.pptx + bundle
fakenews grade
```

---

## 🤖 The agent (mandatory agentic component)

A **deterministic FSM** with **five decision points** acting on the model's own intermediate outputs,
plus an *optional* LLM brain (`anthropic`, opt-in, validated — it only rephrases the rule decision over
real evidence, **never invents a verdict or a citation**):

- **D1** claim routing (article vs short claim; extract the central claim)
- **D2** check-worthiness / classifier-confidence gate (skip retrieval when very confident on a non-claim article)
- **D3** evidence-coverage gate (too little relevant evidence → abstain)
- **D4** stance-aggregation / verdict gate (evidence support→real / refute→fake, classifier prior is a soft nudge)
- **D5** confidence / abstain gate ("unverified — needs human review")

Every step is timed + traced; the classifier signal and the evidence verdict are reported **separately**.
See [`docs/agent_architecture.md`](docs/agent_architecture.md).

## ☁️ Colab / H100 training

Open [`notebooks/Fake_News_Detection_Colab_Training_H100_AUTOPILOT.ipynb`](notebooks/Fake_News_Detection_Colab_Training_H100_AUTOPILOT.ipynb)
— mounts Drive, installs Colab-safe deps, auto-profiles the GPU, fine-tunes resume-safely, evaluates vs
baselines + fact-check, and generates the report/slides. Step-by-step:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md). **Always report the cross-domain number** (PolitiFact→GossipCop)
— the source-style-leakage signal.

## 🧪 Tests

```bash
pytest -q        # CPU-only, no model/network downloads (seed + TF-IDF + lexical stance + BM25)
```

## 📚 Docs index

`docs/`: problem_definition · data_description · data_card · model_selection · evaluation ·
agent_architecture · deployment · continual_learning_monitoring · privacy_robustness ·
project_plan · ethics_statement · architecture · model_card · slide_deck_outline · DESIGN_BRIEF.

## 📝 License

MIT — see [`LICENSE`](LICENSE). Pretrained models keep their own licenses (table above). Training data
licenses vary (GonzaloA *unknown*; LIAR2 Apache; FEVER cc-by-sa+gpl) — flag before redistribution.
**This is an advisory, human-in-the-loop tool — never wire it to automated content takedown.**
