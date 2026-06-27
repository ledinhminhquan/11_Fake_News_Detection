# ☁️ Colab Training Guide — Fake News Detection

Fine-tune the **fake-news classifier** (and train the TF-IDF baseline) on Colab
(Pro/Pro+), then test classification + the evidence-grounded fact-check and collect
the deliverables. The notebook auto-adapts to **H100 / A100 / L4 / T4** and
**resumes** after a disconnect. The stance/NLI model + the evidence embedder are
pretrained (zero-shot) — not trained.

---

## 0. What you need
- Google **Colab** (Pro+ for H100/A100; T4/L4 also work — the classifier is light).
- (Recommended) a **public GitHub repo** with this project, or upload the folder to Drive.

## 1. Get the project onto Colab
- **Option A (GitHub):** push this folder to `https://github.com/<you>/fakenews`, set `GIT_REPO_URL` in cell 0.
- **Option B (Drive):** upload `11_Fake_News_Detection/` to `MyDrive/fakenews/fakenews/`.

## 2. Drive layout (artifacts persist here → training survives disconnects)
Auto-created under `MyDrive/fakenews/artifacts/`:
```
artifacts/
├── models/
│   ├── classifier/<version>/   # fine-tuned transformer classifier (+ latest pointer)
│   └── classifier/tfidf_logreg.joblib   # the TF-IDF + LogReg baseline
├── runs/         # eval / error-analysis / benchmark / monitoring JSON
├── submission/   # report.pdf + slides.pptx + submission_bundle.zip
└── hf_cache/     # HuggingFace model + dataset cache
```

## 3. Configure & run
1. Open `notebooks/Fake_News_Detection_Colab_Training_H100_AUTOPILOT.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU` (H100 if available).
3. **Cell 0 (Controls):**
   - `CLF_BASE` — `distilbert-base-uncased` (default, T4-light) / `microsoft/deberta-v3-base` (MIT, strongest) / `answerdotai/ModernBERT-base`,
   - `CLF_DATASET` — `GonzaloA/fake_news` (default; **set `FAKE_LABEL_VALUE=0`** for this mirror) / `ErfanMoosaviMonazzah/...` / `chengxuphd/liar2`,
   - ⚠️ **`FAKE_LABEL_VALUE`** — the raw value that means *fake* in your dataset (GonzaloA=0, but LittleFish mirrors use 1). The loader normalizes to internal 1=fake.
   - `MAX_TRAIN_SAMPLES`, `EPOCHS`.
4. `Runtime → Run all` → installs Colab-safe deps (never touches torch), auto-profiles the GPU, then runs
   **autopilot** (cell 9): train baseline + transformer → evaluate vs baselines + fact-check → analysis →
   `report.pdf` + `slides.pptx` + grade + bundle.
5. **Disconnected?** Re-run **cell 9** — it resumes from the last checkpoint on Drive.

## 4. Verify it worked
- **Cell 10c / 11** — `evaluate` should show the fine-tuned classifier **beating majority-class and the
  TF-IDF baseline** on macro-F1, and report **ROC-AUC + ECE** (calibration). The fact-check accuracy is reported too.
- **Cell 12** — `classify` gives a style signal; `factcheck` returns a verdict + evidence with stance.
- **Cell 13** — find `report.pdf` + `slides.pptx` + `submission_bundle.zip` in `…/submission/`.

## 5. ⚠️ Source-style leakage (read this)
Fake-news datasets often leak the **outlet/style**, not the **truth**. Always report the **cross-domain**
number (train on PolitiFact, test on GossipCop via the `LittleFish-Coder/*` mirrors) — a large macro-F1 drop
is the signature of source leakage and belongs in your report. The agentic **evidence layer** exists to
compensate; the classifier is only a *prior*.

## 6. Use the model later
```python
from transformers import pipeline
clf = pipeline("text-classification", model=".../models/classifier/latest")
clf("Miracle pill melts thirty pounds in a single day.")
```
or simply: `fakenews factcheck --claim "..."` / `fakenews serve --ui`.

## Troubleshooting
- **OOM** → lower `per_device_train_batch_size` (the GPU profile sets it) or `MAX_TRAIN_SAMPLES`.
- **Wrong labels** → check `FAKE_LABEL_VALUE` matches your dataset's polarity (the #1 fake-news pitfall).
- **License** → `GonzaloA/fake_news` license is *unknown* (flag); `chengxuphd/liar2` is Apache and
  `mohammadjavadpirhadi/...-english` is MIT if you need a clean license. FEVER is cc-by-sa+gpl (copyleft).
- **No torch** → the system falls back to TF-IDF + LogReg + a lexical stance (still runs, lower quality).
- ⚖️ The tool **flags for review, never auto-censors**; `unverified` (abstain) is a valid, safe outcome.
