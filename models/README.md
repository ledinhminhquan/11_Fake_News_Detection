# Models directory

Trained model checkpoints live here at runtime. **Large artifacts are git-ignored**
(see `.gitignore`) — only this README is committed.

## What lands here after training

```
models/
├── classifier/                     # fine-tuned fake-news classifier (real vs fake)
│   ├── final/                      #   Trainer.save_model(...)
│   │   ├── config.json, model.safetensors, tokenizer.*
│   │   └── model_meta.json         #   {"version": "...", "base_model": "...", "metrics": {...}}
│   ├── tfidf_logreg.joblib         #   the TF-IDF + LogReg baseline (sklearn)
│   └── checkpoint-*/               #   intermediate HF checkpoints
└── stance/                         # optional fine-tuned stance / NLI model (else pretrained zero-shot)
    └── final/
```

| Path | Produced by |
|---|---|
| `classifier/` | `fakenews train-classifier` (or the Colab notebook autopilot) |
| `classifier/tfidf_logreg.joblib` | `fakenews train-baseline` (sklearn, no GPU) |
| `stance/` | `fakenews train-stance` (optional; the agent uses a pretrained MNLI model by default) |

Paths are configurable via `FAKENEWS_MODEL_DIR` (default under
`FAKENEWS_ARTIFACTS_DIR`). The stance/NLI model is **pretrained** (zero-shot MNLI)
unless you fine-tune it on FEVER.

If no fine-tuned model is present, the system **degrades gracefully**: the
classifier falls back to the **TF-IDF + LogReg** baseline (trained on the seed /
real data, sklearn-only, no torch), and the fact-check stance falls back to a
**lexical-overlap heuristic** — so a verdict with evidence is still produced offline.
