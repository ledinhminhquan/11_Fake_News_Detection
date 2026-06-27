# Data directory

**No large datasets are committed** (per the assignment rules and `.gitignore`).
This directory holds download/preparation scripts and small cached artifacts. The
committed, redistributable seed lives under `../sample_data/`.

## Data sources (see `docs/data_description.md` and `docs/data_card.md`)

| Role | Source | Notes |
|---|---|---|
| **Classifier training** | a clean real/fake article dataset (HF) | title + text → label (0=real, 1=fake) — the trainable core |
| **Short-claim credibility** | LIAR (HF) | 6-way truthfulness of short political statements |
| **Fact-check / evidence** | FEVER / climate-fever / health_fact (HF) | claim + evidence + {SUPPORTS, REFUTES, NEI} for the stance model + evidence corpus |
| **Offline fallback** | `sample_data/seed_news.jsonl` + `seed_evidence.jsonl` + `seed_claims.jsonl` | labeled samples + evidence snippets + gold-verdict claims |

(Exact verified ids + licenses are pinned in `docs/data_card.md`.)

## How to fetch

```bash
fakenews data            # download/sanity-check the classification + fact-check datasets
python -m fakenews.cli data
```

The loader **degrades gracefully**: if the HF datasets or network are unavailable
it falls back to the bundled seed, so every command (and the test-suite) runs
fully offline — TF-IDF + LogReg classifier + lexical-overlap fact-check, no torch.

## ⚠️ Note on labels

Fake-news labels are **noisy and contested**, and many datasets leak the *source/
style* rather than the *truth of the claim* (see `docs/model_selection.md`). The
system **flags content for human review — it never auto-removes** and always shows
its evidence and confidence.

## Layout at runtime (git-ignored)

```
data/
├── cache/        # HF datasets cache
├── processed/    # tokenised splits, the evidence index
└── *.parquet     # materialised tables
```
