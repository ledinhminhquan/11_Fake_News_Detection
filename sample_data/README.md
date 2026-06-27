# Sample data (committed, redistributable)

Small, self-contained artifacts so the demo, the agent and the test-suite run with
**zero network and no torch**. Mirrors the built-in seed in
`src/fakenews/data/samples.py` (original/synthetic content — no copied dataset).

| File | Rows | What |
|---|---|---|
| `seed_news.jsonl` | 40 | Labeled news samples — `{id, title, text, label}` (label **0=real, 1=fake**) |
| `seed_evidence.jsonl` | 16 | Evidence snippets for the fact-check — `{id, source, text}` |
| `seed_claims.jsonl` | 8 | Claims with gold verdicts — `{claim, verdict}` (real/fake) |
| `sample_requests.json` | 2 | Example `/classify` + `/factcheck` request bodies |

These power the offline ground truth used by `tests/` and `fakenews evaluate`
(classifier accuracy / macro-F1 / ROC-AUC / ECE, and fact-check verdict accuracy).
On Colab the real datasets (`GonzaloA/fake_news`, `chengxuphd/liar2`, FEVER) are
used instead.

Quick try:

```bash
fakenews factcheck --claim "Drinking bleach cures every virus overnight." --fast
# -> verdict: FAKE, with WHO evidence refuting the claim
```

> ⚖️ The system **flags content for human review and shows its evidence — it never
> auto-removes content.** `unverified` (abstain) is a first-class outcome.
