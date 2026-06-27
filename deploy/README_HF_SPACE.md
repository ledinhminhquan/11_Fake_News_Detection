# Deploying to a Hugging Face Space (Docker SDK)

The Fake News Detection system ships as a single Docker image serving the FastAPI
REST API **and** the Gradio UI (mounted at `/ui`) on port `7860`.

## 1. Create the Space

1. New Space → **SDK: Docker** → name e.g. `fakenews`.
2. Push this repo to the Space (or point it at your GitHub repo).
3. The Space builds the root `Dockerfile` and runs
   `uvicorn fakenews.api.app_combined:app --host 0.0.0.0 --port 7860`.

## 2. Files the Space needs (already in this repo)

```
Dockerfile                 # CPU image: classifier + BM25 run on CPU
src/fakenews/...           # the package (installed with .[ml,api,report])
configs/infer.yaml         # serving config
sample_data/               # offline seed so the Space boots with zero network
```

## 3. Secrets (optional)

| Secret | Effect |
|---|---|
| `FAKENEWS_LLM_API_KEY` | Enables the optional LLM rationale brain (it only rephrases the rule decision over real evidence — it never invents a verdict or a citation). **Unset → the brain auto-disables** and the system runs fully on deterministic rules/models. |

## 4. First boot

On first request the Space pulls the classifier base + the NLI stance model
(`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) + the evidence embedder and
caches them. With no torch it serves the **TF-IDF + LogReg** classifier and the
**lexical-overlap** fact-check over the bundled seed evidence — so the demo always
works.

## 5. Endpoints

- `GET  /ui` — the Gradio app (Fact-check + Classify tabs).
- `POST /classify` — text → {label, probability} (fast style signal).
- `POST /factcheck` — claim → {verdict, evidence, rationale, abstained}.
- `GET  /healthz` · `GET /version`.

## 6. ⚖️ Responsible-use posture

This is an **advisory** tool: it **flags content for human review and always shows
its evidence — it never auto-removes, auto-blocks or auto-censors**. `unverified`
(abstain) is a first-class outcome. The classifier signal (style/source) and the
evidence-grounded verdict are reported **separately** so a reviewer sees when they
disagree. Do not wire this to any automated takedown.
