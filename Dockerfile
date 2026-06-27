# Fake News Detection — API + UI image.
# CPU image by default (classifier + BM25 run on CPU). For GPU, base off an
# nvidia/cuda runtime and install a CUDA torch wheel via `.[ml]`.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    FAKENEWS_ARTIFACTS_DIR=/artifacts

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ml,api,report]"

COPY configs ./configs
COPY docs ./docs
COPY sample_data ./sample_data

EXPOSE 7860
# Combined REST API + Gradio UI (UI mounted at /ui)
CMD ["uvicorn", "fakenews.api.app_combined:app", "--host", "0.0.0.0", "--port", "7860"]
