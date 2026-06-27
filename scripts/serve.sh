#!/usr/bin/env bash
# Start the FastAPI REST API + Gradio UI (UI mounted at /ui) on port 7860.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec python -m fakenews.cli --config configs/infer.yaml serve --ui --host 0.0.0.0 --port 7860
