#!/usr/bin/env bash
# Offline demo: classifier + agentic fact-check on the seed claims (no GPU, no downloads).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m fakenews.cli demo-agent --fast
echo
echo "--- one fact-check ---"
python -m fakenews.cli factcheck --claim "5G towers secretly control the weather." --fast
