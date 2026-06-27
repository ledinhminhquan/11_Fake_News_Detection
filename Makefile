.PHONY: help install install-all data train train-baseline evaluate demo classify factcheck serve ui test grade report slides autopilot clean

help:
	@echo "Fake News & Misinformation Detection — common tasks"
	@echo "  make install          core install (pip install -e .)"
	@echo "  make install-all      full install (.[all])"
	@echo "  make data             prefetch/sanity-check the datasets"
	@echo "  make train            fine-tune the transformer classifier (macro-F1)"
	@echo "  make train-baseline   train the TF-IDF + LogReg baseline (sklearn)"
	@echo "  make evaluate         classifier vs baselines + fact-check"
	@echo "  make demo             run the fact-check agent on the seed claims (offline)"
	@echo "  make serve / ui       FastAPI server / + Gradio UI at /ui"
	@echo "  make test report slides autopilot grade"

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

data:
	fakenews data

train:
	fakenews --config configs/train.yaml train-classifier

train-baseline:
	fakenews --config configs/train.yaml train-baseline

evaluate:
	fakenews evaluate

demo:
	fakenews demo-agent --fast

classify:
	fakenews classify --text "Scientists confirm clouds are made of cotton candy you can harvest." --fast

factcheck:
	fakenews factcheck --claim "Drinking bleach cures every virus overnight." --fast

serve:
	fakenews --config configs/infer.yaml serve --host 0.0.0.0 --port 8000

ui:
	fakenews serve --ui --host 0.0.0.0 --port 7860

test:
	pytest -q

grade:
	fakenews grade

report:
	fakenews generate-report

slides:
	fakenews generate-slides

autopilot:
	fakenews autopilot --no-train

clean:
	rm -rf artifacts __pycache__ .pytest_cache src/*.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
