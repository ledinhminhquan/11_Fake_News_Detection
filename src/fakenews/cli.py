"""Command-line interface — the single entrypoint for the Fake News Detection system.

    fakenews <command> [options]

Commands: data, train-classifier, train-baseline, tune, evaluate, classify,
factcheck, demo-agent, serve, benchmark, error-analysis, monitor, generate-report,
generate-slides, autopilot, grade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Fake News & Misinformation Detection System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def _read_text(val: str) -> str:
    if not val:
        return ""
    p = Path(val)
    return p.read_text(encoding="utf-8") if p.exists() else val


def cmd_data(args):
    from .data.download_dataset import download_all
    print(json.dumps(download_all(_load(args)), indent=2, ensure_ascii=False))


def cmd_train_classifier(args):
    from .training.train_classifier import train_classifier
    print(json.dumps(train_classifier(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_train_baseline(args):
    from .training.train_baseline import train_baseline
    print(json.dumps(train_baseline(_load(args), limit=args.limit), indent=2))


def cmd_tune(args):
    from .training.tune import tune_classifier
    print(json.dumps(tune_classifier(_load(args), n_trials=args.n_trials, limit=args.limit), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    print(json.dumps(evaluate(_load(args), limit=args.limit).get("summary", {}), indent=2))


def cmd_classify(args):
    from .agent.fakenews_agent import FakeNewsAgent
    agent = FakeNewsAgent(_load(args), load_model=not args.fast)
    print(json.dumps(agent.classify(_read_text(args.text), title=args.title or ""), indent=2, ensure_ascii=False))


def cmd_factcheck(args):
    from .agent.fakenews_agent import FakeNewsAgent
    agent = FakeNewsAgent(_load(args), load_model=not args.fast)
    job = agent.run(_read_text(args.claim), title=args.title or "", mode=args.mode, save=False)
    print(json.dumps(job.to_dict(), indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.fakenews_agent import FakeNewsAgent
    from .data import samples
    agent = FakeNewsAgent(_load(args), load_model=not args.fast)
    for c in samples.claims():
        job = agent.run(c["claim"], mode="factcheck", save=False)
        sd = job.to_dict()
        ok = "OK " if sd["verdict"] == c["verdict"] else ("~  " if sd["verdict"] == "unverified" else "XX ")
        print(f"\n[{ok} gold={c['verdict']}] {c['claim'][:60]}")
        print(f"  verdict={sd['verdict']} conf={sd['confidence']} decisions={[(d['id'],d['branch']) for d in sd['decisions']]}")
        print(f"  evidence: {[(e['source'], e['stance']) for e in sd['evidence'][:2]]}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["FAKENEWS_INFER_CONFIG"] = str(args.config)
    target = "fakenews.api.app_combined:app" if args.ui else "fakenews.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_monitor(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fakenews", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="prefetch/sanity-check the datasets"); sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("train-classifier", help="fine-tune the transformer classifier (macro-F1)")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None)
    sp.set_defaults(func=cmd_train_classifier)
    sp = sub.add_parser("train-baseline", help="train the TF-IDF + LogReg baseline (sklearn, no GPU)")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_baseline)
    sp = sub.add_parser("tune", help="basic LR hyperparameter search")
    sp.add_argument("--n-trials", type=int, default=3); sp.add_argument("--limit", type=int, default=4000)
    sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="classifier vs baselines + fact-check (macro-F1, ROC-AUC, ECE)")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("classify", help="classify an article/headline (fast style signal)")
    sp.add_argument("--text", default=""); sp.add_argument("--title", default="")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_classify)
    sp = sub.add_parser("factcheck", help="agentic evidence-grounded fact-check of a claim")
    sp.add_argument("--claim", required=True); sp.add_argument("--title", default="")
    sp.add_argument("--mode", default="auto"); sp.add_argument("--fast", action="store_true")
    sp.set_defaults(func=cmd_factcheck)
    sp = sub.add_parser("demo-agent", help="run the fact-check agent on the seed claims")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server")
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--ui", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark"); sp.add_argument("--n", type=int, default=30)
    sp.add_argument("--warmup", type=int, default=3); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="per-item misclassification analysis")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("monitor", help="monitoring report from request logs")
    sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor)
    sp = sub.add_parser("generate-report", help="generate the PDF report")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check")
    sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
