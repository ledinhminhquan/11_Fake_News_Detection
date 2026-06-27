"""Evaluation, error analysis, monitoring, report/slides generation and grading."""

from __future__ import annotations

from pathlib import Path

from fakenews.analysis.error_analysis import error_analysis
from fakenews.autoreport.report_pdf import generate_report
from fakenews.autoreport.slides_pptx import generate_slides
from fakenews.grading.checklist import build_checklist
from fakenews.monitoring.drift_report import monitoring_report
from fakenews.training.evaluate import evaluate


def test_evaluate_structure(cfg):
    res = evaluate(cfg, save=False)
    assert "model" in res and "baseline" in res
    assert "macro_f1" in res["model"] and "ece" in res["model"]
    assert "summary" in res and "beats_baseline" in res["summary"]
    assert "factcheck" in res


def test_error_analysis_structure(cfg):
    res = error_analysis(cfg, save=False)
    assert isinstance(res, dict)


def test_monitoring_handles_empty(cfg):
    res = monitoring_report(cfg, log_path="/nonexistent/requests.jsonl", save=False)
    assert isinstance(res, dict)
    assert res.get("n_requests", 0) == 0 or res.get("status") in {"no_data", "empty"}


def test_report_and_slides_generate(cfg, tmp_path):
    rep = generate_report(cfg, out_path=str(tmp_path / "report.pdf"))
    sli = generate_slides(cfg, out_path=str(tmp_path / "slides.pptx"))
    assert Path(rep).exists() and Path(rep).stat().st_size > 1000
    assert Path(sli).exists() and Path(sli).stat().st_size > 1000


def test_grade_repo():
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["FAIL"] == 0, [i for i in res["items"] if i["status"] == "FAIL"]
