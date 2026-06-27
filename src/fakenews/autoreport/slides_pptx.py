"""Generate the submission slides.pptx (python-pptx) — ~12 concise visual slides
matching the assignment's required slide list for the Fake News & Misinformation
Detection system. Degrades to a Markdown outline if python-pptx is unavailable.

Internal label convention: 0 = real, 1 = fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger
from . import charts as charts_mod
from .artifact_loader import (classifier_name, clf_metric, factcheck_metric,
                              load_artifacts, model_version)

logger = get_logger(__name__)


def _slides(cfg: AppConfig, arts: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    mf = clf_metric(arts, "model", "macro_f1")
    bf = clf_metric(arts, "baseline", "macro_f1")
    acc = clf_metric(arts, "model", "accuracy")
    auc = clf_metric(arts, "model", "roc_auc")
    ece = clf_metric(arts, "model", "ece")
    res = (f"classifier macro-F1 {mf:.3f} vs TF-IDF+LogReg {bf:.3f}"
           if (mf is not None and bf is not None) else "train + evaluate to populate results")
    detail = []
    if acc is not None:
        detail.append(f"accuracy {acc:.3f}")
    if auc is not None:
        detail.append(f"ROC-AUC {auc:.3f}")
    if ece is not None:
        detail.append(f"ECE {ece:.3f}")
    res2 = ("Headline: accuracy / macro-F1 / per-class P/R/F1 / ROC-AUC / ECE"
            if not detail else "Model: " + ", ".join(detail) + " (macro-F1 headline; ECE = calibration)")
    fc = factcheck_metric(arts, "accuracy")
    cov = factcheck_metric(arts, "coverage")
    fact = (f"fact-check verdict accuracy {fc:.3f} (coverage {cov:.2f})"
            if (fc is not None and cov is not None)
            else "fact-check: verdict accuracy + coverage + abstain-rate (evidence-grounded)")
    return [
        ("Fake News & Misinformation Detection",
         [f"{cfg.author} — Student {cfg.student_id}", "NLP in Industry — Final Assignment",
          "Article / claim → fake-vs-real + evidence-grounded fact-check",
          "Transformer classifier → agentic retrieval + NLI stance + verdict (D1–D5)",
          "Flags for human review — never auto-censors. cf. KaiDMML/FakeNewsNet"]),
        ("Business Problem & Motivation",
         ["Misinformation spreads faster than humans can fact-check it",
          "Manual review does not scale; lexical rules overfit to a source/era",
          "Wrongly flagging real news (false positive) is a censorship risk",
          "Goal: triage likely-fake content + ground verdicts in retrieved evidence"]),
        ("Proposed NLP Solution",
         ["Core: trainable transformer classifier (DistilBERT, fake-vs-real)",
          "Calibrated probability (ECE-tracked) — confidence can gate a review queue",
          "Agentic fact-check: retrieve evidence → NLI stance → verdict / abstain",
          "Two complementary signals: style (classifier) + evidence (fact-check)"]),
        ("System Architecture",
         ["ingest → classify → route claim (D1) → check-worthiness gate (D2)",
          "→ retrieve evidence (BM25 + dense + re-ranker) → NLI stance",
          "→ evidence-coverage gate (D3) → verdict gate (D4) → abstain gate (D5)",
          "→ assemble (label + prob + verdict + evidence + decisions trace)"]),
        ("Data Overview",
         ["Train (clf): GonzaloA/fake_news — licence unspecified (research-only flag)",
          "Secondary: chengxuphd/liar2 — Apache (permissive, redistribution-safe)",
          "Fact-check: fever/fever — CC-BY-SA + GPL (copyleft flag)",
          "Normalize to internal 1=fake/0=real; offline seed = 40 news + 16 evidence + 8 claims"]),
        ("Model & Evaluation Results",
         [res, res2, fact,
          "Watch per-class real-recall (false-flag rate) + ECE (calibration)"]),
        ("Agentic AI Component",
         ["Deterministic FSM + optional LLM brain (OFF by default, rule fallback)",
          "D1 claim routing (article vs short claim) · D2 check-worthiness gate",
          "D3 evidence-coverage gate (abstain on thin evidence)",
          "D4 verdict gate (support vs refute + prior) · D5 abstain gate (low confidence → unverified)"]),
        ("Deployment Overview",
         ["FastAPI /analyze · Gradio UI (paste-article + check-a-claim) · CLI `fakenews`",
          "Docker + HF Space (Docker SDK); lazy heavy deps, numpy/sklearn fallback",
          "Evidence + decisions trace shown per request; verdicts cite their sources",
          "model_version pinned; metadata-only request logging"]),
        ("Ethics, Privacy & Risks",
         ["Flags content for human review — NEVER auto-censors, hides, or deletes",
          "False positives on real news + political bias → watch per-class real-recall",
          "Source leakage (style memorisation) → strip artefacts, split by article, OOD transfer",
          "Calibration (ECE) + abstention (D5) → confident-but-wrong flags can't gate review"]),
        ("Key Takeaways & Future Work",
         ["Fine-tuned classifier beats TF-IDF+LogReg and the majority floor on macro-F1",
          "Agentic fact-check adds evidence-grounded verdicts with calibrated abstention",
          "Future: stronger retrieval/re-ranking, multilingual + cross-source transfer",
          "Future: bias audits, continual re-training on drifting misinformation"]),
    ]


def generate_slides(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    arts = load_artifacts(cfg)
    out_path = Path(out_path) if out_path else run_dir() / "report" / "slides.pptx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slides = _slides(cfg, arts)
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches, Pt
    except Exception as exc:
        logger.warning("python-pptx unavailable (%s); writing markdown outline", exc)
        md = "\n\n".join(f"## {t}\n" + "\n".join(f"- {b}" for b in bs) for t, bs in slides)
        alt = out_path.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    try:
        chart = charts_mod.classification_chart(arts.get("eval") or {},
                                                run_dir() / "report" / "slide_classification.png")
    except Exception:
        chart = None
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    accent = RGBColor(0x2B, 0x6C, 0xB0)
    for i, (t, bullets) in enumerate(slides):
        slide = prs.slides.add_slide(blank)
        bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        bar.fill.solid(); bar.fill.fore_color.rgb = accent; bar.line.fill.background()
        tf = bar.text_frame; tf.text = t
        tf.paragraphs[0].font.size = Pt(28); tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        body = slide.shapes.add_textbox(Inches(0.6), Inches(1.5),
                                        Inches(8.3 if (i == 5 and chart) else 12), Inches(5.4))
        bt = body.text_frame; bt.word_wrap = True
        for j, bp in enumerate(bullets):
            p = bt.paragraphs[0] if j == 0 else bt.add_paragraph()
            p.text = "•  " + bp; p.font.size = Pt(20); p.space_after = Pt(10)
        if i == 5 and chart:
            slide.shapes.add_picture(str(chart), Inches(8.9), Inches(1.7), width=Inches(4.0))
        foot = slide.shapes.add_textbox(Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.4))
        foot.text_frame.text = f"{title or cfg.project_title} — {author or cfg.author} ({cfg.student_id})"
        foot.text_frame.paragraphs[0].font.size = Pt(9)
    prs.save(str(out_path))
    logger.info("Slides -> %s", out_path)
    return str(out_path)


__all__ = ["generate_slides"]
