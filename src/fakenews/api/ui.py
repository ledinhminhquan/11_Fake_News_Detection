"""Gradio UI for the Fake News Detection system.

Two tabs: (1) Classify — paste an article/headline → a fake/real style signal with
a confidence and an explicit "this is a probability signal, not a verdict" notice;
(2) Fact-check — paste a claim → a verdict chip (real / fake / **unverified**) with
a highlighted evidence list (source + stance + score) and the rationale.
``gradio`` is imported lazily.
"""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_STANCE_EMOJI = {"support": "🟢 support", "refute": "🔴 refute", "neutral": "⚪ neutral"}


def build_ui(cfg: Optional[AppConfig] = None):
    import gradio as gr  # lazy
    from ..agent.fakenews_agent import FakeNewsAgent

    cfg = cfg or AppConfig()
    agent = FakeNewsAgent(cfg, load_model=True)

    def do_classify(text, title):
        if not (text.strip() or title.strip()):
            return "Enter an article or a headline."
        r = agent.classify(text, title=title)
        bar = "🔴" if r["label"] == "fake" else "🟢"
        return (f"### {bar} {r['label'].upper()}  ·  P(fake) = {r['prior_fake']:.2f}\n\n"
                f"_This is a style/probability signal, **not** a verdict — use the Fact-check tab "
                f"for an evidence-grounded check._  \n<sub>model: {r['model_version']}</sub>")

    def do_factcheck(claim, title):
        if not (claim.strip() or title.strip()):
            return "Enter a claim.", []
        job = agent.run(claim, title=title, mode="factcheck", save=False)
        sd = job.to_dict()
        chip = {"fake": "🔴 FAKE", "real": "🟢 REAL", "unverified": "⚪ UNVERIFIED"}.get(sd["verdict"], sd["verdict"])
        head = (f"### {chip}  ·  confidence {sd['confidence'] or 0:.2f}\n\n"
                f"**Rationale:** {sd['rationale']}  \n"
                f"<sub>classifier prior: {sd['clf_label']} (P(fake)={sd['clf_prior_fake'] or 0:.2f}) · "
                f"decisions: {' · '.join(d['id']+':'+d['branch'] for d in sd['decisions'])}</sub>\n\n"
                "_Advisory only — flagged for human review. The system never auto-removes content._")
        table = [[e.get("source", ""), _STANCE_EMOJI.get(e.get("stance"), e.get("stance")),
                  round(e.get("score", 0), 2), round(e.get("relevance", 0), 2), e.get("text", "")[:160]]
                 for e in sd["evidence"]]
        return head, table

    with gr.Blocks(title=cfg.project_title) as demo:
        gr.Markdown(f"# 🕵️ {cfg.project_title}\nA fake-news **classifier** (fast style signal) wrapped by "
                    "an agentic, evidence-grounded **fact-check** (retrieve → stance → verdict with citations → "
                    "abstain). **It flags content for human review and shows its evidence — it never auto-censors.**")
        with gr.Tab("Fact-check (recommended)"):
            with gr.Row():
                claim = gr.Textbox(label="Claim", lines=3,
                                   value="Drinking bleach cures every virus overnight.")
                title2 = gr.Textbox(label="Title (optional)", value="")
            btn2 = gr.Button("Fact-check", variant="primary")
            verdict = gr.Markdown()
            evidence = gr.Dataframe(headers=["Source", "Stance", "Score", "Relevance", "Evidence"],
                                    label="Evidence", wrap=True)
            btn2.click(do_factcheck, [claim, title2], [verdict, evidence])
        with gr.Tab("Classify (style signal)"):
            text = gr.Textbox(label="Article / headline", lines=5)
            title = gr.Textbox(label="Title (optional)")
            btn = gr.Button("Classify", variant="primary")
            out = gr.Markdown()
            btn.click(do_classify, [text, title], [out])
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
    build_ui().launch(server_name=server_name, server_port=server_port, share=share)


__all__ = ["build_ui", "launch"]
