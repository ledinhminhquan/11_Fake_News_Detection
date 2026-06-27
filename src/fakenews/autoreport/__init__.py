"""Auto-report layer: load run artifacts and render the report PDF + slides PPTX
(classification + fact-check metrics tables, baseline-comparison charts, decision
trace summaries). All heavy deps (reportlab, python-pptx, matplotlib) are
lazy-imported; every entrypoint degrades gracefully and never raises."""

from .artifact_loader import load_artifacts
from .report_pdf import generate_report
from .slides_pptx import generate_slides

__all__ = ["load_artifacts", "generate_report", "generate_slides"]
