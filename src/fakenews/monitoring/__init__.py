"""Monitoring layer: drift / quality reporting (latency, label distribution,
abstain rate, confidence/calibration drift, input drift)."""

from .drift_report import monitoring_report

__all__ = ["monitoring_report"]
