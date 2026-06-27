"""Production monitoring & input-drift report from the request log (JSONL).

Reads the append-only request log written by the serving agent
(:class:`fakenews.logging_utils.JsonlLogger` at ``cfg.serving.request_log_path``)
and turns raw ``decision`` events into an at-a-glance health picture: request
volume, verdict distribution (fake / real / unverified), abstain rate, the
classifier-label distribution (fake vs real), latency (mean + p95), a
confidence / calibration-drift signal (mean verdict confidence over a recent
window vs an earlier reference window), and a simple *input-drift* signal that
compares a recent window of traffic against an earlier baseline window. A rising
abstain rate, a shifting verdict / classifier-label mix, a latency regression, or
a drop in mean confidence is the tell-tale of a claim-distribution shift or a
degrading model.

Each logged event has the shape::

    {"ts": ..., "event": "decision", "mode": "auto"|"classify"|"factcheck",
     "verdict": "real"|"fake"|"unverified", "clf_label": "fake"|"real",
     "abstained": true|false,
     "metrics": {"latency_ms": ..., "confidence": ..., ...}}

``confidence`` is read from ``metrics["confidence"]`` or a top-level
``confidence`` field when present; the calibration-drift signal degrades to
``available: False`` when neither is logged.

The function never raises past its entrypoint: a missing / empty / corrupt log
yields ``{"status": "no_data", ...}`` so callers (CLI, autoreport, API) can rely
on a dict always coming back. Heavy deps are avoided entirely; only the stdlib
is used so the module imports cleanly offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

# event field this monitor cares about
_DECISION = "decision"
# verdicts the agent emits
_FAKE = "fake"
_REAL = "real"
_UNVERIFIED = "unverified"
_VERDICTS = (_REAL, _FAKE, _UNVERIFIED)
# classifier labels
_CLF_LABELS = (_REAL, _FAKE)


def _read_logs(path: Path) -> List[Dict[str, Any]]:
    """Load and parse the JSONL request log, skipping blank/corrupt lines."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:  # unreadable file => behave like "no data"
        logger.warning("could not read request log %s: %s", path, exc)
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile (stdlib only). ``pct`` in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[rank]), 1)


def _safe_mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _confidence_of(r: Dict[str, Any]) -> Optional[float]:
    """Pull a verdict confidence from the event, tolerating where it was logged."""
    metrics = r.get("metrics") or {}
    if isinstance(metrics, dict):
        c = metrics.get("confidence")
        if _is_num(c):
            return float(c)
    c = r.get("confidence")
    return float(c) if _is_num(c) else None


def _window_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one window of ``decision`` events into monitoring metrics."""
    n = len(rows)
    if n == 0:
        return {"n": 0}

    verdicts: Dict[str, int] = {}
    clf_labels: Dict[str, int] = {}
    modes: Dict[str, int] = {}
    lats: List[float] = []
    confs: List[float] = []
    abstained = 0
    fake_v = real_v = unverified_v = 0
    clf_fake = clf_real = 0

    for r in rows:
        v = str(r.get("verdict", "?"))
        verdicts[v] = verdicts.get(v, 0) + 1
        if v == _FAKE:
            fake_v += 1
        elif v == _REAL:
            real_v += 1
        elif v == _UNVERIFIED:
            unverified_v += 1

        clf = str(r.get("clf_label", "?"))
        clf_labels[clf] = clf_labels.get(clf, 0) + 1
        if clf == _FAKE:
            clf_fake += 1
        elif clf == _REAL:
            clf_real += 1

        mode = str(r.get("mode", "?"))
        modes[mode] = modes.get(mode, 0) + 1

        if bool(r.get("abstained")):
            abstained += 1

        metrics = r.get("metrics") or {}
        if isinstance(metrics, dict):
            lat = metrics.get("latency_ms")
            if _is_num(lat):
                lats.append(float(lat))

        c = _confidence_of(r)
        if c is not None:
            confs.append(c)

    def _rate(count: int) -> float:
        return round(count / n, 4)

    stats: Dict[str, Any] = {
        "n": n,
        "verdict_distribution": verdicts,
        "fake_rate": _rate(fake_v),
        "real_rate": _rate(real_v),
        "unverified_rate": _rate(unverified_v),
        "abstain_rate": _rate(abstained),
        "clf_label_distribution": clf_labels,
        "clf_fake_rate": _rate(clf_fake),
        "clf_real_rate": _rate(clf_real),
        "mode_distribution": modes,
        "mean_latency_ms": _safe_mean(lats),
        "p95_latency_ms": _percentile(lats, 95),
    }
    # verdict-confidence figures power the calibration-drift signal
    if confs:
        stats["mean_confidence"] = _safe_mean(confs)
        stats["n_confidence"] = len(confs)
    return stats


def _delta(base: Dict[str, Any], recent: Dict[str, Any], key: str) -> Optional[float]:
    a, b = base.get(key), recent.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(b) - float(a), 4)
    return None


def _drift(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare an earlier baseline window with the recent window for drift."""
    if len(rows) < 6:
        return {
            "available": False,
            "reason": "need >=6 events to split baseline/recent windows",
        }
    half = len(rows) // 2
    base = _window_stats(rows[:half])
    recent = _window_stats(rows[half:])

    d_abstain = _delta(base, recent, "abstain_rate")
    d_fake = _delta(base, recent, "fake_rate")
    d_unverified = _delta(base, recent, "unverified_rate")
    d_clf_fake = _delta(base, recent, "clf_fake_rate")
    d_latency = _delta(base, recent, "mean_latency_ms")
    d_conf = _delta(base, recent, "mean_confidence")

    flags: List[str] = []
    # --- output / behaviour drift -------------------------------------------
    if (d_abstain or 0) > 0.10:
        flags.append("rising_abstain_rate")
    if d_unverified is not None and abs(d_unverified) > 0.15:
        flags.append("verdict_distribution_shift")
    # --- input drift: the classifier's fake-vs-real call is the input proxy --
    if d_clf_fake is not None and abs(d_clf_fake) > 0.20:
        flags.append("input_label_shift")
    if d_fake is not None and abs(d_fake) > 0.20:
        flags.append("fake_verdict_shift")
    # --- latency regression --------------------------------------------------
    if d_latency is not None and base.get("mean_latency_ms"):
        base_lat = base["mean_latency_ms"] or 1.0
        if d_latency / base_lat > 0.50:  # latency up >50% vs baseline
            flags.append("latency_regression")
    # --- confidence / calibration drift -------------------------------------
    if d_conf is not None and d_conf < -0.10:  # mean confidence dropped >0.10
        flags.append("confidence_drop")

    return {
        "available": True,
        "baseline_window": base,
        "recent_window": recent,
        "delta_abstain_rate": d_abstain,
        "delta_fake_rate": d_fake,
        "delta_unverified_rate": d_unverified,
        "delta_clf_fake_rate": d_clf_fake,
        "delta_mean_latency_ms": d_latency,
        "delta_mean_confidence": d_conf,
        "flags": flags,
        "alert": bool(flags),
    }


def _recommendations(overall: Dict[str, Any], drift: Dict[str, Any]) -> List[str]:
    """Turn the computed metrics into plain-English operator guidance."""
    recs: List[str] = []
    abstain = overall.get("abstain_rate") or 0.0
    unverified = overall.get("unverified_rate") or 0.0
    fake = overall.get("fake_rate") or 0.0
    p95 = overall.get("p95_latency_ms")
    mean_conf = overall.get("mean_confidence")
    flags = drift.get("flags") or []

    if abstain > 0.40:
        recs.append(
            "High abstain rate ({:.0%}): the D5 confidence gate is firing often — evidence "
            "coverage is thin. Consider lowering agent.min_verdict_confidence, raising "
            "retrieval.top_k, or expanding the evidence corpus.".format(abstain))
    if unverified > 0.50:
        recs.append(
            "Most requests resolve to 'unverified' ({:.0%}): the D3 coverage gate likely lacks "
            "relevant evidence — check the retriever index and agent.min_evidence / "
            "min_evidence_relevance.".format(unverified))
    if fake > 0.70:
        recs.append(
            "Fake-verdict share is high ({:.0%}): confirm this matches the incoming traffic and "
            "is not a classifier-prior bias (review classifier calibration / threshold).".format(fake))
    if isinstance(p95, (int, float)) and p95 > 4000:
        recs.append(
            "p95 latency is {:.0f} ms: profile retrieval + stance/NLI; consider a smaller stance "
            "model, fewer reranked passages, or caching the embedder.".format(p95))
    if isinstance(mean_conf, (int, float)) and mean_conf < 0.55:
        recs.append(
            "Low mean verdict confidence ({:.2f}): the verdict aggregation is borderline — "
            "re-check stance thresholds (stance.support_threshold / refute_threshold) and the "
            "classifier prior weight (agent.prior_weight).".format(mean_conf))
    if "confidence_drop" in flags:
        recs.append(
            "Confidence is drifting down vs the baseline window: a possible calibration drift — "
            "re-run training/evaluate and inspect the model ECE.")
    if "input_label_shift" in flags or "fake_verdict_shift" in flags:
        recs.append(
            "Input/verdict distribution is shifting: the incoming claim mix changed — re-evaluate "
            "the classifier on a fresh slice and consider refreshing fine-tuning data.")
    if drift.get("alert") and not recs:
        recs.append(
            "Drift flags {}: re-evaluate the classifier/stance on a fresh slice and watch the "
            "next monitoring window.".format(flags))
    if not recs:
        recs.append("No action needed: metrics within healthy operating ranges.")
    return recs


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    """Compute a production monitoring + drift report from the request log.

    Parameters
    ----------
    cfg : AppConfig
        Provides ``cfg.serving.request_log_path`` (the JSONL written at serve time).
    log_path : str, optional
        Override the log location (useful for tests / ad-hoc analysis).
    save : bool
        When ``True`` (default) writes ``run_dir()/monitoring/{latest,monitor-<stamp>}.json``.

    Returns
    -------
    dict
        ``{"status": "no_data"|"ok", "log_path", "n_events", "n_requests",
           "request_volume", "overall", "drift", "recommendations", ...}``.
        Degrades gracefully — a missing / empty log returns ``status == "no_data"``
        rather than raising.
    """
    path = Path(log_path) if log_path else cfg.serving.request_log_path

    rows = _read_logs(path)
    # only aggregate genuine decision events (ignore any other event types)
    events = [r for r in rows if r.get("event", _DECISION) == _DECISION]

    if not events:
        logger.info("monitoring: no decision events at %s", path)
        result = {
            "status": "no_data",
            "log_path": str(path),
            "n_events": 0,
            "n_requests": 0,
            "request_volume": 0,
            "overall": {"n": 0},
            "drift": {"available": False, "reason": "no events"},
            "recommendations": ["No request logs yet: exercise the agent / API to populate the log."],
            "note": "no request logs found yet",
            "generated_at": utc_stamp(),
        }
    else:
        overall = _window_stats(events)
        drift = _drift(events)
        result = {
            "status": "ok",
            "log_path": str(path),
            "n_events": len(events),
            "n_requests": len(events),
            "request_volume": len(events),
            "overall": overall,
            "drift": drift,
            "recommendations": _recommendations(overall, drift),
            "note": "",
            "generated_at": utc_stamp(),
        }

    if save:
        try:
            out = run_dir() / "monitoring"
            out.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            (out / f"monitor-{utc_stamp()}.json").write_text(payload, encoding="utf-8")
            (out / "latest.json").write_text(payload, encoding="utf-8")
        except Exception as exc:  # persistence is best-effort, never fatal
            logger.warning("monitoring: could not save report: %s", exc)

    logger.info(
        "monitoring: %s events, abstain=%.0f%% unverified=%.0f%% fake=%.0f%% p95=%s ms, drift_alert=%s",
        result["n_events"],
        100 * (result["overall"].get("abstain_rate") or 0.0),
        100 * (result["overall"].get("unverified_rate") or 0.0),
        100 * (result["overall"].get("fake_rate") or 0.0),
        result["overall"].get("p95_latency_ms"),
        result.get("drift", {}).get("alert", False),
    )
    return result


__all__ = ["monitoring_report"]
