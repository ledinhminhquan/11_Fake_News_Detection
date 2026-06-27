"""Latency benchmark: end-to-end fake-news / fact-check agent throughput.

Builds a single ``FakeNewsAgent`` (TF-IDF + LogReg classifier + lexical stance over
the seed evidence, ``load_model=False`` so it stays offline/deterministic) and times
``.run(text, mode='factcheck')`` over the seed claims. Reports p50 / p95 / p99 / mean
of the job ``latency_ms``, throughput, how often each decision (D1..D5) fires, and the
classifier / stance names — a cheap, reproducible performance profile of the agent FSM.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _pct(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "mean": round(float(a.mean()), 2)}


def _job_latency_ms(job, fallback_ms: float) -> float:
    """Prefer the agent-reported latency; fall back to the wall-clock measurement."""
    try:
        v = job.metrics.get("latency_ms")
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        pass
    return float(fallback_ms)


def _seed_claims() -> List[str]:
    """A deterministic workload of claim strings from the seed gold claims."""
    try:
        from ..data import samples  # lazy
        out = [str(c.get("claim", "")).strip() for c in samples.claims()]
        out = [c for c in out if c]
        if out:
            return out
    except Exception as exc:
        logger.info("benchmark: could not load seed claims (%s)", exc)
    return ["Drinking bleach cures every virus overnight."]


def benchmark(cfg: AppConfig = None, n: int = 30, warmup: int = 3, save: bool = True) -> Dict:
    """Time ``FakeNewsAgent.run`` over the seed claims (fact-check mode).

    Builds the agent ONCE (offline, deterministic, TF-IDF + lexical stance), runs
    ``warmup`` discarded iterations, then times ``n`` runs. Degrades gracefully —
    any failure is logged and returns a stub dict rather than raising.
    """
    cfg = cfg or AppConfig()
    # Force the offline seed evidence + offline (TF-IDF / lexical) stack for determinism.
    try:
        cfg.data.use_hf = False
    except Exception:
        pass

    device = "cpu"
    try:
        import torch  # lazy
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    try:
        from ..agent.fakenews_agent import FakeNewsAgent  # lazy (pulls classifier + stance)
        agent = FakeNewsAgent(cfg, load_model=False)
    except Exception as exc:
        logger.warning("benchmark: could not build agent (%s)", exc)
        out = {"device": device, "classifier": "?", "stance": "?", "n_claims": 0,
               "n": n, "warmup": warmup, "error": str(exc),
               "latency_ms": _pct([]), "throughput_per_s": 0.0,
               "decision_presence": {}, "statuses": {}}
        if save:
            _save(out)
        return out

    pool = _seed_claims()
    if not pool:
        pool = ["Drinking bleach cures every virus overnight."]
    total = n + warmup
    workload = (pool * (total // len(pool) + 1))[:total]

    def _run(text: str):
        return agent.run(text=text, mode="factcheck", save=False)

    # Warmup (discarded).
    for text in workload[:warmup]:
        try:
            _run(text)
        except Exception as exc:
            logger.info("benchmark warmup failed (%s)", exc)

    lat: List[float] = []
    decision_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    n_ok = 0
    for text in workload[warmup:]:
        t0 = time.perf_counter()
        try:
            job = _run(text)
            wall = (time.perf_counter() - t0) * 1000.0
            ms = _job_latency_ms(job, wall)
            lat.append(ms)
            n_ok += 1
            try:
                status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1
                for d in job.decisions:
                    decision_counts[d.id] = decision_counts.get(d.id, 0) + 1
            except Exception:
                pass
        except Exception as exc:
            logger.info("benchmark iteration failed (%s)", exc)

    mean_ms = float(np.mean(lat)) if lat else 0.0
    decision_presence = {d: round(decision_counts[d] / n_ok, 3) for d in sorted(decision_counts)} if n_ok else {}

    out = {
        "device": device,
        "classifier": getattr(agent.classifier, "name", "?"),
        "classifier_version": getattr(agent.classifier, "version", "?"),
        "stance": getattr(agent.stance, "name", "?"),
        "stance_version": getattr(agent.stance, "version", "?"),
        "n_claims": len(pool),
        "n": n_ok, "warmup": warmup,
        "latency_ms": _pct(lat),
        "throughput_per_s": round(1000.0 / max(1e-6, mean_ms), 2) if lat else 0.0,
        "decision_presence": decision_presence,
        "statuses": status_counts,
    }
    if save:
        _save(out)
    logger.info("benchmark: n=%d p50=%.1fms p95=%.1fms throughput=%.1f/s",
                n_ok, out["latency_ms"]["p50"], out["latency_ms"]["p95"], out["throughput_per_s"])
    return out


def _save(out: Dict) -> None:
    try:
        d = run_dir() / "benchmark"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"benchmark-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("benchmark: could not save (%s)", exc)


__all__ = ["benchmark"]
