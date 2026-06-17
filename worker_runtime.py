"""
worker_runtime.py -- ops/runtime hardening helpers (no heavy deps).

Bundles the cross-cutting production concerns:
  * build_sha + uptime
  * degradation ladder (full -> no_risk -> detect_only -> down)
  * graceful-shutdown flag (SIGTERM: stop accepting frames, drain, exit)
  * structured JSON logging with redaction (never log imagery/secrets/tokens)
  * a tiny, dependency-free metrics registry rendered as Prometheus text

Pure stdlib so importing it can never break server boot.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

log = logging.getLogger("safelens-vision-worker.runtime")

_BOOT = time.time()

# Degradation ladder, most-healthy first.
LADDER = ("full", "no_risk", "detect_only", "down")
_LADDER_RANK = {m: i for i, m in enumerate(LADDER)}


def build_sha() -> str:
    return os.getenv("BUILD_SHA", "unknown")


def uptime_s() -> float:
    return round(time.time() - _BOOT, 2)


# -- graceful shutdown --------------------------------------------------------

_SHUTTING_DOWN = threading.Event()


def begin_shutdown() -> None:
    _SHUTTING_DOWN.set()


def accepting() -> bool:
    """False once shutdown has begun -> stop accepting new frames."""
    return not _SHUTTING_DOWN.is_set()


def is_shutting_down() -> bool:
    return _SHUTTING_DOWN.is_set()


# -- degradation state --------------------------------------------------------

_DEGRADE: Dict[str, str] = {"mode": "full"}
_LOCK = threading.RLock()


def set_degradation(mode: str) -> None:
    if mode not in _LADDER_RANK:
        mode = "full"
    with _LOCK:
        _DEGRADE["mode"] = mode
    set_gauge("degraded", float(_LADDER_RANK[mode] > 0))
    set_gauge("degradation_rank", float(_LADDER_RANK[mode]))


def degradation_mode() -> str:
    with _LOCK:
        return _DEGRADE["mode"]


# -- structured logging (redacted) --------------------------------------------

# Keys/▸substrings whose values must never be logged.
_SENSITIVE = ("image_b64", "frame_b64", "authorization", "x-worker-secret",
              "worker_shared_secret", "hf_token", "token", "secret", "password",
              "api_key", "apikey")


def redact(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Drop/redact sensitive values; truncate anything huge (e.g. stray b64)."""
    out: Dict[str, Any] = {}
    for k, v in (fields or {}).items():
        lk = str(k).lower()
        if any(s in lk for s in _SENSITIVE):
            out[k] = "[redacted]"
            continue
        if isinstance(v, str) and len(v) > 512:
            out[k] = v[:64] + "...[truncated]"
        else:
            out[k] = v
    return out


def log_event(event: str, level: int = logging.INFO, *, logger: Optional[logging.Logger] = None,
              **fields: Any) -> None:
    """Emit one structured JSON log line with build_sha + redacted fields."""
    rec = {"event": event, "build_sha": build_sha(), "ts": int(time.time() * 1000)}
    rec.update(redact(fields))
    try:
        (logger or log).log(level, json.dumps(rec, default=str))
    except Exception:  # noqa: BLE001 -- logging must never raise
        (logger or log).log(level, "event=%s (log serialize failed)", event)


# -- metrics registry (Prometheus text; no client dep) ------------------------

_COUNTERS: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_GAUGES: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_LAT: Dict[str, Deque[float]] = {}
_MLOCK = threading.RLock()
_LAT_MAX = 512


def _key(name: str, labels: Optional[Dict[str, str]]):
    lab = tuple(sorted((labels or {}).items()))
    return (name, lab)


def inc(name: str, labels: Optional[Dict[str, str]] = None, n: float = 1.0) -> None:
    with _MLOCK:
        k = _key(name, labels)
        _COUNTERS[k] = _COUNTERS.get(k, 0.0) + n


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    with _MLOCK:
        _GAUGES[_key(name, labels)] = float(value)


def observe_latency(name: str, ms: float) -> None:
    with _MLOCK:
        dq = _LAT.get(name)
        if dq is None:
            dq = deque(maxlen=_LAT_MAX)
            _LAT[name] = dq
        dq.append(float(ms))


def _quantile(values, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[idx], 3)


def _fmt_labels(lab: Tuple[Tuple[str, str], ...]) -> str:
    if not lab:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in lab)
    return "{" + inner + "}"


def render_prometheus(extra_gauges: Optional[Dict[str, float]] = None) -> str:
    """Render all metrics as Prometheus text. extra_gauges are scrape-time
    live values (e.g. active_sessions) injected by the /metrics route."""
    lines = []
    with _MLOCK:
        if extra_gauges:
            for name, val in extra_gauges.items():
                set_gauge(name, val)
        seen = set()
        for (name, lab), val in sorted(_GAUGES.items()):
            metric = f"safelens_{name}"
            if metric not in seen:
                lines.append(f"# TYPE {metric} gauge")
                seen.add(metric)
            lines.append(f"{metric}{_fmt_labels(lab)} {val}")
        for (name, lab), val in sorted(_COUNTERS.items()):
            metric = f"safelens_{name}"
            if metric not in seen:
                lines.append(f"# TYPE {metric} counter")
                seen.add(metric)
            lines.append(f"{metric}{_fmt_labels(lab)} {val}")
        for name, dq in sorted(_LAT.items()):
            vals = list(dq)
            metric = f"safelens_{name}"
            lines.append(f"# TYPE {metric} summary")
            for q in (0.5, 0.95, 0.99):
                lines.append(f'{metric}{{quantile="{q}"}} {_quantile(vals, q)}')
            lines.append(f"{metric}_count {len(vals)}")
    return "\n".join(lines) + "\n"


def reset_metrics() -> None:
    with _MLOCK:
        _COUNTERS.clear()
        _GAUGES.clear()
        _LAT.clear()


def reset_shutdown() -> None:
    """Test helper: clear the shutdown flag."""
    _SHUTTING_DOWN.clear()
