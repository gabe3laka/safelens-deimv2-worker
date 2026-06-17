"""
risk/risk_matrix.py -- configurable, versioned severity x likelihood matrix.

Score = severity * likelihood (both 1..5). The score maps to a colour band
(GREEN/YELLOW/ORANGE/RED) defined by a JSON profile. The profile is validated on
load (B2): bands must be monotonic, contiguous, cover the full score range, and
use known colours/levels -- a malformed matrix raises ValueError so the caller
(config validation / readiness) can fail fast instead of silently guessing.

Profile resolution:
  1. RISK_MATRIX_PROFILE env var (path to a JSON profile), else
  2. the bundled risk/risk_matrix_profile.json next to this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

_VALID_LEVELS = {"GREEN", "YELLOW", "ORANGE", "RED"}
_BUNDLED = Path(__file__).with_name("risk_matrix_profile.json")


class RiskMatrix:
    """Loaded, validated risk matrix with score() and band() lookups."""

    def __init__(self, profile: Dict[str, Any]):
        validate_profile(profile)
        self.profile_name: str = profile.get("profile", "unnamed")
        self.version: str = profile.get("version", "0")
        scale = profile.get("scale", {})
        self.severity_max: int = int(scale.get("severity_max", 5))
        self.likelihood_max: int = int(scale.get("likelihood_max", 5))
        self.bands: List[Dict[str, Any]] = list(profile["bands"])
        self._raw = profile

    def clamp(self, severity: int, likelihood: int) -> "tuple[int, int]":
        s = max(1, min(self.severity_max, int(severity)))
        likely = max(1, min(self.likelihood_max, int(likelihood)))
        return s, likely

    def score(self, severity: int, likelihood: int) -> int:
        s, likely = self.clamp(severity, likelihood)
        return s * likely

    def band(self, score: int) -> Dict[str, Any]:
        for b in self.bands:
            if b["min"] <= score <= b["max"]:
                return b
        # Defensive: clamp to the nearest band (validation guarantees coverage
        # of 1..severity_max*likelihood_max, so this only triggers for 0).
        return self.bands[0] if score < self.bands[0]["min"] else self.bands[-1]

    def level(self, severity: int, likelihood: int) -> str:
        return self.band(self.score(severity, likelihood))["level"]

    def evaluate(self, severity: int, likelihood: int) -> Dict[str, Any]:
        s, likely = self.clamp(severity, likelihood)
        score = s * likely
        b = self.band(score)
        return {
            "severity": s,
            "likelihood": likely,
            "risk_score": score,
            "risk_level": b["level"],
            "should_alert": bool(b.get("alert", False)),
            "color": b.get("color"),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "profile": self.profile_name,
            "version": self.version,
            "severity_max": self.severity_max,
            "likelihood_max": self.likelihood_max,
            "bands": [{"level": b["level"], "min": b["min"], "max": b["max"],
                       "alert": bool(b.get("alert", False))} for b in self.bands],
        }


def validate_profile(profile: Dict[str, Any]) -> None:
    """Raise ValueError if the matrix profile is malformed (B2 fail-fast)."""
    if not isinstance(profile, dict):
        raise ValueError("risk matrix profile must be a JSON object")
    bands = profile.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("risk matrix profile must define a non-empty 'bands' list")
    scale = profile.get("scale", {})
    smax = int(scale.get("severity_max", 5))
    lmax = int(scale.get("likelihood_max", 5))
    if smax < 1 or lmax < 1:
        raise ValueError("risk matrix scale maxima must be >= 1")
    full_max = smax * lmax

    prev_max = 0
    for i, b in enumerate(bands):
        if not isinstance(b, dict):
            raise ValueError(f"band {i} must be an object")
        if b.get("level") not in _VALID_LEVELS:
            raise ValueError(f"band {i} has invalid level {b.get('level')!r}; "
                             f"expected one of {sorted(_VALID_LEVELS)}")
        lo, hi = b.get("min"), b.get("max")
        if not isinstance(lo, int) or not isinstance(hi, int):
            raise ValueError(f"band {i} min/max must be integers")
        if lo > hi:
            raise ValueError(f"band {i} min ({lo}) > max ({hi})")
        if i == 0 and lo != 1:
            raise ValueError("first band must start at 1")
        if i > 0 and lo != prev_max + 1:
            raise ValueError(f"band {i} not contiguous: starts {lo}, expected {prev_max + 1}")
        prev_max = hi
    if prev_max != full_max:
        raise ValueError(f"bands must cover the full score range 1..{full_max}; "
                         f"last band ends at {prev_max}")


def load_profile(path: str = "") -> Dict[str, Any]:
    """Load a matrix profile JSON from `path` / RISK_MATRIX_PROFILE / bundled."""
    p = path or os.getenv("RISK_MATRIX_PROFILE", "")
    target = Path(p) if p else _BUNDLED
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_CACHE: "RiskMatrix | None" = None
_CACHE_KEY: str = ""


def get_matrix() -> RiskMatrix:
    """Return the active (cached) RiskMatrix, reloading if the env path changes."""
    global _CACHE, _CACHE_KEY
    key = os.getenv("RISK_MATRIX_PROFILE", "")
    if _CACHE is None or key != _CACHE_KEY:
        _CACHE = RiskMatrix(load_profile(key))
        _CACHE_KEY = key
    return _CACHE


def reset_cache() -> None:
    """Drop the cached matrix (tests / hot config reload)."""
    global _CACHE, _CACHE_KEY
    _CACHE, _CACHE_KEY = None, ""
