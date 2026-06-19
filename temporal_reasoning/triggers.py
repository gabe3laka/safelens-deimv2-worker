"""
temporal_reasoning/triggers.py -- decide WHEN the event-driven VLM should run.

The detector runs every frame; the VLM does NOT. This module returns the set of
trigger reasons for the current frame. Rate-limiting / one-in-flight / global cap
are enforced by async_reasoning, not here.

Trigger reasons:
  low_conf_stable     -- a track is stable but persistently low-confidence
  label_instability   -- a track's label flipped within the flip window
  scene_mismatch      -- detector labels conflict with the scene_hint/site context
  object_near_edge    -- an edge risk is present this frame
  person_in_danger_zone -- a person is involved in an active ORANGE+ risk
  risk_escalation     -- highest risk level is ORANGE or above
  user_request        -- reasoning_preferences.force_reason
  periodic_refresh    -- scene_context is stale past SCENE_CONTEXT_REFRESH_MS
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import session_memory as mem
from .session_memory import _bool_env, _float_env, _int_env

_LEVEL = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}

# Labels that are out of place indoors -> candidates for scene_mismatch.
_OUTDOOR_VEHICLE = {"bus", "truck", "car", "train", "airplane", "boat",
                    "motorcycle", "fire hydrant", "traffic light", "stop sign"}
_INDOOR_HINTS = {"cafe", "office", "indoor", "restaurant", "shop", "store",
                 "home", "house", "classroom", "warehouse", "kitchen", "lab"}


def scene_context_enabled() -> bool:
    return _bool_env("SCENE_CONTEXT_ENABLED", True)


def scene_context_refresh_ms() -> int:
    return _int_env("SCENE_CONTEXT_REFRESH_MS", 2000)


def _low_conf_threshold() -> float:
    return _float_env("SEMANTIC_CORRECTION_LOW_CONF_THRESHOLD", 0.35)


def _scene_hint_env_enabled() -> bool:
    return _bool_env("SCENE_HINT_ENABLED", True)


def _environment_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not _scene_hint_env_enabled():
        return None
    hint = (payload or {}).get("scene_hint")
    site = (payload or {}).get("site_context") or {}
    env = site.get("environment_type") or hint
    if not env:
        return None
    env = str(env).lower()
    for h in _INDOOR_HINTS:
        if h in env:
            return h
    return env


def _scene_mismatch(entities: List[Dict[str, Any]], payload: Dict[str, Any]) -> bool:
    env = _environment_from_payload(payload)
    if not env or env not in _INDOOR_HINTS:
        return False
    for e in entities or []:
        if str(e.get("label", "")).lower() in _OUTDOOR_VEHICLE:
            return True
    return False


def evaluate(session_id: Optional[str], *, entities: List[Dict[str, Any]],
             tracks: List[Dict[str, Any]], highest_level: str,
             deterministic_risks: List[Dict[str, Any]],
             edge_risks: List[Dict[str, Any]], payload: Dict[str, Any]) -> List[str]:
    """Return the list of trigger reasons for this frame. Never raises."""
    reasons: List[str] = []
    try:
        prefs = (payload or {}).get("reasoning_preferences") or {}
        if prefs.get("force_reason"):
            reasons.append("user_request")

        if _LEVEL.get((highest_level or "GREEN").upper(), 0) >= _LEVEL["ORANGE"]:
            reasons.append("risk_escalation")

        if edge_risks:
            reasons.append("object_near_edge")

        if _scene_mismatch(entities, payload):
            reasons.append("scene_mismatch")

        # person involved in an active ORANGE+ deterministic risk
        for r in deterministic_risks or []:
            lvl = _LEVEL.get(str(r.get("risk_level", "GREEN")).upper(), 0)
            if lvl >= _LEVEL["ORANGE"] and str(r.get("risk_state")) == "active":
                ht = str(r.get("hazard_type", "")).lower()
                if "person" in ht or "pedestrian" in ht:
                    reasons.append("person_in_danger_zone")
                    break

        # per-track temporal signals
        thr = _low_conf_threshold()
        flip_win = mem.label_flip_window_frames()
        for tid in mem.all_track_ids(session_id):
            labels = mem.label_history(session_id, tid)
            if len(labels) >= 2 and len(set(labels[-flip_win:])) > 1:
                reasons.append("label_instability")
            confs = mem.confidence_history(session_id, tid)
            if len(confs) >= 3 and max(confs[-5:]) < thr:
                reasons.append("low_conf_stable")
            if "label_instability" in reasons and "low_conf_stable" in reasons:
                break

        # periodic scene-context refresh
        if scene_context_enabled():
            snap = mem.snapshot(session_id)
            ctx = snap.get("latest_scene_context") or {}
            last_checked = int(ctx.get("last_checked_ms", 0) or 0)
            if (int(time.time() * 1000) - last_checked) >= scene_context_refresh_ms():
                reasons.append("periodic_refresh")

        # de-dupe, stable order
        seen = set()
        ordered = []
        for reason in reasons:
            if reason not in seen:
                seen.add(reason)
                ordered.append(reason)
        return ordered
    except Exception:  # noqa: BLE001
        return reasons
