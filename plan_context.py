"""
plan_context.py -- selected-crop Plan context for SafeLens Plan Mode.

Turns the already-computed Build/Plan crop geometry (YOLO26 detections +
mask/contour, or the fallback Canny contour) into structured, RULE-BASED context
the app + the app-side DeepSeek Edge Function can reason over:

    selectedLabel, cropEntities, cropSegments, suggestedGoals,
    virtualBlueprintPoints (2D rules, no depth required), planContext,
    + optional depthPoints / open-vocab / known-part-pose / assembly-state.

Design rules (per the SafeLens worker spec):
  * The worker's job is GEOMETRY + CONTEXT, not final reasoning. DeepSeek (on
    the app side) enriches steps/overlays later; the worker never calls it.
  * Depth Anything / GroundingDINO+SAM2 / FoundationPose / assembly-state are
    OPTIONAL and DISABLED by default -- safe stubs that fail gracefully.
  * Point-E is never used in the live loop.
  * All x/y are normalized to selected-crop coords 0..1 (clamped); z is optional
    pseudo-depth 0..1. virtualBlueprintPoints is capped (default 12).
  * Pure/CPU and fast -- it runs on the event loop from already-computed geom.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("safelens-vision-worker.plan")

MAX_VIRTUAL_POINTS = 12

# Labels that should trigger electronics-aware goals / safety hints.
_ELECTRONICS_WORDS = (
    "pcb", "circuit board", "circuit", "arduino", "raspberry pi", "raspberry",
    "microcontroller", "wire", "wiring", "cable", "connector", "battery",
    "sensor", "module", "solder", "voltage", "volt", "power", "capacitor",
    "resistor", "breadboard", "chip", "ic", "transistor", "diode",
)
# Subset that warrants an explicit electrical safety warning.
_SAFETY_WORDS = (
    "power", "voltage", "volt", "battery", "solder", "live circuit", "mains",
    "electrical", "wire", "wiring", "current", "capacitor",
)


# -- Config -------------------------------------------------------------------

def _flag(name, default):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")

def _int(name, default):
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def config() -> Dict[str, Any]:
    """Non-sensitive plan-context config block for /debug/state."""
    return {
        "enabled": _flag("PLAN_CONTEXT_ENABLED", "true"),
        "depth_enabled": _flag("PLAN_DEPTH_ENABLED", "false"),
        "depth_backend": os.getenv("PLAN_DEPTH_BACKEND", "none"),
        "depth_model_loaded": _depth_model_loaded(),
        "depth_model_id": os.getenv("DEPTH_MODEL_ID", "depth-anything-v2-small"),
        "depth_sample_points": _int("PLAN_DEPTH_SAMPLE_POINTS", 120),
        "depth_max_res": _int("PLAN_DEPTH_MAX_RES", 384),
        "depth_every_n": _int("PLAN_DEPTH_EVERY_N", 3),
        "open_vocab_enabled": _flag("PLAN_OPEN_VOCAB_ENABLED", "false"),
        "open_vocab_backend": os.getenv("PLAN_OPEN_VOCAB_BACKEND", "none"),
        "known_part_pose_enabled": _flag("PLAN_KNOWN_PART_POSE_ENABLED", "false"),
        "known_part_pose_backend": os.getenv("PLAN_KNOWN_PART_POSE_BACKEND", "none"),
        "assembly_state_enabled": _flag("PLAN_ASSEMBLY_STATE_ENABLED", "false"),
    }


def is_enabled() -> bool:
    return _flag("PLAN_CONTEXT_ENABLED", "true")


# -- Small helpers -------------------------------------------------------------

def _clamp(v: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(v))))
    except (TypeError, ValueError):
        return 0.5


def _bbox_center(b: Dict[str, float]) -> Tuple[float, float]:
    return _clamp(b["x"] + b["w"] / 2.0), _clamp(b["y"] + b["h"] / 2.0)


def _outline_bbox(outline: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    if not outline:
        return 0.0, 0.0, 1.0, 1.0
    xs = [p["x"] for p in outline]
    ys = [p["y"] for p in outline]
    return min(xs), min(ys), max(xs), max(ys)


def is_electronics(label: Optional[str], crop_entities: List[Dict[str, Any]],
                   intent_text: str = "") -> bool:
    blob = " ".join([
        (label or ""),
        intent_text or "",
        " ".join(str(e.get("label", "")) for e in (crop_entities or [])),
    ]).lower()
    return any(w in blob for w in _ELECTRONICS_WORDS)


def _is_safety(label: Optional[str], crop_entities, intent_text: str) -> bool:
    blob = " ".join([
        (label or ""), intent_text or "",
        " ".join(str(e.get("label", "")) for e in (crop_entities or [])),
    ]).lower()
    return any(w in blob for w in _SAFETY_WORDS)


# -- selectedLabel -------------------------------------------------------------

def selected_label(crop_entities: List[Dict[str, Any]], hint: Optional[str],
                   mask_source: Optional[str]) -> str:
    # 1. highest-confidence crop entity label
    if crop_entities:
        best = max(crop_entities, key=lambda e: e.get("confidence", 0.0))
        lbl = str(best.get("label") or "").strip()
        if lbl:
            return lbl
    # 2. explicit hint from the app (selectedLabel / source label)
    if hint and str(hint).strip():
        return str(hint).strip()
    # 3/4. strong segment present -> generic; else "selected item"
    if mask_source == "yolo26-seg":
        return "selected object"
    return "selected item"


# -- suggestedGoals ------------------------------------------------------------

def suggested_goals(label: Optional[str], crop_entities: List[Dict[str, Any]],
                    electronics: bool) -> List[str]:
    if electronics:
        return [
            "Identify board orientation",
            "Locate connector points",
            "Plan cable connection",
            "Inspect for damage",
            "Check safety before powering",
        ]
    count = len(crop_entities)
    if count > 1:
        return [
            "Identify these parts",
            "Help assemble these pieces",
            "Inspect for damage",
            "Plan the next build step",
        ]
    return [
        "Identify this item",
        "Inspect this item",
        "Troubleshoot this item",
        "Explain what this is",
    ]


# -- virtualBlueprintPoints (2D rules, no depth) -------------------------------

def virtual_blueprint_points(geom: Dict[str, Any], electronics: bool,
                             safety: bool) -> List[Dict[str, Any]]:
    outline = geom.get("outline") or []
    center = geom.get("center") or {"x": 0.5, "y": 0.5}
    parts = geom.get("detected_parts") or []
    minx, miny, maxx, maxy = _outline_bbox(outline)
    cx, cy = _clamp(center["x"]), _clamp(center["y"])
    width, height = maxx - minx, maxy - miny

    pts: List[Dict[str, Any]] = []

    def add(role, x, y, label, point_id, instruction=None):
        if len(pts) >= MAX_VIRTUAL_POINTS:
            return
        pts.append({"id": point_id, "role": role, "x": _clamp(x), "y": _clamp(y),
                    "label": label, "instruction": instruction})

    # 1. anchor at the main part center
    add("anchor", cx, cy, "Main part", "vp-main-center")

    # 2. alignment points along the longest bbox edge
    if width >= height:
        add("alignment-point", (minx + maxx) / 2.0, miny, "Alignment edge", "vp-align-top")
        add("alignment-point", (minx + maxx) / 2.0, maxy, "Alignment edge", "vp-align-bottom")
    else:
        add("alignment-point", minx, (miny + maxy) / 2.0, "Alignment edge", "vp-align-left")
        add("alignment-point", maxx, (miny + maxy) / 2.0, "Alignment edge", "vp-align-right")

    # 3. inspection points on the detected parts (or a couple of contour points)
    if parts:
        for i, p in enumerate(sorted(parts, key=lambda e: -e.get("confidence", 0.0))[:3]):
            px, py = _bbox_center(p.get("bbox", {"x": 0.5, "y": 0.5, "w": 0, "h": 0}))
            add("inspection-point", px, py, str(p.get("label") or "Inspect here"),
                f"vp-inspect-{i}", "Inspect this part")
    elif outline:
        n = len(outline)
        for i in range(min(2, n)):
            p = outline[(i * n) // max(2, n)]
            add("inspection-point", p["x"], p["y"], "Inspect here", f"vp-inspect-{i}")

    # 4. a target-position + connection-point near the object edges
    add("target-position", _clamp(cx + (0.18 if cx < 0.5 else -0.18)), cy,
        "Target area", "vp-target")
    add("connection-point", maxx, cy, "Connection area", "vp-connect")

    # 5. electronics: alignment along the longest edge + a connector inspection
    if electronics:
        add("connection-point", maxx, miny + height * 0.25, "Likely connector side",
            "vp-connector", "Locate connector points here")
        add("inspection-point", minx, miny + height * 0.25, "Inspect board edge",
            "vp-edge-inspect")

    # 6. warning point for power/electrical contexts
    if safety:
        add("warning-point", cx, _clamp(miny + height * 0.12),
            "Verify unpowered", "vp-warning", "Ensure unpowered before handling")

    return pts[:MAX_VIRTUAL_POINTS]


# -- planContext summary -------------------------------------------------------

_TASK_TO_USE = {
    "build": "assemble", "assemble": "assemble", "inspect": "inspect",
    "identify": "identify", "repair": "repair", "troubleshoot": "troubleshoot",
}


def plan_context_summary(label, crop_entities, mask_source, electronics, safety,
                         task_type) -> Dict[str, Any]:
    count = len(crop_entities)
    warnings: List[str] = []
    if safety:
        warnings.append("Possible electrical/powered context -- verify it is safe to handle.")
    source = "yolo26" if (crop_entities or mask_source == "yolo26-seg") else "rules"
    return {
        "selectedLabel": label,
        "objectCount": count,
        "hasMultipleParts": count > 1,
        "likelyUse": _TASK_TO_USE.get((task_type or "").lower(), "unknown"),
        "contextSource": source,
        "warnings": warnings,
    }


# -- Optional depth (disabled stub by default) ---------------------------------

_DEPTH_STATE: Dict[str, Any] = {"attempted": False, "model": None}


def _depth_model_loaded() -> bool:
    return _DEPTH_STATE["model"] is not None


def _load_depth():
    """Lazily try to load a depth backend. Returns None when unavailable.

    Depth Anything is intentionally NOT a hard dependency of the image; until a
    real backend is wired this returns None so the enabled path degrades to a
    clear warning (per spec) instead of fabricating depth.
    """
    if _DEPTH_STATE["attempted"]:
        return _DEPTH_STATE["model"]
    _DEPTH_STATE["attempted"] = True
    backend = os.getenv("PLAN_DEPTH_BACKEND", "none").strip().lower()
    if backend in ("", "none"):
        return None
    try:  # pragma: no cover -- real backend not installed in the default image
        if backend == "depth-anything-v2":
            import transformers  # noqa: F401
            from transformers import pipeline
            _DEPTH_STATE["model"] = pipeline(
                "depth-estimation", model=os.getenv("DEPTH_MODEL_ID", "depth-anything-v2-small"))
        # video-depth-anything / others: future
    except Exception as exc:  # noqa: BLE001
        log.warning("[plan] depth backend %s unavailable: %s", backend, exc)
        _DEPTH_STATE["model"] = None
    return _DEPTH_STATE["model"]


def _geom_sample_points(geom: Dict[str, Any]) -> List[Dict[str, float]]:
    """Candidate 2D points to sample depth at (contour + mask + centroid + corners)."""
    pts: List[Dict[str, float]] = []
    for key in ("maskContour", "outline", "sparsePoints"):
        for p in (geom.get(key) or []):
            pts.append({"x": _clamp(p["x"]), "y": _clamp(p["y"])})
    c = geom.get("center") or {"x": 0.5, "y": 0.5}
    pts.append({"x": _clamp(c["x"]), "y": _clamp(c["y"])})
    minx, miny, maxx, maxy = _outline_bbox(geom.get("outline") or [])
    pts.extend([{"x": minx, "y": miny}, {"x": maxx, "y": miny},
                {"x": maxx, "y": maxy}, {"x": minx, "y": maxy}])
    for p in (geom.get("detected_parts") or []):
        cx, cy = _bbox_center(p.get("bbox", {"x": 0.5, "y": 0.5, "w": 0, "h": 0}))
        pts.append({"x": cx, "y": cy})
    return pts


def _downsample(points: List[Any], n: int) -> List[Any]:
    """Evenly down-sample to at most n points (preserves order)."""
    if n <= 0 or len(points) <= n:
        return list(points)
    stride = len(points) / float(n)
    return [points[int(i * stride)] for i in range(n)]


def maybe_depth(geom: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, Optional[float], Optional[str]]:
    """Return (depthPoints, depthSource, depthConfidence, warning). Never raises.

    Default (PLAN_DEPTH_ENABLED=false): ([], "none", None, None).
    Enabled but model unavailable: ([], backend, None, warning).
    """
    if not _flag("PLAN_DEPTH_ENABLED", "false"):
        return [], "none", None, None
    backend = os.getenv("PLAN_DEPTH_BACKEND", "none").strip().lower()
    model = _load_depth()
    if model is None:
        return [], backend, None, "depth backend configured but model unavailable"
    try:  # pragma: no cover -- real depth backend not installed in the default image
        n = _int("PLAN_DEPTH_SAMPLE_POINTS", 120)
        sample_xy = _downsample(_geom_sample_points(geom), n)
        # A real backend would read z from the depth map at each (x, y); here we
        # only reach this branch when a model is actually wired up.
        z_default = 0.5
        pts = [{"x": p["x"], "y": p["y"], "z": z_default, "confidence": 0.5} for p in sample_xy]
        return pts, os.getenv("DEPTH_MODEL_ID", backend), 0.5, None
    except Exception as exc:  # noqa: BLE001
        log.warning("[plan] depth sampling failed: %s", exc)
        return [], backend, None, "depth sampling failed"


# -- Optional open-vocab / known-part-pose / assembly-state (disabled stubs) ---

def open_vocab_parts(geom, prompts=None) -> List[Dict[str, Any]]:
    """GroundingDINO/Grounded-SAM path -- disabled by default; [] otherwise."""
    if not _flag("PLAN_OPEN_VOCAB_ENABLED", "false"):
        return []
    return []  # backend not bundled -> no-op (fails gracefully)


def known_part_pose(geom, session=None) -> Optional[Dict[str, Any]]:
    """FoundationPose/MegaPose path -- disabled by default; None otherwise."""
    if not _flag("PLAN_KNOWN_PART_POSE_ENABLED", "false"):
        return None
    return None  # requires CAD/reference + backend -> skip


def assembly_state(geom, session=None) -> Optional[Dict[str, Any]]:
    """IndustReal/ASDF-style step-state path -- disabled by default; None."""
    if not _flag("PLAN_ASSEMBLY_STATE_ENABLED", "false"):
        return None
    return None


# -- Main entry: build the Plan-context field block ----------------------------

def build(geom: Dict[str, Any], region: Dict[str, float],
          user_intent: Optional[Dict[str, Any]], payload_label: Optional[str] = None
          ) -> Dict[str, Any]:
    """Build the optional Plan-context fields from already-computed crop geom.

    Returns a dict of BlueprintFrame-compatible optional fields. Never raises --
    on any internal error it returns an empty/minimal block.
    """
    if not is_enabled():
        return {}
    try:
        crop_entities = geom.get("detected_parts") or []
        mask_source = geom.get("maskSource")
        intent = user_intent if isinstance(user_intent, dict) else {}
        intent_text = str(intent.get("text") or "")
        task_type = str(intent.get("taskType") or "")

        label = selected_label(crop_entities, payload_label, mask_source)
        electronics = is_electronics(label, crop_entities, intent_text)
        safety = _is_safety(label, crop_entities, intent_text)

        # cropSegments from the chosen YOLO26 mask (when present).
        crop_segments: List[Dict[str, Any]] = []
        if mask_source == "yolo26-seg" and geom.get("maskContour"):
            crop_segments.append({
                "label": label, "class_id": -1,
                "confidence": float(geom.get("confidence", 0.0)),
                "maskContour": geom["maskContour"], "source": "yolo26-seg",
            })

        depth_points, depth_source, depth_conf, depth_warning = maybe_depth(geom)
        ctx = plan_context_summary(label, crop_entities, mask_source, electronics,
                                    safety, task_type)

        out: Dict[str, Any] = {
            "selectedLabel": label,
            "cropEntities": crop_entities,
            "cropSegments": crop_segments,
            "suggestedGoals": suggested_goals(label, crop_entities, electronics),
            "virtualBlueprintPoints": virtual_blueprint_points(geom, electronics, safety),
            "reasoningSource": "rules",
            "depthPoints": depth_points,
            "depthSource": depth_source,
            "depthConfidence": depth_conf,
            "depthWarning": depth_warning,
            "planContext": ctx,
            "knownPartPose": known_part_pose(geom),
            "assemblyState": assembly_state(geom),
        }
        # electronics safety hint (additive; the caller may already set one).
        if safety:
            out["electronicsSafety"] = (
                "Ensure the board is unpowered and safe to handle before "
                "connecting or soldering.")
        return out
    except Exception as exc:  # noqa: BLE001 -- plan context must never break a frame
        log.warning("[plan] build_plan_context failed: %s", exc)
        return {"planContext": {"contextSource": "rules", "warnings": []}}
