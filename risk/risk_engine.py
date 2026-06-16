"""
risk/risk_engine.py -- deterministic, rule-based risk-aware perception.

The deterministic engine is the SAFETY SIGNAL. It turns detector entities (+ a
per-session tracker + a deterministic scene graph) into scored, controlled,
provenance-stamped risk items. It is fully additive and gated by
RISK_ENGINE_ENABLED (default off): when disabled, responses are byte-for-byte
the legacy shape. It never raises into the live path -- on any internal failure
it degrades to the plain detection result plus a `warning` (matching the
existing backend_fallback behaviour), never a 500.

15 deterministic rules (R01..R15) span PPE, pedestrian/vehicle separation, fire/
smoke, escape routes, edge/fall, height, slips, electrical, and overhead loads.
Each rule yields severity+likelihood, which the configurable 5x5 risk_matrix
turns into a score + colour band; controls.py adds hierarchy-ordered controls;
provenance.py stamps producer/rule/version/timestamp.

The VLM reasoner / open-vocab scanner are deliberately NOT here -- they are a
separate event-driven path that only enriches `scene_risks` as AI drafts.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from . import controls, provenance, risk_matrix, scene_graph, tracking
from .risk_schema import (
    Control,
    RiskEngineMeta,
    RiskItem,
    RiskResult,
    SCHEMA_VERSION,
    Track,
)

# -- label vocabulary (normalized, lowercase) ---------------------------------

PERSON = {"person", "worker", "people", "pedestrian"}
NO_HARDHAT = {"no_hardhat", "no-hardhat", "nohardhat", "head"}
NO_VEST = {"no_safety_vest", "no-safety_vest", "no-safety vest", "no_vest", "no-vest"}
NO_GLOVES = {"no_gloves", "no-gloves"}
VEHICLE = {"car", "truck", "bus", "vehicle", "van", "excavator", "machinery"}
FORKLIFT = {"forklift", "mhe_vehicle"}
FIRE = {"fire", "flame"}
SMOKE = {"smoke"}
BLOCKED_EXIT = {"blocked_exit", "blocked-exit"}
LADDER = {"ladder"}
SCAFFOLD = {"scaffold", "scaffolding"}
SPILL = {"spill", "liquid", "puddle", "water_on_floor"}
EXPOSED_ELEC = {"exposed_wire", "cable", "wire", "open_panel", "electrical_panel"}
SUSPENDED = {"suspended_load", "load", "hook"}
OPEN_HOLE = {"open_hole", "manhole", "floor_opening", "hole", "trench"}
# Structural/static classes we do NOT treat as "near edge -> could fall".
_STRUCTURAL = PERSON | LADDER | SCAFFOLD | OPEN_HOLE | BLOCKED_EXIT | {"floor", "wall", "table", "bench"}

STANDARD_REF = {
    "ppe_missing_hardhat": "OSHA 29 CFR 1926.100 (Head Protection)",
    "ppe_missing_vest": "OSHA 29 CFR 1926.201 / hi-vis policy",
    "person_forklift_proximity": "OSHA 1926.602 / pedestrian-vehicle segregation",
    "person_vehicle_proximity": "OSHA 1926.601 / traffic management",
    "fire": "OSHA 1926.150 (Fire Protection)",
    "smoke": "OSHA 1926.150 (Fire Protection)",
    "blocked_exit": "OSHA 1910.36 (Means of Egress)",
    "working_at_height_ladder": "UK HSE INDG401 / Work at Height Regs 2005",
    "scaffold_proximity": "OSHA 1926 Subpart L (Scaffolds)",
    "spill_slip": "UK HSE slips & trips / housekeeping",
    "electrical_exposed": "OSHA 1926 Subpart K (Electrical) / LOTO",
    "suspended_load_overhead": "OSHA 1926.251 / lifting operations",
    "open_hole_fall": "OSHA 1926.501 (Fall Protection)",
    "object_near_edge": "Housekeeping / dropped-object prevention",
    "ppe_missing_gloves": "Task PPE assessment",
}

# Per-rule severity/likelihood. risk_matrix turns these into score + band.
# (severity, likelihood) on a 1..5 scale.
_SL = {
    "ppe_missing_hardhat": (3, 4),
    "ppe_missing_vest": (2, 3),
    "ppe_missing_gloves": (2, 2),
    "person_forklift_proximity_near": (4, 3),
    "person_forklift_proximity_overlap": (4, 4),
    "person_vehicle_proximity_near": (4, 2),
    "person_vehicle_proximity_overlap": (4, 3),
    "fire": (5, 4),
    "smoke": (3, 3),
    "blocked_exit": (4, 3),
    "object_near_edge": (2, 2),
    "working_at_height_ladder": (4, 3),
    "scaffold_proximity": (3, 3),
    "spill_slip": (2, 3),
    "spill_slip_person": (3, 4),
    "electrical_exposed": (4, 2),
    "electrical_exposed_person": (4, 3),
    "suspended_load_overhead": (5, 3),
    "open_hole_fall": (4, 2),
    "open_hole_fall_person": (4, 4),
}


def _norm(label: Any) -> str:
    return str(label or "").strip().lower()


def _indices(entities: List[Dict[str, Any]], labels: set) -> List[int]:
    return [i for i, e in enumerate(entities) if _norm(e.get("label")) in labels]


def _proximity(entities, scene, set_a: set, set_b: set):
    """Yield (i_a, j_b, relation, distance) for near/overlap pairs across sets."""
    out = []
    for r in scene.get("relations", []):
        if r["relation"] not in ("near", "overlaps"):
            continue
        i, j = r["subject"], r["object"]
        li, lj = _norm(entities[i].get("label")), _norm(entities[j].get("label"))
        if li in set_a and lj in set_b:
            out.append((i, j, r["relation"], r.get("distance")))
        elif lj in set_a and li in set_b:
            out.append((j, i, r["relation"], r.get("distance")))
    return out


# -- the 15 deterministic rules -----------------------------------------------
# Each returns a list of partial-risk dicts:
#   {rule_id, hazard_type, risk_state, sl_key, involved_entities, reason, bbox}

def _ppe_rules(entities, scene):
    out = []
    for lbl_set, rule_id, hz in (
        (NO_HARDHAT, "R01_ppe_hardhat", "ppe_missing_hardhat"),
        (NO_VEST, "R02_ppe_vest", "ppe_missing_vest"),
        (NO_GLOVES, "R03_ppe_gloves", "ppe_missing_gloves"),
    ):
        for i in _indices(entities, lbl_set):
            out.append({"rule_id": rule_id, "hazard_type": hz, "risk_state": "active",
                        "sl_key": hz, "involved_entities": [i],
                        "reason": f"Detected '{entities[i].get('label')}' indicating missing PPE.",
                        "bbox": entities[i].get("bbox")})
    return out


def _proximity_rule(entities, scene, set_a, set_b, rule_id, hz, sl_near, sl_overlap):
    out = []
    for i, j, rel, dist in _proximity(entities, scene, set_a, set_b):
        overlap = rel == "overlaps"
        out.append({
            "rule_id": rule_id, "hazard_type": hz,
            "risk_state": "active" if overlap else "latent",
            "sl_key": sl_overlap if overlap else sl_near,
            "involved_entities": sorted([i, j]),
            "reason": (f"'{entities[i].get('label')}' {rel} '{entities[j].get('label')}'"
                       + (f" (dist {dist})." if dist is not None else ".")),
            "bbox": entities[i].get("bbox"),
        })
    return out


def _presence_rule(entities, indices, rule_id, hz, risk_state="active"):
    out = []
    for i in indices:
        out.append({"rule_id": rule_id, "hazard_type": hz, "risk_state": risk_state,
                    "sl_key": hz, "involved_entities": [i],
                    "reason": f"Detected '{entities[i].get('label')}'.",
                    "bbox": entities[i].get("bbox")})
    return out


def _object_near_edge(entities, scene):
    out = []
    for node in scene.get("nodes", []):
        i = node["index"]
        lbl = _norm(entities[i].get("label"))
        if lbl in _STRUCTURAL:
            continue
        edges = node.get("edges", {})
        if edges.get("bottom") or edges.get("left") or edges.get("right"):
            out.append({"rule_id": "R09_object_edge", "hazard_type": "object_near_edge",
                        "risk_state": "latent", "sl_key": "object_near_edge",
                        "involved_entities": [i],
                        "reason": (f"'{entities[i].get('label')}' is near a surface/frame edge "
                                   "and may fall if displaced."),
                        "bbox": entities[i].get("bbox")})
    return out


def _overhead_rule(entities, scene, set_over, set_under, rule_id, hz, sl_key):
    """Fire only when an 'over' object is ABOVE an 'under' object (overhead)."""
    out = []
    for r in scene.get("relations", []):
        if r.get("relation") != "above":
            continue
        i, j = r["subject"], r["object"]  # i is above j
        if _norm(entities[i].get("label")) in set_over and _norm(entities[j].get("label")) in set_under:
            out.append({"rule_id": rule_id, "hazard_type": hz, "risk_state": "active",
                        "sl_key": sl_key, "involved_entities": sorted([i, j]),
                        "reason": (f"'{entities[i].get('label')}' is overhead of "
                                   f"'{entities[j].get('label')}'."),
                        "bbox": entities[i].get("bbox")})
    return out


def _run_rules(entities: List[Dict[str, Any]], scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    partials: List[Dict[str, Any]] = []
    # R01-R03 PPE
    partials += _ppe_rules(entities, scene)
    # R04 person <-> forklift
    partials += _proximity_rule(entities, scene, PERSON, FORKLIFT,
                                "R04_person_forklift", "person_forklift_proximity",
                                "person_forklift_proximity_near", "person_forklift_proximity_overlap")
    # R05 person <-> vehicle
    partials += _proximity_rule(entities, scene, PERSON, VEHICLE,
                                "R05_person_vehicle", "person_vehicle_proximity",
                                "person_vehicle_proximity_near", "person_vehicle_proximity_overlap")
    # R06 fire / R07 smoke (fire dominates: skip smoke when fire present)
    fire_idx = _indices(entities, FIRE)
    partials += _presence_rule(entities, fire_idx, "R06_fire", "fire")
    if not fire_idx:
        partials += _presence_rule(entities, _indices(entities, SMOKE), "R07_smoke", "smoke")
    # R08 blocked exit
    partials += _presence_rule(entities, _indices(entities, BLOCKED_EXIT),
                               "R08_blocked_exit", "blocked_exit")
    # R09 object near edge
    partials += _object_near_edge(entities, scene)
    # R10 working at height (person near ladder)
    partials += _proximity_rule(entities, scene, PERSON, LADDER,
                                "R10_height_ladder", "working_at_height_ladder",
                                "working_at_height_ladder", "working_at_height_ladder")
    # R11 scaffold proximity
    partials += _proximity_rule(entities, scene, PERSON, SCAFFOLD,
                                "R11_scaffold", "scaffold_proximity",
                                "scaffold_proximity", "scaffold_proximity")
    # R12 spill (latent) + person near spill (active/escalated)
    spill_idx = _indices(entities, SPILL)
    spill_person = _proximity(entities, scene, PERSON, SPILL)
    if spill_person:
        for i, j, rel, dist in spill_person:
            partials.append({"rule_id": "R12_spill", "hazard_type": "spill_slip",
                             "risk_state": "active", "sl_key": "spill_slip_person",
                             "involved_entities": sorted([i, j]),
                             "reason": "A person is near a spill/slip hazard.",
                             "bbox": entities[j].get("bbox")})
    else:
        partials += _presence_rule(entities, spill_idx, "R12_spill", "spill_slip",
                                   risk_state="latent")
    # R13 electrical exposed (+ person escalation)
    elec_idx = _indices(entities, EXPOSED_ELEC)
    elec_person = _proximity(entities, scene, PERSON, EXPOSED_ELEC)
    if elec_person:
        for i, j, rel, dist in elec_person:
            partials.append({"rule_id": "R13_electrical", "hazard_type": "electrical_exposed",
                             "risk_state": "active", "sl_key": "electrical_exposed_person",
                             "involved_entities": sorted([i, j]),
                             "reason": "A person is near an exposed electrical hazard.",
                             "bbox": entities[j].get("bbox")})
    else:
        for i in elec_idx:
            partials.append({"rule_id": "R13_electrical", "hazard_type": "electrical_exposed",
                             "risk_state": "latent", "sl_key": "electrical_exposed",
                             "involved_entities": [i],
                             "reason": f"Detected '{entities[i].get('label')}' (electrical).",
                             "bbox": entities[i].get("bbox")})
    # R14 suspended load overhead of a person
    partials += _overhead_rule(entities, scene, SUSPENDED, PERSON,
                               "R14_suspended_load", "suspended_load_overhead",
                               "suspended_load_overhead")
    # R15 open hole / fall opening (+ person escalation)
    hole_idx = _indices(entities, OPEN_HOLE)
    hole_person = _proximity(entities, scene, PERSON, OPEN_HOLE)
    if hole_person:
        for i, j, rel, dist in hole_person:
            partials.append({"rule_id": "R15_open_hole", "hazard_type": "open_hole_fall",
                             "risk_state": "active", "sl_key": "open_hole_fall_person",
                             "involved_entities": sorted([i, j]),
                             "reason": "A person is near an unprotected opening.",
                             "bbox": entities[j].get("bbox")})
    else:
        for i in hole_idx:
            partials.append({"rule_id": "R15_open_hole", "hazard_type": "open_hole_fall",
                             "risk_state": "latent", "sl_key": "open_hole_fall",
                             "involved_entities": [i],
                             "reason": f"Detected unprotected opening '{entities[i].get('label')}'.",
                             "bbox": entities[i].get("bbox")})
    return partials


# -- flags ---------------------------------------------------------------------

def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _flag("RISK_ENGINE_ENABLED", "false")


def tracking_enabled() -> bool:
    return _flag("RISK_TRACKING_ENABLED", "true")


def scene_graph_enabled() -> bool:
    return _flag("RISK_SCENE_GRAPH_ENABLED", "true")


_LEVEL_ORDER = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
# Last-evaluation snapshot for /debug/state.
_LAST: Dict[str, Any] = {"risk_count": 0, "highest_level": "GREEN",
                         "alerting_count": 0, "active_tracks": 0}


def _entity_track_ids(entities, tracks) -> Dict[int, str]:
    """Map entity index -> track_id by best IoU (tracks were built this frame)."""
    out: Dict[int, str] = {}
    for ei, e in enumerate(entities):
        best, best_iou = None, 0.0
        for t in tracks:
            ov = scene_graph.iou(e.get("bbox") or {}, t.get("bbox") or {})
            if ov > best_iou:
                best, best_iou = t.get("track_id"), ov
        if best is not None and best_iou >= 0.5:
            out[ei] = best
    return out


def evaluate(*, entities: List[Dict[str, Any]], img_w: int, img_h: int,
             session_id: Optional[str], frame_id: Optional[str] = None,
             ts_ms: Optional[int] = None) -> Dict[str, Any]:
    """Run tracking + scene graph + rules; return a RiskResult dict (JSON-safe).

    Never raises for normal inputs; callers still guard for the degradation
    ladder. Deterministic: identical entities -> identical risks (ids included).
    """
    t0 = time.perf_counter()
    entities = entities or []

    # 1) tracking (per-session)
    tracks: List[Dict[str, Any]] = []
    if tracking_enabled():
        tracks = tracking.update(session_id, entities, ts_ms)
    t_track = time.perf_counter()

    # 2) scene graph
    scene = scene_graph.build(entities, img_w, img_h) if scene_graph_enabled() else {
        "nodes": [], "relations": [], "object_count": len(entities)}
    t_scene = time.perf_counter()

    # 3) rules -> scored, controlled, stamped risk items
    ent_track = _entity_track_ids(entities, tracks)
    matrix = risk_matrix.get_matrix()
    partials = _run_rules(entities, scene)

    risks: List[RiskItem] = []
    highest = "GREEN"
    alerting = 0
    for p in partials:
        sev, likely = _SL.get(p["sl_key"], (1, 1))
        ev = matrix.evaluate(sev, likely)
        hz = p["hazard_type"]
        involved_e = p.get("involved_entities", [])
        tids = sorted({ent_track[i] for i in involved_e if i in ent_track})
        key = "-".join(tids) if tids else "-".join(str(i) for i in involved_e)
        ctrls = [Control(**c) for c in controls.controls_for(hz)]
        item = {
            "risk_id": f"rsk_{p['rule_id']}_{key}",
            "hazard_type": hz,
            "risk_state": p.get("risk_state", "active"),
            "involved_track_ids": tids,
            "involved_entities": involved_e,
            "severity": ev["severity"],
            "likelihood": ev["likelihood"],
            "risk_score": ev["risk_score"],
            "risk_level": ev["risk_level"],
            "reason": p.get("reason", ""),
            "bbox": p.get("bbox"),
            "recommended_controls": ctrls,
            "recommended_action": controls.primary_action(hz),
            "standard_reference": STANDARD_REF.get(hz),
            "confidence": 1.0,
            "should_alert": ev["should_alert"],
        }
        provenance.stamp(item, rule_id=p["rule_id"], ts_ms=ts_ms,
                         produced_by="risk_engine", requires_human_review=False)
        risks.append(RiskItem(**item))
        if _LEVEL_ORDER[ev["risk_level"]] > _LEVEL_ORDER[highest]:
            highest = ev["risk_level"]
        if ev["should_alert"]:
            alerting += 1

    # Stable ordering: highest risk first, then rule_id, then id.
    risks.sort(key=lambda r: (-_LEVEL_ORDER[r.risk_level], r.rule_id, r.risk_id))
    t_risk = time.perf_counter()

    _LAST.update(risk_count=len(risks), highest_level=highest,
                 alerting_count=alerting, active_tracks=len(tracks))

    meta = RiskEngineMeta(
        enabled=True, degraded=False,
        matrix_profile=matrix.profile_name, matrix_version=matrix.version,
        tracking_enabled=tracking_enabled(), scene_graph_enabled=scene_graph_enabled(),
        provenance_enabled=provenance.enabled(),
        privacy_blur_enabled=_flag("PRIVACY_BLUR_ENABLED", "false"),
        model_version=provenance.model_version(), session_id=session_id,
        active_tracks=len(tracks), risk_count=len(risks),
        highest_level=highest, alerting_count=alerting,
        stage_timings_ms={
            "track": round((t_track - t0) * 1000, 3),
            "scene": round((t_scene - t_track) * 1000, 3),
            "risk": round((t_risk - t_scene) * 1000, 3),
        },
    )
    result = RiskResult(
        schema_version=SCHEMA_VERSION, risk_engine=meta,
        tracks=[Track(**t) for t in tracks], scene_graph=scene,
        risks=risks, scene_risks=[], highest_risk_level=highest,
    )
    return result.model_dump()


def attach_risk(resp_dict: Dict[str, Any], *, session_id: Optional[str],
                frame_id: Optional[str] = None, ts_ms: Optional[int] = None) -> Dict[str, Any]:
    """Merge the additive risk block into a detection response dict.

    No-op when RISK_ENGINE_ENABLED is false (legacy shape preserved). On any
    failure, degrades: keeps the detection result and attaches a `warning`
    plus a degraded risk_engine meta -- never raises, never a 500.
    """
    if not enabled():
        return resp_dict
    try:
        block = evaluate(
            entities=resp_dict.get("entities", []) or [],
            img_w=int(resp_dict.get("img_w", 0) or 0),
            img_h=int(resp_dict.get("img_h", 0) or 0),
            session_id=session_id, frame_id=frame_id, ts_ms=ts_ms,
        )
        resp_dict.update(block)
    except Exception as exc:  # noqa: BLE001 -- degradation ladder, never break detection
        msg = "risk_engine_error: " + type(exc).__name__ + ": " + str(exc)
        if not resp_dict.get("warning"):
            resp_dict["warning"] = msg
        resp_dict["schema_version"] = SCHEMA_VERSION
        resp_dict["risk_engine"] = RiskEngineMeta(
            enabled=True, degraded=True, error=msg,
            session_id=session_id).model_dump()
    return resp_dict


def config() -> Dict[str, Any]:
    """Risk-engine config + last-eval snapshot for GET /debug/state (no secrets)."""
    out: Dict[str, Any] = {
        "enabled": enabled(),
        "tracking_enabled": tracking_enabled(),
        "scene_graph_enabled": scene_graph_enabled(),
        "provenance_enabled": provenance.enabled(),
        "privacy_blur_enabled": _flag("PRIVACY_BLUR_ENABLED", "false"),
        "model_version": provenance.model_version(),
        "session_ttl_ms": tracking.session_ttl_ms(),
        "session_max_active": tracking.session_max_active(),
        "active_sessions": tracking.active_session_count(),
        "near_threshold": scene_graph._near_threshold(),
        "last_eval": dict(_LAST),
    }
    try:
        out["matrix"] = risk_matrix.get_matrix().summary()
        out["matrix_valid"] = True
    except Exception as exc:  # noqa: BLE001
        out["matrix_valid"] = False
        out["matrix_error"] = type(exc).__name__ + ": " + str(exc)
    return out
