# Temporal reasoning integration (event-triggered VLM perception)

`temporal_reasoning/` adds short per-session memory, deterministic
object-near-edge risk, and an **event-triggered, non-blocking** VLM perception
pass to `/detect`. It is additive and gated by `TEMPORAL_REASONING_ENABLED`.

## Where it runs

`server.py /detect` calls, after detection + the deterministic risk engine:

```python
import temporal_reasoning
if temporal_reasoning.enabled():
    resp_dict = temporal_reasoning.attach_temporal(
        resp_dict, session_id=..., frame_id=..., frame_b64=image_b64, payload=payload)
```

`attach_temporal` never raises and never blocks (it submits async work and reads
the latest cached result). When disabled, `/detect` is byte-for-byte the legacy
shape.

## Detector every frame, VLM only on a trigger

The detector/tracking/risk run every frame (cheap, deterministic). The VLM runs
**only** when `triggers.evaluate` returns a reason:

- `low_conf_stable` — a track is stable but persistently low-confidence
- `label_instability` — a track's label flipped within the flip window
- `scene_mismatch` — detector labels conflict with `scene_hint`/`site_context`
  (e.g. detector says "bus" but the scene is a cafe)
- `object_near_edge` — an edge risk is present this frame
- `person_in_danger_zone` — a person is in an active ORANGE+ deterministic risk
- `risk_escalation` — highest risk level is ORANGE or above
- `user_request` — `reasoning_preferences.force_reason`
- `periodic_refresh` — scene context is stale past `SCENE_CONTEXT_REFRESH_MS`

Triggers are rate-limited (`max(TEMPORAL_REASONING_TRIGGER_MIN_INTERVAL_MS,
REASONER_MIN_INTERVAL_MS)`), one in-flight job per session, global cap
`TEMPORAL_REASONING_MAX_ASYNC_JOBS`, and each job needs a free GPU slot
(`gpu_vision.gpu_reasoner_slot`) or it is dropped. With
`REASONER_LATEST_WINS=true`, only the newest pending frame is kept per session
while a job runs (old pending frames are replaced/dropped). **No unbounded queue.**

## Non-blocking contract

`/detect` never waits for the VLM. It attaches the most recent cached
`scene_context` / `semantic_corrections` and a `reasoner_status` and returns. A
slow or failing reasoner can never break `/detect` (covered by tests).

## Perception correction vs safety draft (authority)

This is the key safety distinction:

- **Perception correction** — fixes what the camera mis-saw (e.g. ceiling-mounted
  "bus" -> "ceiling panel", suppressed from HSE alerts). Marked
  `purpose="perception_correction"`, `authority="advisory_perception"`,
  `requires_human_review=false`. No human approval needed — it is not a safety
  action. Raw detector labels are **always preserved** (`raw_label`).
- **Safety/compliance draft** — if the VLM creates or escalates a real hazard, it
  flows through `scene_risks` as `purpose="safety_draft"`,
  `authority="advisory_safety"`, `requires_human_review=true`.

A scene hint (e.g. `cafe`) suppresses vehicle/construction false positives but
**never** suppresses a real hazard (person, knife, spill, object-near-edge).

## Object-near-edge risk

`edge_risk.py` is deterministic. It uses bbox motion over recent frames plus, if a
support surface (table/desk/bench/...) is detected, the surface's top edge.
**If no surface is detected it uses the frame edge and marks
`edge_reference="frame_fallback"`** — it never invents surface geometry. Persons
and vehicles are never flagged as falling objects.

## Additive `/detect` response fields

```json
{
  "temporal_reasoning": { "enabled": true, "session_id": "cam_123",
    "memory_frames": 32, "active_tracks": 5, "triggered": true,
    "trigger_reasons": ["scene_mismatch", "object_near_edge"] },
  "scene_context": { "scene_type": "cafe", "environment_type": "indoor_public",
    "confidence": 0.82, "source": "vlm_reasoner", "reason": "...", "last_checked_ms": 0 },
  "semantic_corrections": [ { "track_id": "trk_7", "raw_label": "bus",
    "corrected_label": "ceiling panel", "correction_type": "false_positive",
    "action": "suppress_from_hse_alerts", "requires_human_review": false } ],
  "reasoner_status": { "enabled": true, "mode": "gemini", "state": "ready",
    "last_trigger": "scene_mismatch", "result_age_ms": 2000, "stale": false }
}
```

`reasoner_status.state` vocabulary is compatible with existing clients and may be:
`idle`, `queued`, `queued_latest`, `running`, `ready`, `cached`, `throttled`,
`timeout`, `error`, `unavailable`, `disabled`.

`object_near_edge` risks are appended to the deterministic `risks[]` list.

## Session memory (a sub-record, not a competing lifecycle)

`session_memory.py` is keyed by the **same `session_id`** the risk tracker uses,
with the same TTL/eviction pattern (`SESSION_TTL_MS`/`TEMPORAL_MEMORY_TTL_MS`,
`TEMPORAL_MAX_ACTIVE_SESSIONS`). It stores only metadata (bbox/centroid/label/
confidence history + timestamps) and the latest scene context/corrections.
**No raw frames are persisted** (`TEMPORAL_STORE_KEYFRAMES=false`); a frame is
passed transiently to a reasoner job and dropped.

## Privacy

When `PRIVACY_BLUR_ENABLED=true`, frames are blurred (persons) **before** they
reach the VLM. The temporal real path calls `risk.vlm_reasoner.generate_json`,
which routes through `_decode_blurred` — no un-blurred frame reaches the model.

## Modes

The temporal VLM reuses `risk.vlm_reasoner` (`REASONER_MODE`):
`mock` (no weights; deterministic; used by tests + CPU integration),
`gemini` (Google GenAI API). Removed transformer modes return unavailable and do not load weights.
When `VLM_REASONER_ENABLED=false`, the deterministic temporal layer (memory, edge
risk, triggers, blocks) still runs; `reasoner_status.state="disabled"`.

Env reference: `runpod_env_reference.md`.
