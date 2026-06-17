# App / Cloudflare -> RunPod worker contract

This documents what the app and Cloudflare gateway should send/forward. **No
changes were made to the app or Cloudflare repo in this task** — this file is the
contract to implement there.

All requests flow `App -> Cloudflare Gateway -> RunPod worker :8000`. Auth is the
shared secret (`WORKER_SHARED_SECRET`) on the `x-worker-secret` header for every
route except `/health` and `/ping` (see `worker_security.py`). Cloudflare should
forward that header (and the WebSocket `?token=` for `/ws/*`).

## `POST /detect` — additive request fields to forward

The base contract (`image_b64`, `conf`, `img_size`, `classes`) is unchanged.
**Forward these additive fields when available** so temporal perception can use
context (all optional; the worker is backward-compatible without them):

```json
{
  "image_b64": "<base64 JPEG/PNG>",
  "session_id": "cam_123",
  "frame_id": "frame_456",
  "scene_hint": "cafe_demo",
  "site_context": { "environment_type": "cafe", "mode": "demo",
    "allowed_hazard_focus": ["object_near_edge", "spill", "trip", "falling_object"] },
  "camera_context": { "camera_name": "phone_camera", "location_name": "cafe_table" },
  "reasoning_preferences": { "force_reason": false }
}
```

- `session_id` / `frame_id` — required for temporal memory + provenance linking.
- `scene_hint` / `site_context.environment_type` — drives scene-mismatch detection
  and contextual suppression (a cafe hint suppresses vehicle false positives but
  never real hazards).
- `camera_context` — labels the temporal session (camera name/location).
- `reasoning_preferences.force_reason` — force a VLM pass this frame.

Machine-readable example: `shared/contracts/cloudflare_to_runpod_detect.json`.

## `POST /detect` — additive response fields the app may consume

Base vision fields are unchanged. When enabled, the response also carries:
`schema_version`, `risks`, `scene_risks`, `highest_risk_level` (risk engine);
`temporal_reasoning`, `scene_context`, `semantic_corrections`, `reasoner_status`
(temporal layer). See `shared/contracts/runpod_detect_response.json` and
`temporal_reasoning_integration.md`.

The app should treat `semantic_corrections` (perception, `requires_human_review:
false`) differently from `scene_risks` (safety drafts, `requires_human_review:
true`): the former may auto-suppress an HSE alert; the latter must go to a human.

## `/agent/*` — agentic routes

The app can call the CPU agent routes (same worker, same auth) to draft QHSE
artefacts from a detection result. The app should:

1. Send the detection JSON it received from `/detect` as `detection_context`.
2. Show the returned draft (`status: pending_approval`) to a human.
3. Call `POST /agent/approvals/execute` with `approved: true` + `approved_by` only
   after a human approves.

Example draft: `shared/contracts/agent_action_preview.json`. Routes + approval
model: `agentic_cpu_inside_runpod.md`.

## Follow-up fields to forward (summary checklist)

- [ ] `session_id`, `frame_id` on every `/detect`
- [ ] `scene_hint` and/or `site_context.environment_type`
- [ ] `camera_context.camera_name` / `location_name`
- [ ] `reasoning_preferences.force_reason` when the user asks "why?"
- [ ] forward `x-worker-secret` (and WS `?token=`) through Cloudflare
- [ ] for `/agent/*`: pass the detection JSON as `detection_context`; never
      auto-approve — require a human `approved_by` on execute
