# Pedestrian 3D Tracking - Planning Note (NOT Implemented)

> Status: **planning only**. No tracking code ships in this PR. This document
> describes how a *future* analytics layer would consume EdgeCrafter outputs.
> Do not implement 3D tracking until EdgeCrafter returns reliable entities and
> poses in the live dry-run.

## Why this is a later layer

Pedestrian 3D tracking can eventually provide persistent person IDs, movement
trails, in/out counting, distance approximation, trajectory analysis, and zone
crossing. But true 3D tracking depends on three things that must come first:

1. **Stable 2D detections** - reliable ECDet-S person boxes frame to frame.
2. **Camera calibration or explicit assumptions** - without these, "meters" are
   not meaningful.
3. **Motion filtering** - association + smoothing across frames.

Until EdgeCrafter boxes and poses are proven stable in the visual dry-run, any
3D/world-coordinate output would be speculative. The first production value is
**stable IDs, trails, and counting** - not true 3D.

## Inputs (already available from the worker)

The future tracker consumes the existing `/detect` response plus per-frame metadata:

- `entities` - person bounding boxes (`class_id == 0`), normalized 0..1.
- `poses` - COCO-17 keypoints + skeleton edges (optional, when a person is visible).
- `timestamp` - frame capture time (supplied by the caller).
- `img_w` / `img_h` - camera frame size (already returned by `/detect`).
- *optional, later:* camera calibration (intrinsics/extrinsics or a homography
  to a ground plane).

## Future outputs (target shape - illustrative only)

```json
{
  "tracks": [
    {
      "track_id": "p1",
      "bbox": { "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.6 },
      "confidence": 0.87,
      "centroid": { "x": 0.2, "y": 0.7 },
      "trail": [{ "x": 0.2, "y": 0.7, "t": 12345 }],
      "state": "active"
    }
  ],
  "tracks_3d": [
    {
      "track_id": "p1",
      "estimated_depth_m": null,
      "world_position": null,
      "confidence": "not_calibrated"
    }
  ]
}
```

Note that `tracks_3d` fields are `null` / `not_calibrated` by design until a
calibration step exists. The system must never report a number of meters it
cannot justify.

## Staged plan

- **Stage 1 - 2D tracker.** Associate ECDet-S person boxes across frames using
  IoU and centroid matching; assign persistent `track_id`s; maintain a short
  trail buffer per track. Output `tracks` with `state` (active / lost).
- **Stage 2 - line crossing / in-out counting.** Add user-defined lines/zones in
  normalized coordinates; count crossings using trail direction.
- **Stage 3 - calibrated ground-plane mapping.** With a homography or camera
  calibration, map foot-point (e.g. between the ankles from pose, or bbox bottom
  centre) onto a ground plane for top-down trajectories.
- **Stage 4 - approximate distance / depth.** Only *after* Stage 3, and only with
  clear assumptions (known camera height/tilt or calibration), produce coarse
  distance/depth estimates. Label confidence honestly.
- **Stage 5 - safety mapping.** Only once tracking is stable do tracks feed any
  safety logic. This remains **blocked** (see below).

## Hard constraints (carried from the sprint scope)

- 3D tracking is **not reliable** without calibration or explicit assumptions.
- Do **not** claim accurate meters until Stage 3/4 calibration exists.
- No safety alerts, no RiskEngine mapping, no PPE/forklift/blocked-exit alerts,
  and no incident saving are introduced by this tracking layer in its early
  stages. Sprint 4B safety mapping stays blocked until the visual dry-run
  (EdgeCrafter boxes + poses) is stable.
