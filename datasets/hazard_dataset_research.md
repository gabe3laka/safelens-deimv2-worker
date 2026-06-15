# SafeLens Hazard Dataset Research

## Use Policy

Only sources with an explicitly verified commercial-safe license are eligible
for product training. In this bundle that means CC BY 4.0, CC0, Apache-2.0, or
MIT. CC BY-NC-SA, AGPL, missing-license, and unclear-license data are excluded
from product datasets. Private customer frames require approval, access control,
anonymization, and an auditable retention policy.

The detector direction is DEIM/DEIMv2 (Apache-2.0), with RT-DETR as an
alternative. The current worker still defaults to `yolo26`, so this is a
candidate-model pipeline rather than an assertion that DEIMv2 is deployed.

## 1. PPE Non-Compliance

- Public data: PPE Combined Model, Hard Hats, Safety Vests, Construction Site
  Safety, and SHEL5K.
- License: CC BY 4.0; commercial use allowed with attribution.
- Annotation: bounding boxes.
- Target labels: `person`, `hardhat`, `no_hardhat`, `safety_vest`,
  `no_safety_vest`, `gloves`, `no_gloves`, `goggles`, `no_goggles`, `mask`,
  `no_mask`.
- Synthetic value: medium; useful for rare PPE colors, occlusion, camera angle,
  and lighting, but real negative examples remain essential.
- Difficulty: medium to high because absence labels require person-to-PPE
  association, not only object detection.
- Starting size: 20,000-40,000 reviewed images, balanced by PPE item and site.
- Model: DEIMv2 detector plus association/post-processing.
- Safety note: do not declare non-compliance where the site rule, work zone, or
  task does not require that PPE.

## 2. Working At Height And Falls

- Public data: Fall Video Dataset (CC0) for temporal fall research; general
  construction data can seed `person`, `ladder`, and `scaffold`.
- Annotation: video classification/temporal clips plus custom bounding boxes,
  pose, and height-zone labels.
- Target labels: `person`, `ladder`, `scaffold`; add scene metadata for edge,
  platform, guardrail, and estimated height.
- Synthetic value: high for controlled height, guardrail, harness, and drop-path
  variations.
- Difficulty: high; a person standing is not a height hazard without geometry.
- Starting size: 8,000-15,000 site-relevant frames plus temporal sequences.
- Model: DEIMv2 + pose/depth/zone reasoning; temporal model for fall state.
- Safety note: distinguish a fall event from exposure to falling; the latter
  requires contextual reasoning before an incident occurs.

## 3. Ladder And Scaffold Safety

- Public data: Construction Site Safety (CC BY 4.0) provides some ladder and
  scaffold examples; no broad commercial-safe condition-quality dataset was
  identified in the bundle.
- Annotation: bounding boxes plus custom keypoints/attributes for angle,
  extension, footing, guardrails, deck completeness, and access.
- Target labels: `ladder`, `scaffold`, `person`.
- Synthetic value: high for ladder angle, top support, missing rails, incomplete
  platforms, and workers above/below.
- Difficulty: high because unsafe setup is relational.
- Starting size: 6,000-10,000 reviewed scenes.
- Model: DEIMv2 + geometry rules or VLM reasoning.
- Safety note: object presence is not a violation; setup, condition, use, and
  fall distance determine risk.

## 4. Open Holes, Manholes, And Floor Openings

- Public data: no verified commercial-safe specialist dataset identified.
- Annotation: bounding boxes or segmentation for `open_hole`, covers,
  barricades, and people; depth/plane metadata is valuable.
- Synthetic value: very high using BlenderProc and approved assets.
- Difficulty: high due to shadows, dark surfaces, covers, and perspective.
- Starting size: 5,000-8,000 scenes with hard negatives.
- Model: segmentation-capable DEIMv2/RT-DETR variant plus depth/geometry.
- Safety note: an opening in a locked area differs from an unprotected opening
  in a travel route; encode access and fall distance.

## 5. Slips, Trips, And Housekeeping

- Public data: no complete commercial-safe hazard set identified; general
  construction data may provide partial clutter examples.
- Annotation: segmentation for `spill`, boxes/lines for `trailing_cable`,
  floor-plane and traffic-zone metadata.
- Target labels: `spill`, `trailing_cable`, `safety_cone`, `person`.
- Synthetic value: high for spill shape, reflectivity, cable routing, lighting,
  and traffic placement.
- Difficulty: high because small floor hazards and location drive severity.
- Starting size: 10,000-20,000 floor-focused images.
- Model: segmentation + DEIMv2 detector + scene reasoning.
- Safety note: substance, floor surface, visibility, foot traffic, and proximity
  to stairs or edges must influence risk.

## 6. Electrical Safety

- Public data: no verified commercial-safe dataset covering open panels,
  exposed conductors, access, and isolation state.
- Annotation: `open_panel`, panel door, barriers, person, warning signage, and
  energized/de-energized metadata supplied by the site.
- Synthetic value: high for open/closed states and approach boundaries.
- Difficulty: critical; imagery cannot reliably prove de-energization.
- Starting size: 5,000-10,000 scenes plus site-authenticated state labels.
- Model: DEIMv2 + rules/RAG; never infer lockout solely from pixels.
- Safety note: visual evidence supports triage only. Qualified-person and LOTO
  verification remain authoritative.

## 7. Fire, Smoke, And Hot Work

- Public data: D-Fire is listed as unverified and excluded until its license and
  provenance are confirmed.
- Annotation: boxes/segmentation for `fire`, `smoke`, combustibles, hot-work
  source, extinguisher, and exclusion area.
- Synthetic value: medium to high, especially for hot-work relationships; use
  physically plausible overlays and preserve negatives such as steam/dust.
- Difficulty: high due to false positives and severe consequence.
- Starting size: 15,000-30,000 images and clips.
- Model: DEIMv2 detector plus temporal confirmation and QHSE reasoning.
- Safety note: hot work near combustibles can be critical before visible flame;
  permit and fire-watch state must come from operational data.

## 8. Lifting And Suspended Loads

- Public data: no verified commercial-safe specialist set identified.
- Annotation: `suspended_load`, lifting equipment, rigging, people, exclusion
  zone, and load path.
- Synthetic value: very high for drop paths, swing radius, and people placement.
- Difficulty: high because load state and geometry matter more than category.
- Starting size: 6,000-12,000 scenes with close and wide views.
- Model: DEIMv2 + tracking + geometric/VLM reasoning.
- Safety note: people under or near the load path create the dominant exposure;
  a radio or PPE is not a substitute for exclusion.

## 9. Confined Space

- Public data: no verified commercial-safe complete dataset identified.
- Annotation: entry opening, person, attendant, barriers, signage, gas monitor,
  and permit-state metadata.
- Synthetic value: high for entrances and control configurations.
- Difficulty: critical because atmosphere and permit state are not visible.
- Starting size: 4,000-8,000 entry scenes plus structured operational records.
- Model: detector + RAG/rules, not vision alone.
- Safety note: the system must never declare a confined space safe from imagery;
  atmospheric testing, isolation, rescue planning, and permit approval control.

## 10. Manual Handling And Ergonomics

- Public data: no selected commercial-safe construction-specific set in the
  manifest; approved pose datasets may be added only after separate license
  review.
- Annotation: person pose/keypoints, load box, carry height, reach, twist, and
  repetition metadata.
- Synthetic value: medium for pose diversity but limited for realistic effort.
- Difficulty: high because mass, frequency, individual capability, and duration
  are not visually reliable.
- Starting size: 10,000-20,000 poses/clips with task metadata.
- Model: pose estimation + ergonomic scoring, with DEIMv2 for load/person boxes.
- Safety note: output is a screening prompt, not a medical or definitive manual
  handling assessment.

## 11. Forklift And Vehicle-Pedestrian Interaction

- Public data: Thalos Forklift Safety v1 is AGPL-3.0 and benchmark-only.
- Annotation: `forklift`, `person`, load, lane, crossing, blind corner, and
  proximity/trajectory.
- Synthetic value: very high for near-miss geometry and rare collision paths.
- Difficulty: high; tracking and intent matter.
- Starting size: 12,000-25,000 frames plus sequences.
- Model: DEIMv2 + multi-object tracking and trajectory risk.
- Safety note: avoid a single distance threshold; speed, visibility, separation,
  right-of-way, and load obstruction change risk.

## 12. Machine Guarding

- Public data: no verified commercial-safe specialist dataset identified.
- Annotation: machine, hazard zone, guard present/open/missing, person hands,
  and machine operating state.
- Synthetic value: high for guard configurations.
- Difficulty: high and machine-specific.
- Starting size: 5,000-10,000 images per machine family.
- Model: DEIMv2 + site-specific fine-tune and rule context.
- Safety note: do not infer isolation or zero energy from a still image.

## 13. Chemical Storage And Labeling

- Public data: no selected commercial-safe HSE-specific dataset.
- Annotation: container, label present/readable, cabinet, secondary containment,
  spill, incompatible storage, and gas cylinder.
- Target labels: `spill`, `gas_cylinder`, with custom label attributes.
- Synthetic value: medium; text/label rendering and storage layouts help.
- Difficulty: high because chemical identity and compatibility require records.
- Starting size: 8,000-15,000 images with OCR-ready crops.
- Model: DEIMv2 + OCR + RAG over SDS/site rules.
- Safety note: never identify an unknown substance solely by container color.

## 14. Compressed Gas Cylinders

- Public data: general construction sources may contain cylinders, but no
  specialist commercial-safe condition set was identified.
- Annotation: `gas_cylinder`, cap, chain/restraint, valve protection, upright
  state, heat source, and storage separation.
- Synthetic value: high for restraint and placement variants.
- Difficulty: medium to high.
- Starting size: 5,000-8,000 scenes.
- Model: DEIMv2 + attribute classification.
- Safety note: cylinder contents and compatibility must come from readable labels
  or inventory data, not appearance alone.

## 15. Excavation And Trench Safety

- Public data: no verified commercial-safe specialist set identified.
- Annotation: excavation boundary, depth estimate, access ladder, spoil pile,
  shoring/benching, person, vehicle, and barricade.
- Synthetic value: very high.
- Difficulty: high due to scale and depth ambiguity.
- Starting size: 6,000-12,000 wide scenes.
- Model: detection/segmentation + depth and geometry.
- Safety note: competent-person inspection, soil classification, utilities, and
  atmosphere are external facts the model cannot establish visually.

## 16. Blocked Exits And Fire Equipment

- Public data: no verified complete commercial-safe dataset identified.
- Annotation: `blocked_exit`, `extinguisher`, exit sign, obstruction, access
  width, and visibility.
- Synthetic value: high for obstruction placement and lighting.
- Difficulty: medium; scene context and route continuity matter.
- Starting size: 5,000-10,000 images.
- Model: DEIMv2 + segmentation/route reasoning.
- Safety note: an extinguisher being visible does not prove inspection status,
  suitability, or unobstructed access.

## 17. Lighting, Lone Work, And Restricted Areas

- Public data: no selected commercial-safe specialist set.
- Annotation: illumination proxy, person, restricted-zone boundary, signage,
  occupancy, and time/authorization metadata.
- Synthetic value: high for lighting and zone geometry.
- Difficulty: high because authorization and lone-worker controls are nonvisual.
- Starting size: 5,000-10,000 scenes plus access-control events.
- Model: detector + zone rules + operational integrations.
- Safety note: camera exposure is not a calibrated lux reading; use sensor or
  survey data where illumination compliance matters.

## Recommended Data Workflow

1. Record source URL, version, license, attribution, hash, and acquisition date.
2. Download only `usage: product` sources.
3. Map labels to `schemas/model_classes.json`; reject ambiguous mappings.
4. Deduplicate mirrors and near-identical images before splitting.
5. Split by site/video/source group to prevent leakage.
6. Anonymize approved private frames and retain them in restricted storage.
7. Review labels in CVAT or FiftyOne.
8. Export both COCO and YOLO representations.
9. Evaluate per class and on relational safety slices, not only aggregate mAP.
10. Require human approval before a candidate model enters the registry.
