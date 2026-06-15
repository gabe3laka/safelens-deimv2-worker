# RunPod Training Plan

## Purpose

Train and evaluate a commercial-safe DEIMv2 candidate without changing the live
worker automatically. OpenClaw/Codex authors this handoff; RunPod performs the
GPU work after the user supplies approved data and credentials.

## GPU Sizing

- Development/smoke training: 16-24 GB GPU, reduced image size and batch.
- Baseline fine-tune: one A40, L40S, RTX 4090, or A5000-class 24-48 GB GPU.
- Larger images or backbones: A100 40/80 GB or equivalent.
- Multi-GPU is optional after one-GPU reproducibility is established.

Start at 640 px and batch 16 on a 24 GB class GPU. Reduce batch first when
memory-bound; use gradient accumulation to preserve effective batch. Increase
image size only when small-object AP justifies the latency and memory cost.

## Inputs

- `datasets/dataset_manifest.json`: source/license catalog, not the training set.
- `datasets/raw/`: locally acquired approved public datasets in COCO form.
- `datasets/training-candidates/`: private approved images plus JSON sidecars.
- `schemas/model_classes.json`: canonical 25 classes.
- `runpod_training/train_config.yaml`: experiment configuration.

Do not upload rejected, unverified, NC, or AGPL dataset sources.

## Preparation

Run:

```bash
python prepare_dataset.py --output-dir prepared
```

The script:

1. selects only `commercial_safe=true` and `usage=product` sources
2. accepts only private sidecars with `approved=true`
3. validates class names and bounding boxes
4. hashes and deduplicates images
5. groups by source/site/video/sequence before deterministic split assignment
6. writes YOLO labels, COCO JSON, `data.yaml`, and a preparation report

Review the preparation report and visually sample every class before training.

## Training

1. Pin the DEIMv2 repository and base checkpoint by commit/hash.
2. Create a clean environment and record CUDA, PyTorch, driver, and package versions.
3. Train with mixed precision and deterministic seed where supported.
4. Save best and last checkpoints plus optimizer/config metadata.
5. Record per-epoch loss, AP, recall, confusion matrix, and GPU utilization.
6. Do not overwrite incumbent weights.

The exact upstream DEIMv2 command depends on the selected repository. Adapt its
data loader to the COCO export and preserve `train_config.yaml` alongside the
run.

## Evaluation

Evaluate the candidate and current `yolo26` incumbent on the same held-out test
set and scenario slices:

- aggregate mAP50 and mAP50-95
- AP and recall per hazard class
- false-negative rate for `open_hole`, `suspended_load`, `fire`, `smoke`,
  `blocked_exit`, and negative PPE classes
- site, camera, lighting, occlusion, and distance slices
- inference latency, VRAM, model size, and cold-start time
- relational-context tests handled by the reasoning layer

Aggregate mAP alone cannot approve a safety model. A candidate that increases
rare critical false negatives must be rejected even if global mAP improves.

## Promotion Gate

Training output enters `pending_approval`. The evaluation report becomes the
preview payload for the LangGraph human-approval flow:

`Preview metrics -> interrupt() -> approve/reject/revise -> registry update -> log`

Approval must identify the reviewer, dataset version, candidate hash, incumbent
hash, metrics, known limitations, and rollback target. Only a user-controlled
process may update `schemas/model_registry.json` and publish a restricted
weights URL.

Never auto-deploy or auto-promote a training result.

## Rollout

1. Offline evaluation.
2. Shadow inference beside the incumbent.
3. Review disagreement samples.
4. Limited site/camera rollout with rollback.
5. Monitor drift, critical-class recall, latency, and alert burden.
6. Expand only after signed review.

## Reproducibility Artifacts

Retain:

- source manifest and attribution
- prepared dataset report and hashes
- split lists
- code commit and environment lock
- train config and random seed
- base/candidate checkpoint hashes
- raw and summarized metrics
- approval record
- deployment/rollback record
