"""Prepare commercial-safe SafeLens data as YOLO and COCO exports.

Public datasets must already be present under ``datasets/raw/<source>/`` with a
COCO ``annotations.json`` file and image paths referenced by that file. The
folder may include ``source.json`` containing ``{"dataset": "<manifest name>"}``.

Approved private images live under ``datasets/training-candidates/`` and require
a same-name JSON sidecar:

    {
      "approved": true,
      "annotations": [{"label": "open_hole", "bbox": [x, y, width, height]}]
    }

Network acquisition is opt-in and accepts only a manifest ``direct_url`` that
points to a ZIP archive. Current catalog page URLs are never scraped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "datasets" / "dataset_manifest.json"
DEFAULT_CLASSES = ROOT / "schemas" / "model_classes.json"
DEFAULT_REFERENCE_CATALOG = ROOT / "datasets" / "dataset_reference_catalog.json"
DEFAULT_RAW = ROOT / "datasets" / "raw"
DEFAULT_PRIVATE = ROOT / "datasets" / "training-candidates"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "prepared"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    source: str
    group_id: str
    image: Path
    width: int
    height: int
    annotations: list[dict[str, Any]]
    digest: str
    provenance: dict[str, Any]


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    safe = [s for s in data["sources"] if s.get("commercial_safe") and s.get("usage") == "product"]
    skipped = [s for s in data["sources"] if s not in safe]
    return safe, skipped


def download_source(source: dict, raw_dir: Path) -> Path:
    """Download an explicitly declared direct ZIP URL. Never scrape catalog pages."""
    direct_url = source.get("direct_url")
    if not direct_url or not str(direct_url).lower().endswith(".zip"):
        raise ValueError(
            f"{source['dataset']}: no approved direct ZIP URL in manifest; "
            "download manually under datasets/raw after license review"
        )
    target = raw_dir / slug(source["dataset"])
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "source.zip"
    urllib.request.urlretrieve(direct_url, archive)
    with zipfile.ZipFile(archive) as bundle:
        root = target.resolve()
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"unsafe ZIP path in {source['dataset']}: {member.filename}")
        bundle.extractall(target)
    return target


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Private image sizing requires Pillow: pip install Pillow") from exc
    with Image.open(path) as image:
        return image.size


def _validate_bbox(bbox: list[float], width: int, height: int, context: str) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"{context}: bbox must contain x, y, width, height")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"{context}: bbox dimensions must be positive")
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise ValueError(f"{context}: bbox is outside image bounds {width}x{height}")
    return [x, y, box_width, box_height]


def _source_metadata(folder: Path, source: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "dataset": source["dataset"],
        "url": source["url"],
        "license": source["license"],
        "version": source.get("version", "unspecified"),
    }
    source_file = folder / "source.json"
    if source_file.exists():
        supplied = json.loads(source_file.read_text(encoding="utf-8"))
        for protected in ("dataset", "url", "license"):
            if protected in supplied and supplied[protected] != metadata[protected]:
                raise ValueError(f"{source_file}: {protected} does not match the approved manifest")
        metadata.update(supplied)
    return metadata


def load_coco_source(folder: Path, source: dict[str, Any], class_names: list[str]) -> list[Sample]:
    annotation_path = folder / "annotations.json"
    if not annotation_path.exists():
        return []
    source_metadata = _source_metadata(folder, source)
    source_name = str(source_metadata.get("dataset") or source["dataset"])
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c["name"] for c in data.get("categories", [])}
    images = {int(i["id"]): i for i in data.get("images", [])}
    grouped: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for annotation in data.get("annotations", []):
        name = categories.get(int(annotation["category_id"]))
        if name not in class_names:
            continue
        image = images[int(annotation["image_id"])]
        bbox = _validate_bbox(
            [float(v) for v in annotation["bbox"]],
            int(image["width"]),
            int(image["height"]),
            f"{annotation_path}: annotation {annotation.get('id', '?')}",
        )
        grouped.setdefault(int(annotation["image_id"]), []).append({
            "label": name,
            "bbox": bbox,
            "iscrowd": int(annotation.get("iscrowd", 0)),
        })

    samples: list[Sample] = []
    for image_id, image in images.items():
        path = folder / image["file_name"]
        if not path.exists():
            raise FileNotFoundError(f"COCO image missing: {path}")
        samples.append(Sample(
            source=source_name,
            group_id=str(
                image.get("video_id")
                or image.get("sequence_id")
                or image.get("site_id")
                or f"{source_name}:{Path(image['file_name']).parent.as_posix()}"
            ),
            image=path,
            width=int(image["width"]),
            height=int(image["height"]),
            annotations=grouped.get(image_id, []),
            digest=sha256(path),
            provenance={**source_metadata, "source_file": image["file_name"]},
        ))
    return samples


def load_private_candidates(folder: Path, class_names: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for image in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        sidecar = image.with_suffix(".json")
        if not sidecar.exists():
            print(f"WARN private candidate skipped without sidecar: {image}")
            continue
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("approved") is not True:
            print(f"WARN private candidate skipped without approved=true: {image}")
            continue
        annotations = []
        width, height = _image_size(image)
        for annotation in metadata.get("annotations", []):
            label = annotation.get("label")
            bbox = annotation.get("bbox")
            if label not in class_names or not isinstance(bbox, list):
                raise ValueError(f"invalid annotation in {sidecar}: {annotation}")
            annotations.append({
                "label": label,
                "bbox": _validate_bbox(bbox, width, height, str(sidecar)),
                "iscrowd": 0,
            })
        samples.append(Sample(
            source="private-approved",
            group_id=str(metadata.get("group_id") or metadata.get("site_id") or image.parent.as_posix()),
            image=image,
            width=width,
            height=height,
            annotations=annotations,
            digest=sha256(image),
            provenance={
                "dataset": "private-approved",
                "license": "company-owned/user-approved",
                "source_file": image.name,
                "sidecar": sidecar.name,
            },
        ))
    return samples


def deduplicate(samples: list[Sample]) -> list[Sample]:
    unique: dict[str, Sample] = {}
    for sample in samples:
        unique.setdefault(sample.digest, sample)
    return list(unique.values())


def split_samples(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id, []).append(sample)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    total = len(samples)
    targets = {"train": total * 0.8, "val": total * 0.1, "test": total * 0.1}
    priority = {"train": 2, "val": 1, "test": 0}
    for group_id in group_ids:
        split = max(
            splits,
            key=lambda name: (targets[name] - len(splits[name]), priority[name]),
        )
        splits[split].extend(groups[group_id])
    return splits


def _yolo_line(annotation: dict, sample: Sample, class_to_id: dict[str, int]) -> str:
    x, y, width, height = annotation["bbox"]
    cx = (x + width / 2) / sample.width
    cy = (y + height / 2) / sample.height
    return f"{class_to_id[annotation['label']]} {cx:.8f} {cy:.8f} {width / sample.width:.8f} {height / sample.height:.8f}"


def export_dataset(splits: dict[str, list[Sample]], output: Path, class_names: list[str]) -> dict:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    report: dict[str, Any] = {"splits": {}, "classes": class_names}
    for split, samples in splits.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        annotation_dir = output / "annotations"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(class_names)],
        }
        annotation_id = 1
        for image_id, sample in enumerate(samples, start=1):
            filename = f"{sample.digest[:16]}{sample.image.suffix.lower()}"
            shutil.copy2(sample.image, image_dir / filename)
            lines = [_yolo_line(a, sample, class_to_id) for a in sample.annotations]
            (label_dir / f"{Path(filename).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            coco["images"].append({
                "id": image_id,
                "file_name": f"images/{split}/{filename}",
                "width": sample.width,
                "height": sample.height,
                "source": sample.source,
                "group_id": sample.group_id,
                "provenance": sample.provenance,
            })
            for annotation in sample.annotations:
                x, y, width, height = annotation["bbox"]
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[annotation["label"]] + 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": annotation.get("iscrowd", 0),
                })
                annotation_id += 1
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(coco, indent=2) + "\n",
            encoding="utf-8",
        )
        report["splits"][split] = {
            "images": len(samples),
            "annotations": len(coco["annotations"]),
        }

    yaml_lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ] + [f"  {index}: {name}" for index, name in enumerate(class_names)]
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--reference-catalog", type=Path, default=DEFAULT_REFERENCE_CATALOG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    safe_sources, skipped_sources = load_catalog(args.manifest)
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))["classes"]
    samples: list[Sample] = []
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    for source in safe_sources:
        folder = args.raw_dir / slug(source["dataset"])
        if args.download and not folder.exists():
            folder = download_source(source, args.raw_dir)
        if folder.exists():
            samples.extend(load_coco_source(folder, source, class_names))

    samples.extend(load_private_candidates(args.private_dir, class_names))
    unique = deduplicate(samples)
    splits = split_samples(unique, args.seed)
    report = export_dataset(splits, args.output_dir, class_names)
    report.update({
        "manifest": str(args.manifest),
        "commercial_sources_eligible": [source["dataset"] for source in safe_sources],
        "non_dataset_tools": [source["dataset"] for source in skipped_sources],
        "reference_catalog": str(args.reference_catalog),
        "split_strategy": "grouped by source/site/video/sequence before image allocation",
        "samples_before_dedup": len(samples),
        "samples_after_dedup": len(unique),
        "auto_deploy": False,
        "promotion_requires_human_approval": True,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
