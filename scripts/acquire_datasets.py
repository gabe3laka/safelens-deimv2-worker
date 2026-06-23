#!/usr/bin/env python3
"""
SafeLens HSE dataset acquisition script.

Downloads approved CC BY 4.0 PPE/safety datasets from Roboflow and Mendeley
into a local staging directory for subsequent Drive upload.

Usage:
    export ROBOFLOW_API_KEY="your_key_here"
    python scripts/acquire_datasets.py [--output /path/to/staging]

Approved sources (all CC BY 4.0):
    SRC-001  construction-site-safety          (Roboflow)
    SRC-002  personal-protective-equipment-combined-model (Roboflow)
    SRC-003  hard-hats-fhbh5                   (Roboflow)
    SRC-004  safety-vests                      (Roboflow)
    SRC-005  ppe-detection-hardhat-vest        (Roboflow)
    SRC-006  construction-safety-object-detection — SKIP (HF mirror of SRC-001)
    SRC-007  SHEL5K v4                         (Mendeley)

DO NOT commit image/label files to git. Stage to /tmp or an external drive.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"ERROR: environment variable {name} is not set")
    return val


def write_log(log_path: Path, entries: list[dict]) -> None:
    with open(log_path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


# ── Roboflow downloads ────────────────────────────────────────────────────────

ROBOFLOW_SOURCES = [
    {
        "src_id": "SRC-001",
        "workspace": "roboflow-universe-projects",
        "project": "construction-site-safety",
        "version": None,  # auto-detect latest
        "license": "CC BY 4.0",
        "classes": [
            "Barricade", "Dumpster", "EXCAVATORS", "Gloves", "Hardhat", "Mask",
            "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Net",
            "Safety Shoes", "Safety Vest", "dump truck", "mini-van", "truck", "wheel loader",
        ],
    },
    {
        "src_id": "SRC-002",
        "workspace": "roboflow-universe-projects",
        "project": "personal-protective-equipment-combined-model",
        "version": None,
        "license": "CC BY 4.0",
        "classes": [
            "Fall-Detected", "Gloves", "Goggles", "Hardhat", "Ladder", "Mask",
            "NO-Gloves", "NO-Goggles", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
            "Person", "Safety Cone", "Safety Vest",
        ],
    },
    {
        "src_id": "SRC-003",
        "workspace": "roboflow-universe-projects",
        "project": "hard-hats-fhbh5",
        "version": None,
        "license": "CC BY 4.0",
        "classes": ["Hardhat", "NO-Hardhat"],
    },
    {
        "src_id": "SRC-004",
        "workspace": "roboflow-universe-projects",
        "project": "safety-vests",
        "version": None,
        "license": "CC BY 4.0",
        "classes": ["NO-Safety Vest", "Safety Vest"],
    },
    {
        "src_id": "SRC-005",
        "workspace": "roboflow-universe-projects",
        "project": "ppe-detection-hardhat-vest",
        "version": 4,  # only v4 exists
        "license": "CC BY 4.0",
        "classes": ["helmet", "no-helmet", "no-vest", "person", "vest"],
    },
]


def download_roboflow(src: dict, output_dir: Path, api_key: str) -> dict:
    from roboflow import Roboflow

    dest = output_dir / src["src_id"]
    if dest.exists():
        files = list(dest.rglob("*.jpg")) + list(dest.rglob("*.jpeg")) + list(dest.rglob("*.png"))
        if files:
            print(f"  {src['src_id']}: already downloaded ({len(files)} images) — skipping")
            return _count_entry(src, dest)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(src["workspace"]).project(src["project"])

    if src["version"] is not None:
        version_obj = project.version(src["version"])
    else:
        # Try versions 10..1 until one succeeds
        version_obj = None
        for v in range(10, 0, -1):
            try:
                version_obj = project.version(v)
                break
            except Exception:
                continue
        if version_obj is None:
            raise RuntimeError(f"No accessible version found for {src['project']}")

    dataset = version_obj.download("yolov8", location=str(dest))
    return _count_entry(src, dest)


def _count_entry(src: dict, dest: Path) -> dict:
    images = list(dest.rglob("*.jpg")) + list(dest.rglob("*.jpeg")) + list(dest.rglob("*.png"))
    labels = list(dest.rglob("*.txt"))
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024 / 1024
    return {
        "src_id": src["src_id"],
        "name": src["project"],
        "license": src["license"],
        "date": time.strftime("%Y-%m-%d"),
        "images": len(images),
        "labels": len(labels),
        "size_mb": round(size_mb, 1),
        "classes": src.get("classes", []),
        "status": "downloaded",
        "drive_file_id": None,
        "note": "",
    }


# ── Mendeley SRC-007 ──────────────────────────────────────────────────────────

MENDELEY_DATASET_ID = "9rcv8mm682"
MENDELEY_VERSION = 4


def download_mendeley(output_dir: Path) -> dict:
    """
    Download SHEL5K v4 from Mendeley Data API.

    NOTE: The Mendeley API has a pagination bug — it always returns the same
    100 files regardless of page/offset params. The recommended workaround is
    to download the full dataset ZIP directly from:
        https://data.mendeley.com/datasets/9rcv8mm682/4
    That requires a Mendeley account. If you have one, log in, download the ZIP,
    and place it at: <output_dir>/SRC-007/SHEL5K_v4.zip
    """
    import urllib.request

    dest = output_dir / "SRC-007"
    dest.mkdir(parents=True, exist_ok=True)

    api_url = (
        f"https://data.mendeley.com/api/datasets/{MENDELEY_DATASET_ID}"
        f"/files?version={MENDELEY_VERSION}"
    )

    req = urllib.request.Request(api_url, headers={"User-Agent": "SafeLens/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        files = json.loads(resp.read())

    downloaded = 0
    failed = 0
    for item in files:
        filename = item.get("filename", "")
        download_url = item.get("content_details", {}).get("download_url", "")
        if not download_url or not filename:
            continue

        out_path = dest / filename
        if out_path.exists():
            continue

        try:
            req2 = urllib.request.Request(download_url, headers={"User-Agent": "SafeLens/1.0"})
            with urllib.request.urlopen(req2, timeout=60) as r, open(out_path, "wb") as fh:
                fh.write(r.read())
            downloaded += 1
        except Exception as e:
            print(f"    WARNING: could not download {filename}: {e}")
            failed += 1

    images = list(dest.rglob("*.jpg")) + list(dest.rglob("*.jpeg")) + list(dest.rglob("*.png"))
    labels = list(dest.rglob("*.txt"))
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024 / 1024

    note = (
        "Mendeley API pagination broken — returns same 100 files regardless of "
        "page/offset params. Full dataset (~10k images) requires manual download from "
        "https://data.mendeley.com/datasets/9rcv8mm682/4"
    )

    return {
        "src_id": "SRC-007",
        "name": "SHEL5K v4",
        "license": "CC BY 4.0",
        "date": time.strftime("%Y-%m-%d"),
        "images": len(images),
        "labels": len(labels),
        "size_mb": round(size_mb, 1),
        "classes": [],
        "status": "partial" if failed > 0 or len(images) < 1000 else "downloaded",
        "drive_file_id": None,
        "note": note,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download SafeLens HSE datasets")
    parser.add_argument("--output", default="/tmp/safelens_dl", help="Staging directory")
    parser.add_argument(
        "--sources", nargs="+",
        default=["SRC-001", "SRC-002", "SRC-003", "SRC-004", "SRC-005", "SRC-007"],
        help="Which sources to download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = require_env("ROBOFLOW_API_KEY")

    log_path = output_dir / "acquisition_log.jsonl"
    entries = []

    for src in ROBOFLOW_SOURCES:
        if src["src_id"] not in args.sources:
            continue
        print(f"\n[{src['src_id']}] Downloading {src['project']}...")
        try:
            entry = download_roboflow(src, output_dir, api_key)
            print(f"  OK — {entry['images']} images, {entry['size_mb']} MB")
        except Exception as e:
            print(f"  FAILED: {e}")
            entry = {
                "src_id": src["src_id"], "name": src["project"],
                "license": src["license"], "date": time.strftime("%Y-%m-%d"),
                "images": 0, "labels": 0, "size_mb": 0,
                "classes": src.get("classes", []),
                "status": "failed", "drive_file_id": None, "note": str(e),
            }
        entries.append(entry)

    if "SRC-006" in args.sources:
        entries.append({
            "src_id": "SRC-006",
            "name": "construction-safety-object-detection",
            "license": "CC BY 4.0",
            "date": time.strftime("%Y-%m-%d"),
            "images": 0, "labels": 0, "size_mb": 0, "classes": [],
            "status": "skipped",
            "reason": "HuggingFace Xet CDN blocked; same data as SRC-001 (Roboflow mirror)",
            "drive_file_id": None, "note": "",
        })

    if "SRC-007" in args.sources:
        print("\n[SRC-007] Downloading SHEL5K v4 from Mendeley...")
        try:
            entry = download_mendeley(output_dir)
            print(f"  OK — {entry['images']} images, {entry['size_mb']} MB ({entry['status']})")
        except Exception as e:
            print(f"  FAILED: {e}")
            entry = {
                "src_id": "SRC-007", "name": "SHEL5K v4",
                "license": "CC BY 4.0", "date": time.strftime("%Y-%m-%d"),
                "images": 0, "labels": 0, "size_mb": 0, "classes": [],
                "status": "failed", "drive_file_id": None, "note": str(e),
            }
        entries.append(entry)

    write_log(log_path, entries)
    print(f"\nAcquisition log written to {log_path}")


if __name__ == "__main__":
    main()
