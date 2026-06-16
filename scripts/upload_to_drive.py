#!/usr/bin/env python3
"""
SafeLens HSE dataset Drive uploader.

Zips each downloaded dataset and uploads it to the designated Google Drive folder
using the Drive API v3 with resumable uploads (handles files of any size).

Prerequisites:
    pip install google-auth-oauthlib google-api-python-client

Authentication:
    1. Create an OAuth 2.0 Client ID in Google Cloud Console
       (Application type: Desktop app)
    2. Download credentials.json and place it in the project root or pass via
       --credentials
    3. On first run, a browser window opens for consent; token is cached to
       token.json (or --token-file path)

Usage:
    python scripts/upload_to_drive.py \\
        --staging /tmp/safelens_dl \\
        --credentials credentials.json \\
        [--sources SRC-001 SRC-002 ...]

Drive folder layout (IDs from SafeLens brief):
    Roboflow datasets  →  1U0Ks4U6UB1bmZIBRJo7OW_f49SUvXVuf
    Mendeley SHEL5K    →  1yQLu2QwLQ5Iz07fdnYZN7xkDeD8R83jf
    Manifests          →  1RJXSNxqaVPXpYvJMKIQGfyGF1OtcknW4

After all uploads complete, drive_file_id fields in data/acquisition_log.jsonl
are updated automatically.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DRIVE_FOLDERS = {
    "roboflow": "1U0Ks4U6UB1bmZIBRJo7OW_f49SUvXVuf",
    "mendeley": "1yQLu2QwLQ5Iz07fdnYZN7xkDeD8R83jf",
    "manifests": "1RJXSNxqaVPXpYvJMKIQGfyGF1OtcknW4",
}

SOURCE_FOLDERS = {
    "SRC-001": DRIVE_FOLDERS["roboflow"],
    "SRC-002": DRIVE_FOLDERS["roboflow"],
    "SRC-003": DRIVE_FOLDERS["roboflow"],
    "SRC-004": DRIVE_FOLDERS["roboflow"],
    "SRC-005": DRIVE_FOLDERS["roboflow"],
    "SRC-007": DRIVE_FOLDERS["mendeley"],
}

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_service(credentials_path: str, token_path: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def zip_dataset(src_id: str, staging: Path) -> Path:
    """Zip dataset directory. Returns path to zip file."""
    src_dir = staging / src_id
    zip_path = staging / f"{src_id}.zip"

    if zip_path.exists():
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  {zip_path.name} already exists ({size_mb:.1f} MB) — reusing")
        return zip_path

    if not src_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {src_dir}")

    print(f"  Zipping {src_id}...", flush=True)
    subprocess.run(
        ["zip", "-r", "-q", str(zip_path), src_id],
        cwd=str(staging),
        check=True,
    )
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  {zip_path.name} = {size_mb:.1f} MB", flush=True)
    return zip_path


def upload_file(service, local_path: Path, title: str, parent_id: str) -> str:
    """Resumable upload. Returns Drive file ID."""
    from googleapiclient.http import MediaFileUpload

    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"  Uploading {title} ({size_mb:.1f} MB)...", flush=True)

    file_metadata = {"name": title, "parents": [parent_id]}
    media = MediaFileUpload(
        str(local_path),
        mimetype="application/zip",
        resumable=True,
        chunksize=8 * 1024 * 1024,  # 8 MB chunks
    )

    request = service.files().create(body=file_metadata, media_body=media, fields="id")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    {pct}%...", end="\r", flush=True)

    file_id = response.get("id")
    print(f"    Done → {file_id}      ")
    return file_id


def update_log(log_path: Path, updates: dict[str, str]) -> None:
    """Update drive_file_id fields in acquisition_log.jsonl."""
    if not log_path.exists():
        return
    entries = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            src_id = entry.get("src_id", "")
            if src_id in updates:
                entry["drive_file_id"] = updates[src_id]
                if entry.get("status") in ("downloaded", "partial"):
                    entry["status"] = "uploaded"
            entries.append(entry)
    with open(log_path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    print(f"  Acquisition log updated: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Upload SafeLens datasets to Google Drive")
    parser.add_argument("--staging", default="/tmp/safelens_dl")
    parser.add_argument("--credentials", default="credentials.json")
    parser.add_argument("--token-file", default="token.json")
    parser.add_argument(
        "--sources", nargs="+",
        default=list(SOURCE_FOLDERS.keys()),
    )
    parser.add_argument("--log", default="data/acquisition_log.jsonl")
    args = parser.parse_args()

    staging = Path(args.staging)
    log_path = Path(args.log)

    service = get_service(args.credentials, args.token_file)

    drive_ids: dict[str, str] = {}

    for src_id in args.sources:
        if src_id not in SOURCE_FOLDERS:
            print(f"\n{src_id}: unknown source — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"{src_id}")

        try:
            zip_path = zip_dataset(src_id, staging)
            folder_id = SOURCE_FOLDERS[src_id]
            file_id = upload_file(service, zip_path, zip_path.name, folder_id)
            drive_ids[src_id] = file_id
            print(f"  {src_id} DONE — Drive ID: {file_id}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if drive_ids:
        update_log(log_path, drive_ids)

    print("\n=== SUMMARY ===")
    for src_id in args.sources:
        status = "uploaded" if src_id in drive_ids else "FAILED/SKIPPED"
        fid = drive_ids.get(src_id, "—")
        print(f"  {src_id}: {status}  {fid}")

    also_upload_log = input("\nUpload acquisition_log.jsonl to manifests folder? [y/N] ")
    if also_upload_log.strip().lower() == "y":
        if log_path.exists():
            print("Uploading acquisition log...")
            fid = upload_file(
                service, log_path, "acquisition_log.jsonl",
                DRIVE_FOLDERS["manifests"],
            )
            print(f"  acquisition_log.jsonl → {fid}")
        else:
            print(f"  {log_path} not found — skipping")


if __name__ == "__main__":
    main()
