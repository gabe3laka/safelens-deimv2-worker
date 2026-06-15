"""Build a local SafeLens document index and a user-loadable pgvector export.

OpenClaw/Codex never uploads the result. The user may review
the document/chunk CSV exports with ``db/pgvector_loader.sql``.

Install the optional runtime:
    pip install sentence-transformers faiss-cpu numpy pypdf python-docx
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "source-documents"
DEFAULT_INDEX = ROOT / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LICENSE = "Apache-2.0"
NATIVE_DIMENSIONS = 384
EXPORT_DIMENSIONS = NATIVE_DIMENSIONS
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
NIL_UUID = "00000000-0000-0000-0000-000000000000"


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    content: str
    metadata: dict


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst", ".csv", ".log", ".yaml", ".yml"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(value, ensure_ascii=False, indent=2)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("PDF found; install pypdf: pip install pypdf") from exc
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError("DOCX found; install python-docx: pip install python-docx") from exc
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError(f"unsupported document type: {path}")


def _files(source_dir: Path) -> Iterable[Path]:
    allowed = {".txt", ".md", ".rst", ".csv", ".log", ".yaml", ".yml", ".json", ".pdf", ".docx"}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed and path.name != ".gitkeep":
            yield path


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    clean = "\n".join(line.rstrip() for line in text.replace("\x00", "").splitlines()).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = max(clean.rfind("\n", start, end), clean.rfind(" ", start, end))
            if boundary > start + size // 2:
                end = boundary
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def collect_chunks(source_dir: Path, chunk_size: int, overlap: int) -> tuple[list[Chunk], list[dict]]:
    chunks: list[Chunk] = []
    documents: list[dict] = []
    for path in _files(source_dir):
        relative = path.relative_to(source_dir).as_posix()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"safelens-document:{relative}:{digest}"))
        text = _read_text(path)
        parts = _chunk_text(text, chunk_size, overlap)
        documents.append({
            "document_id": document_id,
            "path": relative,
            "sha256": digest,
            "bytes": len(raw),
            "chunks": len(parts),
        })
        for index, content in enumerate(parts):
            chunks.append(Chunk(
                document_id=document_id,
                chunk_index=index,
                content=content,
                metadata={"source_path": relative, "sha256": digest},
            ))
    return chunks, documents


def _embed(contents: list[str], model_name: str):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embedding dependencies missing. Install: "
            "pip install sentence-transformers numpy"
        ) from exc
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        contents,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    if vectors.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape: {vectors.shape}")
    return np, vectors


def _export_vector(vector, dimensions: int) -> list[float]:
    values = vector.tolist()
    if len(values) > dimensions:
        raise ValueError(f"native embedding dimension {len(values)} exceeds export dimension {dimensions}")
    return values + [0.0] * (dimensions - len(values))


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_index(
    source_dir: Path,
    output_dir: Path,
    owner_id: str,
    org_id: str | None,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> dict:
    chunks, documents = collect_chunks(source_dir, chunk_size, overlap)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "embeddings_manifest.json"

    if not chunks:
        manifest = {
            "status": "awaiting_documents",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {"name": model_name, "license": MODEL_LICENSE},
            "dimensions": {"native": NATIVE_DIMENSIONS, "pgvector_export": EXPORT_DIMENSIONS},
            "chunking": {"strategy": "character_window", "size": chunk_size, "overlap": overlap},
            "index_backend": "faiss",
            "source_documents": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    np, vectors = _embed([chunk.content for chunk in chunks], model_name)
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("FAISS missing. Install: pip install faiss-cpu") from exc

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(output_dir / "documents.faiss"))

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    documents_jsonl = ROOT / "documents_export.jsonl"
    documents_csv = ROOT / "documents_export.csv"
    with documents_jsonl.open("w", encoding="utf-8") as json_handle, documents_csv.open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        fieldnames = [
            "id", "owner_id", "org_id", "title", "document_type",
            "storage_path", "status",
        ]
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for document in documents:
            row = {
                "id": document["document_id"],
                "owner_id": owner_id,
                "org_id": org_id or "",
                "title": Path(document["path"]).name,
                "document_type": Path(document["path"]).suffix.lower().lstrip(".") or "text",
                "storage_path": document["path"],
                "status": "indexed",
            }
            json_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow(row)

    chunks_jsonl = ROOT / "chunks_export.jsonl"
    chunks_csv = ROOT / "chunks_export.csv"
    with chunks_jsonl.open("w", encoding="utf-8") as json_handle, chunks_csv.open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        fieldnames = [
            "owner_id", "org_id", "document_id", "chunk_index",
            "content", "embedding_json", "metadata",
        ]
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk, vector in zip(chunks, vectors, strict=True):
            row = {
                "owner_id": owner_id,
                "org_id": org_id or "",
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding_json": json.dumps(_export_vector(vector, EXPORT_DIMENSIONS)),
                "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
            }
            json_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow(row)

    manifest = {
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"name": model_name, "license": MODEL_LICENSE},
        "dimensions": {"native": int(vectors.shape[1]), "pgvector_export": EXPORT_DIMENSIONS},
        "chunking": {"strategy": "character_window", "size": chunk_size, "overlap": overlap},
        "index_backend": "faiss.IndexFlatIP",
        "index_path": _portable_path(output_dir / "documents.faiss"),
        "metadata_path": _portable_path(output_dir / "metadata.jsonl"),
        "documents_export": [documents_jsonl.name, documents_csv.name],
        "chunks_export": [chunks_jsonl.name, chunks_csv.name],
        "source_documents": documents,
        "chunk_count": len(chunks),
        "privacy": "local_only_user_review_required",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--owner-id", default=os.getenv("SAFELENS_OWNER_ID"))
    parser.add_argument("--org-id", default=os.getenv("SAFELENS_ORG_ID"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    args.source_dir.mkdir(parents=True, exist_ok=True)
    has_documents = any(_files(args.source_dir))
    if has_documents and not args.owner_id:
        parser.error("--owner-id or SAFELENS_OWNER_ID is required when documents are present")
    manifest = build_index(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        owner_id=args.owner_id or NIL_UUID,
        org_id=args.org_id,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
