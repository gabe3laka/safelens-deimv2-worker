# Local RAG Index

Place approved company HSE documents in `source-documents/`, then run:

```bash
pip install sentence-transformers faiss-cpu numpy pypdf python-docx
python build_local_index.py --owner-id <user-uuid>
```

Outputs when documents are supplied:

- `index/documents.faiss`: local cosine-similarity index
- `index/metadata.jsonl`: chunk metadata for retrieval
- `documents_export.jsonl` / `.csv`: parent document rows
- `chunks_export.jsonl` / `.csv`: chunk rows with `embedding_json`
- `embeddings_manifest.json`: model, license, dimensions, chunking, and source hashes

The local model and pgvector export both use the native 384 dimensions. The CSV
columns match the staging tables in `db/pgvector_loader.sql`.

OpenClaw/Codex does not upload documents or embeddings. The user may load a
reviewed export with `db/pgvector_loader.sql`.
