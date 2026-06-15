-- USER-RUN LOADER ONLY.
-- Run supabase_schema_proposal.sql first, review the CSV files, then execute
-- the two \copy commands from psql under the user's own credentials.

begin;

create extension if not exists vector;

alter table document_chunks
  add column if not exists embedding vector(384);

do $$
declare
  embedding_type text;
begin
  if to_regclass('document_chunks') is null or to_regclass('documents') is null then
    raise exception 'documents and document_chunks must exist before loading RAG exports';
  end if;
  select format_type(a.atttypid, a.atttypmod)
  into embedding_type
  from pg_attribute a
  where a.attrelid = 'document_chunks'::regclass
    and a.attname = 'embedding'
    and not a.attisdropped;
  if embedding_type <> 'vector(384)' then
    raise exception 'document_chunks.embedding must be vector(384), found %', embedding_type;
  end if;
end $$;

create index if not exists document_chunks_embedding_hnsw_idx
  on document_chunks
  using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create unique index if not exists document_chunks_document_chunk_idx
  on document_chunks(document_id, chunk_index);

create temporary table staged_documents (
  id uuid not null,
  owner_id uuid not null,
  org_id text null,
  title text not null,
  document_type text not null,
  storage_path text null,
  status text not null
) on commit drop;

create temporary table staged_document_chunks (
  owner_id uuid not null,
  org_id text null,
  document_id uuid not null,
  chunk_index integer not null,
  content text not null,
  embedding_json text not null,
  metadata text not null
) on commit drop;

-- Run these from psql after replacing the paths:
-- \copy staged_documents(id,owner_id,org_id,title,document_type,storage_path,status)
--   from 'rag/documents_export.csv' with (format csv, header true);
-- \copy staged_document_chunks(owner_id,org_id,document_id,chunk_index,content,embedding_json,metadata)
--   from 'rag/chunks_export.csv' with (format csv, header true);

insert into documents (
  id, owner_id, org_id, title, document_type, storage_path, status
)
select
  id,
  owner_id,
  nullif(org_id, '')::uuid,
  title,
  document_type,
  nullif(storage_path, ''),
  status
from staged_documents
on conflict (id) do update
set
  owner_id = excluded.owner_id,
  org_id = excluded.org_id,
  title = excluded.title,
  document_type = excluded.document_type,
  storage_path = excluded.storage_path,
  status = excluded.status;

insert into document_chunks (
  owner_id, org_id, document_id, chunk_index, content, embedding, metadata
)
select
  owner_id,
  nullif(org_id, '')::uuid,
  document_id,
  chunk_index,
  content,
  embedding_json::vector(384),
  metadata::jsonb
from staged_document_chunks
on conflict (document_id, chunk_index) do update
set
  owner_id = excluded.owner_id,
  org_id = excluded.org_id,
  content = excluded.content,
  embedding = excluded.embedding,
  metadata = excluded.metadata;

commit;
