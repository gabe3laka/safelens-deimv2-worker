-- USER-RUN PROPOSAL ONLY.
-- OpenClaw/Codex did not connect to or write to Supabase.
-- Review in a staging project and take a backup before applying.

begin;

create extension if not exists vector;

-- Additive tenant-readiness for the existing SafeLens tables. These statements
-- do not change existing primary keys, ownership columns, enums, or policies.
alter table if exists profiles add column if not exists org_id uuid null;
alter table if exists alert_settings add column if not exists org_id uuid null;
alter table if exists monitoring_sessions add column if not exists org_id uuid null;
alter table if exists detections add column if not exists org_id uuid null;
alter table if exists incidents add column if not exists org_id uuid null;
alter table if exists hazard_zones add column if not exists org_id uuid null;
alter table if exists blueprints add column if not exists org_id uuid null;
alter table if exists risk_register add column if not exists org_id uuid null;
alter table if exists risk_actions add column if not exists org_id uuid null;
alter table if exists compliance_items add column if not exists org_id uuid null;

create table if not exists company_profiles (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  name text not null,
  industry text,
  risk_matrix_size int not null default 5 check (risk_matrix_size in (3, 4, 5)),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists site_profiles (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  company_profile_id uuid references company_profiles(id) on delete cascade,
  name text not null,
  site_type text,
  created_at timestamptz not null default now()
);

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  site_profile_id uuid null references site_profiles(id) on delete set null,
  title text not null,
  document_type text not null,
  storage_path text,
  status text not null default 'pending'
    check (status in ('pending', 'indexed', 'rejected', 'archived')),
  created_at timestamptz not null default now()
);

create table if not exists document_chunks (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  document_id uuid not null references documents(id) on delete cascade,
  chunk_index int not null check (chunk_index >= 0),
  content text not null,
  embedding vector(384),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create table if not exists agent_memory (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  site_profile_id uuid null references site_profiles(id) on delete set null,
  memory_type text not null,
  content jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists agent_actions_log (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  session_id text,
  action_type text not null,
  status text not null
    check (status in ('preview', 'pending_approval', 'approved', 'rejected', 'held', 'executed', 'logged', 'failed')),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists approval_records (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  action_log_id uuid references agent_actions_log(id) on delete cascade,
  approver_id uuid,
  decision text not null check (decision in ('approve', 'reject', 'revise')),
  notes text,
  decided_at timestamptz not null default now()
);

create table if not exists dataset_candidates (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  site_profile_id uuid null references site_profiles(id) on delete set null,
  source_ref text not null,
  hazard_tags text[] not null default '{}',
  review_status text not null default 'pending'
    check (review_status in ('pending', 'approved', 'rejected', 'exported')),
  privacy_flags jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists model_versions (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  name text not null,
  model_family text not null,
  weights_uri text,
  status text not null default 'draft'
    check (status in ('draft', 'evaluating', 'pending_approval', 'approved', 'rejected', 'active', 'retired')),
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists model_evaluations (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  org_id uuid null,
  model_version_id uuid not null references model_versions(id) on delete cascade,
  eval_name text not null,
  metrics jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists company_profiles_owner_idx on company_profiles(owner_id);
create index if not exists site_profiles_owner_idx on site_profiles(owner_id);
create index if not exists documents_owner_idx on documents(owner_id);
create index if not exists document_chunks_owner_idx on document_chunks(owner_id);
create unique index if not exists document_chunks_document_chunk_idx
  on document_chunks(document_id, chunk_index);
create index if not exists agent_memory_owner_idx on agent_memory(owner_id);
create index if not exists agent_actions_log_owner_idx on agent_actions_log(owner_id);
create index if not exists approval_records_owner_idx on approval_records(owner_id);
create index if not exists dataset_candidates_owner_idx on dataset_candidates(owner_id);
create index if not exists model_versions_owner_idx on model_versions(owner_id);
create index if not exists model_evaluations_owner_idx on model_evaluations(owner_id);

create index if not exists company_profiles_org_idx on company_profiles(org_id);
create index if not exists site_profiles_org_idx on site_profiles(org_id);
create index if not exists documents_org_idx on documents(org_id);
create index if not exists document_chunks_org_idx on document_chunks(org_id);
create index if not exists agent_memory_org_idx on agent_memory(org_id);
create index if not exists agent_actions_log_org_idx on agent_actions_log(org_id);
create index if not exists approval_records_org_idx on approval_records(org_id);
create index if not exists dataset_candidates_org_idx on dataset_candidates(org_id);
create index if not exists model_versions_org_idx on model_versions(org_id);
create index if not exists model_evaluations_org_idx on model_evaluations(org_id);

alter table company_profiles enable row level security;
alter table site_profiles enable row level security;
alter table documents enable row level security;
alter table document_chunks enable row level security;
alter table agent_memory enable row level security;
alter table agent_actions_log enable row level security;
alter table approval_records enable row level security;
alter table dataset_candidates enable row level security;
alter table model_versions enable row level security;
alter table model_evaluations enable row level security;

-- Policies are intentionally owner-scoped. org_id is present for a later
-- organization-membership policy once the application defines its membership
-- table and role model; guessing that contract here would weaken isolation.
drop policy if exists company_profiles_select_own on company_profiles;
drop policy if exists company_profiles_insert_own on company_profiles;
drop policy if exists company_profiles_update_own on company_profiles;
drop policy if exists company_profiles_delete_own on company_profiles;
create policy company_profiles_select_own on company_profiles for select using (auth.uid() = owner_id);
create policy company_profiles_insert_own on company_profiles for insert with check (auth.uid() = owner_id);
create policy company_profiles_update_own on company_profiles for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy company_profiles_delete_own on company_profiles for delete using (auth.uid() = owner_id);

drop policy if exists site_profiles_select_own on site_profiles;
drop policy if exists site_profiles_insert_own on site_profiles;
drop policy if exists site_profiles_update_own on site_profiles;
drop policy if exists site_profiles_delete_own on site_profiles;
create policy site_profiles_select_own on site_profiles for select using (auth.uid() = owner_id);
create policy site_profiles_insert_own on site_profiles for insert with check (auth.uid() = owner_id);
create policy site_profiles_update_own on site_profiles for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy site_profiles_delete_own on site_profiles for delete using (auth.uid() = owner_id);

drop policy if exists documents_select_own on documents;
drop policy if exists documents_insert_own on documents;
drop policy if exists documents_update_own on documents;
drop policy if exists documents_delete_own on documents;
create policy documents_select_own on documents for select using (auth.uid() = owner_id);
create policy documents_insert_own on documents for insert with check (auth.uid() = owner_id);
create policy documents_update_own on documents for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy documents_delete_own on documents for delete using (auth.uid() = owner_id);

drop policy if exists document_chunks_select_own on document_chunks;
drop policy if exists document_chunks_insert_own on document_chunks;
drop policy if exists document_chunks_update_own on document_chunks;
drop policy if exists document_chunks_delete_own on document_chunks;
create policy document_chunks_select_own on document_chunks for select using (auth.uid() = owner_id);
create policy document_chunks_insert_own on document_chunks for insert with check (auth.uid() = owner_id);
create policy document_chunks_update_own on document_chunks for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy document_chunks_delete_own on document_chunks for delete using (auth.uid() = owner_id);

drop policy if exists agent_memory_select_own on agent_memory;
drop policy if exists agent_memory_insert_own on agent_memory;
drop policy if exists agent_memory_update_own on agent_memory;
drop policy if exists agent_memory_delete_own on agent_memory;
create policy agent_memory_select_own on agent_memory for select using (auth.uid() = owner_id);
create policy agent_memory_insert_own on agent_memory for insert with check (auth.uid() = owner_id);
create policy agent_memory_update_own on agent_memory for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy agent_memory_delete_own on agent_memory for delete using (auth.uid() = owner_id);

drop policy if exists agent_actions_log_select_own on agent_actions_log;
drop policy if exists agent_actions_log_insert_own on agent_actions_log;
drop policy if exists agent_actions_log_update_own on agent_actions_log;
drop policy if exists agent_actions_log_delete_own on agent_actions_log;
create policy agent_actions_log_select_own on agent_actions_log for select using (auth.uid() = owner_id);
create policy agent_actions_log_insert_own on agent_actions_log for insert with check (auth.uid() = owner_id);
create policy agent_actions_log_update_own on agent_actions_log for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy agent_actions_log_delete_own on agent_actions_log for delete using (auth.uid() = owner_id);

drop policy if exists approval_records_select_own on approval_records;
drop policy if exists approval_records_insert_own on approval_records;
drop policy if exists approval_records_update_own on approval_records;
drop policy if exists approval_records_delete_own on approval_records;
create policy approval_records_select_own on approval_records for select using (auth.uid() = owner_id);
create policy approval_records_insert_own on approval_records for insert with check (auth.uid() = owner_id);
create policy approval_records_update_own on approval_records for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy approval_records_delete_own on approval_records for delete using (auth.uid() = owner_id);

drop policy if exists dataset_candidates_select_own on dataset_candidates;
drop policy if exists dataset_candidates_insert_own on dataset_candidates;
drop policy if exists dataset_candidates_update_own on dataset_candidates;
drop policy if exists dataset_candidates_delete_own on dataset_candidates;
create policy dataset_candidates_select_own on dataset_candidates for select using (auth.uid() = owner_id);
create policy dataset_candidates_insert_own on dataset_candidates for insert with check (auth.uid() = owner_id);
create policy dataset_candidates_update_own on dataset_candidates for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy dataset_candidates_delete_own on dataset_candidates for delete using (auth.uid() = owner_id);

drop policy if exists model_versions_select_own on model_versions;
drop policy if exists model_versions_insert_own on model_versions;
drop policy if exists model_versions_update_own on model_versions;
drop policy if exists model_versions_delete_own on model_versions;
create policy model_versions_select_own on model_versions for select using (auth.uid() = owner_id);
create policy model_versions_insert_own on model_versions for insert with check (auth.uid() = owner_id);
create policy model_versions_update_own on model_versions for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy model_versions_delete_own on model_versions for delete using (auth.uid() = owner_id);

drop policy if exists model_evaluations_select_own on model_evaluations;
drop policy if exists model_evaluations_insert_own on model_evaluations;
drop policy if exists model_evaluations_update_own on model_evaluations;
drop policy if exists model_evaluations_delete_own on model_evaluations;
create policy model_evaluations_select_own on model_evaluations for select using (auth.uid() = owner_id);
create policy model_evaluations_insert_own on model_evaluations for insert with check (auth.uid() = owner_id);
create policy model_evaluations_update_own on model_evaluations for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy model_evaluations_delete_own on model_evaluations for delete using (auth.uid() = owner_id);

commit;
