-- ============================================================================
-- NexBot — initial schema
-- ============================================================================
-- Run once in Supabase: SQL Editor → New query → paste → Run.
-- Requires the pgvector extension (Supabase has it pre-installed).
-- ============================================================================

create extension if not exists vector;

-- ── documents : source material + 384-dim embeddings ───────────────────────
create table if not exists public.documents (
    id          bigserial primary key,
    title       text        not null default '',
    description text        not null default '',
    code        text        not null default '',
    tags        text[]      not null default '{}',
    text        text        not null default '',
    source_file text        not null default '',
    chunk_index int         not null default 0,
    embedding   vector(384),
    created_at  timestamptz not null default now(),

    -- Deduplication key: same (source_file, chunk_index) ⇒ upsert
    constraint documents_source_chunk_unique
        unique (source_file, chunk_index)
);

create index if not exists documents_embedding_hnsw_idx
    on public.documents
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

create index if not exists documents_tags_gin_idx
    on public.documents
    using gin (tags);

-- ── query_cache : MD5(query) → response JSON, TTL 24 h ─────────────────────
create table if not exists public.query_cache (
    id            bigserial primary key,
    query_hash    text        not null unique,
    query_text    text        not null,
    response      jsonb       not null,
    hit_count     int         not null default 1,
    created_at    timestamptz not null default now(),
    last_accessed timestamptz not null default now()
);

create index if not exists query_cache_last_accessed_idx
    on public.query_cache (last_accessed desc);

-- ── query_history : analytics ───────────────────────────────────────────────
create table if not exists public.query_history (
    id                bigserial primary key,
    query             text        not null,
    retrieved_doc_ids bigint[]    not null default '{}',
    response_title    text        not null default '',
    cache_hit         boolean     not null default false,
    latency_ms        int         not null default 0,
    top_k             int         not null default 5,
    created_at        timestamptz not null default now()
);

create index if not exists query_history_created_at_idx
    on public.query_history (created_at desc);

create index if not exists query_history_cache_hit_idx
    on public.query_history (cache_hit);
