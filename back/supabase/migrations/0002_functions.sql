-- ============================================================================
-- NexBot — RPC functions called from main.py
-- ============================================================================
-- These match the call sites:
--   sb.rpc("match_documents", {query_embedding, match_count, match_threshold})
--   sb.rpc("fts_documents",   {query_text, match_count})
--   sb.rpc("increment_cache_hit", {qhash})
-- ============================================================================

-- ── Dense retrieval: pgvector cosine similarity via HNSW ───────────────────
create or replace function public.match_documents(
    query_embedding vector(384),
    match_count     int    default 8,
    match_threshold float  default 0.20
)
returns table (id bigint, similarity float)
language sql stable
as $$
    select
        d.id,
        1 - (d.embedding <=> query_embedding) as similarity
    from public.documents d
    where d.embedding is not null
      and 1 - (d.embedding <=> query_embedding) > match_threshold
    order by d.embedding <=> query_embedding
    limit match_count;
$$;

-- ── Sparse retrieval: PostgreSQL full-text search on `text` column ─────────
create or replace function public.fts_documents(
    query_text  text,
    match_count int default 8
)
returns table (id bigint, rank float)
language sql stable
as $$
    with q as (select websearch_to_tsquery('english', query_text) as tsq)
    select
        d.id,
        ts_rank_cd(to_tsvector('english', coalesce(d.text, '')), q.tsq) as rank
    from public.documents d, q
    where to_tsvector('english', coalesce(d.text, '')) @@ q.tsq
    order by rank desc
    limit match_count;
$$;

-- ── Atomic cache hit increment (replaces read-then-write in main.py) ──────
create or replace function public.increment_cache_hit(qhash text)
returns void
language sql
as $$
    update public.query_cache
       set hit_count     = hit_count + 1,
           last_accessed = now()
     where query_hash = qhash;
$$;
