-- ============================================================================
-- NexBot — chat sessions
-- ============================================================================
-- Adds a session_id column to query_history so individual queries can be
-- grouped into chat threads. The default (gen_random_uuid) means existing
-- rows become singleton sessions automatically — we make no attempt to
-- retroactively merge past queries into multi-message chats.
--
-- Run once in Supabase: SQL Editor → New query → paste → Run.
-- ============================================================================

alter table public.query_history
    add column if not exists session_id uuid not null default gen_random_uuid();

create index if not exists query_history_session_id_idx
    on public.query_history (session_id, created_at);