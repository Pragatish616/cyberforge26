-- ============================================================================
-- NexBot — persist bot responses on query_history
-- ============================================================================
-- Until now the bot's explanation / code / unit_tests / mcu_target were
-- discarded after each query; the frontend's history-replay fell back to
-- re-firing /query and hoping the 24h query_cache still had the original
-- answer. That broke for any session older than a day.
--
-- This migration adds four text columns to query_history so we can render
-- the full thread from a single GET, with no replay and no cache dependency.
--
-- All columns default to '' / 'Generic' so existing rows remain valid.
-- Run once in Supabase: SQL Editor → New query → paste → Run.
-- ============================================================================

alter table public.query_history
    add column if not exists response_explanation text not null default '',
    add column if not exists response_code        text not null default '',
    add column if not exists response_unit_tests  text not null default '',
    add column if not exists response_mcu_target  text not null default 'Generic';
