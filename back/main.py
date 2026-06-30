"""
NexBot Backend — Hybrid RAG for Optimized C/C++ Microcontroller Code Generation
================================================================================
Improved Architecture v2 (Render-optimized — no torch/sentence-transformers):
  Persistence      : Supabase PostgreSQL — documents, embeddings, cache, history
  Dense retrieval   : pgvector HNSW index via Supabase RPC (replaces FAISS)
  Sparse retrieval : BM25 (rank_bm25, in-memory) + PostgreSQL Full-Text Search
  Fusion           : 3-way Reciprocal Rank Fusion (RRF)
  Boosting         : Tag-overlap boost on fused scores
  Embeddings       : Google Gemini text-embedding-004 (API-based, no local model)
  Generation       : Google Gemini (gemini-2.5-flash, free tier)
  Caching          : MD5 query-hash → Supabase query_cache (TTL 24 h)
  Analytics        : query_history table with latency tracking

Environment variables required:
  GEMINI_API_KEY         — Google AI Studio free API key (aistudio.google.com)
  SUPABASE_URL           — e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY   — service-role key (not anon key)

Dataset format (place files in ./dataset/):
  - JSON  : [{"title":"...", "description":"...", "code":"...", "tags":["uart","stm32",...]}]
  - .c/.h : Raw source files chunked automatically with overlap

SQL migrations required (run once via Supabase dashboard or CLI):
  See supabase/migrations/ in your project — both migration files must be applied.
"""

import os
import json
import re
import hashlib
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from supabase import create_client
from google import genai as ggenai
from google.genai import types
# Load .env from the project root (one level above this file) so we don't
# require environment variables to be set by the shell.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional if the user exports env vars
    pass

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR   = Path(__file__).parent / "dataset"
EMBED_MODEL   = "models/text-embedding-001"  # Gemini embedding model
EMBED_DIM     = 384   # matches pgvector column; Gemini supports output_dimensionality

TOP_K_DENSE   = 8    # candidates from pgvector
TOP_K_SPARSE  = 8    # candidates from BM25
TOP_K_FTS     = 8    # candidates from PostgreSQL FTS
TOP_K_FINAL   = 5    # docs fed to Gemini after RRF + boosting
RRF_K         = 60   # RRF constant (higher = less aggressive rank penalty)

CACHE_TTL_HOURS = 24        # query cache time-to-live
CHUNK_SIZE      = 60        # lines per C/H chunk
CHUNK_OVERLAP   = 10        # overlapping lines between chunks

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
# Free-tier model — check AI Studio for the current alias.
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are NexBot, an expert embedded-systems firmware engineer specialising \
in C and C++ for microcontrollers (STM32, AVR, ESP32, RP2040, RISC-V, etc.).

When given a user query and retrieved context snippets, you MUST:
1. Generate complete, compilable, well-structured C/C++ firmware code.
2. Use ST HAL / CMSIS / Arduino / ESP-IDF / Pico SDK APIs correctly for the target MCU.
3. Handle common pitfalls: volatile for ISR-shared variables, proper NVIC config, \
   DMA alignment, watchdog resets, and power-mode transitions.
4. Add terse inline comments only where logic is non-obvious.
5. After the code, output a "## Unit Tests" section with ≥3 Unity-framework or \
   CppUTest stubs covering key functions.
6. Keep explanations concise: 3–5 sentences before the code block.

Respond ONLY with a valid JSON object (no markdown fences, no preamble) using these keys:
  "title"       : short descriptive title (≤ 60 chars)
  "explanation" : concise explanation string (3–5 sentences)
  "code"        : complete firmware code string
  "unit_tests"  : unit-test code string
  "mcu_target"  : detected or inferred MCU family, e.g. "STM32F4", "ESP32", "Generic"
"""

# ── FastAPI app ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    1. Configure Gemini API.
    2. Load dataset files → upsert into Supabase (skips existing via on_conflict).
    3. Rebuild BM25 index from Supabase.
    """
    _configure_gemini()  # just sets API key, no model download

    docs = load_dataset()
    if docs:
        upserted = upsert_documents_to_supabase(docs)
        print(f"[startup] Upserted {upserted} documents to Supabase.")

    rebuild_bm25_from_supabase()
    print("[startup] NexBot v2 ready (lightweight mode — Gemini embeddings).")
    yield


app = FastAPI(title="NexBot Hybrid RAG Backend", version="2.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ──────────────────────────────────────────────────────────────────
supabase_client:           Optional[object]      = None
_gemini_configured:        bool                  = False

# In-memory BM25 (rebuilt from Supabase at startup / reload)
_bm25_docs:          list[dict]       = []
_bm25_index:         Optional[BM25Okapi] = None
_bm25_tokenized:     list[list[str]]  = []


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser for BM25."""
    return re.findall(r"\w+", text.lower())


def _doc_text(doc: dict) -> str:
    """Flatten a document dict into a single retrieval string."""
    parts = [
        doc.get("title", ""),
        doc.get("description", ""),
        " ".join(doc.get("tags", [])),
    ]
    code = doc.get("code", "")
    if code:
        parts.append(code[:400])
    return " ".join(p for p in parts if p)


def _query_hash(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _extract_mcu_tags(query: str) -> list[str]:
    """Extract MCU-related keywords from the query to boost tag matching."""
    patterns = [
        r"\b(stm32\w*)\b", r"\b(esp32\w*)\b", r"\b(avr\w*|atmega\w*|attiny\w*)\b",
        r"\b(rp2040|pico)\b", r"\b(nrf\d+\w*)\b", r"\b(uart|spi|i2c|can|usb|adc|dac|pwm|dma|rtos|freertos)\b",
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, query.lower()))
    return list(set(found))


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Embeddings (replaces sentence-transformers — zero local model loading)
# ─────────────────────────────────────────────────────────────────────────────
def _configure_gemini():
    global _gemini_client, _gemini_configured
    if not _gemini_configured:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is required.")
        _gemini_client = ggenai.Client(api_key=GEMINI_API_KEY)
        _gemini_configured = True
        print("[gemini] API configured (embedding + generation).")


def _embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embed a list of texts using Gemini text-embedding-004.
    Uses output_dimensionality=384 to match the existing pgvector column.

    task_type should be:
      - "RETRIEVAL_DOCUMENT" for indexing documents
      - "RETRIEVAL_QUERY"    for encoding search queries
    """
    _configure_gemini()

    embeddings = []
    # Gemini embedding API supports batches of up to 100 texts
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        )
        # result["embedding"] is a list of lists when input is a list
        if isinstance(result["embedding"][0], list):
            embeddings.extend(result["embedding"])
        else:
            # Single text input returns a flat list
            embeddings.append(result["embedding"])

        if len(texts) > batch_size:
            print(f"[embed] Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts …")

    return embeddings


def _embed_query(query: str) -> list[float]:
    """Embed a single query using the RETRIEVAL_QUERY task type."""
    result = _embed_texts([query], task_type="RETRIEVAL_QUERY")
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# Supabase initialisation
# ─────────────────────────────────────────────────────────────────────────────

def get_supabase():
    global supabase_client
    if supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise HTTPException(
                status_code=500,
                detail="SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required.",
            )
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return supabase_client


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading & chunking
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _chunk_c_file(path: Path) -> list[dict]:
    """
    Split a .c/.h file into overlapping line-window chunks.
    Uses a sliding window (CHUNK_SIZE lines, CHUNK_OVERLAP overlap) instead of
    the original regex split, producing more consistent chunk sizes and better
    context continuity across function boundaries.
    """
    lines   = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    docs    = []
    step    = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    i       = 0
    chunk_i = 0

    while i < len(lines):
        window = lines[i: i + CHUNK_SIZE]
        chunk  = "\n".join(window).strip()
        if len(chunk) >= 40:
            first_line = next((l.strip() for l in window if l.strip()), "")[:80]
            docs.append({
                "title":       f"{path.name} — chunk {chunk_i + 1}",
                "description": first_line,
                "code":        chunk,
                "tags":        [path.suffix.lstrip("."), path.stem.lower()],
                "source_file": str(path),
                "chunk_index": chunk_i,
            })
            chunk_i += 1
        i += step

    return docs


def load_dataset() -> list[dict]:
    docs = []
    if not DATASET_DIR.exists():
        print(f"[WARNING] Dataset directory not found: {DATASET_DIR}")
        return docs

    for json_path in DATASET_DIR.glob("**/*.json"):
        try:
            loaded = _load_json_file(json_path)
            for idx, d in enumerate(loaded):
                d.setdefault("source_file", str(json_path))
                d.setdefault("chunk_index", idx)
            docs.extend(loaded)
            print(f"[dataset] Loaded {len(loaded)} entries from {json_path.name}")
        except Exception as e:
            print(f"[dataset] Failed to parse {json_path.name}: {e}")

    for c_path in list(DATASET_DIR.glob("**/*.c")) + list(DATASET_DIR.glob("**/*.h")):
        try:
            chunks = _chunk_c_file(c_path)
            docs.extend(chunks)
            print(f"[dataset] Chunked {len(chunks)} segments from {c_path.name}")
        except Exception as e:
            print(f"[dataset] Failed to chunk {c_path.name}: {e}")

    # Compute retrieval text for all docs
    for d in docs:
        d["text"] = _doc_text(d)

    print(f"[dataset] Total documents to index: {len(docs)}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Supabase document upsert + embedding
# ─────────────────────────────────────────────────────────────────────────────

def upsert_documents_to_supabase(docs: list[dict]) -> int:
    """
    Encode documents and upsert into Supabase.
    Deduplication key: (source_file, chunk_index).
    Returns the number of rows upserted.
    """
    if not docs:
        return 0

    sb = get_supabase()

    print(f"[supabase] Encoding {len(docs)} documents via Gemini embeddings …")
    texts      = [d["text"] for d in docs]
    embeddings = _embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    rows = []
    for i, doc in enumerate(docs):
        rows.append({
            "title":       doc.get("title", ""),
            "description": doc.get("description", ""),
            "code":        doc.get("code", ""),
            "tags":        doc.get("tags", []),
            "text":        doc["text"],
            "source_file": doc.get("source_file", ""),
            "chunk_index": doc.get("chunk_index", 0),
            "embedding":   embeddings[i],
        })

    # Upsert in batches of 100 to avoid request-size limits
    batch_size = 100
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start: start + batch_size]
        sb.table("documents").upsert(
            batch,
            on_conflict="source_file,chunk_index",
        ).execute()
        total += len(batch)
        print(f"[supabase] Upserted {total}/{len(rows)} documents …")

    return total


# ─────────────────────────────────────────────────────────────────────────────
# BM25 in-memory index (seeded from Supabase)
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_bm25_from_supabase():
    """Load all document texts from Supabase and rebuild the BM25 index."""
    global _bm25_docs, _bm25_index, _bm25_tokenized

    sb   = get_supabase()
    resp = sb.table("documents").select("id,title,description,code,tags,text").execute()
    _bm25_docs = resp.data or []

    if not _bm25_docs:
        _bm25_index     = None
        _bm25_tokenized = []
        print("[bm25] No documents in Supabase — BM25 disabled.")
        return

    _bm25_tokenized = [_tokenize(d["text"]) for d in _bm25_docs]
    _bm25_index     = BM25Okapi(_bm25_tokenized)
    print(f"[bm25] Index rebuilt with {len(_bm25_docs)} documents.")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval — three-way hybrid with RRF fusion
# ─────────────────────────────────────────────────────────────────────────────

def _dense_retrieve(query: str) -> list[tuple[int, float]]:
    """
    pgvector HNSW cosine similarity search via Supabase RPC.
    Returns list of (supabase_row_id, similarity).
    """
    q_emb   = _embed_query(query)
    sb      = get_supabase()
    resp    = sb.rpc("match_documents", {
        "query_embedding": q_emb,
        "match_count":     TOP_K_DENSE,
        "match_threshold": 0.20,
    }).execute()
    return [(int(row["id"]), float(row["similarity"])) for row in (resp.data or [])]


def _fts_retrieve(query: str) -> list[tuple[int, float]]:
    """
    PostgreSQL full-text search via Supabase RPC.
    Returns list of (supabase_row_id, ts_rank).
    """
    sb   = get_supabase()
    resp = sb.rpc("fts_documents", {
        "query_text":  query,
        "match_count": TOP_K_FTS,
    }).execute()
    return [(int(row["id"]), float(row["rank"])) for row in (resp.data or [])]


def _bm25_retrieve(query: str) -> list[tuple[int, float]]:
    """
    In-memory BM25 search.
    Returns list of (supabase_row_id, bm25_score).
    """
    if _bm25_index is None or not _bm25_docs:
        return []
    tokens      = _tokenize(query)
    bm25_scores = _bm25_index.get_scores(tokens)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_SPARSE]
    return [
        (int(_bm25_docs[i]["id"]), float(bm25_scores[i]))
        for i in top_indices
        if bm25_scores[i] > 0
    ]


def _rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> dict[int, float]:
    """Reciprocal Rank Fusion across multiple ranked lists of (doc_id, score)."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _tag_boost(
    fused_scores: dict[int, float],
    docs_by_id:   dict[int, dict],
    query_tags:   list[str],
    boost:        float = 0.05,
) -> dict[int, float]:
    """Add a small bonus for each matching tag between query and document."""
    if not query_tags:
        return fused_scores
    boosted = {}
    for doc_id, score in fused_scores.items():
        doc       = docs_by_id.get(doc_id, {})
        doc_tags  = [t.lower() for t in doc.get("tags", [])]
        overlap   = len(set(query_tags) & set(doc_tags))
        boosted[doc_id] = score + overlap * boost
    return boosted


def retrieve_hybrid(query: str, top_k_final: int = TOP_K_FINAL) -> list[dict]:
    """
    Three-way hybrid retrieval:
      1. Dense   — pgvector HNSW (semantic)
      2. Sparse  — BM25 in-memory (lexical)
      3. FTS     — PostgreSQL full-text search (lexical + stemming)
    Fused with RRF, then boosted by tag overlap.
    """
    dense_results = _dense_retrieve(query)
    bm25_results  = _bm25_retrieve(query)
    fts_results   = _fts_retrieve(query)

    fused = _rrf_fuse([dense_results, bm25_results, fts_results])

    if not fused:
        return []

    # Fetch full doc data for all candidates from Supabase
    all_ids = list(fused.keys())
    sb      = get_supabase()
    resp    = sb.table("documents").select(
        "id,title,description,code,tags,text"
    ).in_("id", all_ids).execute()

    docs_by_id = {int(d["id"]): d for d in (resp.data or [])}

    # Tag boost
    query_tags = _extract_mcu_tags(query)
    fused      = _tag_boost(fused, docs_by_id, query_tags)

    # Sort and return top-k
    ranked_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k_final]
    return [docs_by_id[doc_id] for doc_id in ranked_ids if doc_id in docs_by_id]


# ─────────────────────────────────────────────────────────────────────────────
# Query cache
# ─────────────────────────────────────────────────────────────────────────────

def cache_lookup(query: str) -> Optional[dict]:
    qhash = _query_hash(query)
    sb    = get_supabase()
    try:
        resp = sb.table("query_cache").select("response").eq("query_hash", qhash).limit(1).execute()
        if resp.data:
            try:
                sb.rpc("increment_cache_hit", {"qhash": qhash}).execute()
            except Exception:
                sb.table("query_cache").update({"last_accessed": "now()"}).eq("query_hash", qhash).execute()
            return resp.data[0]["response"]
    except Exception:
        pass
    return None


def cache_store(query: str, response: dict):
    qhash = _query_hash(query)
    sb    = get_supabase()
    try:
        sb.table("query_cache").upsert({
            "query_hash": qhash,
            "query_text": query,
            "response":   response,
        }, on_conflict="query_hash").execute()
    except Exception as e:
        print(f"[cache] Store failed: {e}")


def log_query_history(
    query: str,
    retrieved_doc_ids: list[int],
    response: dict,
    cache_hit: bool,
    latency_ms: int,
    top_k: int,
    session_id: Optional[str] = None,
):
    sb = get_supabase()
    try:
        row = {
            "query":                query,
            "retrieved_doc_ids":    retrieved_doc_ids,
            "response_title":       response.get("title", ""),
            "response_explanation": response.get("explanation", ""),
            "response_code":        response.get("code", ""),
            "response_unit_tests":  response.get("unit_tests", ""),
            "response_mcu_target":  response.get("mcu_target", "Generic"),
            "cache_hit":            cache_hit,
            "latency_ms":           latency_ms,
            "top_k":                top_k,
        }
        if session_id:
            row["session_id"] = session_id
        sb.table("query_history").insert(row).execute()
    except Exception as e:
        print(f"[history] Log failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Gemini generation
# ─────────────────────────────────────────────────────────────────────────────

def build_context_block(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""
    parts = ["### Retrieved Context\n"]
    for i, doc in enumerate(retrieved, 1):
        parts.append(f"**[{i}] {doc.get('title', 'Snippet')}**")
        if doc.get("description"):
            parts.append(f"_{doc['description']}_")
        tags = doc.get("tags", [])
        if tags:
            parts.append(f"Tags: `{'`, `'.join(tags)}`")
        if doc.get("code"):
            code_preview = doc["code"][:1000]
            parts.append(f"```c\n{code_preview}\n```")
        parts.append("")
    return "\n".join(parts)


def generate_with_gemini(query: str, context: str) -> dict:
    _configure_gemini()

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )

    user_message = (
        f"{context}\n\n---\n\n"
        f"**User Request:** {query}\n\n"
        "Respond with ONLY a valid JSON object. No markdown fences, no extra text."
    )

    response = None
    for attempt in range(3):
        try:
            response = model.generate_content(user_message)
            break
        except Exception as e:
            if attempt == 2:
                raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
            time.sleep(1.5 * (attempt + 1))

    raw = response.text.strip() if response else ""

    json_fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if json_fence:
        raw = json_fence.group(1)

    try:
        parsed = json.loads(raw)
        return {
            "title":       parsed.get("title", "Generated Firmware"),
            "explanation": parsed.get("explanation", ""),
            "code":        parsed.get("code", ""),
            "unit_tests":  parsed.get("unit_tests", ""),
            "mcu_target":  parsed.get("mcu_target", "Generic"),
        }
    except json.JSONDecodeError:
        return {
            "title":       "Generated Firmware",
            "explanation": raw,
            "code":        "",
            "unit_tests":  "",
            "mcu_target":  "Generic",
        }


# ─────────────────────────────────────────────────────────────────────────────
# API schemas
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str
    top_k:      int  = Field(default=TOP_K_FINAL, ge=1, le=20)
    use_cache:  bool = True
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    title:       str
    explanation: str
    code:        str
    unit_tests:  str
    mcu_target:  str
    sources:     list[dict]
    cache_hit:   bool
    latency_ms:  int


class UpsertRequest(BaseModel):
    documents: list[dict]


class DeleteResponse(BaseModel):
    deleted: bool
    id:      int


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    sb       = get_supabase()
    count    = sb.table("documents").select("id", count="exact").execute()
    doc_count = count.count if hasattr(count, "count") else len(count.data or [])
    return {
        "status":        "ok",
        "documents":     doc_count,
        "bm25_loaded":   _bm25_index is not None,
        "bm25_docs":     len(_bm25_docs),
        "embed_model":   EMBED_MODEL,
        "gemini_model":  GEMINI_MODEL,
        "version":       "2.1.0",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    t0        = time.monotonic()
    cache_hit = False

    # 1. Cache lookup
    if req.use_cache:
        cached = cache_lookup(req.query)
        if cached:
            cache_hit  = True
            latency_ms = int((time.monotonic() - t0) * 1000)
            background_tasks.add_task(
                log_query_history, req.query, [], cached,
                True, latency_ms, req.top_k, req.session_id,
            )
            return QueryResponse(
                cache_hit=True,
                latency_ms=latency_ms,
                sources=[],
                **cached,
            )

    # 2. Hybrid retrieval
    retrieved = retrieve_hybrid(req.query, top_k_final=req.top_k)

    # 3. Build context
    context = build_context_block(retrieved)

    # 4. Generate with Gemini
    result = generate_with_gemini(req.query, context)

    latency_ms = int((time.monotonic() - t0) * 1000)

    # 5. Sources metadata
    sources = [
        {"id": d["id"], "title": d.get("title", ""), "tags": d.get("tags", [])}
        for d in retrieved
    ]

    # 6. Cache store + history log (background, non-blocking)
    if req.use_cache:
        background_tasks.add_task(cache_store, req.query, result)

    doc_ids = [int(d["id"]) for d in retrieved]
    background_tasks.add_task(
        log_query_history, req.query, doc_ids,
        result, False, latency_ms, req.top_k, req.session_id,
    )

    return QueryResponse(
        **result,
        sources=sources,
        cache_hit=False,
        latency_ms=latency_ms,
    )


@app.get("/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    sb   = get_supabase()
    resp = sb.table("documents").select(
        "id,title,tags,source_file,chunk_index,created_at"
    ).range(offset, offset + limit - 1).order("id").execute()

    count_resp = sb.table("documents").select("id", count="exact").execute()
    total = count_resp.count if hasattr(count_resp, "count") else len(count_resp.data or [])

    return {"total": total, "offset": offset, "limit": limit, "documents": resp.data}


@app.post("/documents/upsert")
async def upsert_documents(req: UpsertRequest):
    """Add or update documents via the API (no need for dataset files)."""
    docs = req.documents
    for d in docs:
        d["text"] = _doc_text(d)
    upserted = upsert_documents_to_supabase(docs)
    rebuild_bm25_from_supabase()
    return {"status": "ok", "upserted": upserted}


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: int):
    sb   = get_supabase()
    resp = sb.table("documents").delete().eq("id", doc_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    rebuild_bm25_from_supabase()
    return DeleteResponse(deleted=True, id=doc_id)


@app.post("/reload")
async def reload_dataset():
    """Hot-reload dataset files and rebuild all indices."""
    docs     = load_dataset()
    upserted = upsert_documents_to_supabase(docs)
    rebuild_bm25_from_supabase()
    return {"status": "reloaded", "upserted": upserted, "bm25_docs": len(_bm25_docs)}


@app.get("/stats")
async def stats():
    """Query analytics from history table."""
    sb = get_supabase()

    total_resp   = sb.table("query_history").select("id", count="exact").execute()
    total        = total_resp.count if hasattr(total_resp, "count") else 0

    cache_resp   = sb.table("query_history").select("id", count="exact").eq("cache_hit", True).execute()
    cache_hits   = cache_resp.count if hasattr(cache_resp, "count") else 0

    recent_resp  = sb.table("query_history").select(
        "query,response_title,cache_hit,latency_ms,created_at"
    ).order("created_at", desc=True).limit(10).execute()

    return {
        "total_queries":   total,
        "cache_hits":      cache_hits,
        "cache_hit_rate":  round(cache_hits / total, 3) if total else 0,
        "recent_queries":  recent_resp.data or [],
    }


@app.delete("/cache")
async def clear_cache():
    """Clear the entire query cache."""
    sb      = get_supabase()
    resp    = sb.table("query_cache").delete().neq("id", 0).execute()
    cleared = len(resp.data) if resp.data else 0
    return {"status": "cleared", "entries_removed": cleared}


# ─────────────────────────────────────────────────────────────────────────────
# Chat sessions (grouped query history)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/history/sessions")
async def list_chat_sessions(limit: int = 50):
    sb = get_supabase()
    resp = sb.table("query_history").select(
        "id,query,response_title,cache_hit,latency_ms,created_at,session_id"
    ).order("created_at", desc=True).limit(500).execute()

    rows = resp.data or []
    sessions: dict[str, dict] = {}
    for r in rows:
        sid = r.get("session_id")
        if not sid:
            continue
        s = sessions.setdefault(sid, {
            "session_id":    sid,
            "title":         "",
            "message_count": 0,
            "last_active":   r.get("created_at"),
            "preview":       "",
            "queries":       [],
        })
        s["message_count"] += 1
        s["queries"].append(r)
        if not s["preview"]:
            s["preview"] = r.get("response_title") or r.get("query", "")

    out = []
    for s in sessions.values():
        s["queries"].reverse()
        oldest = s["queries"][0]
        title_src = (oldest.get("query") or "").strip()
        s["title"] = (title_src[:60] + "…") if len(title_src) > 60 else (title_src or "Untitled")
        out.append(s)

    out.sort(key=lambda x: x["last_active"] or "", reverse=True)
    return {"sessions": out[:limit]}


@app.get("/history/sessions/{session_id}")
async def get_chat_session(session_id: str):
    sb = get_supabase()
    resp = sb.table("query_history").select(
        "id,query,response_title,response_explanation,response_code,"
        "response_unit_tests,response_mcu_target,cache_hit,latency_ms,"
        "created_at,session_id"
    ).eq("session_id", session_id).order("created_at", desc=False).execute()

    rows = resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    messages = []
    for r in rows:
        messages.append({
            "role":        "user",
            "content":     r.get("query", ""),
            "created_at":  r.get("created_at"),
            "cache_hit":   bool(r.get("cache_hit")),
            "latency_ms":  r.get("latency_ms", 0),
        })
        messages.append({
            "role":        "bot",
            "title":       r.get("response_title", ""),
            "content":     r.get("response_explanation", ""),
            "code":        r.get("response_code", ""),
            "unit_tests":  r.get("response_unit_tests", ""),
            "mcu_target":  r.get("response_mcu_target", "Generic"),
            "created_at":  r.get("created_at"),
        })

    return {
        "session_id":    session_id,
        "title":         (rows[0].get("query", "")[:60] + "…") if len(rows[0].get("query", "")) > 60 else rows[0].get("query", "Untitled"),
        "message_count": len(rows),
        "last_active":   rows[-1].get("created_at"),
        "messages":      messages,
    }


@app.delete("/history/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    sb = get_supabase()
    resp = sb.table("query_history").delete().eq("session_id", session_id).execute()
    deleted = len(resp.data or [])
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return {"deleted": True, "id": session_id, "rows": deleted}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
