# NexBot — Hybrid RAG for Embedded Firmware Generation

NexBot is a chatbot that generates C/C++ microcontroller firmware. It uses a
hybrid retrieval pipeline (BM25 + pgvector + Postgres FTS) over a corpus of
embedded-systems reference code, fused with Reciprocal Rank Fusion, then asks
Google Gemini to produce a complete, compilable firmware snippet plus unit tests.

```
┌────────────┐    POST /query     ┌──────────────────┐    RPC     ┌──────────────┐
│  Frontend  │ ─────────────────▶ │  FastAPI backend │ ─────────▶ │   Supabase   │
│  (HTML/JS) │ ◀─────  JSON  ─── │   (main.py)      │            │ pgvector+FTS │
└────────────┘                    └──────────────────┘            └──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Gemini 2.5   │
                                   │ (free tier)  │
                                   └──────────────┘
```

## Repo layout

```
cyberforge/
├── .env                  # secrets — fill in (see "Configuration")
├── .gitignore
├── back/                 # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   └── dataset/          # optional — drop JSON or .c/.h files here
├── front/                # static frontend
│   ├── index.html
│   ├── README.md
│   └── nexbot_robot_character_concept.gltf
└── supabase/
    └── migrations/       # SQL — run these in the Supabase SQL editor
        ├── 0001_init.sql
        └── 0002_functions.sql
```

## 1 — Supabase setup

1. Create a project at [supabase.com](https://supabase.com).
2. Open **SQL Editor → New query** and paste the contents of:
   - `supabase/migrations/0001_init.sql` → Run
   - `supabase/migrations/0002_functions.sql` → Run
3. Grab your project URL and **service_role** key from
   **Settings → API**. (Not the anon key — the backend needs full DB access.)

## 2 — Backend setup

```powershell
cd C:\Users\welle\OneDrive\cyberforge\back
py -m pip install -r requirements.txt
```

Run it (the `--env-file` flag auto-loads your `.env`):

```powershell
cd C:\Users\welle\OneDrive\cyberforge\back
uvicorn "main (1):app" --env-file ..\.env --reload
```

> Tip: rename `main (1).py` to `main.py` and run `uvicorn main:app` instead —
> spaces in module names confuse some tooling.

Health check:

```
http://localhost:8000/health
```

should return `{"status": "ok", ...}`. If `documents` is `0`, either drop files
into `back/dataset/` (JSON or `.c`/`.h`) and restart, or `POST` to
`/documents/upsert` to add entries by hand.

## 3 — Frontend setup

The frontend is a single static HTML file. Serve it from anywhere:

```powershell
cd C:\Users\welle\OneDrive\cyberforge\front
py -m http.server 8080
```

Then open <http://localhost:8080>. The 3D robot character is just a UI prop —
the chat input at the bottom is what calls the backend.

If you serve the frontend on a different origin than `localhost:8000`, the
backend already allows all CORS origins (see `app.add_middleware`).

## Configuration — `.env`

```
GEMINI_API_KEY=...        # https://aistudio.google.com/app/apikey
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=...
GEMINI_MODEL=gemini-2.5-flash   # optional override
```

`.env` is already in `.gitignore`. **Never commit it.**

## API surface

| Method | Path                    | Purpose                                    |
| ------ | ----------------------- | ------------------------------------------ |
| GET    | `/health`               | Liveness + document count                  |
| POST   | `/query`                | Run RAG + Gemini                           |
| GET    | `/documents`            | List indexed documents                     |
| POST   | `/documents/upsert`     | Add/update documents (skip dataset files)  |
| DELETE | `/documents/{id}`       | Remove one document                        |
| POST   | `/reload`               | Re-read `dataset/` and reindex             |
| GET    | `/stats`                | Recent queries + cache hit rate            |
| DELETE | `/cache`                | Clear query cache                          |

## Dataset format

Place files under `back/dataset/`:

- **JSON** — `[{"title":"...", "description":"...", "code":"...", "tags":["uart","stm32"]}]`
- **.c / .h** — chunked automatically (60-line sliding window, 10-line overlap)

## Known limitations

- No auth — anyone with the URL can call `/query` and run up your Gemini quota.
- In-memory BM25 is rebuilt on every `/reload` and `/documents/upsert`; fine for
  ≤ ~50 k documents, slow above that.
- Gemini response is parsed as JSON; malformed outputs fall back to a stub
  object with the raw text in `explanation`.