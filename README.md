---
title: GeneSight API
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_port: 7860
pinned: false
---

# GeneSight · Genomics Hypothesis Engine

> An agentic AI system that automatically generates, validates, and ranks scientific hypotheses from genomics literature and biological databases — powered by an 8-agent async pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/GPT--4o-OpenAI-412991?logo=openai&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5-008CC1?logo=neo4j&logoColor=white)
![Neon](https://img.shields.io/badge/Neon-PostgreSQL-00E5BF?logo=postgresql&logoColor=white)

---

## What it does

Enter a genomics research question (e.g. *"BRCA1 and breast cancer risk"*) and GeneSight:

1. Searches **33M+ PubMed papers** for relevant literature
2. Extracts genes, diseases, and biological entities using biomedical NER
3. Enriches genes with **NCBI Gene** and **UniProt** metadata
4. Builds a **Neo4j knowledge graph** of gene–disease–protein relationships
5. Generates ranked **research hypotheses** using GPT-4o
6. Validates each hypothesis against the retrieved evidence
7. Scores overall analysis quality (0–100)
8. Produces a structured report with an interactive D3 knowledge graph

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
User Query ─── FastAPI ─┤         8-Agent Async Pipeline          │
                        │                                         │
                        │  1. Task Planner      → GPT-4o          │
                        │  2. Literature        → PubMed/Entrez   │
                        │  3. Extractor         → spaCy NER       │
                        │  4. Genomics DB       → NCBI + UniProt  │
                        │  5. Knowledge Graph   → Neo4j           │
                        │  6. Hypothesis Gen    → GPT-4o          │
                        │  7. Validator         → Evidence Scoring│
                        │  8. Evaluator+Report  → GPT-4o          │
                        └────────────────────┬────────────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  Structured JSON Report      │
                              │  • Hypotheses + confidence   │
                              │  • D3 knowledge graph        │
                              │  • Supporting papers         │
                              │  • Quality score (0–100)     │
                              └─────────────────────────────┘
```

**Tech stack:**

| Layer | Technology |
|---|---|
| Backend API | FastAPI + asyncio background tasks |
| AI / Agents | OpenAI GPT-4o, LangChain, LangGraph |
| Literature | NCBI Entrez / Biopython (PubMed) |
| Gene enrichment | NCBI Gene REST API, UniProt REST API |
| Knowledge graph | Neo4j AuraDB (Bolt/async driver) |
| Auth | Clerk (RS256 JWT, JWKS-cached) |
| User database | Neon (PostgreSQL, connection pooled) |
| Frontend | Vanilla JS, D3.js v7 force graph |

---

## Local setup

### Prerequisites

- Python 3.11
- Neo4j AuraDB free instance — [console.neo4j.io](https://console.neo4j.io)
- OpenAI API key with GPT-4o access
- NCBI Entrez API key — [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/)

### Option A — Direct (no Docker)

```bash
# 1. Clone
git clone <repo-url>
cd "genomic hypothesis"

# 2. Virtual environment
python3.11 -m venv env311
source env311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your keys

# 5. Start backend
uvicorn main:app --reload --port 8001

# 6. Serve frontend (separate terminal)
cd frontend && python3 -m http.server 5500
```

Open **http://localhost:5500/index.html**

### Option B — Docker Compose (includes local Neo4j)

```bash
cp .env.example .env
# Add OPENAI_API_KEY and ENTREZ_* to .env
# Neo4j is provided by Docker — no AuraDB needed

docker compose up --build
```

API available at `http://localhost:8001` · Neo4j browser at `http://localhost:7474`

---

## Environment variables

See [`.env.example`](.env.example) for the full list. Required variables:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | GPT-4o access |
| `ENTREZ_EMAIL` | Yes | PubMed identity |
| `ENTREZ_API_KEY` | Yes | Higher PubMed rate limit |
| `NEO4J_URI` | Yes | AuraDB bolt URI |
| `NEO4J_USER` | Yes | AuraDB username (instance ID) |
| `NEO4J_PASSWORD` | Yes | AuraDB password |
| `DATABASE_URL` | No | Neon PostgreSQL (enables history & rate limiting) |
| `CLERK_SECRET_KEY` | No | Enables user auth (skipped in dev) |

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Start a pipeline job. Body: `{"query": "..."}` |
| `GET` | `/status/{job_id}` | Poll status: `queued \| running \| done \| error` |
| `GET` | `/stream/{job_id}` | SSE stream — real-time agent progress events |
| `GET` | `/result/{job_id}` | Fetch completed result JSON |
| `GET` | `/trace/{job_id}` | Per-agent execution trace with timing + audit log |
| `GET` | `/usage` | Queries used today (authenticated) |
| `GET` | `/history` | User query history (authenticated) |
| `GET` | `/health` | Lightweight health check |
| `GET` | `/health/deep` | Deep health — validates Neo4j, Neon, circuit breakers |
| `GET` | `/metrics` | Job counts, cache size, avg duration, circuit breaker states |

Interactive docs: **http://localhost:8001/docs**

**Built-in production features:**
- Result caching — identical queries return instantly from cache
- Input guardrails — prompt injection detection + genomics topic enforcement
- Hallucination guard — hypothesis genes verified against NCBI Gene DB
- Faithfulness check — genes must appear in retrieved papers
- Circuit breakers — auto-open on repeated failures to OpenAI / NCBI / Neo4j
- Concurrency limiter — max 3 simultaneous pipelines (semaphore)
- Pipeline timeout — 5-minute hard limit per run
- IP rate limiting — 10 queries/min per IP (middleware)
- Request IDs — every response carries `X-Request-ID` for tracing
- Per-agent tracing — timing, token counts, flags on every run

---

## Running tests

```bash
source env311/bin/activate
pip install pytest httpx
pytest tests/ -v
```

---

## Project structure

```
genomic hypothesis/
├── main.py                  # FastAPI app, routes, job store, caching
├── config.py                # Settings from env vars
├── agents/
│   ├── task_planner.py      # GPT-4o: decompose query into subtasks
│   ├── literature.py        # PubMed search via Biopython Entrez
│   ├── extractor.py         # spaCy biomedical NER
│   ├── genomics_db.py       # NCBI Gene + UniProt enrichment (async httpx)
│   ├── knowledge_graph.py   # Neo4j graph builder
│   ├── hypothesis.py        # GPT-4o hypothesis generation
│   ├── validator.py         # Evidence-based hypothesis scoring
│   ├── evaluator.py         # Overall quality scoring
│   └── reporter.py          # Final structured report
├── db/
│   ├── neo4j_client.py      # Neo4j driver (sync, pooled)
│   └── neon.py              # PostgreSQL via psycopg2 connection pool
├── models/schemas.py        # Pydantic request/response models
├── frontend/
│   ├── index.html           # Landing page (Clerk auth, query input)
│   └── results.html         # Results page (D3 graph, hypothesis cards)
├── tests/
│   └── test_api.py          # API integration tests
├── Dockerfile
├── docker-compose.yml
└── render.yaml              # Render.com deployment config
```

---

## Deployment

### Backend → Render

The `render.yaml` is pre-configured. Push to GitHub, connect the repo on [render.com](https://render.com), and set secret env vars in the Render dashboard.

### Frontend → Vercel

The `frontend/vercel.json` is pre-configured.

```bash
cd frontend
vercel --prod
```

Set `FRONTEND_ORIGIN` in Render to your Vercel URL to allow CORS.
