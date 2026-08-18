from __future__ import annotations
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from collections import defaultdict
import uuid, os, traceback, httpx, json, asyncio, time, hashlib

from models.schemas import QueryRequest, PipelineStatus, PipelineResult
from agents.task_planner import TaskPlannerAgent
from agents.literature import LiteratureAgent
from agents.extractor import ExtractionAgent
from agents.genomics_db import GenomicsDBAgent
from agents.knowledge_graph import KnowledgeGraphAgent
from agents.hypothesis import HypothesisAgent
from agents.validator import ValidatorAgent
from agents.evaluator import EvaluatorAgent
from agents.reporter import ReporterAgent
from agents.guardrails import InputGuardrail
from agents.hallucination_guard import HallucinationGuard
from agents.citation_validator import CitationValidator
from agents.paper_sanitiser import sanitise as sanitise_papers
from utils.tracer import PipelineTracer
from utils.circuit_breaker import openai_breaker, ncbi_breaker, neo4j_breaker
from utils.job_queue import job_queue
from utils.rag_metrics import compute as compute_rag_metrics
from db.neon import init_db, upsert_user, save_query, get_usage_today, get_history
from config import settings

JOB_TTL_SECONDS        = 3600   # evict completed/failed jobs after 1 hour
PIPELINE_TIMEOUT_S     = 300    # kill a pipeline after 5 minutes
MAX_CONCURRENT_PIPELINES = 3    # max simultaneous pipeline runs
IP_RATE_LIMIT          = 10     # max /query requests per IP per minute
IP_RATE_WINDOW_S       = 60.0

# ── Concurrency limiter ───────────────────────────────────────────────────────
_pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)

# ── IP sliding-window rate limiter ────────────────────────────────────────────
_ip_timestamps: dict[str, list[float]] = defaultdict(list)

# ── Result cache: sha256(normalised query) → job_id ──────────────────────────
_query_cache: dict[str, str] = {}

def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()

# ── Clerk JWT verification ────────────────────────────────────────────────────
_clerk_jwks: dict | None = None
_clerk_jwks_lock = asyncio.Lock()

async def _get_jwks() -> dict:
    global _clerk_jwks
    if _clerk_jwks:
        return _clerk_jwks
    async with _clerk_jwks_lock:
        if _clerk_jwks:  # re-check after acquiring lock
            return _clerk_jwks
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.clerk.com/v1/jwks",
                headers={"Authorization": f"Bearer {settings.clerk_secret_key}"},
            )
            r.raise_for_status()
            _clerk_jwks = r.json()
    return _clerk_jwks

async def verify_clerk_token(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
) -> dict | None:
    """Verify Clerk JWT. Returns payload dict or None if no/invalid token."""
    if not credentials:
        return None
    if not settings.clerk_secret_key:
        return {"sub": "dev_user", "email": "dev@genesight.app", "name": "Dev User"}
    try:
        import jwt as pyjwt
        jwks = await _get_jwks()
        token = credentials.credentials
        header = pyjwt.get_unverified_header(token)
        kid = header.get("kid")
        key = None
        for k in jwks.get("keys", []):
            if k.get("kid") == kid:
                key = pyjwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
                break
        if not key:
            return None
        payload = pyjwt.decode(token, key, algorithms=["RS256"], options={"verify_aud": False})
        return payload
    except Exception as e:
        print(f"JWT verify error: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GeneSight API starting up...")
    init_db()
    job_queue.set_handler(run_pipeline)
    await job_queue.start()
    yield
    await job_queue.stop()
    print("Shutting down...")

app = FastAPI(title="GeneSight API", version="1.0.0", lifespan=lifespan)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://localhost:8001",
    *([os.environ["FRONTEND_ORIGIN"]] if os.environ.get("FRONTEND_ORIGIN") else []),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Adds X-Request-ID, IP rate limiting, and security response headers."""
    request_id = str(uuid.uuid4())[:8]

    # ── IP rate limit on /query only ─────────────────────────────────────────
    if request.url.path == "/query" and request.method == "POST":
        ip = (request.client.host if request.client else "unknown")
        now = time.monotonic()
        _ip_timestamps[ip] = [t for t in _ip_timestamps[ip] if now - t < IP_RATE_WINDOW_S]
        if len(_ip_timestamps[ip]) >= IP_RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit: max {IP_RATE_LIMIT} queries/min per IP"},
                headers={"X-Request-ID": request_id, "Retry-After": "60"},
            )
        _ip_timestamps[ip].append(now)

    response = await call_next(request)

    # ── Security headers (OWASP baseline) ────────────────────────────────────
    response.headers["X-Request-ID"]          = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]     = "geolocation=(), microphone=(), camera=()"
    # HSTS only for HTTPS (Render, production) — skip on localhost
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# In-memory job store — evict finished jobs older than JOB_TTL_SECONDS
jobs: dict = {}

def _evict_old_jobs():
    now = time.monotonic()
    to_delete = [
        jid for jid, j in jobs.items()
        if j["status"] in ("done", "error") and now - j.get("finished_at", now) > JOB_TTL_SECONDS
    ]
    for jid in to_delete:
        del jobs[jid]


async def run_pipeline(job_id: str, query: str, clerk_id: str | None = None):
    try:
        await asyncio.wait_for(
            _run_pipeline(job_id, query, clerk_id),
            timeout=PIPELINE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        jobs[job_id].update({
            "status": "error",
            "error": f"Pipeline timed out after {PIPELINE_TIMEOUT_S}s",
            "finished_at": time.monotonic(),
        })


async def _run_pipeline(job_id: str, query: str, clerk_id: str | None = None):
    tracer = PipelineTracer(job_id)
    jobs[job_id]["status"] = "running"

    async with _pipeline_semaphore:
        try:
            # ── Task Planner ─────────────────────────────────────────────────
            t = tracer.start("Task Planner", f"query={query[:80]}")
            jobs[job_id]["agent"] = "Task Planner"
            subtasks = await TaskPlannerAgent().run(query)
            t.finish(f"search_terms={subtasks.get('search_terms', [])[:3]}")

            # ── Literature Retrieval ──────────────────────────────────────────
            t = tracer.start("Literature Retrieval", f"terms={subtasks['search_terms']}")
            jobs[job_id]["agent"] = "Literature Retrieval"
            papers = await LiteratureAgent().run(subtasks["search_terms"])
            t.finish(f"{len(papers)} papers retrieved")

            # ── Paper Sanitisation ────────────────────────────────────────────
            papers, sanitise_flags = sanitise_papers(papers)
            if sanitise_flags:
                jobs[job_id].setdefault("audit_log", []).extend(sanitise_flags)

            # ── Info Extraction ───────────────────────────────────────────────
            t = tracer.start("Info Extraction", f"{len(papers)} papers")
            jobs[job_id]["agent"] = "Info Extraction"
            entities = await ExtractionAgent().run(papers)
            t.finish(f"{len(entities.get('genes', []))} genes, {len(entities.get('diseases', []))} diseases")

            # ── Genomics DB ───────────────────────────────────────────────────
            t = tracer.start("Genomics DB", f"genes={entities.get('genes', [])[:5]}")
            jobs[job_id]["agent"] = "Genomics DB"
            db_data = await GenomicsDBAgent().run(entities["genes"])
            verified = [d["gene"] for d in db_data if isinstance(d, dict) and d.get("ncbi_id")]
            t.finish(
                f"{len(verified)}/{len(db_data)} genes verified in NCBI",
                flags=[f"unverified: {[d['gene'] for d in db_data if isinstance(d,dict) and not d.get('ncbi_id')][:5]}"]
                if len(verified) < len(db_data) else [],
            )

            # ── Knowledge Graph ───────────────────────────────────────────────
            t = tracer.start("Knowledge Graph", f"{len(db_data)} enriched genes")
            jobs[job_id]["agent"] = "Knowledge Graph"
            graph = await KnowledgeGraphAgent().run(entities, db_data)
            t.finish(f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")

            # ── Hypothesis Generation ─────────────────────────────────────────
            t = tracer.start("Hypothesis Generation", f"graph={len(graph.nodes)} nodes")
            jobs[job_id]["agent"] = "Hypothesis Generation"
            hypotheses = await HypothesisAgent().run(query, graph, papers)
            t.finish(f"{len(hypotheses)} hypotheses generated")

            # ── Hallucination + Faithfulness Guard ────────────────────────────
            guard = HallucinationGuard()
            hypotheses, audit_log = guard.run(hypotheses, db_data, papers)
            if audit_log:
                jobs[job_id]["audit_log"] = audit_log

            # ── Citation Validation ───────────────────────────────────────────
            hypotheses, citation_flags = CitationValidator().run(hypotheses, papers)
            if citation_flags:
                jobs[job_id].setdefault("audit_log", []).extend(citation_flags)

            # ── Evidence Validation ───────────────────────────────────────────
            t = tracer.start("Evidence Validation", f"{len(hypotheses)} hypotheses, {len(papers)} papers")
            jobs[job_id]["agent"] = "Evidence Validation"
            validated = await ValidatorAgent().run(hypotheses, papers)
            strong = sum(1 for h in validated if h.status == "Strong")
            t.finish(f"{strong} strong, {len(validated)-strong} exploratory/moderate")

            # ── Quality Evaluation ────────────────────────────────────────────
            t = tracer.start("Evaluating Quality")
            jobs[job_id]["agent"] = "Evaluating Quality"
            evaluation = await EvaluatorAgent().run(query, validated, papers, graph)
            t.finish(f"score={evaluation.health_score}/100 grade={evaluation.grade}")

            # ── Report Generation ─────────────────────────────────────────────
            t = tracer.start("Report Generation")
            jobs[job_id]["agent"] = "Report Generation"
            report = await ReporterAgent().run(query, validated, graph, papers, evaluation)
            t.finish("report built")

            # ── RAG Metrics ───────────────────────────────────────────────────
            hyps_as_dicts = [
                {"genes": h.genes, "statement": h.statement, "title": h.title}
                for h in validated
            ]
            papers_as_dicts = [
                {"title": p.title, "abstract": p.abstract}
                for p in report.papers
            ]
            rag_metrics = compute_rag_metrics(papers_as_dicts, hyps_as_dicts, query)

            jobs[job_id].update({
                "status": "done",
                "result": report,
                "finished_at": time.monotonic(),
                "trace": tracer.to_dict(),
                "total_tokens": tracer.total_tokens,
                "total_cost_usd": sum(
                    t.get("estimated_cost_usd", 0.0) for t in tracer.to_dict()
                ),
                "pipeline_duration_s": tracer.total_duration_s,
                "rag_metrics": rag_metrics,
            })
            _query_cache[_cache_key(query)] = job_id
            print(f"✓ Pipeline complete — job={job_id} duration={tracer.total_duration_s}s")

            # ── Persist to Neon ───────────────────────────────────────────────
            if clerk_id:
                top_hyp = validated[0].title if validated else ""
                save_query(
                    clerk_id=clerk_id,
                    query=query,
                    hyp_count=len(validated),
                    paper_count=len(papers),
                    health_score=evaluation.health_score if evaluation else 0,
                    top_hyp=top_hyp,
                )

        except Exception as e:
            error_msg = traceback.format_exc()
            print("=" * 60)
            print("PIPELINE ERROR in agent:", jobs[job_id].get("agent"))
            print(error_msg)
            print("=" * 60)
            jobs[job_id].update({
                "status": "error",
                "error": error_msg,
                "finished_at": time.monotonic(),
                "trace": tracer.to_dict(),
            })


# ── Routes ────────────────────────────────────────────────────────────────────

_guardrail = InputGuardrail()

@app.post("/query", response_model=dict)
async def start_query(
    req: QueryRequest,
    user: dict | None = Depends(verify_clerk_token),
):
    if not req.query or not req.query.strip():
        raise HTTPException(400, "Query must not be empty")

    # ── Input guardrails ──
    valid, reason = _guardrail.check(req.query)
    if not valid:
        raise HTTPException(422, reason)

    clerk_id = user.get("sub") if user else None

    # ── Rate limiting ──
    if clerk_id and settings.database_url:
        used_today = get_usage_today(clerk_id)
        if used_today >= settings.free_queries_per_day:
            raise HTTPException(429, f"Daily limit reached ({settings.free_queries_per_day} queries/day on free plan)")
        # Upsert user record
        email = user.get("email", "")
        name = user.get("name", "")
        if user.get("first_name") and user.get("last_name"):
            name = f"{user['first_name']} {user['last_name']}"
        upsert_user(clerk_id, email, name)

    _evict_old_jobs()

    # Return cached result if same query already completed
    ck = _cache_key(req.query)
    cached_jid = _query_cache.get(ck)
    if cached_jid and jobs.get(cached_jid, {}).get("status") == "done":
        return {"job_id": cached_jid, "cached": True}
    elif cached_jid:
        del _query_cache[ck]  # job was evicted, remove stale cache entry

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "agent": None, "result": None, "error": None, "finished_at": None}
    accepted = await job_queue.enqueue(job_id, req.query, clerk_id)
    if not accepted:
        del jobs[job_id]
        raise HTTPException(503, "Server is at capacity — please try again shortly")
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "current_agent": job.get("agent"),
        "error": job.get("error"),
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not complete — status: {job['status']}")
    result = job["result"]
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


@app.get("/usage")
async def get_usage(user: dict | None = Depends(verify_clerk_token)):
    """Return how many queries the user has used today."""
    if not user:
        return {"used": 0, "limit": settings.free_queries_per_day, "authenticated": False}
    clerk_id = user.get("sub")
    used = get_usage_today(clerk_id) if clerk_id else 0
    return {
        "used": used,
        "limit": settings.free_queries_per_day,
        "remaining": max(0, settings.free_queries_per_day - used),
        "authenticated": True,
    }


@app.get("/history")
async def get_user_history(user: dict | None = Depends(verify_clerk_token)):
    """Return the authenticated user's query history."""
    if not user:
        raise HTTPException(401, "Not authenticated")
    clerk_id = user.get("sub")
    history = get_history(clerk_id)
    for item in history:
        if item.get("created_at"):
            item["created_at"] = str(item["created_at"])
        if item.get("id"):
            item["id"] = str(item["id"])
    return {"history": history}


@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    """
    Server-Sent Events stream — pushes agent progress in real time.
    Frontend can use EventSource('/stream/<job_id>') instead of polling.
    """
    async def event_gen():
        last_agent = None
        last_status = None
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
                return

            status = job["status"]
            agent  = job.get("agent")

            # Push on first event and whenever agent or status changes
            if agent != last_agent or status != last_status:
                last_agent  = agent
                last_status = status
                payload = json.dumps({"status": status, "agent": agent})
                yield f"event: progress\ndata: {payload}\n\n"

            if status == "done":
                yield f"event: done\ndata: {json.dumps({'job_id': job_id})}\n\n"
                return
            if status == "error":
                err = str(job.get("error", "Unknown error"))[:300]
                yield f"event: error\ndata: {json.dumps({'message': err})}\n\n"
                return

            await asyncio.sleep(1)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stream/summary/{job_id}")
async def stream_summary(job_id: str):
    """
    Streams the GPT-4o research summary token-by-token as SSE.
    Call this after /stream/{job_id} emits a 'done' event.
    Frontend subscribes with EventSource and renders tokens as they arrive.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not complete — status: {job['status']}")

    result = job["result"]
    hypotheses = result.hypotheses if hasattr(result, "hypotheses") else []
    papers_raw = [p.model_dump() for p in result.papers] if hasattr(result, "papers") else []
    query = result.query if hasattr(result, "query") else ""

    async def token_gen():
        try:
            reporter = ReporterAgent()
            async for token in reporter.stream_summary(query, hypotheses, papers_raw):
                payload = json.dumps({"token": token})
                yield f"event: token\ndata: {payload}\n\n"
            yield f"event: done\ndata: {{}}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)[:200]})}\n\n"

    return StreamingResponse(
        token_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health/deep")
async def health_deep():
    """Ops-grade health check that validates all external dependencies."""
    checks: dict[str, str | dict] = {}

    # Neo4j
    try:
        from db.neo4j_client import Neo4jClient
        client = Neo4jClient()
        with client.driver.session() as s:
            s.run("RETURN 1")
        client.close()
        checks["neo4j"] = "ok"
        neo4j_breaker.record_success()
    except Exception as e:
        checks["neo4j"] = f"error: {str(e)[:120]}"
        neo4j_breaker.record_failure()

    # Neon (PostgreSQL)
    if settings.database_url:
        try:
            from db.neon import _get_pool
            conn = _get_pool().getconn()
            _get_pool().putconn(conn)
            checks["neon"] = "ok"
        except Exception as e:
            checks["neon"] = f"error: {str(e)[:120]}"
    else:
        checks["neon"] = "not configured"

    # OpenAI key present
    checks["openai_key"] = "ok" if settings.openai_api_key else "missing"

    # Circuit breakers
    checks["circuit_breakers"] = {
        "openai": openai_breaker.state,
        "ncbi":   ncbi_breaker.state,
        "neo4j":  neo4j_breaker.state,
    }

    # Job store
    checks["job_store"] = {
        "total_jobs": len(jobs),
        "running":    sum(1 for j in jobs.values() if j["status"] == "running"),
        "slots_free": _pipeline_semaphore._value,
    }

    all_ok = all(
        v == "ok"
        for k, v in checks.items()
        if isinstance(v, str) and k not in ("circuit_breakers",)
    )
    return {"status": "ok" if all_ok else "degraded", "checks": checks}


@app.get("/metrics")
async def metrics():
    """System metrics — job counts, cache size, circuit breaker states, throughput."""
    status_counts: dict[str, int] = defaultdict(int)
    durations: list[float] = []

    for j in jobs.values():
        status_counts[j["status"]] += 1
        if d := j.get("pipeline_duration_s"):
            durations.append(d)

    return {
        "jobs": dict(status_counts),
        "cache_entries": len(_query_cache),
        "avg_pipeline_duration_s": round(sum(durations) / len(durations), 1) if durations else 0,
        "concurrent_slots_used": MAX_CONCURRENT_PIPELINES - _pipeline_semaphore._value,
        "concurrent_slots_total": MAX_CONCURRENT_PIPELINES,
        "circuit_breakers": {
            "openai": openai_breaker.stats(),
            "ncbi":   ncbi_breaker.stats(),
            "neo4j":  neo4j_breaker.stats(),
        },
        "queue": {
            "depth": job_queue.depth,
            "workers": job_queue.workers,
            "capacity": job_queue.max_size,
        },
    }


@app.get("/trace/{job_id}")
async def get_trace(job_id: str):
    """Return the per-agent execution trace for a completed or failed job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "pipeline_duration_s": job.get("pipeline_duration_s"),
        "total_tokens": job.get("total_tokens", 0),
        "total_cost_usd": job.get("total_cost_usd", 0.0),
        "rag_metrics": job.get("rag_metrics", {}),
        "agents": job.get("trace", []),
        "audit_log": job.get("audit_log", []),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
