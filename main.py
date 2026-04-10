from __future__ import annotations
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uuid, os, traceback, httpx, json

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
from db.neon import init_db, upsert_user, save_query, get_usage_today, get_history
from config import settings


# ── Clerk JWT verification ────────────────────────────────────────────────────
_clerk_jwks: dict | None = None

async def _get_jwks() -> dict:
    global _clerk_jwks
    if _clerk_jwks:
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
    yield
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

# In-memory job store
jobs: dict = {}


async def run_pipeline(job_id: str, query: str, clerk_id: str | None = None):
    jobs[job_id]["status"] = "running"
    try:
        jobs[job_id]["agent"] = "Task Planner"
        planner = TaskPlannerAgent()
        subtasks = await planner.run(query)

        jobs[job_id]["agent"] = "Literature Retrieval"
        lit = LiteratureAgent()
        papers = await lit.run(subtasks["search_terms"])

        jobs[job_id]["agent"] = "Info Extraction"
        extractor = ExtractionAgent()
        entities = await extractor.run(papers)

        jobs[job_id]["agent"] = "Genomics DB"
        db_agent = GenomicsDBAgent()
        db_data = await db_agent.run(entities["genes"])

        jobs[job_id]["agent"] = "Knowledge Graph"
        kg = KnowledgeGraphAgent()
        graph = await kg.run(entities, db_data)

        jobs[job_id]["agent"] = "Hypothesis Generation"
        hyp = HypothesisAgent()
        hypotheses = await hyp.run(query, graph, papers)

        jobs[job_id]["agent"] = "Evidence Validation"
        validator = ValidatorAgent()
        validated = await validator.run(hypotheses, papers)

        jobs[job_id]["agent"] = "Evaluating Quality"
        evaluator = EvaluatorAgent()
        evaluation = await evaluator.run(query, validated, papers, graph)

        jobs[job_id]["agent"] = "Report Generation"
        reporter = ReporterAgent()
        report = await reporter.run(query, validated, graph, papers, evaluation)

        jobs[job_id].update({"status": "done", "result": report})
        print(f"✓ Pipeline complete for job {job_id}")

        # ── Persist to Neon ──
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
        jobs[job_id].update({"status": "error", "error": error_msg})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=dict)
async def start_query(
    req: QueryRequest,
    bg: BackgroundTasks,
    user: dict | None = Depends(verify_clerk_token),
):
    if not req.query or not req.query.strip():
        raise HTTPException(400, "Query must not be empty")

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

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "agent": None, "result": None, "error": None}
    bg.add_task(run_pipeline, job_id, req.query, clerk_id)
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


@app.get("/health")
async def health():
    return {"status": "ok"}
