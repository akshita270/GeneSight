from __future__ import annotations
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import os
import traceback

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
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GeneSight API starting up...")
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


async def run_pipeline(job_id: str, query: str):
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

    except Exception as e:
        error_msg = traceback.format_exc()
        print("=" * 60)
        print("PIPELINE ERROR in agent:", jobs[job_id].get("agent"))
        print(error_msg)
        print("=" * 60)
        jobs[job_id].update({"status": "error", "error": error_msg})


@app.post("/query", response_model=dict)
async def start_query(req: QueryRequest, bg: BackgroundTasks):
    if not req.query or not req.query.strip():
        raise HTTPException(400, "Query must not be empty")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "agent": None, "result": None, "error": None}
    bg.add_task(run_pipeline, job_id, req.query)
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


@app.get("/health")
async def health():
    return {"status": "ok"}
