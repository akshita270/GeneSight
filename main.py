from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import traceback

from models.schemas import QueryRequest, PipelineStatus, PipelineResult
from agents.task_planner import TaskPlannerAgent
from agents.literature import LiteratureAgent
from agents.extractor import ExtractionAgent
from agents.genomics_db import GenomicsDBAgent
from agents.knowledge_graph import KnowledgeGraphAgent
from agents.hypothesis import HypothesisAgent
from agents.validator import ValidatorAgent
from agents.reporter import ReporterAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup — preload heavy models
    print("Preloading agents...")
    ExtractionAgent()  # triggers spaCy model load
    print("All agents ready!")
    yield
    # Runs on shutdown
    print("Shutting down...")

app = FastAPI(title="Genomics AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis in production)
jobs: dict = {}


async def run_pipeline(job_id: str, query: str):
    jobs[job_id]["status"] = "running"
    try:
        # 1. Task Planner
        jobs[job_id]["agent"] = "Task Planner"
        planner = TaskPlannerAgent()
        subtasks = await planner.run(query)

        # 2. Literature Retrieval
        jobs[job_id]["agent"] = "Literature Retrieval"
        lit = LiteratureAgent()
        papers = await lit.run(subtasks["search_terms"])

        # 3. Information Extraction
        jobs[job_id]["agent"] = "Info Extraction"
        extractor = ExtractionAgent()
        entities = await extractor.run(papers)

        # 4. Genomics DB
        jobs[job_id]["agent"] = "Genomics DB"
        db_agent = GenomicsDBAgent()
        db_data = await db_agent.run(entities["genes"])

        # 5. Knowledge Graph
        jobs[job_id]["agent"] = "Knowledge Graph"
        kg = KnowledgeGraphAgent()
        graph = await kg.run(entities, db_data)

        # 6. Hypothesis Generation
        jobs[job_id]["agent"] = "Hypothesis Generation"
        hyp = HypothesisAgent()
        hypotheses = await hyp.run(query, graph, papers)

        # 7. Evidence Validation
        jobs[job_id]["agent"] = "Evidence Validation"
        validator = ValidatorAgent()
        validated = await validator.run(hypotheses, papers)

        # 8. Report
        jobs[job_id]["agent"] = "Report Generation"
        reporter = ReporterAgent()
        report = await reporter.run(query, validated, graph, papers)

        jobs[job_id].update({"status": "done", "result": report})

    except Exception as e:
        error_msg = traceback.format_exc()
        print("=" * 60)
        print("PIPELINE ERROR in agent:", jobs[job_id].get("agent"))
        print(error_msg)
        print("=" * 60)
        jobs[job_id].update({"status": "error", "error": error_msg})


@app.post("/query", response_model=dict)
async def start_query(req: QueryRequest, bg: BackgroundTasks):
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
    # Serialize result to dict to avoid Pydantic validation errors
    result = job["result"]
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}