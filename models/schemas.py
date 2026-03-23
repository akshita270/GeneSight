from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str

class PipelineStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    current_agent: Optional[str] = None

class Paper(BaseModel):
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: int
    abstract: str
    relevance_score: float = 0.0

class Entity(BaseModel):
    name: str
    type: str  # gene | disease | protein | pathway

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict = {}

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    confidence: float = 1.0

class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]

class Hypothesis(BaseModel):
    id: str
    title: str
    statement: str
    genes: list[str]
    pathway: str
    confidence: float
    evidence_count: int = 0
    supporting_pmids: list[str] = []
    status: str = "Exploratory"  # Strong | Moderate | Exploratory

class PipelineResult(BaseModel):
    query: str
    papers: list[Paper]
    graph: KnowledgeGraph
    hypotheses: list[Hypothesis]
    summary: str