from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings
from models.schemas import KnowledgeGraph, Hypothesis, Paper, PipelineResult

class ReporterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
        self.prompt = ChatPromptTemplate.from_template("""
Summarize the following genomics research findings in 3-4 sentences for a scientist audience.

Query: {query}
Top hypothesis: {top_hyp}
Total papers analyzed: {paper_count}
Total hypotheses generated: {hyp_count}
""")

    def _safe_year(self, year) -> int:
        try:
            return int(str(year))
        except Exception:
            return 0

    async def run(
        self,
        query: str,
        hypotheses: list[Hypothesis],
        graph: KnowledgeGraph,
        papers: list[dict],
    ) -> PipelineResult:
        chain = self.prompt | self.llm
        resp = await chain.ainvoke({
            "query": query,
            "top_hyp": hypotheses[0].statement if hypotheses else "None",
            "paper_count": len(papers),
            "hyp_count": len(hypotheses),
        })

        # Safely convert raw paper dicts to Paper models
        paper_models = []
        for p in papers:
            try:
                paper_models.append(Paper(
                    pmid=str(p.get("pmid", "")),
                    title=str(p.get("title", "")),
                    authors=[str(a) for a in p.get("authors", [])],
                    journal=str(p.get("journal", "")),
                    year=self._safe_year(p.get("year", 0)),
                    abstract=str(p.get("abstract", "")),
                    relevance_score=float(p.get("relevance_score", 0.7)),
                ))
            except Exception as e:
                print(f"Skipping paper due to error: {e}")
                continue

        return PipelineResult(
            query=query,
            papers=paper_models,
            graph=graph,
            hypotheses=hypotheses,
            summary=resp.content,
        )