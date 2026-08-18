from __future__ import annotations
from typing import Optional, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import AsyncOpenAI
from config import settings
from models.schemas import KnowledgeGraph, Hypothesis, Paper, PipelineResult, EvaluationReport

SUMMARY_PROMPT = """Summarize the following genomics research findings in 3-4 sentences for a scientist audience.

Query: {query}
Top hypothesis: {top_hyp}
Total papers analyzed: {paper_count}
Total hypotheses generated: {hyp_count}
"""

class ReporterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
        self.prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def _build_prompt(query: str, hypotheses: list, papers: list) -> str:
        return SUMMARY_PROMPT.format(
            query=query,
            top_hyp=hypotheses[0].statement if hypotheses else "None",
            paper_count=len(papers),
            hyp_count=len(hypotheses),
        )

    async def stream_summary(
        self,
        query: str,
        hypotheses: list[Hypothesis],
        papers: list[dict],
    ) -> AsyncIterator[str]:
        """Stream the summary generation token-by-token via OpenAI streaming API."""
        prompt_text = self._build_prompt(query, hypotheses, papers)
        stream = await self._openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_text}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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
        evaluation: Optional[EvaluationReport] = None,
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
            evaluation=evaluation,
        )