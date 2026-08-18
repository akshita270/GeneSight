from __future__ import annotations
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import AsyncOpenAI
from config import settings
from utils.text import strip_md_fences
from utils.llm_utils import parse_json_with_retry

logger = logging.getLogger("genesight")


class TaskPlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.prompt = ChatPromptTemplate.from_template("""
You are a genomics research planning assistant.
Given a research question, extract:
1. search_terms: 3-5 PubMed search terms (MeSH-style)
2. target_disease: the primary disease of interest
3. focus_genes: any specific genes mentioned (empty list if none)

Respond ONLY in valid JSON with no extra text, no markdown, no backticks.
Example:
{{"search_terms":["Alzheimer disease genes","neuroinflammation GWAS"],"target_disease":"Alzheimer's disease","focus_genes":[]}}

Question: {query}
""")

    async def run(self, query: str) -> dict:
        chain   = self.prompt | self.llm
        resp    = await chain.ainvoke({"query": query})
        content = strip_md_fences(resp.content)

        if not content:
            raise ValueError("OpenAI returned an empty response. Check your API key and billing.")

        logger.info("TaskPlanner raw response: %s", repr(content[:200]))

        # Uses the same JSON repair utility as HypothesisAgent — auto-fixes
        # malformed output via a GPT-4o-mini follow-up call instead of crashing.
        result = await parse_json_with_retry(content, "task planning", self._openai)
        return result
