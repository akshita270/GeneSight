from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings
import json
import re

class TaskPlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
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
        chain = self.prompt | self.llm
        resp = await chain.ainvoke({"query": query})

        content = resp.content.strip()
        print("Task Planner raw response:", repr(content))

        # Strip markdown code fences if present
        content = re.sub(r"```json|```", "", content).strip()

        if not content:
            raise ValueError("OpenAI returned an empty response. Check your API key and billing.")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse LLM response as JSON: {content}") from e