from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import AsyncOpenAI
from config import settings
from models.schemas import KnowledgeGraph, Hypothesis
from utils.llm_utils import parse_json_with_retry
import json
import uuid
import re

class HypothesisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=0.4,
        )
        # Fallback to cheaper/faster model if primary fails
        self.llm_fallback = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.4,
        )
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.prompt = ChatPromptTemplate.from_template("""
You are a genomics research scientist generating novel hypotheses.

Research query: {query}

Knowledge graph entities:
Genes: {genes}
Diseases: {diseases}
Pathways: {pathways}

Recent literature abstracts (top 3):
{abstracts}

Generate {n} novel gene-disease hypotheses.
STRICT RULES — follow exactly:
1. genes field: pick exactly 2 symbols from the Genes list above. Use them EXACTLY as written.
   REJECT any symbol that is: an abbreviation (RAS, VIP, CP, NF), a pathway name, a disease name,
   or not a standard HGNC gene symbol. Valid examples: TREM2, APOE, BRCA1, TP53, EGFR, PTEN, KRAS.
2. Each hypothesis must use a DIFFERENT pair of genes — no repeated genes across hypotheses.
3. statement: 1-2 sentences describing the specific molecular mechanism linking those 2 genes to the disease.
4. confidence: integer 50-90 reflecting scientific plausibility.
5. pathway: the specific biological pathway (e.g. "PI3K/AKT Signaling", "DNA Damage Response").
You MUST respond ONLY with a raw JSON array. No markdown, no backticks, no extra text.
Each object must have exactly: title, statement, genes (array of exactly 2), pathway, confidence.

Example:
[{{"title":"TREM2-APOE interaction in neuroinflammation","statement":"TREM2 and APOE jointly regulate microglial lipid clearance; loss of function in both accelerates amyloid accumulation in Alzheimer's disease.","genes":["TREM2","APOE"],"pathway":"Microglial Lipid Metabolism","confidence":74}}]
""")

    async def run(self, query: str, graph: KnowledgeGraph, papers: list[dict]) -> list[Hypothesis]:
        # Filter out obvious non-gene strings (long phrases, lowercase common words)
        NON_GENES = {
            "cholesterol", "amyloid", "tau", "lipid", "glucose", "insulin",
            "dopamine", "serotonin", "cortisol", "protein", "pathway",
            "mitochondrial", "inflammation", "oxidative stress"
        }
        graph_genes = [n.id for n in graph.nodes if n.type == "gene"]
        genes = [
            g for g in graph_genes
            if g.lower() not in NON_GENES       # exclude known non-genes
            and len(g.split()) == 1             # single word only
            and len(g) <= 10                    # gene symbols are short
            and not g[0].islower()              # gene symbols start with uppercase
        ] or graph_genes
        genes = genes[:10]

        diseases = [n.id for n in graph.nodes if n.type == "disease"]
        pathways = [n.id for n in graph.nodes if n.type == "pathway"]
        abstracts = "\n\n".join(p.get("abstract", "")[:300] for p in papers[:8])

        invoke_args = {
            "query": query,
            "genes": genes,
            "diseases": diseases,
            "pathways": pathways,
            "abstracts": abstracts,
            "n": settings.hypothesis_max,
        }

        # Try primary model, fall back to gpt-4o-mini on failure
        for attempt, llm in enumerate([self.llm, self.llm_fallback]):
            try:
                chain = self.prompt | llm
                resp = await chain.ainvoke(invoke_args)
                content = resp.content.strip()
                print(f"Hypothesis Agent raw response (model attempt {attempt+1}):", repr(content[:300]))
                break
            except Exception as e:
                if attempt == 0:
                    print(f"Hypothesis primary model failed ({e}), falling back to gpt-4o-mini")
                else:
                    raise

        # Use JSON retry helper — asks GPT-4o-mini to fix malformed JSON automatically
        raw = await parse_json_with_retry(
            content, "hypothesis generation", self._openai
        )

        return [Hypothesis(
            id=str(uuid.uuid4()),
            title=h["title"],
            statement=h["statement"],
            genes=h.get("genes", []),
            pathway=h.get("pathway", "Unknown"),
            confidence=float(h.get("confidence", 50)),
            evidence_count=0,
            supporting_pmids=[],
            status="Exploratory",
        ) for h in raw]