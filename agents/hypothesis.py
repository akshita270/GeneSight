from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings
from models.schemas import KnowledgeGraph, Hypothesis
import json
import uuid
import re

class HypothesisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=0.4
        )
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
Use only real human gene symbols (e.g. TREM2, APOE, BIN1) in the genes field.
You MUST respond ONLY with a raw JSON array. No markdown. No backticks. No explanation.
Each object must have: title, statement, genes, pathway, confidence.

Example output:
[{{"title":"TREM2 hypothesis","statement":"TREM2 loss impairs microglial function.","genes":["TREM2"],"pathway":"Neuroinflammation","confidence":80}}]
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
        abstracts = "\n\n".join(p.get("abstract", "")[:300] for p in papers[:3])

        chain = self.prompt | self.llm
        resp = await chain.ainvoke({
            "query": query,
            "genes": genes,
            "diseases": diseases,
            "pathways": pathways,
            "abstracts": abstracts,
            "n": settings.hypothesis_max,
        })

        content = resp.content.strip()
        print("Hypothesis Agent raw response:", repr(content[:300]))

        # Strip markdown fences
        content = re.sub(r"```json|```", "", content).strip()

        # Extract JSON array even if there's surrounding text
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            content = match.group(0)

        if not content:
            raise ValueError("OpenAI returned empty response for hypothesis generation.")

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse hypothesis response as JSON: {content[:300]}") from e

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