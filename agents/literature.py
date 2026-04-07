from __future__ import annotations
from Bio import Entrez
from config import settings
import asyncio

Entrez.email = settings.entrez_email
Entrez.api_key = settings.entrez_api_key

class LiteratureAgent:
    async def run(self, search_terms: list[str]) -> list[dict]:
        query = " OR ".join(f'"{t}"' for t in search_terms)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch, query)

    def _fetch(self, query: str) -> list[dict]:
        search_handle = Entrez.esearch(db="pubmed", term=query,
                                       retmax=settings.pubmed_max_results, sort="relevance")
        try:
            ids = Entrez.read(search_handle)["IdList"]
        finally:
            search_handle.close()

        if not ids:
            return []

        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml")
        try:
            records = Entrez.read(fetch_handle)
        finally:
            fetch_handle.close()

        papers = []
        for i, rec in enumerate(records["PubmedArticle"]):
            art = rec["MedlineCitation"]["Article"]
            # Relevance score: rank-based decay (top result = 1.0, drops to ~0.5)
            relevance_score = round(1.0 - (i / max(len(records["PubmedArticle"]), 1)) * 0.5, 2)
            papers.append({
                "pmid": str(rec["MedlineCitation"]["PMID"]),
                "title": str(art.get("ArticleTitle", "")),
                "abstract": str(art.get("Abstract", {}).get("AbstractText", [""])[0]),
                "authors": [f"{a.get('LastName','')} {a.get('Initials','')}"
                            for a in art.get("AuthorList", [])],
                "journal": str(art["Journal"]["Title"]),
                "year": int(str(art["Journal"]["JournalIssue"]["PubDate"].get("Year", 0))),
                "relevance_score": relevance_score,
            })
        return papers
