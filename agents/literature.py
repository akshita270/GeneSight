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
        handle = Entrez.esearch(db="pubmed", term=query,
                                retmax=settings.pubmed_max_results, sort="relevance")
        ids = Entrez.read(handle)["IdList"]
        if not ids:
            return []
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml")
        records = Entrez.read(handle)
        papers = []
        for rec in records["PubmedArticle"]:
            art = rec["MedlineCitation"]["Article"]
            papers.append({
                "pmid": str(rec["MedlineCitation"]["PMID"]),
                "title": str(art.get("ArticleTitle", "")),
                "abstract": str(art.get("Abstract", {}).get("AbstractText", [""])[0]),
                "authors": [f"{a.get('LastName','')} {a.get('Initials','')}"
                            for a in art.get("AuthorList", [])],
                "journal": str(art["Journal"]["Title"]),
                "year": int(str(art["Journal"]["JournalIssue"]["PubDate"].get("Year", 0))),
            })
        return papers