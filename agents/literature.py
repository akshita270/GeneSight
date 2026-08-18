from __future__ import annotations
import asyncio
import logging
from Bio import Entrez
from config import settings
from utils.retry import with_network_retry
from utils.circuit_breaker import ncbi_breaker

Entrez.email = settings.entrez_email
Entrez.api_key = settings.entrez_api_key
logger = logging.getLogger("genesight")

class LiteratureAgent:
    async def run(self, search_terms: list[str]) -> list[dict]:
        if ncbi_breaker.is_open():
            raise RuntimeError("NCBI circuit breaker is OPEN — too many recent failures")
        query = " OR ".join(f'"{t}"' for t in search_terms)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self._fetch_with_retry, query)
            ncbi_breaker.record_success()
            return result
        except Exception:
            ncbi_breaker.record_failure()
            raise

    def _fetch_with_retry(self, query: str) -> list[dict]:
        """Sync wrapper — tenacity works on sync functions too when called from executor."""
        from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _inner():
            return self._fetch(query)
        return _inner()

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
