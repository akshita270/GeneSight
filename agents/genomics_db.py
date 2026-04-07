from __future__ import annotations
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class GenomicsDBAgent:
    NCBI_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2))
    async def _ncbi_gene(self, gene: str) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.get(self.NCBI_URL, params={
                "db": "gene",
                "term": f"{gene}[Gene Name] AND Homo sapiens[Organism]",
                "retmode": "json",
                "retmax": 1,
            })
            data = r.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            return {"gene": gene, "ncbi_id": ids[0] if ids else None}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2))
    async def _uniprot(self, gene: str) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.get(self.UNIPROT_URL, params={
                "query": f"gene:{gene} AND organism_id:9606",
                "format": "json",
                "size": 1,
            })
            results = r.json().get("results", [])
            if results:
                entry = results[0]
                return {
                    "gene": gene,
                    "uniprot_id": entry.get("primaryAccession"),
                    "function": entry.get("comments", [{}])[0].get("texts", [{}])[0].get("value", ""),
                }
            return {"gene": gene, "uniprot_id": None, "function": ""}

    async def run(self, genes: list[str]) -> list[dict]:
        genes = genes[:15]  # cap to avoid rate limits
        ncbi_results = await asyncio.gather(
            *[self._ncbi_gene(g) for g in genes], return_exceptions=True
        )
        uni_results = await asyncio.gather(
            *[self._uniprot(g) for g in genes], return_exceptions=True
        )
        merged = {}
        for n in ncbi_results:
            if isinstance(n, dict):
                merged[n["gene"]] = n
        for u in uni_results:
            if isinstance(u, dict) and u["gene"] in merged:
                merged[u["gene"]].update(u)
        return list(merged.values())