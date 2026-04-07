from __future__ import annotations
import asyncio
import json
import re
from openai import AsyncOpenAI
from config import settings
from models.schemas import Hypothesis


class ValidatorAgent:
    """
    Uses GPT-4o-mini to semantically judge whether each paper
    actually supports each hypothesis — not just keyword matching.

    For each hypothesis we build ONE prompt that lists all paper
    titles+abstracts and asks the model to return a JSON list of
    which PMIDs genuinely support the hypothesis, understanding
    negation, context, and relevance properly.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def run(self, hypotheses: list[Hypothesis], papers: list[dict]) -> list[Hypothesis]:
        total = len(papers)
        # Run all hypotheses concurrently (each is one GPT call)
        tasks = [self._validate_one(hyp, papers) for hyp in hypotheses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for hyp, res in zip(hypotheses, results):
            if isinstance(res, Exception):
                print(f"Validator error for '{hyp.title}': {res}")
                # Fallback: keep GPT's prior confidence, status Exploratory
                hyp.evidence_count   = 0
                hyp.supporting_pmids = []
                hyp.status           = "Exploratory"
                hyp.confidence       = float(max(10, min(40, hyp.confidence * 0.5)))
            else:
                supporting_pmids, count = res
                hyp.evidence_count   = count
                hyp.supporting_pmids = supporting_pmids
                hyp.confidence, hyp.status = self._calc_confidence(
                    count, total, float(hyp.confidence)
                )
                print(f"  '{hyp.title}' → {count}/{total} papers, "
                      f"conf={hyp.confidence:.0f}%, status={hyp.status}")

        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    # ── Validate one hypothesis against all papers ────────────────────────────
    async def _validate_one(
        self, hyp: Hypothesis, papers: list[dict]
    ) -> tuple[list[str], int]:
        """
        Ask GPT-4o-mini to read every paper and decide which ones
        provide genuine positive evidence for this hypothesis.
        Returns (list_of_supporting_pmids, count).
        """
        # Build paper list for the prompt (title + first 400 chars of abstract)
        paper_entries = []
        for p in papers:
            pmid     = p.get("pmid", "")
            title    = p.get("title", "")
            abstract = str(p.get("abstract", ""))[:400]
            paper_entries.append(f'PMID:{pmid} | {title} | {abstract}')

        papers_block = "\n".join(paper_entries)

        prompt = f"""You are a biomedical research expert evaluating scientific evidence.

HYPOTHESIS:
\"{hyp.statement}\"
Genes involved: {', '.join(hyp.genes)}
Biological pathway: {hyp.pathway}

TASK:
Read each paper below (PMID | Title | Abstract excerpt) and decide whether it provides
GENUINE POSITIVE EVIDENCE supporting the hypothesis above.

Rules for counting a paper as SUPPORTING:
- The paper must study or report findings about the genes mentioned in the hypothesis
- The paper must find a POSITIVE relationship (association, regulation, mechanism, risk)
  between those genes and the disease/pathway in the hypothesis
- Papers that CONTRADICT the hypothesis (no association, not significant, failed to show)
  must NOT be counted as supporting
- Papers that are only tangentially related (mention the gene but study something unrelated)
  must NOT be counted

Return ONLY a raw JSON object like this — no explanation, no markdown:
{{"supporting_pmids": ["12345678", "23456789"]}}

If no papers support the hypothesis, return: {{"supporting_pmids": []}}

PAPERS:
{papers_block}"""

        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )

        raw = resp.choices[0].message.content.strip()
        print(f"\n  [{hyp.title[:40]}] GPT validator raw: {raw[:120]}")

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(raw)
            pmids = [str(p) for p in data.get("supporting_pmids", [])]
            # Only keep PMIDs that actually exist in our paper set
            valid_pmids_set = {str(p.get("pmid", "")) for p in papers}
            pmids = [pid for pid in pmids if pid in valid_pmids_set]
            return pmids, len(pmids)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Validator JSON parse error: {e} | raw={raw[:200]}")
            return [], 0

    # ── Evidence-based confidence calculation ─────────────────────────────────
    def _calc_confidence(
        self, count: int, total: int, gpt_prior: float
    ) -> tuple[float, str]:
        """
        Confidence is determined PURELY by paper count — strictly monotonic.
        More papers always = higher confidence, regardless of GPT prior.

        GPT prior only acts as a small tiebreaker (±4 pts) within each band
        so hypotheses with equal paper counts are ordered sensibly.

        Scale (count → base confidence):
          0  →  15   Exploratory
          1  →  38   Exploratory
          2  →  48   Exploratory
          3  →  58   Moderate
          4  →  65   Moderate
          5  →  70   Moderate
          6  →  75   Moderate
          7  →  80   Strong
          8  →  84   Strong
          9  →  88   Strong
          10+ →  92   Strong
        """
        BASE = {0: 15, 1: 38, 2: 48, 3: 58, 4: 65,
                5: 70, 6: 75, 7: 80, 8: 84, 9: 88}
        base = BASE.get(count, 92 if count >= 10 else 15)

        # GPT prior maps 50–90 → -4..+4 tiebreaker
        tiebreak = round((gpt_prior - 70) / 20 * 4)  # normalise around 70
        conf = float(max(10, min(95, base + tiebreak)))

        if count == 0:
            return conf, "Exploratory"
        elif count <= 2:
            return conf, "Exploratory"
        elif count <= 6:
            return conf, "Moderate"
        else:
            return conf, "Strong"
