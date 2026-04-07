from __future__ import annotations
from models.schemas import Hypothesis

NEGATIVE_CONTEXT = [
    "not associated", "no association", "not linked", "no evidence",
    "unrelated", "not significant", "failed to", "no correlation",
    "not involved", "independent of", "not observed"
]

RELATIONSHIP_KEYWORDS = [
    "associated with", "linked to", "contributes to", "implicated in",
    "risk factor", "pathway", "mechanism", "involved in", "plays a role",
    "regulates", "influences", "modulates", "increases risk", "variant",
    "mutation", "expression", "interacts with", "co-expressed"
]


class ValidatorAgent:
    async def run(self, hypotheses: list[Hypothesis], papers: list[dict]) -> list[Hypothesis]:
        for hyp in hypotheses:
            count = 0
            pmids = []

            hyp_genes   = [g.lower() for g in hyp.genes]
            hyp_pathway = hyp.pathway.lower()

            # Extract disease keywords from hypothesis statement
            disease_kws = self._extract_disease_keywords(hyp.statement)

            print(f"\nValidating: {hyp.title}")
            print(f"  Genes: {hyp_genes}")
            print(f"  Disease keywords: {disease_kws}")
            print(f"  Pathway: {hyp_pathway}")

            for p in papers:
                text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
                score = self._score_paper(text, hyp_genes, disease_kws, hyp_pathway)
                if score >= 2:   # gene + disease/pathway must both match
                    count += 1
                    pmids.append(p["pmid"])

            print(f"  Evidence count: {count}")

            hyp.evidence_count = count
            hyp.supporting_pmids = pmids

            if count >= 5:
                hyp.status = "Strong"
                hyp.confidence = min(hyp.confidence + 10, 99)
            elif count >= 2:
                hyp.status = "Moderate"
                hyp.confidence = min(hyp.confidence + 5, 95)
            else:
                hyp.status = "Exploratory"
                hyp.confidence = max(hyp.confidence - 5, 10)

        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def _extract_disease_keywords(self, statement: str) -> list[str]:
        """Extract disease-related keywords from hypothesis statement."""
        statement_lower = statement.lower()

        # broad keyword list covering many diseases
        disease_map = {
            "alzheimer":     ["alzheimer", "alzheimer's", "ad"],
            "parkinson":     ["parkinson", "parkinson's", "pd"],
            "cancer":        ["cancer", "tumor", "carcinoma", "malignant", "oncology"],
            "diabetes":      ["diabetes", "diabetic", "insulin resistance", "glucose"],
            "als":           ["als", "amyotrophic", "motor neuron"],
            "multiple sclerosis": ["multiple sclerosis", "ms"],
            "huntington":    ["huntington", "hd"],
            "schizophrenia": ["schizophrenia", "psychosis"],
            "depression":    ["depression", "depressive"],
            "epilepsy":      ["epilepsy", "seizure"],
            "stroke":        ["stroke", "ischemic"],
            "heart failure": ["heart failure", "cardiac"],
            "arthritis":     ["arthritis", "rheumatoid"],
        }

        found = []
        for disease, keywords in disease_map.items():
            if any(kw in statement_lower for kw in keywords):
                found.extend(keywords)

        # Also add individual words from the pathway as fallback
        pathway_words = [w for w in statement_lower.split() if len(w) > 4]
        found.extend(pathway_words[:3])

        return list(set(found)) if found else []

    def _score_paper(self, text: str, genes: list[str],
                     disease_kws: list[str], pathway: str) -> int:
        """
        Score a paper 0–4 based on relevance to the hypothesis.

        +1  gene mentioned
        +1  disease or pathway mentioned
        +1  relationship keyword found
        -1  negative context found
        Threshold: score >= 1 to count as evidence
        """
        score = 0

        # Check 1 — gene must be present
        gene_found = any(g in text for g in genes)
        if not gene_found:
            return 0
        score += 1

        # Check 2 — disease OR pathway mentioned (either is fine)
        disease_found = any(d in text for d in disease_kws) if disease_kws else False
        pathway_found = any(w in text for w in pathway.split() if len(w) > 4)

        if disease_found or pathway_found:
            score += 1

        # Check 3 — relationship keyword
        if any(kw in text for kw in RELATIONSHIP_KEYWORDS):
            score += 1

        # Penalty for negative context
        if any(neg in text for neg in NEGATIVE_CONTEXT):
            score -= 1

        return score