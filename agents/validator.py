from __future__ import annotations
import re
from models.schemas import Hypothesis

NEGATIVE_CONTEXT = [
    "not associated", "no association", "not linked", "no evidence",
    "unrelated", "not significant", "no correlation",
    "not involved", "independent of", "not observed"
]

RELATIONSHIP_KEYWORDS = [
    "associated with", "linked to", "contributes to", "implicated in",
    "risk factor", "pathway", "mechanism", "involved in", "plays a role",
    "regulates", "influences", "modulates", "increases risk", "variant",
    "mutation", "expression", "interacts with", "co-expressed", "promotes",
    "inhibits", "activates", "suppresses", "drives", "mediates", "confers",
    "encodes", "overexpressed", "upregulated", "downregulated", "deletion",
    "amplification", "phosphorylation", "methylation", "signaling",
]


class ValidatorAgent:
    async def run(self, hypotheses: list[Hypothesis], papers: list[dict]) -> list[Hypothesis]:
        total_papers = len(papers)

        for hyp in hypotheses:
            count = 0
            pmids = []

            all_genes    = [g.lower() for g in hyp.genes]
            hyp_pathway  = hyp.pathway.lower()
            disease_kws  = self._extract_disease_keywords(hyp.statement)

            print(f"\nValidating: {hyp.title}")
            print(f"  Genes: {all_genes}  Disease kws: {disease_kws}")

            for p in papers:
                title_text    = p.get("title", "").lower()
                abstract_text = p.get("abstract", "").lower()
                # Weight title higher — include it twice
                full_text = title_text + " " + title_text + " " + abstract_text

                if self._paper_supports(full_text, all_genes, disease_kws, hyp_pathway):
                    count += 1
                    pmids.append(p["pmid"])

            print(f"  Evidence count: {count} / {total_papers}")

            hyp.evidence_count   = count
            hyp.supporting_pmids = pmids

            # ── Confidence: blend GPT prior with evidence ratio ──
            # evidence_ratio = fraction of papers that support this hypothesis
            # GPT gives a prior; evidence shifts it meaningfully
            gpt_prior = float(hyp.confidence)   # 0-100, GPT's subjective estimate

            if total_papers > 0:
                evidence_ratio = count / total_papers   # 0.0 – 1.0
            else:
                evidence_ratio = 0.0

            # evidence_score: 0 papers→0, half papers→65, all papers→85 (log-scaled)
            if count == 0:
                evidence_score = 0.0
            elif count == 1:
                evidence_score = 30.0
            elif count <= 3:
                evidence_score = 45.0 + evidence_ratio * 20
            elif count <= 7:
                evidence_score = 60.0 + evidence_ratio * 20
            else:
                evidence_score = 75.0 + evidence_ratio * 10

            # Blend: 40% GPT prior + 60% evidence score
            blended = 0.40 * gpt_prior + 0.60 * evidence_score

            # Status thresholds
            if count == 0:
                hyp.status = "Exploratory"
                hyp.confidence = float(max(10, min(40, blended)))
            elif count <= 2:
                hyp.status = "Exploratory"
                hyp.confidence = float(max(35, min(64, blended)))
            elif count <= 6:
                hyp.status = "Moderate"
                hyp.confidence = float(max(55, min(79, blended)))
            else:
                hyp.status = "Strong"
                hyp.confidence = float(max(70, min(95, blended)))

        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    # ── Disease keyword extraction ────────────────────────────────────────────
    def _extract_disease_keywords(self, statement: str) -> list[str]:
        statement_lower = statement.lower()
        disease_map = {
            "alzheimer":          ["alzheimer", "alzheimer's", " ad "],
            "parkinson":          ["parkinson", "parkinson's"],
            "cancer":             ["cancer", "tumor", "carcinoma", "malignant", "oncolog"],
            "breast cancer":      ["breast cancer", "breast tumor"],
            "lung cancer":        ["lung cancer", "lung tumor"],
            "diabetes":           ["diabetes", "diabetic", "insulin resistance"],
            "als":                ["amyotrophic lateral sclerosis", "motor neuron disease"],
            "multiple sclerosis": ["multiple sclerosis"],
            "huntington":         ["huntington"],
            "schizophrenia":      ["schizophrenia", "psychosis"],
            "depression":         ["depression", "depressive disorder"],
            "epilepsy":           ["epilepsy", "seizure"],
            "stroke":             ["stroke", "ischemic"],
            "heart failure":      ["heart failure", "cardiac failure"],
            "arthritis":          ["arthritis", "rheumatoid"],
            "glioblastoma":       ["glioblastoma", "glioma"],
            "leukemia":           ["leukemia", "leukaemia"],
            "melanoma":           ["melanoma"],
            "neurodegeneration":  ["neurodegeneration", "neurodegenerative"],
        }
        found = []
        for disease, keywords in disease_map.items():
            if any(kw in statement_lower for kw in keywords):
                found.extend(keywords)
        return list(set(found)) if found else []

    # ── Core: does this paper support the hypothesis? ─────────────────────────
    def _paper_supports(
        self,
        text: str,
        genes: list[str],
        disease_kws: list[str],
        pathway: str,
    ) -> bool:
        """
        A paper supports a hypothesis if it passes the relevance checks below.

        For MULTI-gene hypotheses (e.g. BRCA1 + PALB2):
          - Must mention ALL listed genes (differentiates hypotheses)
          - Disease/pathway match is a bonus, not required

        For SINGLE-gene hypotheses (e.g. GRN alone):
          - Must mention the gene
          - Must mention disease keyword OR meaningful pathway word
          - Must have a relationship keyword (not just incidental mention)

        Negative context anywhere gives a free pass-through penalty.
        """
        # ── Gene check ──
        if not genes:
            return False

        if len(genes) >= 2:
            # Multi-gene: ALL genes must appear
            all_found = all(
                re.search(r'\b' + re.escape(g) + r'\b', text) for g in genes
            )
            if not all_found:
                return False
            # With all genes present, any relationship keyword seals it
            has_relationship = any(kw in text for kw in RELATIONSHIP_KEYWORDS)
            has_negative     = any(neg in text for neg in NEGATIVE_CONTEXT)
            return has_relationship and not has_negative

        else:
            # Single-gene: gene + disease/pathway + relationship
            gene = genes[0]
            gene_found = bool(re.search(r'\b' + re.escape(gene) + r'\b', text))
            if not gene_found:
                return False

            # Disease or pathway context
            disease_found = bool(disease_kws) and any(d in text for d in disease_kws)
            pathway_words = [w for w in pathway.split() if len(w) > 4]
            pathway_found = bool(pathway_words) and any(w in text for w in pathway_words)
            context_found = disease_found or pathway_found

            # Relationship keyword
            has_relationship = any(kw in text for kw in RELATIONSHIP_KEYWORDS)
            has_negative     = any(neg in text for neg in NEGATIVE_CONTEXT)

            return context_found and has_relationship and not has_negative
