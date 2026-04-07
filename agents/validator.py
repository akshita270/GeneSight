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
]


class ValidatorAgent:
    async def run(self, hypotheses: list[Hypothesis], papers: list[dict]) -> list[Hypothesis]:
        for hyp in hypotheses:
            count = 0
            pmids = []

            all_genes   = [g.lower() for g in hyp.genes]
            hyp_pathway = hyp.pathway.lower()

            # Split: primary gene (query gene) vs secondary genes (hypothesis-specific)
            primary_genes   = all_genes[:1]
            secondary_genes = all_genes[1:]

            # Extract disease keywords from statement
            disease_kws = self._extract_disease_keywords(hyp.statement)

            print(f"\nValidating: {hyp.title}")
            print(f"  Primary: {primary_genes}  Secondary: {secondary_genes}")
            print(f"  Disease kws: {disease_kws}")

            for p in papers:
                title_text = p.get("title", "").lower()
                abstract_text = p.get("abstract", "").lower()
                # Title hits are stronger signal — include title twice
                full_text = title_text + " " + title_text + " " + abstract_text

                score = self._score_paper(
                    full_text, title_text,
                    primary_genes, secondary_genes,
                    disease_kws, hyp_pathway
                )
                if score >= 2:
                    count += 1
                    pmids.append(p["pmid"])

            print(f"  Evidence count: {count}")

            hyp.evidence_count = count
            hyp.supporting_pmids = pmids

            # ── Confidence: fully evidence-based, ignore GPT's number ──
            # GPT scores are uncalibrated. We replace with a transparent scale:
            #   0 papers         → 15–25  (Exploratory — no real support found)
            #   1 paper          → 35–45  (Exploratory — minimal)
            #   2–3 papers       → 50–65  (Moderate)
            #   4–6 papers       → 70–80  (Strong)
            #   7+ papers        → 85–92  (Strong — well supported)
            # GPT confidence used only as ±5 tiebreaker within each band
            gpt_conf  = float(hyp.confidence)           # original GPT score 0-100
            gpt_bonus = round((gpt_conf - 50) / 50 * 5) # maps 0-100 → -5..+5

            if count == 0:
                base = 20
                hyp.status = "Exploratory"
            elif count == 1:
                base = 40
                hyp.status = "Exploratory"
            elif count <= 3:
                base = 57
                hyp.status = "Moderate"
            elif count <= 6:
                base = 75
                hyp.status = "Strong"
            else:
                base = 88
                hyp.status = "Strong"

            hyp.confidence = float(max(10, min(95, base + gpt_bonus)))

        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    # ── Disease keyword extraction ────────────────────────────────────────────
    def _extract_disease_keywords(self, statement: str) -> list[str]:
        statement_lower = statement.lower()
        disease_map = {
            "alzheimer":          ["alzheimer", "alzheimer's"],
            "parkinson":          ["parkinson", "parkinson's"],
            "cancer":             ["cancer", "tumor", "carcinoma", "malignant", "oncolog"],
            "breast cancer":      ["breast cancer", "breast tumor"],
            "lung cancer":        ["lung cancer", "lung tumor"],
            "diabetes":           ["diabetes", "diabetic", "insulin resistance"],
            "als":                ["als", "amyotrophic", "motor neuron"],
            "multiple sclerosis": ["multiple sclerosis"],
            "huntington":         ["huntington"],
            "schizophrenia":      ["schizophrenia", "psychosis"],
            "depression":         ["depression", "depressive"],
            "epilepsy":           ["epilepsy", "seizure"],
            "stroke":             ["stroke", "ischemic"],
            "heart failure":      ["heart failure", "cardiac failure"],
            "arthritis":          ["arthritis", "rheumatoid"],
            "glioblastoma":       ["glioblastoma", "glioma"],
            "leukemia":           ["leukemia", "leukaemia"],
            "melanoma":           ["melanoma"],
        }
        found = []
        for disease, keywords in disease_map.items():
            if any(kw in statement_lower for kw in keywords):
                found.extend(keywords)
        return list(set(found)) if found else []

    # ── Paper scoring ─────────────────────────────────────────────────────────
    def _score_paper(
        self,
        text: str,
        title_text: str,
        primary_genes: list[str],
        secondary_genes: list[str],
        disease_kws: list[str],
        pathway: str,
    ) -> int:
        """
        Score 0–4. Need >= 2 to count as supporting evidence.

        +1  primary gene found in text
        +1  secondary gene found (hard requirement if hypothesis has one)
              OR disease keyword found (if no secondary gene)
        +1  relationship keyword (bonus)
        -1  negative context penalty

        Title matches get double weight (title is included twice in `text`).
        """
        score = 0

        # ── Check 1: primary gene (whole word) ──
        if primary_genes:
            primary_found = any(
                re.search(r'\b' + re.escape(g) + r'\b', text)
                for g in primary_genes
            )
            if not primary_found:
                return 0
            score += 1

        # ── Check 2: secondary gene OR disease keyword ──
        if secondary_genes:
            # Secondary gene is REQUIRED — it's what distinguishes this hypothesis
            secondary_found = any(
                re.search(r'\b' + re.escape(g) + r'\b', text)
                for g in secondary_genes
            )
            if not secondary_found:
                return 0   # hard requirement
            score += 1
        else:
            # Single-gene hypothesis: require disease keyword match
            disease_found = (
                any(d in text for d in disease_kws) if disease_kws else False
            )
            # Also check pathway (meaningful words only, length > 5)
            pathway_words = [w for w in pathway.split() if len(w) > 5]
            pathway_found = bool(pathway_words) and any(w in text for w in pathway_words)
            if disease_found or pathway_found:
                score += 1
            else:
                return 0  # no context match — not a supporting paper

        # ── Check 3: relationship keyword (bonus) ──
        if any(kw in text for kw in RELATIONSHIP_KEYWORDS):
            score += 1

        # ── Penalty: negative context ──
        if any(neg in text for neg in NEGATIVE_CONTEXT):
            score -= 1

        return score
