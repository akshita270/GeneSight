from __future__ import annotations
from datetime import datetime
from models.schemas import EvaluationReport, QualityFlag, Hypothesis, KnowledgeGraph


class EvaluatorAgent:
    """
    Rule-based pipeline quality checker — no GPT call needed.
    Scores the run out of 100 across 5 dimensions and returns
    human-readable flags for the results page.
    """

    CURRENT_YEAR = datetime.now().year

    async def run(
        self,
        query: str,
        hypotheses: list[Hypothesis],
        papers: list[dict],
        graph: KnowledgeGraph,
    ) -> EvaluationReport:
        score = 0
        flags: list[QualityFlag] = []

        # ── 1. Paper volume (20 pts) ──────────────────────────────────────────
        paper_count = len(papers)
        if paper_count >= 10:
            score += 20
            flags.append(QualityFlag(level="ok",
                message=f"{paper_count} papers retrieved — good literature coverage"))
        elif paper_count >= 5:
            score += 10
            flags.append(QualityFlag(level="warn",
                message=f"Only {paper_count} papers found — try a more specific query for broader coverage"))
        else:
            flags.append(QualityFlag(level="warn",
                message=f"Very few papers found ({paper_count}) — results may be limited"))

        # ── 2. Gene diversity (20 pts) ────────────────────────────────────────
        all_genes: set[str] = set()
        for h in hypotheses:
            all_genes.update(g.upper() for g in (h.genes or []))
        gene_count = len(all_genes)
        if gene_count >= 5:
            score += 20
            flags.append(QualityFlag(level="ok",
                message=f"{gene_count} unique genes identified — strong biological diversity"))
        elif gene_count >= 3:
            score += 10
            flags.append(QualityFlag(level="warn",
                message=f"Moderate gene diversity ({gene_count} genes) — hypotheses may overlap"))
        else:
            flags.append(QualityFlag(level="warn",
                message=f"Low gene diversity — hypotheses cluster around only {gene_count} gene(s)"))

        # ── 3. Hypothesis strength (20 pts) ──────────────────────────────────
        strong_count = sum(1 for h in hypotheses if h.status == "Strong")
        moderate_count = sum(1 for h in hypotheses if h.status == "Moderate")
        if strong_count >= 1:
            score += 20
            flags.append(QualityFlag(level="ok",
                message=f"{strong_count} strong hypothesis(es) with solid paper support"))
        elif moderate_count >= 1:
            score += 10
            flags.append(QualityFlag(level="warn",
                message="No strong hypotheses found — evidence is moderate at best"))
        else:
            flags.append(QualityFlag(level="warn",
                message="All hypotheses are exploratory — consider a more studied disease/gene"))

        # ── 4. Literature recency (20 pts) ────────────────────────────────────
        if papers:
            cutoff = self.CURRENT_YEAR - 10
            recent = sum(1 for p in papers if isinstance(p, dict) and p.get("year", 0) >= cutoff)
            recency_pct = recent / len(papers) * 100
            if recency_pct >= 50:
                score += 20
                flags.append(QualityFlag(level="ok",
                    message=f"{recency_pct:.0f}% of papers are from the last 10 years — up-to-date literature"))
            elif recency_pct >= 25:
                score += 10
                flags.append(QualityFlag(level="warn",
                    message=f"Only {recency_pct:.0f}% of papers are recent — findings may not reflect latest research"))
            else:
                flags.append(QualityFlag(level="warn",
                    message="Literature skews old — most papers are more than 10 years old"))
        else:
            flags.append(QualityFlag(level="warn", message="No papers retrieved — cannot assess recency"))

        # ── 5. Evidence coverage (20 pts) ─────────────────────────────────────
        if hypotheses:
            avg_evidence = sum(h.evidence_count for h in hypotheses) / len(hypotheses)
            if avg_evidence >= 3:
                score += 20
                flags.append(QualityFlag(level="ok",
                    message=f"Good evidence coverage — {avg_evidence:.1f} supporting papers per hypothesis on average"))
            elif avg_evidence >= 1:
                score += 10
                flags.append(QualityFlag(level="warn",
                    message=f"Low evidence per hypothesis ({avg_evidence:.1f} papers avg) — support is thin"))
            else:
                flags.append(QualityFlag(level="warn",
                    message="Most hypotheses have zero paper support — results are speculative"))
        else:
            flags.append(QualityFlag(level="warn", message="No hypotheses generated"))

        # ── Bonus tip flags (no score impact) ────────────────────────────────
        if len(query.split()) < 4:
            flags.append(QualityFlag(level="tip",
                message="Query is very short — try adding a disease name or gene for richer results"))

        if hypotheses and all(h.status == "Exploratory" for h in hypotheses):
            flags.append(QualityFlag(level="tip",
                message="All results are exploratory — consider querying a well-studied disease (e.g. Alzheimer's, Breast Cancer)"))

        if hypotheses and hypotheses[0].confidence < 50:
            flags.append(QualityFlag(level="tip",
                message=f"Top hypothesis confidence is only {hypotheses[0].confidence:.0f}% — treat results as preliminary"))

        # ── Grade + summary ───────────────────────────────────────────────────
        if score >= 80:
            grade = "Excellent"
            summary_line = "High-quality run — strong evidence and broad gene coverage."
        elif score >= 60:
            grade = "Good"
            summary_line = "Good quality run with minor gaps in evidence or diversity."
        elif score >= 40:
            grade = "Fair"
            summary_line = "Fair quality — some aspects of the pipeline need improvement."
        else:
            grade = "Weak"
            summary_line = "Weak results — try a more specific or well-studied query."

        print(f"✓ EvaluatorAgent: score={score}/100 grade={grade} flags={len(flags)}")

        return EvaluationReport(
            health_score=score,
            grade=grade,
            flags=flags,
            summary_line=summary_line,
        )
