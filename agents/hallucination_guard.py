"""
Hallucination and faithfulness guard — runs after hypothesis generation.

Hallucination:  a gene appears in a hypothesis but has no NCBI Gene ID
                (i.e. the LLM invented a gene symbol that doesn't exist).

Faithfulness:   a gene appears in a hypothesis but is never mentioned in
                any of the retrieved papers (the hypothesis is not grounded
                in the literature the pipeline actually read).

Neither check removes hypotheses; instead they apply confidence penalties
and add audit flags so results are transparent and auditable.
"""
from __future__ import annotations
import re
import logging
from models.schemas import Hypothesis

logger = logging.getLogger("genesight")

# Per-gene confidence penalties
_HALLUCINATION_PENALTY = 15   # gene not in NCBI DB at all
_FAITHFULNESS_PENALTY  =  8   # gene not mentioned in any paper


class HallucinationGuard:

    def run(
        self,
        hypotheses: list[Hypothesis],
        db_data: list[dict],
        papers: list[dict],
    ) -> tuple[list[Hypothesis], list[str]]:
        """
        Returns (updated_hypotheses, audit_log).
        audit_log is a list of human-readable strings describing every
        hallucination and faithfulness violation found.
        """
        # Genes confirmed to exist in NCBI (have a real Gene ID)
        verified_genes: set[str] = {
            d["gene"].upper()
            for d in db_data
            if isinstance(d, dict) and d.get("ncbi_id")
        }

        # All text from retrieved papers (titles + abstracts), uppercased for matching
        paper_corpus = " ".join(
            (p.get("title", "") + " " + p.get("abstract", "")).upper()
            for p in papers
        )

        audit_log: list[str] = []

        for hyp in hypotheses:
            hallucinated: list[str] = []
            unfaithful: list[str] = []

            for gene in hyp.genes:
                g_upper = gene.upper()

                # ── Hallucination check ──────────────────────────────────────
                # Only fire if we actually got gene data back from NCBI.
                # If db_data is empty (API failure), skip this check.
                if verified_genes and g_upper not in verified_genes:
                    hallucinated.append(g_upper)

                # ── Faithfulness check ───────────────────────────────────────
                # Whole-word search in the paper corpus.
                if not re.search(r"\b" + re.escape(g_upper) + r"\b", paper_corpus):
                    unfaithful.append(g_upper)

            if hallucinated:
                msg = (
                    f"HALLUCINATION | '{hyp.title[:60]}' | "
                    f"genes {hallucinated} not found in NCBI Gene DB"
                )
                audit_log.append(msg)
                logger.warning(msg)
                penalty = len(hallucinated) * _HALLUCINATION_PENALTY
                hyp.confidence = max(10.0, hyp.confidence - penalty)
                # Downgrade status one tier
                if hyp.status == "Strong":
                    hyp.status = "Moderate"
                elif hyp.status == "Moderate":
                    hyp.status = "Exploratory"

            if unfaithful:
                msg = (
                    f"FAITHFULNESS | '{hyp.title[:60]}' | "
                    f"genes {unfaithful} not mentioned in any retrieved paper"
                )
                audit_log.append(msg)
                logger.warning(msg)
                penalty = len(unfaithful) * _FAITHFULNESS_PENALTY
                hyp.confidence = max(10.0, hyp.confidence - penalty)

            if not hallucinated and not unfaithful:
                logger.info(
                    "✓ GROUNDED | '%s' | all genes verified + paper-faithful",
                    hyp.title[:60],
                )

        if audit_log:
            logger.warning(
                "HallucinationGuard found %d issue(s) across %d hypotheses",
                len(audit_log), len(hypotheses),
            )
        else:
            logger.info(
                "HallucinationGuard: all %d hypotheses fully grounded", len(hypotheses)
            )

        return hypotheses, audit_log
