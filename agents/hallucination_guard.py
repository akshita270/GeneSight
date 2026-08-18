"""
Hallucination and faithfulness guard — runs after hypothesis generation.

Hallucination:  a gene appears in a hypothesis but has no NCBI Gene ID
                (the LLM invented a gene symbol that doesn't exist).

Faithfulness:   a gene appears in a hypothesis but is never mentioned in
                any of the retrieved papers (the hypothesis is not grounded
                in the literature the pipeline actually read).

Neither check removes hypotheses — they apply confidence penalties and add
audit flags so results are transparent and auditable.
"""
from __future__ import annotations
import re
import logging
from models.schemas import Hypothesis
from utils.text import corpus_words

logger = logging.getLogger("genesight")

_HALLUCINATION_PENALTY = 15
_FAITHFULNESS_PENALTY  =  8


class HallucinationGuard:

    def run(
        self,
        hypotheses: list[Hypothesis],
        db_data: list[dict],
        papers: list[dict],
    ) -> tuple[list[Hypothesis], list[str]]:
        verified_genes: set[str] = {
            d["gene"].upper()
            for d in db_data
            if isinstance(d, dict) and d.get("ncbi_id")
        }

        # Build word set once — O(corpus) — then O(1) per gene instead of
        # running a regex over the joined corpus string per gene.
        paper_words = corpus_words(papers)

        audit_log: list[str] = []

        for hyp in hypotheses:
            hallucinated: list[str] = []
            unfaithful: list[str]   = []

            for gene in hyp.genes:
                g_upper = gene.upper()

                if verified_genes and g_upper not in verified_genes:
                    hallucinated.append(g_upper)

                if g_upper not in paper_words:
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
