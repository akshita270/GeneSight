"""
Citation validation — two checks on hypothesis citations, run after HallucinationGuard.

1. PMID provenance  — every supporting_pmid must come from the retrieved paper set.
                      LLMs sometimes invent plausible-looking PMIDs that don't exist
                      or were never fetched. Penalty: -10% confidence per bad PMID.

2. Citation support — for each valid PMID, the cited paper must mention at least one
                      hypothesis gene (word-boundary, case-insensitive).
                      A paper that never mentions the gene can't support the claim.
                      Penalty: -5% per unsupported citation.

Neither check drops hypotheses; both append to the audit_log so results stay auditable.
"""
from __future__ import annotations
import re
import logging
from models.schemas import Hypothesis

logger = logging.getLogger("genesight")

_HALLUCINATED_PMID_PENALTY = 10  # per PMID not in retrieved set
_UNSUPPORTED_CITE_PENALTY  =  5  # per citation whose paper doesn't mention the genes


class CitationValidator:

    def run(
        self,
        hypotheses: list[Hypothesis],
        papers: list[dict],
    ) -> tuple[list[Hypothesis], list[str]]:
        """
        Returns (updated_hypotheses, citation_flags).
        citation_flags should be appended to the job's audit_log.
        """
        paper_by_pmid: dict[str, dict] = {
            str(p.get("pmid", "")): p for p in papers if p.get("pmid")
        }
        retrieved_pmids: set[str] = set(paper_by_pmid.keys())

        citation_flags: list[str] = []

        for hyp in hypotheses:
            raw = getattr(hyp, "supporting_pmids", None) or []
            supporting = [str(pid) for pid in raw if pid]

            if not supporting:
                continue

            # ── 1. PMID provenance ───────────────────────────────────────────
            hallucinated = [pid for pid in supporting if pid not in retrieved_pmids]
            if hallucinated:
                msg = (
                    f"CITATION_HALLUCINATED | '{hyp.title[:60]}' | "
                    f"PMIDs {hallucinated} not in retrieved paper set"
                )
                citation_flags.append(msg)
                logger.warning(msg)
                penalty = len(hallucinated) * _HALLUCINATED_PMID_PENALTY
                hyp.confidence = max(10.0, hyp.confidence - penalty)

            # ── 2. Citation support ──────────────────────────────────────────
            valid_pmids = [pid for pid in supporting if pid in retrieved_pmids]
            genes_upper = [g.upper() for g in (hyp.genes or [])]

            if valid_pmids and genes_upper:
                unsupported = []
                for pid in valid_pmids:
                    paper = paper_by_pmid[pid]
                    corpus = (
                        paper.get("title", "") + " " + paper.get("abstract", "")
                    ).upper()
                    mentioned = any(
                        re.search(r"\b" + re.escape(g) + r"\b", corpus)
                        for g in genes_upper
                    )
                    if not mentioned:
                        unsupported.append(pid)

                if unsupported:
                    msg = (
                        f"CITATION_UNSUPPORTED | '{hyp.title[:60]}' | "
                        f"{len(unsupported)}/{len(valid_pmids)} cited papers "
                        f"don't mention any hypothesis gene"
                    )
                    citation_flags.append(msg)
                    logger.warning(msg)
                    penalty = len(unsupported) * _UNSUPPORTED_CITE_PENALTY
                    hyp.confidence = max(10.0, hyp.confidence - penalty)

        if citation_flags:
            logger.warning(
                "CitationValidator: %d citation issue(s) across %d hypotheses",
                len(citation_flags), len(hypotheses),
            )
        else:
            logger.info(
                "CitationValidator: all %d hypotheses have clean citations",
                len(hypotheses),
            )

        return hypotheses, citation_flags
