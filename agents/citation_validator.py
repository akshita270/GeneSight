"""
Citation validation — two checks on hypothesis citations, run after HallucinationGuard.

1. PMID provenance  — every supporting_pmid must come from the retrieved paper set.
2. Citation support — each cited paper must mention at least one hypothesis gene.

Neither check drops hypotheses; both append to the audit_log.
"""
from __future__ import annotations
import logging
from models.schemas import Hypothesis
from utils.text import corpus_words

logger = logging.getLogger("genesight")

_HALLUCINATED_PMID_PENALTY = 10
_UNSUPPORTED_CITE_PENALTY  =  5


class CitationValidator:

    def run(
        self,
        hypotheses: list[Hypothesis],
        papers: list[dict],
    ) -> tuple[list[Hypothesis], list[str]]:
        paper_by_pmid: dict[str, dict] = {
            str(p.get("pmid", "")): p for p in papers if p.get("pmid")
        }
        retrieved_pmids: set[str] = set(paper_by_pmid.keys())

        citation_flags: list[str] = []

        for hyp in hypotheses:
            raw       = getattr(hyp, "supporting_pmids", None) or []
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
                hyp.confidence = max(10.0, hyp.confidence - len(hallucinated) * _HALLUCINATED_PMID_PENALTY)

            # ── 2. Citation support ──────────────────────────────────────────
            valid_pmids  = [pid for pid in supporting if pid in retrieved_pmids]
            genes_upper  = [g.upper() for g in (hyp.genes or [])]

            if valid_pmids and genes_upper:
                unsupported = []
                for pid in valid_pmids:
                    # Build word set for this single paper and check membership
                    paper_words = corpus_words([paper_by_pmid[pid]])
                    if not any(g in paper_words for g in genes_upper):
                        unsupported.append(pid)

                if unsupported:
                    msg = (
                        f"CITATION_UNSUPPORTED | '{hyp.title[:60]}' | "
                        f"{len(unsupported)}/{len(valid_pmids)} cited papers "
                        f"don't mention any hypothesis gene"
                    )
                    citation_flags.append(msg)
                    logger.warning(msg)
                    hyp.confidence = max(10.0, hyp.confidence - len(unsupported) * _UNSUPPORTED_CITE_PENALTY)

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
