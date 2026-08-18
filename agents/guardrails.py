"""
Input guardrails — runs before the pipeline starts.
Checks for prompt injection, off-topic queries, and malformed input.
"""
from __future__ import annotations
import re
from utils.text import INJECTION_PATTERNS

# At least one of these must appear for a query to be considered genomics-related
_GENOMICS_KEYWORDS = {
    "gene", "genes", "dna", "rna", "mrna", "protein", "proteins", "disease",
    "mutation", "variant", "snp", "pathway", "expression", "cancer", "tumor",
    "alzheimer", "parkinson", "diabetes", "brca", "genome", "genomic",
    "chromosome", "allele", "phenotype", "genotype", "biomarker", "therapeutic",
    "molecular", "cell", "signaling", "receptor", "kinase", "immune", "neural",
    "cardiovascular", "obesity", "inflammation", "methylation", "epigenetic",
    "crispr", "sequencing", "gwas", "locus", "exome", "transcriptome",
    "proteome", "metabolome", "drug", "therapy", "treatment", "clinical",
    "association", "risk", "mechanism", "biology", "biological", "oncogene",
    "suppressor", "hereditary", "inherited", "somatic", "germline",
}


class InputGuardrail:
    MIN_LEN = 5
    MAX_LEN = 500

    def check(self, query: str) -> tuple[bool, str]:
        """
        Returns (is_valid, rejection_reason).
        is_valid=True and empty reason means the query is safe to run.
        """
        stripped = query.strip()

        if len(stripped) < self.MIN_LEN:
            return False, "Query is too short. Please enter a genomics research question."
        if len(stripped) > self.MAX_LEN:
            return False, f"Query is too long (max {self.MAX_LEN} characters)."

        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", stripped):
            return False, "Query contains invalid characters."

        lower = stripped.lower()
        for pattern in INJECTION_PATTERNS:
            if pattern.search(lower):
                return False, (
                    "Query contains disallowed instructions. "
                    "Please enter a genuine genomics research question."
                )

        words   = set(re.findall(r"\b[a-z][a-z0-9\-]*\b", lower))
        symbols = set(re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", stripped))
        if not words.intersection(_GENOMICS_KEYWORDS) and not symbols:
            return False, (
                "Query does not appear to be genomics-related. "
                "Please ask about genes, diseases, mutations, or biological pathways. "
                "Example: \"BRCA1 and breast cancer risk\" or \"APOE4 Alzheimer's\""
            )

        return True, ""
