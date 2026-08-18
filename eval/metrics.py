"""
Evaluation metrics for the genomics pipeline.

Metrics are computed from a pipeline result against a benchmark spec.
All values are floats in [0.0, 1.0] unless noted.
"""
from __future__ import annotations
import re
from typing import Any


def gene_validity_rate(result: dict) -> float:
    """
    Fraction of hypothesis genes that appear in the NCBI-verified gene set.
    Requires 'db_data' in result (list of {gene, ncbi_id, ...} dicts).
    Returns 1.0 if db_data is empty (guard skipped).
    """
    db_data: list[dict] = result.get("db_data", [])
    hypotheses: list[dict] = result.get("hypotheses", [])

    if not db_data or not hypotheses:
        return 1.0

    verified = {d["gene"].upper() for d in db_data if d.get("ncbi_id")}
    all_genes: list[str] = []
    for h in hypotheses:
        all_genes.extend(g.upper() for g in h.get("genes", []))

    if not all_genes:
        return 1.0

    valid = sum(1 for g in all_genes if g in verified)
    return round(valid / len(all_genes), 4)


def hallucination_rate(audit_log: list[str]) -> float:
    """
    Fraction of audit log entries that are hallucination flags
    (vs faithfulness flags). Lower is better.
    """
    if not audit_log:
        return 0.0
    h_count = sum(1 for entry in audit_log if entry.startswith("HALLUCINATION"))
    return round(h_count / len(audit_log), 4)


def faithfulness_score(audit_log: list[str], total_hypotheses: int) -> float:
    """
    Fraction of hypotheses with NO faithfulness violations.
    Higher is better.
    """
    if total_hypotheses == 0:
        return 1.0
    violated = sum(1 for e in audit_log if e.startswith("FAITHFULNESS"))
    faithful = max(0, total_hypotheses - violated)
    return round(faithful / total_hypotheses, 4)


def citation_accuracy(audit_log: list[str]) -> float:
    """
    Fraction of audit entries that are NOT citation issues.
    Covers CITATION_HALLUCINATED (PMID not in retrieved set) and
    CITATION_UNSUPPORTED (cited paper doesn't mention hypothesis genes).
    Returns 1.0 when audit_log is empty (no issues found). Higher is better.
    """
    if not audit_log:
        return 1.0
    citation_issues = sum(
        1 for e in audit_log
        if e.startswith(("CITATION_HALLUCINATED", "CITATION_UNSUPPORTED"))
    )
    return round(1.0 - citation_issues / len(audit_log), 4)


def paper_relevance_score(papers: list[dict], query: str) -> float:
    """
    Heuristic: fraction of papers whose title/abstract contains at least
    one token from the query (case-insensitive). Proxy for retrieval precision.
    """
    if not papers:
        return 0.0
    query_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", query.lower()))
    relevant = 0
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        if query_tokens & set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text)):
            relevant += 1
    return round(relevant / len(papers), 4)


def hypothesis_disease_relevance(hypotheses: list[dict], disease_keywords: list[str]) -> float:
    """
    Fraction of hypotheses whose statement/title mentions at least one
    expected disease keyword. Proxy for answer relevance.
    """
    if not hypotheses or not disease_keywords:
        return 1.0
    pattern = re.compile(
        "|".join(re.escape(kw) for kw in disease_keywords), re.IGNORECASE
    )
    relevant = sum(
        1 for h in hypotheses
        if pattern.search(h.get("statement", "") + " " + h.get("title", ""))
    )
    return round(relevant / len(hypotheses), 4)


def expected_gene_recall(hypotheses: list[dict], expected_genes: list[str]) -> float:
    """
    Fraction of benchmark expected_genes that appear in any hypothesis gene list.
    """
    if not expected_genes:
        return 1.0
    found_genes: set[str] = set()
    for h in hypotheses:
        found_genes.update(g.upper() for g in h.get("genes", []))
    hits = sum(1 for g in expected_genes if g.upper() in found_genes)
    return round(hits / len(expected_genes), 4)


def compute_all(
    result: dict,
    benchmark: dict,
    audit_log: list[str],
) -> dict[str, Any]:
    """
    Compute all metrics for a single benchmark result.
    Returns a dict with metric names → values + a pass/fail summary.
    """
    papers: list[dict] = result.get("papers", [])
    hypotheses: list[dict] = result.get("hypotheses", [])

    gvr   = gene_validity_rate(result)
    hr    = hallucination_rate(audit_log)
    fs    = faithfulness_score(audit_log, len(hypotheses))
    ca    = citation_accuracy(audit_log)
    prs   = paper_relevance_score(papers, benchmark["query"])
    hdr   = hypothesis_disease_relevance(hypotheses, benchmark.get("expected_disease_keywords", []))
    egr   = expected_gene_recall(hypotheses, benchmark.get("expected_genes", []))

    # Aggregate score (weighted average — citation_accuracy replaces 5% from faithfulness)
    overall = round(
        0.20 * gvr +
        0.15 * (1 - hr) +
        0.15 * fs +
        0.10 * ca +
        0.15 * prs +
        0.10 * hdr +
        0.15 * egr,
        4,
    )

    checks = {
        "paper_count": len(papers) >= benchmark.get("min_papers", 0),
        "hypothesis_count": len(hypotheses) >= benchmark.get("min_hypotheses", 0),
        "top_confidence": (
            hypotheses[0].get("confidence", 0) >= benchmark.get("min_confidence", 0)
            if hypotheses else False
        ),
    }

    return {
        "gene_validity_rate":           gvr,
        "hallucination_rate":           hr,
        "faithfulness_score":           fs,
        "citation_accuracy":            ca,
        "paper_relevance_score":        prs,
        "hypothesis_disease_relevance": hdr,
        "expected_gene_recall":         egr,
        "overall_score":                overall,
        "checks":                       checks,
        "passed":                       all(checks.values()) and overall >= 0.55,
    }
