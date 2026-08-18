"""
RAGAS-inspired RAG evaluation metrics — computed per pipeline run.

Three dimensions:
  context_relevance   — do the retrieved papers match the query?
  answer_groundedness — are hypothesis genes traceable to the paper corpus?
  answer_relevance    — do hypotheses address the disease in the query?

These are stored on the job object and returned via /trace/{job_id}.
All scores are floats in [0.0, 1.0]. Higher is better.
"""
from __future__ import annotations
import re
import math


def context_relevance(papers: list[dict], query: str) -> float:
    """
    TF-IDF-inspired relevance: for each query token, check what fraction
    of papers contain it. Average across tokens → context relevance score.

    Handles edge cases (empty papers, trivial queries) gracefully.
    """
    if not papers or not query.strip():
        return 0.0

    query_tokens = [
        t for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", query.lower())
        if t not in _STOPWORDS
    ]
    if not query_tokens:
        return 0.0

    N = len(papers)
    # Pre-build lowercased paper texts and compile patterns once — avoids
    # re-compiling the same regex inside the inner loop on every call.
    paper_texts = [
        (p.get("title", "") + " " + p.get("abstract", "")).lower()
        for p in papers
    ]
    token_scores: list[float] = []
    for tok in set(query_tokens):
        pat = re.compile(r"\b" + re.escape(tok) + r"\b")
        doc_freq = sum(1 for text in paper_texts if pat.search(text))
        idf = math.log(N / (doc_freq + 1) + 1)
        presence_ratio = doc_freq / N
        token_scores.append(presence_ratio * idf)

    if not token_scores:
        return 0.0

    raw = sum(token_scores) / len(token_scores)
    max_possible = math.log(N + 1) if N > 0 else 1.0
    return round(min(1.0, raw / max_possible), 4)


def answer_groundedness(hypotheses: list[dict], papers: list[dict]) -> float:
    """
    For each hypothesis gene, check if it appears (word boundary match)
    in the paper corpus (combined titles + abstracts).

    Fraction of all hypothesis genes that are grounded in the corpus.
    """
    if not hypotheses or not papers:
        return 0.0

    # Build a word set once — O(corpus) — then lookup is O(1) per gene.
    # Gene symbols are short uppercase tokens so word-boundary splitting works.
    corpus_words: set[str] = set()
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).upper()
        corpus_words.update(re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", text))

    all_genes: list[str] = []
    for h in hypotheses:
        all_genes.extend(g.upper() for g in h.get("genes", []))

    if not all_genes:
        return 0.0

    grounded = sum(1 for g in all_genes if g in corpus_words)
    return round(grounded / len(all_genes), 4)


def answer_relevance(hypotheses: list[dict], query: str) -> float:
    """
    Fraction of hypotheses whose statement references at least one
    meaningful query term (non-stopword, 3+ chars).
    """
    if not hypotheses or not query.strip():
        return 0.0

    query_tokens = {
        t for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", query.lower())
        if t not in _STOPWORDS
    }
    if not query_tokens:
        return 0.0

    relevant = 0
    for h in hypotheses:
        text = (h.get("statement", "") + " " + h.get("title", "")).lower()
        text_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text))
        if query_tokens & text_tokens:
            relevant += 1

    return round(relevant / len(hypotheses), 4)


def compute(
    papers: list[dict],
    hypotheses: list[dict],
    query: str,
) -> dict[str, float]:
    """Convenience wrapper — returns all three metrics plus an aggregate."""
    cr  = context_relevance(papers, query)
    ag  = answer_groundedness(hypotheses, papers)
    ar  = answer_relevance(hypotheses, query)
    agg = round((cr + ag + ar) / 3, 4)
    return {
        "context_relevance":   cr,
        "answer_groundedness": ag,
        "answer_relevance":    ar,
        "rag_aggregate":       agg,
    }


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was", "has",
    "have", "been", "its", "our", "their", "from", "into", "does",
    "not", "but", "also", "some", "any", "all", "can", "may", "more",
    "other", "which", "there", "than", "then", "when", "what", "how",
}
