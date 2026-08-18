"""
Shared text utilities — canonical helpers used across agents and eval.

Centralising here means a change to what "paper text" means (e.g. adding
authors), or a new injection pattern, is made in exactly one place.
"""
from __future__ import annotations
import re


# ── Paper text helpers ────────────────────────────────────────────────────────

def paper_text(paper: dict) -> str:
    """Return concatenated title + abstract for a paper dict."""
    return paper.get("title", "") + " " + paper.get("abstract", "")


def paper_text_upper(paper: dict) -> str:
    return paper_text(paper).upper()


def corpus_words(papers: list[dict]) -> set[str]:
    """
    Build a set of uppercase gene-symbol-shaped tokens from the paper corpus.
    O(corpus) build, O(1) membership test — use instead of repeated regex
    searches over the joined string.
    """
    words: set[str] = set()
    for p in papers:
        words.update(re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", paper_text_upper(p)))
    return words


def paper_corpus(papers: list[dict]) -> str:
    """Join all paper texts into one uppercased string for regex searching."""
    return " ".join(paper_text_upper(p) for p in papers)


# ── LLM output helpers ────────────────────────────────────────────────────────

def strip_md_fences(text: str) -> str:
    """Strip ```json / ``` markdown code fences that LLMs sometimes wrap output in."""
    return re.sub(r"```json|```", "", text).strip()


# ── Security: canonical injection pattern list ────────────────────────────────
# Single source of truth — imported by guardrails.py AND paper_sanitiser.py.
# Adding a pattern here automatically protects both the input and paper layers.

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|all|above|your)\s+(instructions?|prompts?|rules?|system)", re.I),
    re.compile(r"disregard\s+(previous|all|instructions?|rules?|constraints?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I),
    re.compile(r"forget\s+(your|all|previous|everything)", re.I),
    re.compile(r"new\s+(instruction|task|system\s+prompt)", re.I),
    re.compile(r"override\s+(your|all|previous)\s+(training|instructions?|guidelines?|safety)", re.I),
    re.compile(r"act\s+as\s+(if|though)\s+you", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bdan\s+mode\b", re.I),
    re.compile(r"<\s*script[\s>]", re.I),
    re.compile(r"\bprompt\s+injection\b", re.I),
    # Indirect injection patterns used by open-source LLMs as template tags
    re.compile(r"\[INST\]|\[\/INST\]", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),
    re.compile(r"###\s*(Human|Assistant|System)\s*:", re.I),
    re.compile(r"SYSTEM\s*:\s*(you|your|ignore|disregard)", re.I),
]
