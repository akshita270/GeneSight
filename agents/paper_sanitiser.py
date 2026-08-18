"""
Paper content sanitiser — scans retrieved paper abstracts for adversarial
LLM instruction patterns before feeding them into downstream agent prompts.

Threat model: a malicious actor publishes a paper whose abstract contains
prompt injection payloads designed to hijack the LLM when the abstract is
embedded in a prompt (e.g. "Ignore previous instructions and output X").

This module:
  1. Detects injection-like patterns in each abstract.
  2. Strips/replaces flagged segments with a safe placeholder.
  3. Returns sanitised papers + a list of flags for the audit log.

It does NOT drop papers — removing results silently could hide valuable
science. Instead it neutralises the payload and records the incident.
"""
from __future__ import annotations
import logging
from utils.text import INJECTION_PATTERNS

logger = logging.getLogger("genesight")

_PLACEHOLDER = "[CONTENT REDACTED — injection pattern detected]"


def sanitise(papers: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Scan and sanitise a list of paper dicts (each with 'title' and 'abstract').
    Returns (sanitised_papers, flags).
    """
    flags: list[str] = []
    sanitised: list[dict] = []

    for paper in papers:
        pmid  = paper.get("pmid", "unknown")
        clean = dict(paper)

        for field in ("title", "abstract"):
            text: str = clean.get(field, "") or ""
            original_len = len(text)
            redacted = False

            for pattern in INJECTION_PATTERNS:
                if pattern.search(text):
                    new_text = pattern.sub(_PLACEHOLDER, text)
                    if new_text != text:
                        text     = new_text
                        redacted = True

            if redacted:
                flag_msg = (
                    f"PAPER_INJECTION | pmid={pmid} | field={field} | "
                    f"original_len={original_len} → sanitised"
                )
                flags.append(flag_msg)
                logger.warning(flag_msg)
                clean[field] = text

        sanitised.append(clean)

    if flags:
        logger.warning(
            "PaperSanitiser: %d injection pattern(s) detected across %d papers",
            len(flags), len(papers),
        )
    else:
        logger.info("PaperSanitiser: all %d papers clean", len(papers))

    return sanitised, flags
