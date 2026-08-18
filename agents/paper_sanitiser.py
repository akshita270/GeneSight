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
import re
import logging

logger = logging.getLogger("genesight")

# Patterns that look like prompt injection payloads inside paper text.
# Anchored to catch both heading-level and inline injection attempts.
_INJECTION_PATTERNS: list[re.Pattern] = [
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
    # Common indirect injection patterns found in adversarial NLP literature
    re.compile(r"\[INST\]|\[\/INST\]", re.I),       # Llama-style instruction tags
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),  # ChatML tags
    re.compile(r"###\s*(Human|Assistant|System)\s*:", re.I),
    re.compile(r"SYSTEM\s*:\s*(you|your|ignore|disregard)", re.I),
]

_PLACEHOLDER = "[CONTENT REDACTED — injection pattern detected]"


def sanitise(papers: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Scan and sanitise a list of paper dicts (each with 'title' and 'abstract').

    Returns:
        (sanitised_papers, flags)
        flags — human-readable audit strings, one per flagged paper.
    """
    flags: list[str] = []
    sanitised: list[dict] = []

    for paper in papers:
        pmid = paper.get("pmid", "unknown")
        clean = dict(paper)

        for field in ("title", "abstract"):
            text: str = clean.get(field, "") or ""
            original_len = len(text)
            redacted = False

            for pattern in _INJECTION_PATTERNS:
                if pattern.search(text):
                    # Replace the matched span(s) rather than wiping the whole field
                    new_text = pattern.sub(_PLACEHOLDER, text)
                    if new_text != text:
                        text = new_text
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
