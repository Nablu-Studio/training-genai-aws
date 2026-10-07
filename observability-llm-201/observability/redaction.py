"""Redacts emails and AWS access keys (AKIA) from text for safe observability logging.

Parent lab: observability-llm-201.
AWS Services: none (local execution).
Generated artifacts: aucun (fonction utilitaire).
Mode : transformation locale, aucun appel AWS.
"""

import re


def redact(text: str) -> str:
    # Standard email regex + AKIA[0-9A-Z]{16} (fixed length of AWS Access Key IDs).
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)
    text = re.sub(r"AKIA[0-9A-Z]{16}", "[secret]", text)
    return text
