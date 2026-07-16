# =============================================================================
# output_validator.py — Shared helper for validating YES/NO LLM outputs
# Used by hop_classifier.py (Decision 1) and sufficiency_checker.py (Decision 3)
# Pure string logic only — no imports from elsewhere in src/
# =============================================================================

import re

def validate_yes_no(response: str) -> str | None:
    if not response:
        return None

    # Strip complete <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Also strip incomplete think blocks (no closing tag — token limit cut off)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()

    if not cleaned:
        return None

    cleaned = cleaned.upper()
    cleaned = cleaned.lstrip("*_# ")

    if cleaned.startswith("YES"):
        return "YES"
    elif cleaned.startswith("NO"):
        return "NO"
    else:
        return None