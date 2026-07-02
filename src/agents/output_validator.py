# =============================================================================
# output_validator.py — Shared helper for validating YES/NO LLM outputs
# Used by hop_classifier.py (Decision 1) and sufficiency_checker.py (Decision 3)
# Pure string logic only — no imports from elsewhere in src/
# =============================================================================

import re

def validate_yes_no(response: str) -> str | None:
    """
    Validate that an LLM response starts with YES or NO.
    Handles reasoning models that wrap output in <think>...</think> tags.

    Args:
        response: Raw text response from the LLM

    Returns:
        'YES' or 'NO' if valid, None if invalid (signals a retry is needed)
    """
    if not response:
        return None

    # Strip <think>...</think> blocks (reasoning model output)
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # If nothing left after stripping think blocks, use original
    if not cleaned:
        cleaned = response.strip()

    # Strip whitespace and common markdown formatting characters
    cleaned = cleaned.upper()
    cleaned = cleaned.lstrip("*_# ")

    if cleaned.startswith("YES"):
        return "YES"
    elif cleaned.startswith("NO"):
        return "NO"
    else:
        return None