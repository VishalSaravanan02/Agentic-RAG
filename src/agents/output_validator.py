# =============================================================================
# output_validator.py — Shared helper for validating YES/NO LLM outputs
# Used by hop_classifier.py (Decision 1) and sufficiency_checker.py (Decision 3)
# Pure string logic only — no imports from elsewhere in src/
# =============================================================================

def validate_yes_no(response: str) -> str | None:
    """
    Validate that an LLM response starts with YES or NO.

    Args:
        response: Raw text response from the LLM

    Returns:
        'YES' or 'NO' if valid, None if invalid (signals a retry is needed)
    """
    if not response:
        return None

    # Strip whitespace and common markdown formatting characters (*, _, #)
    cleaned = response.strip().upper()
    cleaned = cleaned.lstrip("*_# ")

    if cleaned.startswith("YES"):
        return "YES"
    elif cleaned.startswith("NO"):
        return "NO"
    else:
        return None