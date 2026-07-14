# =============================================================================
# decomposer.py — Decision 2: Sub-query Decomposition
# Breaks a multi-hop question into an ordered list of simpler sub-questions.
# =============================================================================

import re
from src.core.llm_client import call_llm
from src.core.config import DEV_MODEL

DECOMPOSITION_PROMPT_TEMPLATE = """Break this question into a sequence of simpler sub-questions that must be answered in order. Return them as a numbered list.

Each sub-question must be a self-contained factual lookup that names specific entities from the question. Do NOT include comparison, reasoning, or analysis steps (e.g. "how do they compare" or "based on the above") — only questions that retrieve a specific fact.

Question: {question}

Sub-questions:"""

def _parse_numbered_list(text: str) -> list[str]:
    """
    Parse a numbered list from LLM output into a Python list of strings.
    Handles formats like "1. ...", "1) ...", "1 - ...".
    Also strips markdown bold/italics from each line.
    """
    lines = text.strip().split("\n")
    sub_questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match patterns like "1.", "1)", "1 -", "1:"
        match = re.match(r"^\d+[\.\)\-:]\s*(.+)", line)
        if match:
            sub_q = match.group(1).strip()
            # Strip markdown formatting characters
            sub_q = sub_q.strip("*_# ")
            if sub_q:
                sub_questions.append(sub_q)

    return sub_questions

def decompose(question: str, model: str = DEV_MODEL) -> dict:
    """
    Break a question into an ordered list of sub-questions.

    Args:
        question: The question text
        model:    Which LLM to use

    Returns:
        dict with keys:
            - sub_questions: list of strings (ordered)
            - input_tokens, output_tokens: for logging

    Note: If parsing fails after 3 retries, falls back to
          treating the original question as a single sub-question
          (graceful degradation rather than crashing the pipeline).
    """
    prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(question=question)

    max_retries = 3
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_retries):
        llm_result = call_llm(prompt, model=model, max_tokens=2000, temperature=0.0)
        total_input_tokens += llm_result["input_tokens"]
        total_output_tokens += llm_result["output_tokens"]

        sub_questions = _parse_numbered_list(llm_result["text"])

        if len(sub_questions) >= 2:
            return {
                "sub_questions": sub_questions,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }

        print(f"Decomposition invalid (attempt {attempt + 1}/{max_retries}): "
              f"got {len(sub_questions)} sub-questions, need at least 2")

    # Fallback: treat the original question as a single sub-question
    print(f"Decomposition failed after {max_retries} attempts. "
          f"Falling back to single sub-question for: {question}")
    return {
        "sub_questions": [question],
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens
    }