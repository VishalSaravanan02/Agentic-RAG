# =============================================================================
# decomposer.py — Decision 2: Sub-query Decomposition
# Breaks a multi-hop question into an ordered list of simpler sub-questions.
# =============================================================================

import re
from src.core.llm_client import call_llm
from src.core.config import DEV_MODEL

DECOMPOSITION_PROMPT_TEMPLATE = """Break this question into a sequence of simpler sub-questions that must be answered in order. Return them as a numbered list.

Rules:
- Each sub-question must be a self-contained factual lookup that names specific entities.
- Do NOT include comparison, reasoning, or analysis steps — only questions that retrieve a specific fact.
- Every sub-question must be fully written out and searchable on its own. Never refer to another sub-question or its answer. If a later lookup depends on something unknown, describe that thing using the identifying details given in the original question.

Example: for "What year was the university attended by the author of the novel Middlemarch founded?", good sub-questions are:
1. Who is the author of the novel Middlemarch?
2. What university did the author of the novel Middlemarch attend?
3. What year was the university attended by the author of the novel Middlemarch founded?

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

    Validation: at least 2 sub-questions, none containing placeholders
    or references to other sub-questions. On validation failure, the
    retry includes corrective feedback (at temperature 0, retrying an
    identical prompt reproduces the identical output — the prompt must
    change for the retry to be useful). Falls back to the original
    question as a single sub-query after 3 failed attempts.
    """
    base_prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(question=question)
    prompt = base_prompt

    max_retries = 3
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_retries):
        llm_result = call_llm(prompt, model=model, max_tokens=2000, temperature=0.0)
        total_input_tokens += llm_result["input_tokens"]
        total_output_tokens += llm_result["output_tokens"]

        sub_questions = _parse_numbered_list(llm_result["text"])

        # Contract enforcement: reject placeholder/self-referential sub-questions
        bad_patterns = ["[", "]", "sub-question", "identified above",
                        "mentioned above", "from the previous", "the answer to",
                        "in the question", "from the question",
                        "aforementioned", "the above"]
        has_bad = any(
            any(p in sq.lower() for p in bad_patterns)
            for sq in sub_questions
        )

        if len(sub_questions) >= 2 and not has_bad:
            return {
                "sub_questions": sub_questions,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }

        print(f"Decomposition invalid (attempt {attempt + 1}/{max_retries}): "
              f"got {len(sub_questions)} sub-questions"
              f"{', contains placeholder/reference' if has_bad else ''}")

        # Corrective feedback: change the prompt so the retry differs
        if has_bad:
            prompt = base_prompt + (
                "\n\nIMPORTANT: Your previous attempt used placeholders or "
                "references to other sub-questions (such as bracketed text "
                "or phrases like 'from sub-question 1'). That is not allowed. "
                "Rewrite ALL sub-questions so each one is fully self-contained "
                "and searchable, repeating the identifying details from the "
                "original question wherever an unknown entity is needed."
            )

    # Fallback: treat the original question as a single sub-question
    print(f"Decomposition failed after {max_retries} attempts. "
          f"Falling back to single sub-question for: {question}")
    return {
        "sub_questions": [question],
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens
    }