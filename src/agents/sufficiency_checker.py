# =============================================================================
# sufficiency_checker.py — Decision 3: Retrieval Sufficiency Check
# After each hop, decide whether enough evidence has been gathered.
# The 'missing' explanation drives reactive retrieval when sub-questions
# are exhausted but sufficiency is still NO.
# =============================================================================

from src.core.llm_client import call_llm
from src.agents.output_validator import validate_yes_no
from src.core.config import DEV_MODEL

SUFFICIENCY_PROMPT_TEMPLATE = """You are checking whether the retrieved information is enough to fully answer the question.

Original question: {question}

Information retrieved so far:
{context}

Instructions:
1. Identify the specific facts needed to answer the question completely
2. For each required fact, check whether it is EXPLICITLY STATED in the retrieved information above
3. Answer YES only if ALL required facts are explicitly present in the text
4. Answer NO if ANY required fact is missing, implied but not stated, or requires outside knowledge

You MUST begin your response with the single word YES or NO, then list which facts are present and which are missing.

If you answer NO, end your response with a single line in exactly this format:
MISSING: <a short, self-contained search query describing only the missing information>

Begin your answer with YES or NO:"""

def check(question: str, accumulated_context: str, model: str = DEV_MODEL) -> dict:
    """
    Check whether enough evidence has been gathered to answer the question.

    Args:
        question:             The original question text
        accumulated_context:  All chunks retrieved so far, joined into one string
        model:                Which LLM to use

    Returns:
        dict with keys:
            - sufficient: bool (True if YES, False if NO)
            - missing: str (explanation of what's missing, or why it's sufficient)
            - input_tokens, output_tokens: for logging

    Raises:
        Exception if validation fails after 3 retries
    """
    prompt = SUFFICIENCY_PROMPT_TEMPLATE.format(
        question=question,
        context=accumulated_context
    )

    max_retries = 3
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_retries):
        llm_result = call_llm(prompt, model=model, max_tokens=2000, temperature=0.0)
        total_input_tokens += llm_result["input_tokens"]
        total_output_tokens += llm_result["output_tokens"]

        classification = validate_yes_no(llm_result["text"])

        if classification is not None:
            return {
                "sufficient": classification == "YES",
                "missing": llm_result["text"],
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }

        print(f"Sufficiency check invalid (attempt {attempt + 1}/{max_retries}): {llm_result['text'][:100]}")

    raise Exception(
        f"Sufficiency check failed validation after {max_retries} attempts for question: {question}"
    )