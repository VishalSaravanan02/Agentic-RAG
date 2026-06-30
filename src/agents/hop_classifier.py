# =============================================================================
# hop_classifier.py — Decision 1: Hop Necessity Classification
# Before any retrieval, decide whether the question needs multiple hops.
# =============================================================================

from src.core.llm_client import call_llm
from src.agents.output_validator import validate_yes_no
from src.core.config import DEV_MODEL

CLASSIFICATION_PROMPT_TEMPLATE = """Does answering this question require retrieving information from multiple separate sources or reasoning steps? Answer YES or NO and explain why.

Question: {question}

Answer:"""

def classify(question: str, model: str = DEV_MODEL) -> dict:
    """
    Decide whether a question requires multi-hop retrieval.

    Args:
        question: The question text
        model:    Which LLM to use

    Returns:
        dict with keys:
            - classification: 'YES' or 'NO'
            - explanation: the full LLM response text
            - input_tokens, output_tokens: for logging

    Raises:
        Exception if validation fails after 3 retries
    """
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(question=question)

    max_retries = 3
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_retries):
        llm_result = call_llm(prompt, model=model, max_tokens=400, temperature=0.0)
        total_input_tokens += llm_result["input_tokens"]
        total_output_tokens += llm_result["output_tokens"]

        classification = validate_yes_no(llm_result["text"])

        if classification is not None:
            return {
                "classification": classification,
                "explanation": llm_result["text"],
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }

        print(f"Hop classification invalid (attempt {attempt + 1}/{max_retries}): {llm_result['text'][:100]}")

    # All retries failed — log and raise
    raise Exception(
        f"Hop classification failed validation after {max_retries} attempts for question: {question}"
    )