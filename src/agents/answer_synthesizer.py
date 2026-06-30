# =============================================================================
# answer_synthesizer.py — Decision 5: Grounded Answer Synthesis
# Generates the final answer using ONLY retrieved evidence.
# Identical prompt to Baseline A — intentional, isolates retrieval quality
# as the only variable between systems.
# =============================================================================

from src.core.llm_client import call_llm
from src.core.config import DEV_MODEL

SYNTHESIS_PROMPT_TEMPLATE = """Answer the question using only the information retrieved below. Do not use any outside knowledge. If the retrieved information is not enough, say so explicitly.

Question: {question}

Retrieved information:
{context}

Answer:"""

def synthesize(question: str, context: str, model: str = DEV_MODEL) -> dict:
    """
    Generate the final grounded answer from accumulated retrieved context.

    Args:
        question: The original question text
        context:  All accumulated retrieved chunks, joined into one string
        model:    Which LLM to use

    Returns:
        dict with keys:
            - answer: the final answer text
            - input_tokens, output_tokens: for logging
    """
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(question=question, context=context)

    llm_result = call_llm(prompt, model=model, max_tokens=600, temperature=0.0)

    return {
        "answer": llm_result["text"],
        "input_tokens": llm_result["input_tokens"],
        "output_tokens": llm_result["output_tokens"]
    }