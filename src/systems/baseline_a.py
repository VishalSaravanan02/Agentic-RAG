# =============================================================================
# baseline_a.py — Single-Hop RAG (the controlled comparison baseline)
# One retrieval call, one generation call. No decisions, no iteration.
# =============================================================================

import time
from src.core.retriever import retrieve
from src.core.llm_client import call_llm
from src.core.logger import log_result
from src.core.config import TOP_K, EVAL_MODEL, DEV_MODEL

SYNTHESIS_PROMPT_TEMPLATE = """Answer the question using only the information retrieved below. Do not use any outside knowledge. If the retrieved information is not enough, say so explicitly.

Question: {question}

Retrieved information:
{context}

Answer:"""

def run_baseline_a(question: str, question_id: str, gold_answer: str, model: str = DEV_MODEL) -> dict:
    """
    Run the full Baseline A pipeline on a single question.

    Args:
        question:    The question text
        question_id: HotpotQA question ID (for logging)
        gold_answer: The correct answer (for logging, used later in metrics)
        model:       Which LLM to use (DEV_MODEL for Groq, EVAL_MODEL for OpenAI)

    Returns:
        dict with the full log entry (also written to disk via logger.py)
    """
    start_time = time.time()

    # Step 1 + 2: Retrieve top-k documents
    retrieved_docs = retrieve(question, k=TOP_K)
    context = "\n\n".join(
        f"[{doc['metadata']['article_title']}] {doc['text']}"
        for doc in retrieved_docs
    )
    doc_ids = [f"{doc['metadata']['article_title']}_{doc['metadata']['chunk_index']}"
               for doc in retrieved_docs]

    # Step 3: Synthesize answer
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(question=question, context=context)
    llm_result = call_llm(prompt, model=model, max_tokens=256, temperature=0.0)

    total_latency_ms = (time.time() - start_time) * 1000

    # Step 4: Build the log entry (schema must match every other system)
    log_entry = {
        "question_id": question_id,
        "question": question,
        "system_name": "baseline_a",
        "hop_necessity_classification": "N/A",
        "num_hops": 1,
        "sub_queries_generated": [],
        "docs_retrieved_per_hop": [doc_ids],
        "stop_condition_triggered": "N/A",
        "input_tokens": llm_result["input_tokens"],
        "output_tokens": llm_result["output_tokens"],
        "latency_per_hop_ms": [total_latency_ms],
        "total_latency_ms": total_latency_ms,
        "final_answer": llm_result["text"],
        "gold_answer": gold_answer
    }

    return log_entry

def run_and_log_baseline_a(question: str, question_id: str, gold_answer: str,
                             split: str, model: str = DEV_MODEL) -> dict:
    """
    Run Baseline A and log the result to disk in one step.
    This is what run_evaluation.py will call.
    """
    log_entry = run_baseline_a(question, question_id, gold_answer, model=model)
    log_result(log_entry, system_name="baseline_a", split=split)
    return log_entry