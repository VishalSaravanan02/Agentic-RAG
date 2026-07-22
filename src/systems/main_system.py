# =============================================================================
# main_system.py — Agentic Multi-Hop RAG (the Main System)
# Wires together all 5 decision mechanisms.
#
# The retrieve-and-hops pipeline (D1, D2, the iterative loop, D3, D4, and
# context accumulation) lives in _shared_retrieval.py, shared verbatim with
# Ablation 1 so the only difference between the two systems is a single flag
# (use_decomposition). This file owns D5 (synthesis) and log-entry assembly.
# =============================================================================

import time
from src.core.logger import log_result
from src.core.config import DEV_MODEL

from src.agents.answer_synthesizer import synthesize
from src.systems._shared_retrieval import run_retrieval_pipeline, _format_chunk


def run_main_system(question: str, question_id: str, gold_answer: str,
                     model: str = DEV_MODEL) -> dict:
    """
    Run the full agentic multi-hop RAG pipeline on a single question.

    Returns a dict matching the locked log schema (same fields as baseline_a).
    """
    start_time = time.time()

    # --- D1 -> (D2) -> iterative retrieval loop (D3/D4) -> context -----------
    # Shared with Ablation 1. use_decomposition=True runs D2 and follows the
    # planned sub-queries before reactive mode (full Main System behaviour).
    pipeline = run_retrieval_pipeline(question, model=model, use_decomposition=True)

    total_input_tokens = pipeline["input_tokens"]
    total_output_tokens = pipeline["output_tokens"]
    latency_per_hop = pipeline["latency_per_hop"]
    docs_retrieved_per_hop = pipeline["docs_retrieved_per_hop"]
    all_retrieved_chunks = pipeline["all_retrieved_chunks"]

    # --- DECISION 5: Grounded answer synthesis -------------------------------
    final_context = "\n\n".join(_format_chunk(d) for d in all_retrieved_chunks)
    synth_start = time.time()
    synth_result = synthesize(question, final_context, model=model)
    total_input_tokens += synth_result["input_tokens"]
    total_output_tokens += synth_result["output_tokens"]
    latency_per_hop.append((time.time() - synth_start) * 1000)

    total_latency_ms = (time.time() - start_time) * 1000

    log_entry = {
        "question_id": question_id,
        "question": question,
        "system_name": "main_system",
        "hop_necessity_classification": pipeline["hop_necessity_classification"],
        "num_hops": len(docs_retrieved_per_hop),
        "sub_queries_generated": pipeline["sub_queries_generated"],
        "docs_retrieved_per_hop": docs_retrieved_per_hop,
        "queries_per_hop": pipeline["queries_per_hop"],
        "stop_condition_triggered": pipeline["stop_condition"],
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "latency_per_hop_ms": latency_per_hop,
        "total_latency_ms": total_latency_ms,
        "final_answer": synth_result["answer"],
        "gold_answer": gold_answer
    }

    return log_entry


def run_and_log_main_system(question: str, question_id: str, gold_answer: str,
                              split: str, model: str = DEV_MODEL) -> dict:
    """Run the Main System and log the result to disk in one step."""
    log_entry = run_main_system(question, question_id, gold_answer, model=model)
    log_result(log_entry, system_name="main_system", split=split)
    return log_entry