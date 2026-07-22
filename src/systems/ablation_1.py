# =============================================================================
# ablation_1.py — Agentic Multi-Hop RAG WITHOUT query decomposition (D2)
#
# Identical to the Main System in every respect EXCEPT that Decision 2
# (sub-query decomposition) is disabled. It calls the same shared retrieval
# pipeline with use_decomposition=False, so:
#   - D1 (hop necessity), D3 (sufficiency), D4 (stopping), D5 (synthesis) all
#     run exactly as in the Main System.
#   - There is no decomposition plan, so hop 1 retrieves on the ORIGINAL
#     question and every subsequent hop is a reactive query built from D3's
#     MISSING: line. In other words, Ablation 1 is reactive from hop 1.
#
# Purpose (RQ3): the Main System vs Ablation 1 comparison isolates the
# specific contribution of structured query decomposition, because that is the
# ONLY factor that differs between them — guaranteed structurally, since both
# systems run the same _shared_retrieval pipeline and differ only in one flag.
# =============================================================================

import time
from src.core.logger import log_result
from src.core.config import DEV_MODEL

from src.agents.answer_synthesizer import synthesize
from src.systems._shared_retrieval import run_retrieval_pipeline, _format_chunk


def run_ablation_1(question: str, question_id: str, gold_answer: str,
                   model: str = DEV_MODEL) -> dict:
    """
    Run the agentic pipeline WITHOUT decomposition on a single question.

    Returns a dict matching the locked log schema (same fields as main_system).
    sub_queries_generated is always [] — Ablation 1 never decomposes.
    """
    start_time = time.time()

    # --- D1 -> iterative retrieval loop (D3/D4) -> context -------------------
    # use_decomposition=False: D2 is skipped, so sub_queries_generated stays [],
    # hop 1 uses the original question, and later hops are reactive.
    pipeline = run_retrieval_pipeline(question, model=model, use_decomposition=False)

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
        "system_name": "ablation_1",
        "hop_necessity_classification": pipeline["hop_necessity_classification"],
        "num_hops": len(docs_retrieved_per_hop),
        "sub_queries_generated": pipeline["sub_queries_generated"],  # always []
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


def run_and_log_ablation_1(question: str, question_id: str, gold_answer: str,
                            split: str, model: str = DEV_MODEL) -> dict:
    """Run Ablation 1 and log the result to disk in one step."""
    log_entry = run_ablation_1(question, question_id, gold_answer, model=model)
    log_result(log_entry, system_name="ablation_1", split=split)
    return log_entry