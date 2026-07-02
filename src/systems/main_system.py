# =============================================================================
# main_system.py — Agentic Multi-Hop RAG (the Main System)
# Wires together all 5 decision mechanisms + the iterative retrieval loop.
# Decision 4 (adaptive stopping) is inline here, not a separate file —
# it has no logic of its own, it just acts on what Decision 3 returns.
# =============================================================================

import time
from src.core.retriever import retrieve
from src.core.logger import log_result
from src.core.config import TOP_K, MAX_HOPS, MAX_CONTEXT_TOKENS, DEV_MODEL

from src.agents.hop_classifier import classify
from src.agents.decomposer import decompose
from src.agents.sufficiency_checker import check
from src.agents.answer_synthesizer import synthesize


def _format_chunk(doc: dict) -> str:
    """Format a single retrieved chunk for inclusion in context."""
    return f"[{doc['metadata']['article_title']}] {doc['text']}"


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by (article_title, chunk_index)."""
    seen = set()
    unique = []
    for chunk in chunks:
        key = (chunk["metadata"]["article_title"], chunk["metadata"]["chunk_index"])
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def _enforce_context_budget(chunks: list[dict]) -> list[dict]:
    """
    Enforce the MAX_CONTEXT_TOKENS budget. If exceeded, remove the
    lowest cosine_similarity chunks first until under budget.
    Uses a word-count proxy (words * 1.33) for token estimation.
    """
    def estimate_tokens(chunk_list):
        total_words = sum(len(c["text"].split()) for c in chunk_list)
        return total_words * 1.33

    if estimate_tokens(chunks) <= MAX_CONTEXT_TOKENS:
        return chunks

    # Sort by similarity descending, then keep adding until budget is hit
    sorted_chunks = sorted(chunks, key=lambda c: c["cosine_similarity"], reverse=True)
    kept = []
    for chunk in sorted_chunks:
        candidate = kept + [chunk]
        if estimate_tokens(candidate) <= MAX_CONTEXT_TOKENS:
            kept.append(chunk)
        else:
            print(f"Context budget exceeded — dropping low-relevance chunk "
                  f"(similarity={chunk['cosine_similarity']:.3f})")

    return kept


def run_main_system(question: str, question_id: str, gold_answer: str,
                     model: str = DEV_MODEL) -> dict:
    """
    Run the full agentic multi-hop RAG pipeline on a single question.

    Returns a dict matching the locked log schema (same fields as baseline_a).
    """
    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    latency_per_hop = []
    docs_retrieved_per_hop = []
    all_retrieved_chunks = []
    sub_queries_generated = []

    # --- DECISION 1: Hop necessity classification ----------------------------
    hop_check_start = time.time()
    classification_result = classify(question, model=model)
    total_input_tokens += classification_result["input_tokens"]
    total_output_tokens += classification_result["output_tokens"]
    hop_necessity = classification_result["classification"]

    if hop_necessity == "NO":
        # Single-hop path: one retrieval, skip straight to synthesis
        docs = retrieve(question, k=TOP_K)
        all_retrieved_chunks.extend(docs)
        docs_retrieved_per_hop.append(
            [f"{d['metadata']['article_title']}_{d['metadata']['chunk_index']}" for d in docs]
        )
        stop_condition = "single_hop"
        latency_per_hop.append((time.time() - hop_check_start) * 1000)
    else:
        # --- DECISION 2: Sub-query decomposition -----------------------------
        decomposition_result = decompose(question, model=model)
        total_input_tokens += decomposition_result["input_tokens"]
        total_output_tokens += decomposition_result["output_tokens"]
        sub_queries_generated = decomposition_result["sub_questions"]

        # --- ITERATIVE RETRIEVAL LOOP -----------------------------------------
        num_hops = 0
        sub_query_index = 0
        stop_condition = None
        next_query = sub_queries_generated[0] if sub_queries_generated else question

        while True:
            hop_start = time.time()
            num_hops += 1

            # Retrieve for the current query
            docs = retrieve(next_query, k=TOP_K)
            all_retrieved_chunks.extend(docs)
            all_retrieved_chunks = _deduplicate_chunks(all_retrieved_chunks)
            all_retrieved_chunks = _enforce_context_budget(all_retrieved_chunks)

            docs_retrieved_per_hop.append(
                [f"{d['metadata']['article_title']}_{d['metadata']['chunk_index']}" for d in docs]
            )

            accumulated_context = "\n\n".join(_format_chunk(d) for d in all_retrieved_chunks)

            # --- DECISION 3: Sufficiency check --------------------------------
            sufficiency_result = check(question, accumulated_context, model=model)
            total_input_tokens += sufficiency_result["input_tokens"]
            total_output_tokens += sufficiency_result["output_tokens"]

            latency_per_hop.append((time.time() - hop_start) * 1000)

            # --- DECISION 4: Adaptive stopping (inline, not a separate file) -
            if sufficiency_result["sufficient"]:
                stop_condition = "sufficiency"
                break
            if num_hops >= MAX_HOPS:
                stop_condition = "max_hops"
                break

            # Not sufficient and hops remain — determine next query
            sub_query_index += 1
            if sub_query_index < len(sub_queries_generated):
                # Planned sub-queries still remain
                next_query = sub_queries_generated[sub_query_index]
            else:
                # Sub-queries exhausted — reactive mode using "missing" explanation
                next_query = sufficiency_result["missing"]

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
        "hop_necessity_classification": hop_necessity,
        "num_hops": len(docs_retrieved_per_hop),
        "sub_queries_generated": sub_queries_generated,
        "docs_retrieved_per_hop": docs_retrieved_per_hop,
        "stop_condition_triggered": stop_condition,
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