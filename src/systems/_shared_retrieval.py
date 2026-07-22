# =============================================================================
# _shared_retrieval.py — Shared retrieval orchestration for the agentic systems
#
# This module holds the retrieve-and-hops pipeline that the Main System and
# Ablation 1 must run IDENTICALLY. It lives here, imported by both as equals,
# so that the only difference between those systems is a single argument
# (use_decomposition) rather than two separately-maintained copies of the loop.
#
# What this module OWNS:
#   - D1 (hop necessity classification)
#   - D2 (sub-query decomposition) — gated by use_decomposition
#   - the iterative retrieval loop, D3 (sufficiency), D4 (adaptive stopping)
#   - context accumulation (dedup + budget)
#
# What this module does NOT own (stays in each system):
#   - D5 (answer synthesis) — each system calls synthesize() itself
#   - log-entry construction and log_result() — each system builds its own,
#     so system_name and any system-specific fields stay with the system.
#
# The pipeline returns the raw materials; the caller runs D5 and assembles
# the log entry. Behaviour is lifted verbatim from the original inline loop
# in main_system.py (frozen Phase 4). No logic is changed here.
# =============================================================================

import time

from src.core.retriever import retrieve
from src.core.config import TOP_K, MAX_HOPS, MAX_CONTEXT_TOKENS

from src.agents.hop_classifier import classify
from src.agents.decomposer import decompose
from src.agents.sufficiency_checker import check


# --- Context helpers (moved verbatim from main_system.py) --------------------

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


# --- The shared retrieval pipeline -------------------------------------------

def run_retrieval_pipeline(question: str, model: str,
                           use_decomposition: bool) -> dict:
    """
    Run D1 -> (optional D2) -> iterative retrieval loop (D3/D4) -> context.

    This is the retrieve-and-hops stage shared by the Main System
    (use_decomposition=True) and Ablation 1 (use_decomposition=False).
    It does NOT run D5 or build the log entry — the caller does that.

    Args:
        question:          the question text
        model:             which LLM to use
        use_decomposition: if True, run D2 and follow the planned sub-queries
                           before reactive mode (Main System behaviour). If
                           False, skip D2 entirely — every query from hop 1 is
                           the original question then reactive (Ablation 1).

    Returns dict with the raw materials for a log entry:
        hop_necessity_classification, sub_queries_generated,
        docs_retrieved_per_hop, all_retrieved_chunks, stop_condition,
        input_tokens, output_tokens, latency_per_hop
    """
    total_input_tokens = 0
    total_output_tokens = 0
    latency_per_hop = []
    docs_retrieved_per_hop = []
    queries_per_hop = []
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
        queries_per_hop.append(question)
        docs_retrieved_per_hop.append(
            [f"{d['metadata']['article_title']}_{d['metadata']['chunk_index']}" for d in docs]
        )
        stop_condition = "single_hop"
        latency_per_hop.append((time.time() - hop_check_start) * 1000)
    else:
        # --- DECISION 2: Sub-query decomposition -----------------------------
        # Gated by use_decomposition. When False (Ablation 1), D2 is skipped
        # entirely and sub_queries_generated stays [], so the loop below uses
        # the original question for hop 1 and reactive queries thereafter.
        if use_decomposition:
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

            queries_per_hop.append(next_query)
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
                # Reactive mode: use only the extracted missing-information
                # query, not the full sufficiency explanation (which lists
                # facts already present and pollutes the retrieval embedding)
                missing_text = sufficiency_result["missing"]
                if "MISSING:" in missing_text:
                    next_query = missing_text.rsplit("MISSING:", 1)[1].strip()
                else:
                    next_query = missing_text  # fallback: previous behaviour

    return {
        "hop_necessity_classification": hop_necessity,
        "sub_queries_generated": sub_queries_generated,
        "docs_retrieved_per_hop": docs_retrieved_per_hop,
        "queries_per_hop": queries_per_hop,
        "all_retrieved_chunks": all_retrieved_chunks,
        "stop_condition": stop_condition,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "latency_per_hop": latency_per_hop,
    }