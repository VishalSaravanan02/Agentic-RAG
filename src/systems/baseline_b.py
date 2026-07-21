# =============================================================================
# baseline_b.py — Fixed 2-Hop RAG (the fixed-pipeline baseline)
# Exactly two retrieval calls, always. No decision mechanisms.
#
# Hop 1: retrieve on the original question (identical to Baseline A).
# Hop 2: ask the LLM what else needs looking up, retrieve on that.
#
# There is no hop-necessity classification (D1), no decomposition (D2),
# no sufficiency check (D3), and no adaptive stopping (D4). The hop count
# is fixed at 2 regardless of the question. Grounded synthesis (D5) is
# shared with every other system and is imported, not reimplemented.
#
# Purpose (RQ3): isolates the effect of doing MORE retrieval with NO
# decision-making, so that Baseline A vs Baseline B measures extra
# retrieval alone, and Ablation 1 vs Baseline B measures the bundled
# effect of adaptive decision-making.
# =============================================================================

import time

from src.core.retriever import retrieve
from src.core.llm_client import call_llm
from src.core.logger import log_result
from src.core.config import TOP_K, DEV_MODEL

# Synthesis (D5) is imported, never copied — the prompt must remain
# byte-identical across all systems or the comparisons are meaningless.
from src.agents.answer_synthesizer import synthesize, _strip_think_tags

# Context handling must be held constant with the Main System, so these are
# imported rather than duplicated. Their proper home is src/core/ — deferred
# rather than move code inside the frozen Main System. See dissertation
# defect audit.
from src.systems.main_system import (
    _format_chunk,
    _deduplicate_chunks,
    _enforce_context_budget,
)

# Fixed hop count. Baseline B always performs exactly this many retrievals.
NUM_HOPS_FIXED = 2

# The first line is the proposal's specified prompt, verbatim. The remaining
# lines constrain the output format only: asked plainly, the model replies in
# prose, and prose makes a poor embedding query. This mirrors the output
# constraints already used in answer_synthesizer.py and decomposer.py, and
# keeps Baseline B the strongest reasonable fixed-pipeline baseline rather
# than one hobbled by formatting noise.
HOP2_QUERY_PROMPT_TEMPLATE = """Given the original question and what you just retrieved, what else do you need to look up?

Reply with ONLY the search query — a short phrase naming the specific fact still needed. Do not write a sentence, an explanation, or a preamble. Do not repeat what has already been found.

Original question: {question}

Retrieved so far:
{context}

Search query:"""

# Maximum words in a valid hop-2 query. Anything longer is prose, not a query.
MAX_HOP2_QUERY_WORDS = 50


def _clean_query(text: str) -> str:
    """Strip think tags, markdown, quotes and stray whitespace from a query."""
    cleaned = _strip_think_tags(text).strip()
    cleaned = cleaned.strip("*_# ")
    cleaned = cleaned.strip('"\'')
    return cleaned.strip()


def _generate_hop2_query(question: str, context: str, model: str = DEV_MODEL) -> dict:
    """
    Generate the second retrieval query.

    Validation: non-empty, and no longer than MAX_HOP2_QUERY_WORDS words.
    On validation failure the retry includes corrective feedback — at
    temperature 0 an identical prompt reproduces an identical response, so
    the prompt must change for a retry to be worth making (see decomposer.py).

    Falls back to the original question after 3 failed attempts: Baseline B
    is meant to be undiscerning, not fragile. The fallback is announced on
    stdout, consistent with D1/D2 fallback handling in the Main System.

    Returns:
        dict with keys: query, input_tokens, output_tokens
    """
    base_prompt = HOP2_QUERY_PROMPT_TEMPLATE.format(question=question, context=context)
    prompt = base_prompt

    max_retries = 3
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_retries):
        llm_result = call_llm(prompt, model=model, max_tokens=2000, temperature=0.0)
        total_input_tokens += llm_result["input_tokens"]
        total_output_tokens += llm_result["output_tokens"]

        query = _clean_query(llm_result["text"])
        word_count = len(query.split())

        if query and word_count <= MAX_HOP2_QUERY_WORDS:
            return {
                "query": query,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }

        reason = "empty response" if not query else f"{word_count} words (prose, not a query)"
        print(f"Hop-2 query invalid (attempt {attempt + 1}/{max_retries}): {reason}")

        # Corrective feedback: the prompt must change for the retry to differ.
        prompt = base_prompt + (
            "\n\nIMPORTANT: Your previous attempt was not a usable search query "
            f"({reason}). Reply with a short search phrase only — no sentences, "
            "no explanation, no preamble."
        )

    print(f"Hop-2 query generation failed after {max_retries} attempts. "
          f"Falling back to the original question for: {question}")
    return {
        "query": question,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


def run_baseline_b(question: str, question_id: str, gold_answer: str,
                   model: str = DEV_MODEL) -> dict:
    """
    Run the full Baseline B pipeline on a single question.

    Args:
        question:    The question text
        question_id: HotpotQA question ID (for logging)
        gold_answer: The correct answer (for logging, used later in metrics)
        model:       Which LLM to use (DEV_MODEL or EVAL_MODEL)

    Returns:
        dict matching the locked 14-field log schema, plus hop2_query.
    """
    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    latency_per_hop = []
    docs_retrieved_per_hop = []
    all_retrieved_chunks = []

    # --- HOP 1: retrieve on the original question ----------------------------
    # Identical to Baseline A's single retrieval. No query transformation.
    hop1_start = time.time()

    hop1_docs = retrieve(question, k=TOP_K)
    all_retrieved_chunks.extend(hop1_docs)
    docs_retrieved_per_hop.append(
        [f"{d['metadata']['article_title']}_{d['metadata']['chunk_index']}" for d in hop1_docs]
    )

    hop1_context = "\n\n".join(_format_chunk(d) for d in hop1_docs)
    latency_per_hop.append((time.time() - hop1_start) * 1000)

    # --- HOP 2: generate a follow-up query, then retrieve on it ---------------
    # This is the only place Baseline B differs from Baseline A. There is no
    # judgement about whether hop 2 is needed — it always happens.
    hop2_start = time.time()

    hop2_result = _generate_hop2_query(question, hop1_context, model=model)
    total_input_tokens += hop2_result["input_tokens"]
    total_output_tokens += hop2_result["output_tokens"]
    hop2_query = hop2_result["query"]

    hop2_docs = retrieve(hop2_query, k=TOP_K)
    all_retrieved_chunks.extend(hop2_docs)
    docs_retrieved_per_hop.append(
        [f"{d['metadata']['article_title']}_{d['metadata']['chunk_index']}" for d in hop2_docs]
    )

    # Held constant with the Main System: deduplicate across hops, then
    # enforce the context budget by dropping lowest-similarity chunks first.
    all_retrieved_chunks = _deduplicate_chunks(all_retrieved_chunks)
    all_retrieved_chunks = _enforce_context_budget(all_retrieved_chunks)

    latency_per_hop.append((time.time() - hop2_start) * 1000)

    # --- SYNTHESIS (D5): shared with every other system -----------------------
    final_context = "\n\n".join(_format_chunk(d) for d in all_retrieved_chunks)
    synth_start = time.time()

    synth_result = synthesize(question, final_context, model=model)
    total_input_tokens += synth_result["input_tokens"]
    total_output_tokens += synth_result["output_tokens"]

    latency_per_hop.append((time.time() - synth_start) * 1000)

    total_latency_ms = (time.time() - start_time) * 1000

    # --- Log entry: 14 locked fields, in the same order as every system ------
    log_entry = {
        "question_id": question_id,
        "question": question,
        "system_name": "baseline_b",
        "hop_necessity_classification": "N/A",   # no D1 — never classifies
        "num_hops": NUM_HOPS_FIXED,              # always 2, by construction
        "sub_queries_generated": [],             # no D2 — never decomposes
        "docs_retrieved_per_hop": docs_retrieved_per_hop,
        "stop_condition_triggered": "N/A",       # no D4 — never decides to stop
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "latency_per_hop_ms": latency_per_hop,
        "total_latency_ms": total_latency_ms,
        "final_answer": synth_result["answer"],
        "gold_answer": gold_answer,

        # Baseline-B-only diagnostic field. Not read by any metric; exists so
        # that Phase 6 failure analysis can distinguish "sensible query, poor
        # retrieval" from "nonsense query". Read it with .get(), never [].
        "hop2_query": hop2_query,
    }

    return log_entry


def run_and_log_baseline_b(question: str, question_id: str, gold_answer: str,
                           split: str, model: str = DEV_MODEL) -> dict:
    """
    Run Baseline B and log the result to disk in one step.
    This is what run_evaluation.py will call.
    """
    log_entry = run_baseline_b(question, question_id, gold_answer, model=model)
    log_result(log_entry, system_name="baseline_b", split=split)
    return log_entry