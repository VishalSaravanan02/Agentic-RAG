# =============================================================================
# metrics.py — Computes all evaluation metrics from log files
# Reads .jsonl log files produced by run_evaluation.py
# Never touches the systems directly — evaluation is separate from execution
# =============================================================================

import json
import os
import re
import string
from collections import Counter
from src.core.logger import load_results
from src.core.config import RESULTS_DIR

# GPT-4o-mini pricing (USD per 1M tokens)
INPUT_PRICE_PER_1M  = 0.15
OUTPUT_PRICE_PER_1M = 0.60
USD_TO_GBP          = 0.79


# ── Text normalisation (standard HotpotQA/SQuAD approach) ──────────────────

def normalise(text: str) -> str:
    """
    Normalise text for EM and F1 scoring.
    Lowercase, remove punctuation, remove articles (a/an/the), collapse whitespace.
    This is the standard normalisation used in SQuAD and HotpotQA evaluation.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)


# ── Answer quality metrics ──────────────────────────────────────────────────

def exact_match(predicted: str, gold: str) -> int:
    """Binary: 1 if normalised predicted == normalised gold, else 0."""
    return int(normalise(predicted) == normalise(gold))


def f1_score(predicted: str, gold: str) -> float:
    """
    Token-level F1 between predicted and gold answer.
    Gives partial credit for partially correct answers.
    """
    pred_tokens = normalise(predicted).split()
    gold_tokens = normalise(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # Count overlapping tokens
    overlap = sum((pred_counter & gold_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gold_tokens)
    f1        = (2 * precision * recall) / (precision + recall)
    return f1


# ── Retrieval quality metrics ───────────────────────────────────────────────

def recall_at_k(result: dict, k: int = 5) -> int:
    """
    Recall@k: 1 if at least one retrieved chunk contains a gold supporting fact.
    Uses the gold_answer as a proxy for supporting fact content.
    Binary per question: 1 if gold answer text appears in any retrieved chunk title/id.
    Note: Full Supporting Facts Recall requires HotpotQA gold supporting facts,
    computed separately in supporting_facts_recall().
    """
    gold = normalise(result.get("gold_answer", ""))
    docs_per_hop = result.get("docs_retrieved_per_hop", [])

    for hop_docs in docs_per_hop:
        for doc_id in hop_docs:
            if gold and gold in normalise(doc_id):
                return 1
    return 0


def supporting_facts_recall(result: dict, hotpotqa_item: dict) -> float:
    """
    Proportion of HotpotQA gold supporting facts found in retrieved chunks.
    Uses the 'supporting_facts' field from the original HotpotQA data.
    """
    supporting_facts = hotpotqa_item.get("supporting_facts", {})
    gold_titles = supporting_facts.get("title", [])

    if not gold_titles:
        return 0.0

    docs_per_hop = result.get("docs_retrieved_per_hop", [])
    retrieved_titles = set()
    for hop_docs in docs_per_hop:
        for doc_id in hop_docs:
            # doc_id format: "ArticleTitle_chunkIndex"
            title = "_".join(doc_id.split("_")[:-1])
            retrieved_titles.add(title.lower())

    found = sum(
        1 for t in gold_titles
        if t.lower() in retrieved_titles
    )
    return found / len(gold_titles)


def retrieval_precision(result: dict, hotpotqa_item: dict) -> float:
    """
    Proportion of retrieved chunks that are gold supporting facts.
    """
    supporting_facts = hotpotqa_item.get("supporting_facts", {})
    gold_titles = {t.lower() for t in supporting_facts.get("title", [])}

    if not gold_titles:
        return 0.0

    docs_per_hop = result.get("docs_retrieved_per_hop", [])
    all_retrieved = []
    for hop_docs in docs_per_hop:
        all_retrieved.extend(hop_docs)

    if not all_retrieved:
        return 0.0

    relevant = sum(
        1 for doc_id in all_retrieved
        if "_".join(doc_id.split("_")[:-1]).lower() in gold_titles
    )
    return relevant / len(all_retrieved)


# ── Efficiency metrics ──────────────────────────────────────────────────────

def compute_efficiency(results: list[dict], split: str) -> dict:
    """Compute latency, token usage, and cost estimate across all results."""
    latencies    = [r.get("total_latency_ms", 0) for r in results]
    input_tokens = [r.get("input_tokens", 0) for r in results]
    output_tokens= [r.get("output_tokens", 0) for r in results]

    total_input  = sum(input_tokens)
    total_output = sum(output_tokens)
    n = len(results)

    if split == "dev":
        cost = {"note": "Groq dev model — free tier"}
    else:
        input_cost  = (total_input  / 1_000_000) * INPUT_PRICE_PER_1M
        output_cost = (total_output / 1_000_000) * OUTPUT_PRICE_PER_1M
        total_usd   = input_cost + output_cost
        cost = {
            "total_cost_usd":       round(total_usd, 4),
            "total_cost_gbp":       round(total_usd * USD_TO_GBP, 4),
            "cost_per_question_usd":round(total_usd / n, 6) if n else 0,
        }

    return {
        "avg_latency_ms":          round(sum(latencies) / n, 2) if n else 0,
        "avg_input_tokens":        round(total_input / n, 1) if n else 0,
        "avg_output_tokens":       round(total_output / n, 1) if n else 0,
        "total_input_tokens":      total_input,
        "total_output_tokens":     total_output,
        "cost":                    cost,
    }


# ── Main compute function ───────────────────────────────────────────────────

def compute_all(system_name: str, split: str,
                hotpotqa_data: list[dict] = None) -> dict:
    """
    Compute all metrics for a system/split combination.

    Args:
        system_name:   'baseline_a' or 'main_system'
        split:         'dev' or 'eval'
        hotpotqa_data: The original HotpotQA questions (for supporting facts).
                       If None, supporting facts metrics are skipped.

    Returns:
        dict with all computed metrics
    """
    results = load_results(system_name, split)

    if not results:
        print(f"No results found for {system_name} on {split}")
        return {}

    n = len(results)
    print(f"Computing metrics for {system_name} on {split} ({n}