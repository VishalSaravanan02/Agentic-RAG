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

def article_title(doc_id: str) -> str:
    """
    Extract the article title from a logged chunk ID.

    Chunk IDs have the form "{article_title}_{chunk_index}". Article titles may
    themselves contain underscores, so the split is on the LAST underscore only:
    "Foo_2_3" -> "Foo_2".
    """
    return doc_id.rpartition("_")[0]


def recall_at_k(result: dict, hotpotqa_item: dict) -> int:
    """
    Binary per question: 1 if at least one gold supporting-fact article appears
    among the documents retrieved for this question, else 0.

    Scored against HotpotQA's gold supporting facts. An earlier implementation
    used the gold ANSWER string as a proxy for relevance and matched it as a
    substring of the chunk ID; that produced false positives (gold "no" matching
    "Christopher Nolan", "Harley Knoles", "Taxonomy of the Cactaceae") and was
    replaced. HotpotQA provides no relevance annotation other than the gold
    supporting facts, so there is no valid non-gold definition of "relevant".

    Relationship to supporting_facts_recall(): this is the binary "did we find
    ANY required evidence" form; supporting_facts_recall() is the proportional
    "how much of the required evidence did we find" form.

    Scope: computed over the union of documents retrieved across ALL hops
    (top-k = 5 per hop, proposal 5.4). Systems performing more retrieval
    therefore have more opportunity to score, exactly as for
    supporting_facts_recall(); read alongside the efficiency metrics.

    Returns 0 when the item carries no gold titles, matching
    supporting_facts_recall().
    """
    supporting_facts = hotpotqa_item.get("supporting_facts", {})
    gold_titles = {t.lower() for t in supporting_facts.get("title", [])}

    if not gold_titles:
        return 0

    for hop_docs in result.get("docs_retrieved_per_hop", []):
        for doc_id in hop_docs:
            if article_title(doc_id).lower() in gold_titles:
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
            retrieved_titles.add(article_title(doc_id).lower())

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
        if article_title(doc_id).lower() in gold_titles
    )
    return relevant / len(all_retrieved)


# ── Efficiency metrics ──────────────────────────────────────────────────────

def compute_efficiency(results: list[dict]) -> dict:
    """Compute latency, token usage, and cost estimate across all results."""
    latencies    = [r.get("total_latency_ms", 0) for r in results]
    input_tokens = [r.get("input_tokens", 0) for r in results]
    output_tokens= [r.get("output_tokens", 0) for r in results]

    total_input  = sum(input_tokens)
    total_output = sum(output_tokens)
    n = len(results)

    # Both splits now run gpt-4o-mini, so both are priced identically. The dev
    # split was originally a free Groq tier and was special-cased as costless;
    # that special case is gone, because reporting a paid run as free is worse
    # than reporting no cost at all.
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
    results = load_results(system_name, split)

    if not results:
        print(f"No results found for {system_name} on {split}")
        return {}

    n = len(results)
    print(f"Computing metrics for {system_name} on {split} ({n} questions)...")

    # Build lookup for HotpotQA data by question ID
    hpqa_lookup = {}
    if hotpotqa_data:
        hpqa_lookup = {q["id"]: q for q in hotpotqa_data}

    # Per-question metrics
    em_scores  = []
    f1_scores  = []
    r_at_k     = []
    sf_recalls = []
    ret_precs  = []

    for r in results:
        predicted = r.get("final_answer", "")
        gold      = r.get("gold_answer", "")
        qid       = r.get("question_id", "")
        hpqa_item = hpqa_lookup.get(qid, {})

        em_scores.append(exact_match(predicted, gold))
        f1_scores.append(f1_score(predicted, gold))

        # All three retrieval metrics are scored against HotpotQA's gold
        # supporting facts, so all three require the gold item to be present.
        if hpqa_item:
            r_at_k.append(recall_at_k(r, hpqa_item))
            sf_recalls.append(supporting_facts_recall(r, hpqa_item))
            ret_precs.append(retrieval_precision(r, hpqa_item))

    # Aggregate
    metrics = {
        "system":          system_name,
        "split":           split,
        "n_questions":     n,
        "answer_quality": {
            "exact_match": round(sum(em_scores) / n, 4),
            "f1":          round(sum(f1_scores) / n, 4),
        },
        "retrieval_quality": {
            "recall_at_k":             round(sum(r_at_k) / len(r_at_k), 4) if r_at_k else None,
            "supporting_facts_recall": round(sum(sf_recalls) / len(sf_recalls), 4) if sf_recalls else None,
            "retrieval_precision":     round(sum(ret_precs) / len(ret_precs), 4) if ret_precs else None,
            # Retrieval metrics are scored only over questions present in the
            # gold file, so this can be lower than n_questions. Report it.
            "n_scored":                len(sf_recalls),
        },
        "efficiency": compute_efficiency(results),
    }

    # Hop distribution (main system only)
    if system_name == "main_system":
        hop_counts = [r.get("num_hops", 1) for r in results]
        hop_dist   = Counter(hop_counts)
        stop_conds = Counter(r.get("stop_condition_triggered", "unknown") for r in results)
        hop_class  = Counter(r.get("hop_necessity_classification", "N/A") for r in results)
        metrics["hop_analysis"] = {
            "hop_distribution":                dict(sorted(hop_dist.items())),
            "stop_condition_distribution":     dict(stop_conds),
            "hop_classification_distribution": dict(hop_class),
            "avg_hops":                        round(sum(hop_counts) / n, 2),
        }

    return metrics


def print_comparison(baseline_metrics: dict, main_metrics: dict):
    """Print a formatted comparison table of two systems."""
    print(f"\n{'='*65}")
    print(f"{'METRIC':<35} {'BASELINE A':>12} {'MAIN SYSTEM':>14}")
    print(f"{'='*65}")

    print(f"\n--- Answer Quality ---")
    baq = baseline_metrics.get("answer_quality", {})
    maq = main_metrics.get("answer_quality", {})
    print(f"{'Exact Match':<35} {baq.get('exact_match', 0):>12.4f} {maq.get('exact_match', 0):>14.4f}")
    print(f"{'F1 Score':<35} {baq.get('f1', 0):>12.4f} {maq.get('f1', 0):>14.4f}")

    print(f"\n--- Retrieval Quality ---")
    brq = baseline_metrics.get("retrieval_quality", {})
    mrq = main_metrics.get("retrieval_quality", {})

    def _fmt(value, width):
        # Explicit None check: a genuine 0.0 is a real measurement, not a
        # missing one, and must not be printed as "N/A".
        return f"{'N/A' if value is None else round(value, 4)!s:>{width}}"

    for label, key in (("Recall@k", "recall_at_k"),
                       ("Supporting Facts Recall", "supporting_facts_recall"),
                       ("Retrieval Precision", "retrieval_precision")):
        print(f"{label:<35} {_fmt(brq.get(key), 12)} {_fmt(mrq.get(key), 14)}")

    print(f"\n--- Efficiency ---")
    bef = baseline_metrics.get("efficiency", {})
    mef = main_metrics.get("efficiency", {})
    print(f"{'Avg Latency (ms)':<35} {bef.get('avg_latency_ms', 0):>12.1f} {mef.get('avg_latency_ms', 0):>14.1f}")
    print(f"{'Avg Input Tokens':<35} {bef.get('avg_input_tokens', 0):>12.1f} {mef.get('avg_input_tokens', 0):>14.1f}")
    print(f"{'Avg Output Tokens':<35} {bef.get('avg_output_tokens', 0):>12.1f} {mef.get('avg_output_tokens', 0):>14.1f}")

    if "hop_analysis" in main_metrics:
        print(f"\n--- Main System Hop Analysis ---")
        ha = main_metrics["hop_analysis"]
        print(f"{'Avg Hops':<35} {'N/A':>12} {ha.get('avg_hops', 0):>14.2f}")
        print(f"{'Hop Distribution':<35} {'':>12} {str(ha.get('hop_distribution', {})):>14}")
        print(f"{'Stop Conditions':<35} {'':>12} {str(ha.get('stop_condition_distribution', {})):>14}")

    print(f"\n{'='*65}\n")