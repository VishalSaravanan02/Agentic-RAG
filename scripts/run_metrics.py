# =============================================================================
# run_metrics.py — Compute and save evaluation metrics for all four systems
#
# Computes every metric reported in the dissertation and writes them to
# results/metrics_{split}.json, so the figures in the write-up have a dated
# file behind them rather than console output that is lost when the terminal
# closes.
#
# Reads saved .jsonl result files only: no LLM calls, no cost, deterministic.
#
# Retrieval metrics (recall_at_k, supporting_facts_recall, retrieval_precision)
# score against HotpotQA's gold supporting facts and therefore need the
# processed question file, which is gitignored. Run this locally.
#
# Usage:
#   python scripts/run_metrics.py --split eval
#   python scripts/run_metrics.py --split eval --no-save
# =============================================================================

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import DEV_DATA_PATH, EVAL_DATA_PATH
from src.evaluation.metrics import compute_all, save_metrics

SYSTEMS = ["baseline_a", "baseline_b", "ablation_1", "main_system"]


def load_gold(split: str) -> list:
    """Load the split's question file, which carries the gold supporting facts."""
    path = DEV_DATA_PATH if split == "dev" else EVAL_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gold data not found at {path}. Run scripts/download_data.py first. "
            f"Retrieval metrics cannot be computed without it."
        )

    with open(path, "r") as f:
        return json.load(f)


def print_table(all_metrics: dict) -> None:
    """Print all four systems side by side, one row per metric."""
    rows = [
        ("Exact Match",              "answer_quality",    "exact_match"),
        ("F1 Score",                 "answer_quality",    "f1"),
        ("Recall@k",                 "retrieval_quality", "recall_at_k"),
        ("Supporting Facts Recall",  "retrieval_quality", "supporting_facts_recall"),
        ("Retrieval Precision",      "retrieval_quality", "retrieval_precision"),
        ("Duplicate Retrieval Rate", "retrieval_quality", "duplicate_retrieval_rate"),
        ("Avg Latency (ms)",         "efficiency",        "avg_latency_ms"),
        ("Avg Input Tokens",         "efficiency",        "avg_input_tokens"),
        ("Avg Output Tokens",        "efficiency",        "avg_output_tokens"),
        ("Total Cost (USD)",         "efficiency.cost",   "total_cost_usd"),
        ("Total Cost (GBP)",         "efficiency.cost",   "total_cost_gbp"),
    ]

    present = [s for s in SYSTEMS if s in all_metrics]

    print()
    print("=" * 100)
    header = f"{'METRIC':<28}" + "".join(f"{s:>17}" for s in present)
    print(header)
    print("=" * 100)

    for label, section, key in rows:
        cells = ""
        for s in present:
            node = all_metrics[s]
            for part in section.split("."):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            v = node.get(key) if isinstance(node, dict) else None
            # Explicit None check: a genuine 0.0 is a real measurement, not a
            # missing one, and must not be printed as "N/A".
            if v is None:
                cells += f"{'N/A':>17}"
            elif abs(v) >= 100:
                cells += f"{v:>17.1f}"
            else:
                cells += f"{v:>17.4f}"
        print(f"{label:<28}{cells}")

    print("=" * 100)

    # n_scored can be lower than n_questions if any question is missing from
    # the gold file, so report both rather than assuming they match.
    print()
    for s in present:
        m = all_metrics[s]
        n_scored = m.get("retrieval_quality", {}).get("n_scored", "?")
        print(f"  {s:<14} n_questions = {m.get('n_questions')}   "
              f"retrieval metrics scored over {n_scored}")

    if "main_system" in all_metrics and "hop_analysis" in all_metrics["main_system"]:
        ha = all_metrics["main_system"]["hop_analysis"]
        print()
        print("MAIN SYSTEM HOP ANALYSIS")
        print(f"  Average hops:          {ha.get('avg_hops')}")
        print(f"  Hop distribution:      {ha.get('hop_distribution')}")
        print(f"  Stop conditions:       {ha.get('stop_condition_distribution')}")
        print(f"  D1 classifications:    {ha.get('hop_classification_distribution')}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save metrics for all four systems"
    )
    parser.add_argument("--split", choices=["dev", "eval"], default="eval",
                        help="which split to analyse (default: eval)")
    parser.add_argument("--no-save", action="store_true",
                        help="print only; do not write the JSON file")
    args = parser.parse_args()

    split = args.split

    try:
        gold = load_gold(split)
        print(f"Gold data loaded: {len(gold)} questions")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Continuing without gold data — retrieval metrics will be N/A.")
        gold = None

    all_metrics = {}
    for system in SYSTEMS:
        m = compute_all(system, split, hotpotqa_data=gold)
        if m:
            all_metrics[system] = m

    if not all_metrics:
        print(f"\nNo results found for split '{split}'. Run the systems first.")
        return

    print_table(all_metrics)

    if not args.no_save:
        path = save_metrics(all_metrics, split)
        print()
        print(f"Metrics written to {path}")
        print()


if __name__ == "__main__":
    main()