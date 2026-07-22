# =============================================================================
# run_evaluation.py — Runs a system on a question set with checkpointing
# Usage: python scripts/run_evaluation.py --system baseline_a --split dev
#        python scripts/run_evaluation.py --system baseline_b --split dev
#        python scripts/run_evaluation.py --system main_system --split dev
# =============================================================================

import json
import os
import sys
import argparse
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import (
    DEV_DATA_PATH, EVAL_DATA_PATH, DEV_MODEL, EVAL_MODEL, RESULTS_DIR
)
from src.core.logger import get_completed_ids
from src.systems.baseline_a import run_and_log_baseline_a
from src.systems.baseline_b import run_and_log_baseline_b
from src.systems.main_system import run_and_log_main_system
from src.systems.ablation_1 import run_and_log_ablation_1


def get_question_path(split: str) -> str:
    if split == "dev":
        return DEV_DATA_PATH
    elif split == "eval":
        return EVAL_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Use 'dev' or 'eval'.")


def get_model(split: str) -> str:
    """Dev split uses DEV_MODEL. Eval split uses EVAL_MODEL."""
    if split == "dev":
        return DEV_MODEL
    else:
        return EVAL_MODEL


def estimate_cost(system_name: str, split: str, results: list[dict]) -> dict:
    """
    Estimate API cost from logged token usage.
    Uses GPT-4o-mini pricing for eval splits, Groq (free) for dev splits.
    """
    if split == "dev":
        return {"note": "Groq dev model — free tier, no cost"}

    # GPT-4o-mini pricing (as of July 2026)
    input_price_per_1m  = 0.15   # USD per 1M input tokens
    output_price_per_1m = 0.60   # USD per 1M output tokens

    total_input  = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)

    input_cost  = (total_input  / 1_000_000) * input_price_per_1m
    output_cost = (total_output / 1_000_000) * output_price_per_1m
    total_cost  = input_cost + output_cost

    return {
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "input_cost_usd":      round(input_cost, 4),
        "output_cost_usd":     round(output_cost, 4),
        "total_cost_usd":      round(total_cost, 4),
        "total_cost_gbp":      round(total_cost * 0.79, 4),
    }


def run_evaluation(system_name: str, split: str):
    """Run the specified system on the specified question split."""

    print(f"\n{'='*60}")
    print(f"System:  {system_name}")
    print(f"Split:   {split}")
    model = get_model(split)
    print(f"Model:   {model}")
    print(f"{'='*60}\n")

    # Load questions
    question_path = get_question_path(split)
    with open(question_path, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {question_path}")

    # Checkpointing — find already completed question IDs
    completed_ids = get_completed_ids(system_name, split)
    remaining = [q for q in questions if q["id"] not in completed_ids]
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining:         {len(remaining)}")

    if not remaining:
        print("All questions already completed! Nothing to do.")
        return

    # Cost pre-check for eval split
    if split == "eval":
        print("\nWARNING: This is an EVAL run using GPT-4o-mini (paid model).")
        print("Estimated cost based on dev run averages will be shown after completion.")
        response = input("Type 'yes' to proceed: ").strip().lower()
        if response != "yes":
            print("Aborted.")
            return

    # Run evaluation
    print(f"\nStarting evaluation...\n")
    start_time = time.time()
    failed = 0

    for q in tqdm(remaining, desc=f"{system_name} on {split}"):
        try:
            question_id  = q["id"]
            question     = q["question"]
            gold_answer  = q["answer"]

            if system_name == "baseline_a":
                run_and_log_baseline_a(
                    question=question,
                    question_id=question_id,
                    gold_answer=gold_answer,
                    split=split,
                    model=model
                )
            elif system_name == "baseline_b":
                run_and_log_baseline_b(
                    question=question,
                    question_id=question_id,
                    gold_answer=gold_answer,
                    split=split,
                    model=model
                )
            elif system_name == "main_system":
                run_and_log_main_system(
                    question=question,
                    question_id=question_id,
                    gold_answer=gold_answer,
                    split=split,
                    model=model
                )
            elif system_name == "ablation_1":
                run_and_log_ablation_1(
                    question=question,
                    question_id=question_id,
                    gold_answer=gold_answer,
                    split=split,
                    model=model
                )
            else:
                raise ValueError(f"Unknown system: {system_name}")

        except Exception as e:
            failed += 1
            print(f"\nFailed on question {q.get('id', '?')}: {e}")
            print("Continuing to next question...")
            continue

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"System:     {system_name}")
    print(f"Split:      {split}")
    print(f"Completed:  {len(remaining) - failed}/{len(remaining)}")
    print(f"Failed:     {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Load results for cost estimate
    from src.core.logger import load_results
    results = load_results(system_name, split)
    cost = estimate_cost(system_name, split, results)
    print(f"\nCost estimate:")
    for k, v in cost.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a system and split")
    parser.add_argument("--system", required=True,
                        choices=["baseline_a", "baseline_b", "main_system", "ablation_1"],
                        help="Which system to run")
    parser.add_argument("--split", required=True, choices=["dev", "eval"],
                        help="Which question split to use")
    args = parser.parse_args()
    run_evaluation(args.system, args.split)