# =============================================================================
# run_rq2_analysis.py — RQ2 subset analysis
# Splits results by the Main System's hop necessity classification (D1) and
# compares both systems' performance within each subset.
#
# RQ2: Does the performance advantage concentrate on questions the system
#      classifies as requiring multiple hops?
#
# Usage: python scripts/run_rq2_analysis.py --split dev
#        python scripts/run_rq2_analysis.py --split eval   (Phase 5)
# =============================================================================

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logger import load_results
from src.evaluation.metrics import exact_match, f1_score


def analyse_subset(name: str, ba_results: list[dict], ms_results: list[dict]):
    """Compute and print EM/F1 for both systems on one subset of questions."""
    n = len(ms_results)
    if n == 0:
        print(f"\n{name}: no questions in this subset")
        return

    ba_em = sum(exact_match(r["final_answer"], r["gold_answer"]) for r in ba_results) / n
    ms_em = sum(exact_match(r["final_answer"], r["gold_answer"]) for r in ms_results) / n
    ba_f1 = sum(f1_score(r["final_answer"], r["gold_answer"]) for r in ba_results) / n
    ms_f1 = sum(f1_score(r["final_answer"], r["gold_answer"]) for r in ms_results) / n

    print(f"\n{name}  (n = {n})")
    print(f"{'':>14} {'Baseline A':>12} {'Main System':>13} {'Difference':>12}")
    print(f"{'Exact Match':>14} {ba_em:>12.4f} {ms_em:>13.4f} {ms_em - ba_em:>+12.4f}")
    print(f"{'F1 Score':>14} {ba_f1:>12.4f} {ms_f1:>13.4f} {ms_f1 - ba_f1:>+12.4f}")


def run_rq2(split: str):
    ba = load_results("baseline_a", split)
    ms = load_results("main_system", split)

    if not ba or not ms:
        print(f"Missing results for split '{split}'. Run both systems first.")
        return

    # Index Baseline A results by question ID for pairing
    ba_lookup = {r["question_id"]: r for r in ba}

    # Split by the Main System's D1 classification
    yes_ms, yes_ba = [], []
    no_ms, no_ba = [], []
    skipped = 0

    for r in ms:
        qid = r["question_id"]
        if qid not in ba_lookup:
            skipped += 1
            continue
        if r["hop_necessity_classification"] == "YES":
            yes_ms.append(r)
            yes_ba.append(ba_lookup[qid])
        else:
            no_ms.append(r)
            no_ba.append(ba_lookup[qid])

    print(f"{'='*56}")
    print(f"RQ2 SUBSET ANALYSIS — split: {split}")
    print(f"{'='*56}")
    print(f"Total paired questions: {len(yes_ms) + len(no_ms)}"
          + (f" (skipped {skipped} unpaired)" if skipped else ""))

    analyse_subset("Classified YES (multi-hop needed)", yes_ba, yes_ms)
    analyse_subset("Classified NO (single-hop sufficient)", no_ba, no_ms)

    # RQ2 interpretation summary
    if yes_ms and no_ms:
        n_yes = len(yes_ms)
        n_no = len(no_ms)
        yes_gap = (sum(exact_match(r["final_answer"], r["gold_answer"]) for r in yes_ms) / n_yes
                   - sum(exact_match(r["final_answer"], r["gold_answer"]) for r in yes_ba) / n_yes)
        no_gap = (sum(exact_match(r["final_answer"], r["gold_answer"]) for r in no_ms) / n_no
                  - sum(exact_match(r["final_answer"], r["gold_answer"]) for r in no_ba) / n_no)
        print(f"\n{'='*56}")
        print(f"EM advantage on YES subset: {yes_gap:+.4f}")
        print(f"EM advantage on NO subset:  {no_gap:+.4f}")
        if yes_gap > no_gap:
            print("Pattern consistent with RQ2: advantage concentrates on")
            print("questions classified as multi-hop.")
        else:
            print("Pattern NOT consistent with RQ2 on this split.")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 subset analysis")
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    args = parser.parse_args()
    run_rq2(args.split)