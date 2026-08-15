# =============================================================================
# run_rq2_analysis.py — RQ2 subset analysis
# Splits results by the Main System's hop necessity classification (D1) and
# compares both systems' performance within each subset.
#
# RQ2: Does the performance advantage concentrate on questions the system
#      classifies as requiring multiple hops?
#
# The pair is Baseline A vs Main System by design, not by oversight: proposal
# 5.2 defines RQ2 as that same pair, split by Decision 1's classification.
#
# The verdict is decided by a bootstrapped test of the DIFFERENCE between the
# two subset gaps, not by comparing them directly. A bare `yes_gap > no_gap`
# comparison fires roughly half the time on systems with no real advantage at
# all, because both gaps carry sampling noise.
#
# Usage: python scripts/run_rq2_analysis.py --split dev
#        python scripts/run_rq2_analysis.py --split eval
# =============================================================================

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import RANDOM_SEED
from src.core.logger import load_results
from src.evaluation.metrics import exact_match, f1_score

# Matched to src/evaluation/bootstrap.py so every significance figure in the
# project is produced under the same settings.
N_RESAMPLES = 10_000
ALPHA = 0.05


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

    # --- RQ2 verdict, with the difference actually tested --------------------
    print(f"\n{'='*56}")
    print("RQ2 INTERACTION TEST")
    print(f"{'='*56}")

    if not (yes_ms and no_ms):
        print("\nOne subset is empty — the comparison RQ2 asks for cannot be made.")
        print(f"{'='*56}\n")
        return

    for label, metric_fn in (("Exact Match", exact_match), ("F1", f1_score)):
        report_interaction(label, interaction_test(
            yes_ms, yes_ba, no_ms, no_ba, metric_fn))

    print(f"\n  Bootstrap: {N_RESAMPLES:,} resamples, seed {RANDOM_SEED}, "
          f"alpha {ALPHA}.")
    print(f"{'='*56}\n")


def interaction_test(yes_ms, yes_ba, no_ms, no_ba, metric_fn,
                     n_resamples: int = N_RESAMPLES,
                     seed: int = RANDOM_SEED) -> dict:
    """
    Test whether the Main System's advantage is genuinely LARGER on the YES
    subset than on the NO subset, rather than merely larger in this sample.

    This is the question RQ2 actually asks. Comparing the two gaps with a bare
    `>` cannot answer it: both gaps carry sampling noise, so with no real effect
    at all one still exceeds the other about half the time. (Verified by
    simulation at these subset sizes: 48% of runs.) The project's own measured
    3.9% answer-level noise floor makes that concrete rather than theoretical.

    RESAMPLING STRUCTURE — the two subsets are handled differently on purpose:
      - WITHIN a subset, both systems answered the SAME questions, so the
        per-question difference (main - baseline) is taken first and that
        paired difference is what gets resampled.
      - ACROSS subsets, YES and NO are DIFFERENT questions, so they are
        independent groups and are resampled separately.
    Resampling the two subsets independently and taking the difference of their
    means each time is therefore the correct structure here — it is not the
    paired bootstrap used elsewhere in the project, and using a paired test
    across subsets of unequal size would be wrong.

    Returns observed difference-of-gaps, percentile CI, and a two-sided p-value.
    """
    yes_diff = np.array([metric_fn(m["final_answer"], m["gold_answer"])
                         - metric_fn(b["final_answer"], b["gold_answer"])
                         for m, b in zip(yes_ms, yes_ba)], dtype=float)
    no_diff = np.array([metric_fn(m["final_answer"], m["gold_answer"])
                        - metric_fn(b["final_answer"], b["gold_answer"])
                        for m, b in zip(no_ms, no_ba)], dtype=float)

    observed = float(yes_diff.mean() - no_diff.mean())

    # Vectorised: draw all resample indices at once rather than looping.
    # Each row is one resample of that subset, drawn with replacement.
    rng = np.random.default_rng(seed)
    yes_reps = yes_diff[rng.integers(0, yes_diff.size,
                                     size=(n_resamples, yes_diff.size))].mean(axis=1)
    no_reps = no_diff[rng.integers(0, no_diff.size,
                                   size=(n_resamples, no_diff.size))].mean(axis=1)
    reps = yes_reps - no_reps

    ci_low, ci_high = (float(x) for x in np.percentile(reps, [2.5, 97.5]))

    # Two-sided p about zero, with the same tie-splitting and add-one
    # correction used in src/evaluation/bootstrap.py, so p is comparable with
    # every other p-value reported by this project and is never exactly zero.
    n = reps.size
    n_below = int((reps < 0).sum())
    n_above = int((reps > 0).sum())
    n_ties = int((reps == 0).sum())
    tail_low = (n_below + n_ties / 2.0 + 1.0) / (n + 1.0)
    tail_high = (n_above + n_ties / 2.0 + 1.0) / (n + 1.0)
    p = min(2.0 * min(tail_low, tail_high), 1.0)

    return {
        "yes_gap": float(yes_diff.mean()),
        "no_gap": float(no_diff.mean()),
        "difference": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "p_floor": 2.0 / (n + 1.0),
        "significant": p < ALPHA,
        "n_yes": int(yes_diff.size),
        "n_no": int(no_diff.size),
    }


def report_interaction(label: str, r: dict):
    """Print one interaction test and the verdict it does or does not support."""
    p_str = (f"< {r['p_floor']:.4f}" if r["p_value"] <= r["p_floor"]
             else f"= {r['p_value']:.4f}")

    print(f"\n{label}")
    print(f"  Advantage on YES subset (n={r['n_yes']}): {r['yes_gap']:+.4f}")
    print(f"  Advantage on NO subset  (n={r['n_no']}): {r['no_gap']:+.4f}")
    print(f"  Difference:                     {r['difference']:+.4f}   "
          f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]   p {p_str}")

    if r["significant"] and r["difference"] > 0:
        print("  -> Consistent with RQ2: the advantage concentrates on questions")
        print("     classified as multi-hop, by more than sampling noise explains.")
    elif r["significant"] and r["difference"] < 0:
        print("  -> Contrary to RQ2: the advantage is significantly LARGER on")
        print("     questions classified as single-hop.")
    else:
        print("  -> Not resolved on this split. The observed difference is within")
        print("     what sampling noise alone would produce, so this data neither")
        print("     supports nor refutes RQ2. Note this is not evidence of no")
        print("     effect — it may simply be an underpowered subset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 subset analysis")
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    args = parser.parse_args()
    run_rq2(args.split)