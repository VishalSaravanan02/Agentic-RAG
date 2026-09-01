# =============================================================================
# run_significance.py — Bootstrap significance testing for RQ1, RQ2 and RQ3
#
# Runs every paired comparison the dissertation reports, using the shared
# machinery in src/evaluation/bootstrap.py. Reads saved .jsonl result files
# only: no LLM calls, no cost, deterministic given a fixed seed.
#
# The comparisons follow the design in proposal 5.2:
#   RQ1   Baseline A  vs Main System   — combined effect of all five mechanisms
#   RQ2   the same pair, split by D1's classification (see run_rq2_analysis.py
#         for the interaction test; this script reports the within-subset gaps)
#   RQ3a  Baseline B  vs Ablation 1    — isolates adaptive control flow
#   RQ3b  Ablation 1  vs Main System   — isolates query decomposition
#   (context) Baseline A vs Baseline B — raw effect of more retrieval
#
# Usage:
#   python scripts/run_significance.py --split eval
#   python scripts/run_significance.py --split eval --json
#   python scripts/run_significance.py --split dev --answer-only
# =============================================================================

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import DEV_DATA_PATH, EVAL_DATA_PATH, RANDOM_SEED, RESULTS_DIR
from src.core.logger import load_results
from src.evaluation.bootstrap import compare, format_result

# -----------------------------------------------------------------------------
# What gets compared.
#
# Order within each pair matters: (a, b) reports a - b, so the system expected
# to be stronger goes first and a positive difference reads as "a beats b".
# -----------------------------------------------------------------------------

COMPARISONS = [
    # (label, system_a, system_b, note)
    ("RQ1",  "main_system", "baseline_a",
     "All five agentic mechanisms vs single-hop RAG"),
    ("RQ3a", "ablation_1",  "baseline_b",
     "Adaptive control flow (both have adaptive query formulation)"),
    ("RQ3b", "main_system", "ablation_1",
     "Query decomposition (both have adaptive control flow)"),
    ("CTX",  "baseline_b",  "baseline_a",
     "Raw effect of a second retrieval, no decision-making"),
]

# Answer-quality and retrieval-quality metrics. Efficiency metrics are reported
# separately because a "significant" latency difference is expected and not a
# finding — the agentic systems make more LLM calls by construction.
ANSWER_METRICS = ["exact_match", "f1"]

RETRIEVAL_METRICS = [
    "recall_at_k",
    "supporting_facts_recall",
    "retrieval_precision",
    "duplicate_retrieval_rate",
]

EFFICIENCY_METRICS = ["latency_ms", "input_tokens", "output_tokens"]


def load_gold(split: str) -> dict:
    """
    Build {question_id: hotpotqa_item} for the split.

    Required for recall_at_k, supporting_facts_recall and retrieval_precision,
    which score against HotpotQA's gold supporting facts. These files are
    gitignored, so this must be run locally.
    """
    path = DEV_DATA_PATH if split == "dev" else EVAL_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gold data not found at {path}. Run scripts/download_data.py first. "
            f"Retrieval metrics cannot be computed without it."
        )

    with open(path, "r") as f:
        questions = json.load(f)

    return {q["id"]: q for q in questions}


def d1_subsets(split: str) -> dict:
    """
    Return {"YES": {qids...}, "NO": {qids...}} from the Main System's D1
    classification, restricted to questions Baseline A also answered.

    This is the RQ2 split: the same pair as RQ1, partitioned by how the agent
    classified each question. Baseline A never ran D1 — that is the point.
    We are asking whether the Main System's advantage concentrates where its
    own classifier says it should.
    """
    ms = load_results("main_system", split)
    ba_ids = {r["question_id"] for r in load_results("baseline_a", split)}

    subsets = {"YES": set(), "NO": set()}
    for r in ms:
        qid = r["question_id"]
        if qid not in ba_ids:
            continue
        label = r.get("hop_necessity_classification")
        if label in subsets:
            subsets[label].add(qid)

    return subsets


def run_block(title: str, comparisons: list, metrics: list, split: str,
              gold_lookup: dict | None, collected: list,
              question_ids: set | None = None, subset_label: str = "") -> None:
    """Run one group of comparisons across one group of metrics and print it."""
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for label, sys_a, sys_b, note in comparisons:
        header = f"\n{label}  {sys_a} vs {sys_b}"
        if subset_label:
            header += f"  [{subset_label}]"
        print(header)
        print(f"     {note}")
        print()

        for metric in metrics:
            try:
                r = compare(
                    sys_a, sys_b, split, metric,
                    gold_lookup=gold_lookup,
                    question_ids=question_ids,
                    seed=RANDOM_SEED,
                )
            except FileNotFoundError as e:
                print(f"  {metric:<24} skipped — {e}")
                continue
            except ValueError as e:
                # Raised when the two systems cover different question sets, or
                # when gold data is missing for a gold-dependent metric.
                print(f"  {metric:<24} skipped — {e}")
                continue

            r["comparison"] = label
            r["subset"] = subset_label or "all"
            collected.append(r)
            print("  " + format_result(r))


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap significance testing across all research questions"
    )
    parser.add_argument("--split", choices=["dev", "eval"], default="eval",
                        help="which split to analyse (default: eval)")
    parser.add_argument("--answer-only", action="store_true",
                        help="skip retrieval metrics (no gold data needed)")
    parser.add_argument("--skip-efficiency", action="store_true",
                        help="skip latency and token comparisons")
    parser.add_argument("--json", nargs="?", const="AUTO", default=None,
                        help="write results as JSON; omit the path to use "
                             "results/significance_{split}.json")
    args = parser.parse_args()

    split = args.split

    print()
    print("#" * 100)
    print(f"#  BOOTSTRAP SIGNIFICANCE TESTING — split: {split}")
    print(f"#  10,000 resamples, seed {RANDOM_SEED}, paired on question_id")
    print(f"#  A leading * marks p < 0.05. Differences are (system_a - system_b).")
    print("#" * 100)

    # Gold data is only needed for the retrieval metrics. Load it lazily so
    # --answer-only works on a machine without the processed data files.
    gold_lookup = None
    if not args.answer_only:
        try:
            gold_lookup = load_gold(split)
            print(f"\nGold data loaded: {len(gold_lookup)} questions")
        except FileNotFoundError as e:
            print(f"\nWARNING: {e}")
            print("Continuing with answer-quality metrics only.")

    collected = []

    # ---- RQ1, RQ3a, RQ3b, and the extra-retrieval comparison ----------------

    run_block(
        "ANSWER QUALITY  —  RQ1, RQ3a, RQ3b",
        COMPARISONS, ANSWER_METRICS, split, gold_lookup, collected,
    )

    if gold_lookup is not None:
        run_block(
            "RETRIEVAL QUALITY  —  RQ1, RQ3a, RQ3b",
            COMPARISONS, RETRIEVAL_METRICS, split, gold_lookup, collected,
        )
    else:
        # duplicate_retrieval_rate needs no gold data, so it can still run.
        run_block(
            "RETRIEVAL QUALITY  —  duplicate rate only (no gold data)",
            COMPARISONS, ["duplicate_retrieval_rate"], split, None, collected,
        )

    if not args.skip_efficiency:
        run_block(
            "EFFICIENCY  —  expected to differ by construction, reported for completeness",
            COMPARISONS, EFFICIENCY_METRICS, split, gold_lookup, collected,
        )

    # ---- RQ2: the same pair, split by D1 -----------------------------------

    subsets = d1_subsets(split)
    rq2_pair = [("RQ2", "main_system", "baseline_a",
                 "Same pair as RQ1, restricted to one D1 subset")]

    for label in ("YES", "NO"):
        qids = subsets[label]
        if not qids:
            print(f"\nRQ2: no questions classified {label} — skipping")
            continue

        run_block(
            f"RQ2  —  questions D1 classified as {label}  (n = {len(qids)})",
            rq2_pair, ANSWER_METRICS, split, gold_lookup, collected,
            question_ids=qids, subset_label=f"D1={label}",
        )

    print()
    print("-" * 100)
    print("RQ2 verdict: the within-subset gaps above are descriptive. The test of "
          "whether the\n             YES gap genuinely exceeds the NO gap is the "
          "interaction test in\n             scripts/run_rq2_analysis.py --split "
          f"{split}. Run that too.")
    print("-" * 100)

    # ---- Summary -----------------------------------------------------------

    n_sig = sum(1 for r in collected if r["significant"])
    print()
    print("=" * 100)
    print(f"SUMMARY: {len(collected)} comparisons run, {n_sig} significant at p < 0.05")
    print("=" * 100)
    print()
    print("NOTE ON MULTIPLE COMPARISONS")
    print("  These are uncorrected p-values, as pre-specified in proposal 7.4.")
    print("  Running many tests inflates the chance of a false positive, so a")
    print("  correction (e.g. Holm) should be reported alongside. The choice of")
    print("  correction FAMILY materially changes the RQ3b verdict — see the")
    print("  dissertation and confirm the family with the supervisor.")
    print()

    if args.json:
        out_path = (os.path.join(RESULTS_DIR, f"significance_{split}.json")
                    if args.json == "AUTO" else args.json)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        payload = {
            "split": split,
            "seed": RANDOM_SEED,
            "n_comparisons": len(collected),
            "results": collected,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {out_path}")
        print()


if __name__ == "__main__":
    main()