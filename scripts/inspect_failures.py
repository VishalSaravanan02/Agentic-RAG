# =============================================================================
# inspect_failures.py — Pull out failure cases for manual inspection
# Failure = EM 0 AND F1 < 0.3 on the Main System.
# Shows everything needed to categorise each failure by eye.
#
# Usage: python scripts/inspect_failures.py --split dev --n 15
# =============================================================================

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logger import load_results
from src.evaluation.metrics import exact_match, f1_score


def inspect(split: str, n: int):
    ms = load_results("main_system", split)
    ba = load_results("baseline_a", split)
    ba_lookup = {r["question_id"]: r for r in ba}

    failures = []
    for r in ms:
        em = exact_match(r["final_answer"], r["gold_answer"])
        f1 = f1_score(r["final_answer"], r["gold_answer"])
        if em == 0 and f1 < 0.3:
            r["_f1"] = f1
            failures.append(r)

    print(f"Total Main System failures (EM=0, F1<0.3): {len(failures)} / {len(ms)}")
    print(f"Showing first {min(n, len(failures))}\n")

    for i, r in enumerate(failures[:n], 1):
        qid = r["question_id"]
        ba_r = ba_lookup.get(qid)
        print(f"{'='*70}")
        print(f"FAILURE {i}  (id: {qid})")
        print(f"{'='*70}")
        print(f"Question: {r['question']}")
        print(f"Gold answer: {r['gold_answer']}")
        print(f"\nClassification: {r['hop_necessity_classification']}   "
              f"Hops: {r['num_hops']}   Stop: {r['stop_condition_triggered']}")
        if r["sub_queries_generated"]:
            print(f"Sub-queries:")
            for j, sq in enumerate(r["sub_queries_generated"], 1):
                print(f"  {j}. {sq}")
        print(f"\nRetrieved article titles per hop:")
        for h, docs in enumerate(r["docs_retrieved_per_hop"], 1):
            titles = sorted({ "_".join(d.split("_")[:-1]) for d in docs })
            print(f"  Hop {h}: {', '.join(titles[:6])}")
        print(f"\nMain System answer: {r['final_answer'][:150]}")
        if ba_r:
            ba_em = exact_match(ba_r["final_answer"], ba_r["gold_answer"])
            print(f"Baseline A answer:  {ba_r['final_answer'][:150]}   (EM={ba_em})")
        print(f"\nSUGGESTED CATEGORY (circle one):")
        print(f"  [retrieval failure] [decomposition failure] [reasoning failure]")
        print(f"  [error propagation] [grounding failure] [unanswerable]")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()
    inspect(args.split, args.n)