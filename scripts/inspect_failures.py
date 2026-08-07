# =============================================================================
# inspect_failures.py — Pull out failure cases for manual inspection
# Failure = EM 0 AND F1 < 0.3 on the Main System.
# Shows everything needed to categorise each failure by eye.
#
# Usage: python scripts/inspect_failures.py --split dev --n 15
#        python scripts/inspect_failures.py --split dev --n 15 --reactive-only
# =============================================================================
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.logger import load_results
from src.evaluation.metrics import exact_match, f1_score


def inspect(split: str, n: int, reactive_only: bool = False):
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

    # Optional filter: keep only failures that entered reactive mode, i.e.
    # the retrieval loop ran more hops than there were planned sub-queries,
    # so at least one query was generated reactively from D3's MISSING line.
    if reactive_only:
        failures = [r for r in failures
                    if r["num_hops"] > len(r.get("sub_queries_generated", []))]

    label = "reactive failures" if reactive_only else "failures"
    print(f"Total Main System {label} (EM=0, F1<0.3): {len(failures)} / {len(ms)}")
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

        sub_queries = r.get("sub_queries_generated", [])
        if sub_queries:
            print(f"Sub-queries (the plan):")
            for j, sq in enumerate(sub_queries, 1):
                print(f"  {j}. {sq}")

        # queries_per_hop is a diagnostic field added in Phase 5a; older logs
        # (e.g. Baseline A/B) may not have it, so read defensively. When
        # present it shows the ACTUAL query issued at each hop -- including the
        # reactive queries generated after the plan was exhausted, which are
        # not visible in sub_queries_generated.
        queries_per_hop = r.get("queries_per_hop")
        if queries_per_hop:
            n_plan = len(sub_queries)
            print(f"\nQueries actually issued per hop:")
            for h, q in enumerate(queries_per_hop, 1):
                tag = "reactive" if h > n_plan else "planned "
                print(f"  Hop {h} [{tag}]: {q}")

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
    parser.add_argument("--reactive-only", action="store_true",
                        help="Only show failures that entered reactive mode "
                             "(num_hops > number of planned sub-queries)")
    args = parser.parse_args()
    inspect(args.split, args.n, reactive_only=args.reactive_only)