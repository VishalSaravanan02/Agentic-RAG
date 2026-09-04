# =============================================================================
# inspect_failures.py — Pull out failure cases for manual inspection
#
# Failure = EM 0 AND F1 < 0.3 on the Main System. EM alone is too strict:
# a correct answer worded differently scores EM 0 without being a genuine
# failure, and the F1 threshold filters those out.
#
# Sampling follows proposal 7.5: a STRATIFIED RANDOM sample, stratified by
# HotpotQA question type, so the reviewed cases are not dominated by a single
# question type. Allocation is proportional to each type's share of the
# failure population, and the draw is seeded so the same sample is produced
# on every invocation — a manual review spread over several sittings must see
# the same cases each time.
#
# Question type is not present in the result logs, so the split's gold file is
# loaded to supply it. Those files are gitignored; run this locally.
#
# Usage:
#   python scripts/inspect_failures.py --split eval                  # 50 cases
#   python scripts/inspect_failures.py --split eval --n 10           # smaller
#   python scripts/inspect_failures.py --split dev --n 15 --reactive-only
#   python scripts/inspect_failures.py --split eval --save-ids results/failure_sample.json
#   python scripts/inspect_failures.py --split eval --summary        # counts only
# =============================================================================

import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import DEV_DATA_PATH, EVAL_DATA_PATH, RANDOM_SEED
from src.core.logger import load_results
from src.evaluation.metrics import article_title, exact_match, f1_score

# Proposal 7.5 fixes the formal sample size at 50 cases.
DEFAULT_N = 50


def load_gold_items(split: str) -> dict:
    """
    Return {question_id: hotpotqa_item} from the split's gold file.

    Supplies both the question type, needed to stratify the sample, and the
    gold supporting-fact titles, needed to show which required evidence was
    actually retrieved. Neither is present in the result logs.
    """
    path = DEV_DATA_PATH if split == "dev" else EVAL_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gold data not found at {path}. Run scripts/download_data.py first. "
            f"Question type and supporting facts are needed to stratify the "
            f"sample and to show which evidence was retrieved."
        )

    with open(path, "r") as f:
        questions = json.load(f)

    return {q["id"]: q for q in questions}


def stratified_sample(failures: list, n: int, seed: int) -> list:
    """
    Draw n failures, allocated proportionally across question type.

    Proportional allocation preserves the composition of the failure
    population, so the reviewed cases are representative of it rather than of
    any one type. Largest-remainder assignment is used so the parts sum to
    exactly n rather than to n plus or minus a rounding error, and a stratum
    smaller than its allocation contributes everything it has, with the
    shortfall redistributed.
    """
    if n >= len(failures):
        return sorted(failures, key=lambda r: r["question_id"])

    by_type = {}
    for r in failures:
        by_type.setdefault(r["_type"], []).append(r)

    total = len(failures)

    # Integer part of each allocation, then hand out the remaining places to
    # whichever strata were rounded down hardest. Ties are broken on the
    # stratum name rather than on dict insertion order, so the allocation
    # depends only on the counts and not on the order of the input file.
    exact = {t: len(rs) * n / total for t, rs in by_type.items()}
    alloc = {t: int(v) for t, v in exact.items()}
    remaining = n - sum(alloc.values())
    for t in sorted(by_type, key=lambda t: (-(exact[t] - alloc[t]), t)):
        if remaining <= 0:
            break
        alloc[t] += 1
        remaining -= 1

    # Cap any stratum at what it actually contains, then redistribute the
    # shortfall, again in a name-stable order.
    for t in alloc:
        alloc[t] = min(alloc[t], len(by_type[t]))
    shortfall = n - sum(alloc.values())
    while shortfall > 0:
        grew = False
        for t in sorted(by_type, key=lambda t: (-len(by_type[t]), t)):
            if shortfall <= 0:
                break
            if alloc[t] < len(by_type[t]):
                alloc[t] += 1
                shortfall -= 1
                grew = True
        if not grew:
            break

    rng = random.Random(seed)
    sample = []
    # Sort strata and candidates before drawing so the result depends only on
    # the seed, not on dict or file ordering.
    for t in sorted(by_type):
        pool = sorted(by_type[t], key=lambda r: r["question_id"])
        sample.extend(rng.sample(pool, alloc[t]))

    return sorted(sample, key=lambda r: r["question_id"])


def print_case(i: int, r: dict, ba_lookup: dict) -> None:
    qid = r["question_id"]
    ba_r = ba_lookup.get(qid)
    gold_titles = r["_gold_titles"]          # original casing, for display
    gold_lower = {t.lower() for t in gold_titles}

    print("=" * 70)
    print(f"FAILURE {i}  (id: {qid})   type: {r['_type']}")
    print("=" * 70)
    print(f"Question: {r['question']}")
    print(f"Gold answer: {r['gold_answer']}")
    print(f"\nClassification: {r['hop_necessity_classification']}   "
          f"Hops: {r['num_hops']}   Stop: {r['stop_condition_triggered']}   "
          f"F1: {r['_f1']:.3f}")

    sub_queries = r.get("sub_queries_generated", [])
    if sub_queries:
        print("Sub-queries (the plan):")
        for j, sq in enumerate(sub_queries, 1):
            print(f"  {j}. {sq}")

    # queries_per_hop is a diagnostic field added in Phase 5a; older logs may
    # not have it, so read defensively. When present it shows the ACTUAL query
    # issued at each hop -- including reactive queries generated after the plan
    # was exhausted, which are not visible in sub_queries_generated.
    queries_per_hop = r.get("queries_per_hop")
    if queries_per_hop:
        n_plan = len(sub_queries)
        print("\nQueries actually issued per hop:")
        for h, q in enumerate(queries_per_hop, 1):
            tag = "reactive" if h > n_plan else "planned "
            print(f"  Hop {h} [{tag}]: {q}")

    # The gold supporting-fact articles are what the question actually needed.
    # Printing them next to what was retrieved is what separates a retrieval
    # failure from a reasoning or grounding failure.
    print("\nGOLD supporting-fact articles (what was needed):")
    for t in gold_titles:
        print(f"  - {t}")

    print("\nRetrieved article titles per hop  (*** = a gold article):")
    found = set()
    for h, docs in enumerate(r["docs_retrieved_per_hop"], 1):
        titles = sorted({article_title(d) for d in docs})
        marked = []
        for t in titles:
            if t.lower() in gold_lower:
                found.add(t.lower())
                marked.append(f"***{t}***")
            else:
                marked.append(t)
        print(f"  Hop {h}: {', '.join(marked)}")

    missed = [t for t in gold_titles if t.lower() not in found]
    print(f"\n  EVIDENCE COVERAGE: {len(found)}/{len(gold_titles)} gold articles retrieved")
    if missed:
        print(f"  NEVER RETRIEVED: {', '.join(missed)}")
        print("  --> evidence was missing: consider retrieval or decomposition failure")
    else:
        print("  --> all required evidence was retrieved: consider reasoning, "
              "grounding, or error propagation")

    print(f"\nMain System answer: {r['final_answer'][:150]}")
    if ba_r:
        ba_em = exact_match(ba_r["final_answer"], ba_r["gold_answer"])
        print(f"Baseline A answer:  {ba_r['final_answer'][:150]}   (EM={ba_em})")

    print("\nSUGGESTED CATEGORY (circle one):")
    print("  [retrieval failure] [decomposition failure] [reasoning failure]")
    print("  [error propagation] [grounding failure] [unanswerable]")
    print()


def inspect(split: str, n: int, reactive_only: bool = False,
            seed: int = RANDOM_SEED, save_ids: str = None,
            summary_only: bool = False) -> None:
    ms = load_results("main_system", split)
    ba = load_results("baseline_a", split)
    ba_lookup = {r["question_id"]: r for r in ba}
    gold = load_gold_items(split)

    failures = []
    for r in ms:
        em = exact_match(r["final_answer"], r["gold_answer"])
        f1 = f1_score(r["final_answer"], r["gold_answer"])
        if em == 0 and f1 < 0.3:
            item = gold.get(r["question_id"], {})
            r["_f1"] = f1
            r["_type"] = item.get("type", "unknown")
            # HotpotQA annotates supporting facts per SENTENCE, so a title
            # repeats once per supporting sentence drawn from that article.
            # Deduplicate while preserving order, matching
            # metrics.supporting_facts_recall(), so that a two-document
            # question reads as 2 required articles rather than 3 or 4.
            raw_titles = item.get("supporting_facts", {}).get("title", [])
            seen_titles = set()
            distinct = []
            for t in raw_titles:
                if t.lower() not in seen_titles:
                    seen_titles.add(t.lower())
                    distinct.append(t)
            r["_gold_titles"] = distinct
            failures.append(r)

    # Optional filter: keep only failures that entered reactive mode, i.e. the
    # retrieval loop ran more hops than there were planned sub-queries, so at
    # least one query was generated reactively from D3's MISSING line.
    if reactive_only:
        failures = [r for r in failures
                    if r["num_hops"] > len(r.get("sub_queries_generated", []))]

    label = "reactive failures" if reactive_only else "failures"
    print(f"Main System {label} (EM=0, F1<0.3): {len(failures)} / {len(ms)}")

    pop_by_type = {}
    for r in failures:
        pop_by_type[r["_type"]] = pop_by_type.get(r["_type"], 0) + 1
    print("Failure population by question type:")
    for t in sorted(pop_by_type):
        share = pop_by_type[t] / len(failures) * 100 if failures else 0
        print(f"  {t:<12} {pop_by_type[t]:>4}  ({share:.1f}%)")

    if not failures:
        return

    sample = stratified_sample(failures, n, seed)

    samp_by_type = {}
    for r in sample:
        samp_by_type[r["_type"]] = samp_by_type.get(r["_type"], 0) + 1
    print(f"\nStratified random sample of {len(sample)} (seed {seed}):")
    for t in sorted(samp_by_type):
        print(f"  {t:<12} {samp_by_type[t]:>4}")
    print()

    if save_ids:
        os.makedirs(os.path.dirname(os.path.abspath(save_ids)), exist_ok=True)
        payload = {
            "split": split,
            "seed": seed,
            "criterion": "EM == 0 and F1 < 0.3",
            "reactive_only": reactive_only,
            "n_failures": len(failures),
            "n_sampled": len(sample),
            "population_by_type": pop_by_type,
            "sample_by_type": samp_by_type,
            "question_ids": [r["question_id"] for r in sample],
        }
        with open(save_ids, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Sample identifiers written to {save_ids}\n")

    if summary_only:
        return

    for i, r in enumerate(sample, 1):
        print_case(i, r, ba_lookup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect Main System failures for manual categorisation"
    )
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"sample size (default {DEFAULT_N}, per proposal 7.5)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"sampling seed (default {RANDOM_SEED})")
    parser.add_argument("--reactive-only", action="store_true",
                        help="only failures that entered reactive mode "
                             "(num_hops > number of planned sub-queries)")
    parser.add_argument("--save-ids", default=None,
                        help="write the sampled question ids to this JSON file")
    parser.add_argument("--summary", action="store_true",
                        help="print the strata and sample composition only")
    args = parser.parse_args()

    inspect(args.split, args.n,
            reactive_only=args.reactive_only,
            seed=args.seed,
            save_ids=args.save_ids,
            summary_only=args.summary)