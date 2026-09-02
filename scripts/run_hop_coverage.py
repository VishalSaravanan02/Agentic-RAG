# =============================================================================
# run_hop_coverage.py — Cumulative gold-article coverage after each hop
#
# Regenerates the per-hop coverage table underpinning the first-hop deficit
# argument: how much of a question's gold supporting-fact evidence each system
# has accumulated after hop 1, hop 2, and so on.
#
# Definition. Coverage after hop h is the proportion of a question's DISTINCT
# gold supporting-fact articles appearing anywhere in hops 1..h, averaged over
# questions. A question that stopped before hop h contributes its final value
# at every later hop, so the denominator is the same set of questions at every
# hop and the columns are directly comparable. The alternative -- averaging
# only over questions that actually reached hop h -- changes the population
# from column to column and would confound coverage with stopping behaviour.
#
# Subset. The table reported in the dissertation covers the questions the
# agent classified as multi-hop, since that is the only subset where the Main
# System decomposes and therefore the only one where the comparison is
# meaningful. Coverage over all questions is printed alongside for reference:
# there, hop-1 coverage for any system querying the original question equals
# that system's overall Supporting Facts Recall by construction.
#
# Gold titles are deduplicated, matching metrics.supporting_facts_recall().
#
# Usage:
#   python scripts/run_hop_coverage.py --split eval
#   python scripts/run_hop_coverage.py --split eval --json
# =============================================================================

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import DEV_DATA_PATH, EVAL_DATA_PATH, MAX_HOPS, RESULTS_DIR
from src.core.logger import load_results
from src.evaluation.metrics import article_title

SYSTEMS = ["baseline_a", "baseline_b", "ablation_1", "main_system"]


def load_gold(split: str) -> dict:
    path = DEV_DATA_PATH if split == "dev" else EVAL_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gold data not found at {path}. Run scripts/download_data.py first."
        )
    with open(path, "r") as f:
        return {q["id"]: q for q in json.load(f)}


def coverage_by_hop(result: dict, item: dict, max_hops: int) -> list[float]:
    """
    Cumulative proportion of distinct gold articles found after each hop.

    Returns a list of length max_hops. Once a system has stopped, its final
    value is carried forward, so every column is defined for every question.
    """
    gold = {t.lower() for t in item.get("supporting_facts", {}).get("title", [])}
    if not gold:
        return [0.0] * max_hops

    docs_per_hop = result.get("docs_retrieved_per_hop", [])

    seen = set()
    out = []
    for h in range(max_hops):
        if h < len(docs_per_hop):
            for doc_id in docs_per_hop[h]:
                seen.add(article_title(doc_id).lower())
        # Beyond the system's last hop nothing is added, so the value carries.
        out.append(len(gold & seen) / len(gold))
    return out


def compute(split: str, gold: dict, subset_ids: set | None) -> dict:
    """Mean coverage per hop for each system, over the given subset."""
    table = {}
    for system in SYSTEMS:
        rows = []
        for r in load_results(system, split):
            qid = r["question_id"]
            if qid not in gold:
                continue
            if subset_ids is not None and qid not in subset_ids:
                continue
            rows.append(coverage_by_hop(r, gold[qid], MAX_HOPS))
        if not rows:
            continue
        n = len(rows)
        table[system] = {
            "n": n,
            "coverage": [sum(row[h] for row in rows) / n for h in range(MAX_HOPS)],
        }
    return table


def print_table(title: str, table: dict) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    header = f"{'system':<16}" + "".join(f"{'hop '+str(h+1):>12}" for h in range(MAX_HOPS))
    print(header + f"{'n':>8}")
    print("-" * 78)
    for system in SYSTEMS:
        if system not in table:
            continue
        e = table[system]
        cells = "".join(f"{v:>12.4f}" for v in e["coverage"])
        print(f"{system:<16}{cells}{e['n']:>8}")

    # The systems that query the original question at hop 1 retrieve the same
    # documents there, so their hop-1 values must be identical. Reporting the
    # deficit against that shared value is the point of the table.
    shared = [table[s]["coverage"][0] for s in
              ("baseline_a", "baseline_b", "ablation_1") if s in table]
    if len(set(round(v, 4) for v in shared)) == 1 and "main_system" in table:
        base = shared[0]
        ms = table["main_system"]["coverage"][0]
        print("-" * 78)
        print(f"  Hop-1 deficit: {ms:.4f} vs {base:.4f} = "
              f"{(base - ms) * 100:.1f} points")
        gains = table["main_system"]["coverage"]
        print(f"  Main System gain over hops 2-{MAX_HOPS}: "
              f"{gains[-1] - gains[0]:+.4f}")
        if "ablation_1" in table:
            a = table["ablation_1"]["coverage"]
            print(f"  Ablation 1 gain over hops 2-{MAX_HOPS}: {a[-1] - a[0]:+.4f}")
    elif len(set(round(v, 4) for v in shared)) != 1:
        print("-" * 78)
        print("  NOTE: the three original-question systems do not share a hop-1")
        print("        value, which they should by construction. Investigate.")


def main():
    parser = argparse.ArgumentParser(
        description="Cumulative gold-article coverage after each hop"
    )
    parser.add_argument("--split", choices=["dev", "eval"], default="eval")
    parser.add_argument("--json", nargs="?", const="AUTO", default=None,
                        help="write results as JSON; omit the path to use "
                             "results/hop_coverage_{split}.json")
    args = parser.parse_args()

    split = args.split
    gold = load_gold(split)
    print(f"Gold data loaded: {len(gold)} questions")

    # The multi-hop subset is defined by the Main System's D1 classification,
    # the same split used for the RQ2 analysis.
    yes_ids = {r["question_id"] for r in load_results("main_system", split)
               if r.get("hop_necessity_classification") == "YES"}

    subset_table = compute(split, gold, yes_ids)
    all_table = compute(split, gold, None)

    print_table(f"Questions classified multi-hop by D1  (n = {len(yes_ids)})",
                subset_table)
    print_table("All questions, for reference", all_table)

    print()
    print("Note: over all questions, hop-1 coverage for baseline_a, baseline_b")
    print("and ablation_1 equals their Supporting Facts Recall at hop 1 by")
    print("construction, since all three query the original question there.")
    print()

    if args.json:
        out = (os.path.join(RESULTS_DIR, f"hop_coverage_{split}.json")
               if args.json == "AUTO" else args.json)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "split": split,
                "max_hops": MAX_HOPS,
                "definition": "cumulative distinct gold articles found in hops "
                              "1..h, carried forward after a system stops, "
                              "averaged over questions",
                "multi_hop_subset": subset_table,
                "all_questions": all_table,
            }, f, indent=2)
        print(f"Results written to {out}")
        print()


if __name__ == "__main__":
    main()