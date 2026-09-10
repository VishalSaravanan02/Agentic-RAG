# =============================================================================
# case_dossier.py — Assemble the full evidence dossier for one Phase 7 case.
#
# Resolves the chunk identifiers stored in docs_retrieved_per_hop back to the
# chunk text the system actually read, using data/processed/chunks.json, which
# build_index.py writes before indexing. The logs record which chunks were
# retrieved but not what they said; this recovers that exactly, with no
# re-embedding and no dependence on the ChromaDB index.
#
# Usage:
#   python scripts/case_dossier.py --case 3
#   python scripts/case_dossier.py --case 3 --all-systems
# =============================================================================

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import CHUNKS_PATH, EVAL_DATA_PATH

SAMPLE_PATH = "results/failure_sample_eval.json"


def load_chunk_lookup() -> dict:
    """Return {(article_title, chunk_index): text} for every chunk in the corpus."""
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. It is written by scripts/build_index.py."
        )
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    return {(c["article_title"], c["chunk_index"]): c["text"] for c in chunks}


def resolve(chunk_id: str, lookup: dict) -> str:
    """Resolve a 'Title_N' identifier to its chunk text."""
    title, _, idx = chunk_id.rpartition("_")
    try:
        return lookup[(title, int(idx))]
    except (KeyError, ValueError):
        return f"<chunk {chunk_id} not found in corpus>"


def load_system(system: str, qid: str) -> dict:
    path = f"results/{system}_eval.jsonl"
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["question_id"] == qid:
                    return r
    return None


def main(case_n: int, all_systems: bool) -> None:
    qid = json.load(open(SAMPLE_PATH))["question_ids"][case_n - 1]
    gold = {q["id"]: q for q in json.load(open(EVAL_DATA_PATH))}[qid]
    lookup = load_chunk_lookup()

    print("=" * 78)
    print(f"CASE {case_n}   (id: {qid})   type: {gold['type']}")
    print("=" * 78)
    print(f"QUESTION:    {gold['question']}")
    print(f"GOLD ANSWER: {gold['answer']}")

    ctx = dict(zip(gold["context"]["title"], gold["context"]["sentences"]))
    print("\nGOLD SUPPORTING SENTENCES:")
    for t, s in zip(gold["supporting_facts"]["title"], gold["supporting_facts"]["sent_id"]):
        try:
            print(f"  [{t}] {ctx[t][s].strip()}")
        except (KeyError, IndexError):
            print(f"  [{t}] <sentence {s} unavailable>")

    systems = ["main_system"]
    if all_systems:
        systems = ["main_system", "ablation_1", "baseline_b", "baseline_a"]

    for system in systems:
        r = load_system(system, qid)
        if r is None:
            continue
        print("\n" + "-" * 78)
        print(f"{system.upper()}   answer: {r['final_answer']!r}")
        print(f"  classification={r['hop_necessity_classification']}  "
              f"hops={r['num_hops']}  stop={r['stop_condition_triggered']}")
        if r.get("sub_queries_generated"):
            print("  PLAN:")
            for j, sq in enumerate(r["sub_queries_generated"], 1):
                print(f"    {j}. {sq}")

        queries = r.get("queries_per_hop", [])
        seen = set()
        for h, docs in enumerate(r["docs_retrieved_per_hop"], 1):
            q = queries[h - 1] if h <= len(queries) else "<query not logged>"
            print(f"\n  HOP {h} query: {q}")
            for cid in docs:
                if cid in seen:
                    print(f"    [{cid}]  (repeat)")
                    continue
                seen.add(cid)
                print(f"    [{cid}]")
                print(f"      {resolve(cid, lookup)}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Assemble the evidence dossier for one Phase 7 case")
    p.add_argument("--case", type=int, required=True, help="case number, 1-50")
    p.add_argument("--all-systems", action="store_true",
                   help="include the baselines and the ablation, not just the Main System")
    a = p.parse_args()
    main(a.case, a.all_systems)
