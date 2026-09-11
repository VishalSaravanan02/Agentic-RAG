# =============================================================================
# run_supplementary_analyses.py — Supplementary analyses on the evaluation split
#
# Produces every figure the dissertation reports beyond the pre-specified
# comparisons in run_significance.py, so that no reported number exists only
# as a one-off calculation. Reads the saved run logs, the judge logs, the
# evaluation question file and the coded failure cases. No LLM calls, no cost,
# deterministic given the fixed seed.
#
# None of these tests belongs to the Holm-corrected families defined in
# run_significance.py. They are supplementary analyses: each reports an
# uncorrected p-value and should be read as such.
#
# Sections:
#   rq2                    interaction tests and the second-hop counterfactual
#   sufficiency_stops      how often D3 stopped with a gold article missing
#   sfr_by_stop_group      Supporting Facts Recall by how the loop ended
#   judge_baseline_b_vs_main  judge comparison outside the four pre-specified pairs
#   d1_agreement           D1 run-to-run agreement between Main System and Ablation 1
#   plan_length            plans longer than the hop budget, and their cost
#   decomposition_fallback questions where D2 fell back to the original question
#   decomposition_by_type  Main System vs Ablation 1 on bridge and comparison questions
#   retrieval_volume       mean distinct chunks presented to synthesis
#   code_defects           questions touched by the three known defects, and a
#                          sensitivity check excluding them
#   phase7_cross_system    how the comparator systems fared on the 50 coded cases
#
# Usage (from the repository root):
#   python scripts/run_supplementary_analyses.py
# =============================================================================

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from src.core.config import EVAL_DATA_PATH, MAX_HOPS, RANDOM_SEED, RESULTS_DIR
from src.core.logger import load_results
from src.evaluation.bootstrap import N_RESAMPLES, load_judge_results, paired_bootstrap
from src.evaluation.metrics import exact_match, f1_score, normalise, supporting_facts_recall
from run_rq2_analysis import interaction_test

SPLIT = "eval"
SYSTEMS = ["baseline_a", "baseline_b", "ablation_1", "main_system"]
OUTPUT_PATH = os.path.join(RESULTS_DIR, "supplementary_analyses_eval.json")
FAILURE_CASES_PATH = os.path.join(RESULTS_DIR, "failure_analysis.json")

# A reactive query that reproduces D3's whole response rather than the text
# after "MISSING:". D3 is instructed to write the marker in capitals; when it
# writes "Missing:" instead, the case-sensitive check in _shared_retrieval.py
# falls back to passing the full response to the retriever.
UNEXTRACTED_QUERY = re.compile(r"^\s*(YES|NO)\b.*(Missing:|Present:)", re.S)

# A sub-question that refers to an earlier sub-question instead of naming its
# subject. The decomposer's validator rejects "from the previous" and
# "identified above" but not these forms.
SELF_REFERENCE = re.compile(
    r"(identified|named|mentioned|found|given|listed|stated|obtained|determined|"
    r"titled|answered)\s+(in|by)\s+the\s+(previous|prior|preceding|first|second|"
    r"last|earlier)"
    r"|previous (question|sub-?question|step|answer)"
    r"|prior (question|step)|preceding (question|step)",
    re.I,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def r4(x):
    return None if x is None else round(float(x), 4)


def summarise(result: dict) -> dict:
    """Keep the fields of a bootstrap result that the write-up needs."""
    keys = ["observed_difference", "difference", "yes_gap", "no_gap",
            "ci_low", "ci_high", "p_value", "significant", "n_questions",
            "n_yes", "n_no"]
    return {k: (r4(result[k]) if isinstance(result[k], float) else result[k])
            for k in keys if k in result}


def unpaired_bootstrap(values_a, values_b, n_resamples: int = N_RESAMPLES,
                       seed: int = RANDOM_SEED) -> dict:
    """
    Difference in means between two independent groups of questions.

    Used where the two groups are different questions answered by the same
    system, so there is no pairing to preserve. Each group is resampled
    separately. The p-value follows the convention of bootstrap.paired_bootstrap:
    two-sided about zero, ties split between tails, add-one correction.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)
    reps = (a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
            - b[rng.integers(0, b.size, size=(n_resamples, b.size))].mean(axis=1))
    n = reps.size
    below, above, ties = (reps < 0).sum(), (reps > 0).sum(), (reps == 0).sum()
    p = min(2.0 * min((below + ties / 2 + 1) / (n + 1),
                      (above + ties / 2 + 1) / (n + 1)), 1.0)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"observed_difference": float(a.mean() - b.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
            "significant": bool(p < 0.05), "n_a": int(a.size), "n_b": int(b.size)}


def load_all():
    logs = {s: {r["question_id"]: r for r in load_results(s, SPLIT)} for s in SYSTEMS}
    with open(EVAL_DATA_PATH) as f:
        gold = {q["id"]: q for q in json.load(f)}
    for s, rows in logs.items():
        if set(rows) != set(gold):
            raise ValueError(f"{s}: logged questions do not match {EVAL_DATA_PATH}")
    return logs, gold


def em(logs, system, qid):
    r = logs[system][qid]
    return exact_match(r["final_answer"], r["gold_answer"])


def f1(logs, system, qid):
    r = logs[system][qid]
    return f1_score(r["final_answer"], r["gold_answer"])


def paired(logs, a, b, qids, metric):
    fn = em if metric == "exact_match" else f1
    return summarise(paired_bootstrap([fn(logs, a, q) for q in qids],
                                      [fn(logs, b, q) for q in qids]))


def records(logs, system, qids):
    return [logs[system][q] for q in qids]


def reactive_queries(r):
    """Queries issued after the planned sub-queries were exhausted."""
    start = max(len(r["sub_queries_generated"]), 1)
    return r["queries_per_hop"][start:]


# ── analyses ─────────────────────────────────────────────────────────────────

def rq2(logs, gold):
    ms = logs["main_system"]
    yes = sorted(q for q in ms if ms[q]["hop_necessity_classification"] == "YES")
    no = sorted(q for q in ms if ms[q]["hop_necessity_classification"] == "NO")
    out = {"n_yes": len(yes), "n_no": len(no)}

    # Does the Main System's advantage over Baseline A differ between subsets?
    out["interaction_main_vs_baseline_a"] = {
        m: summarise(interaction_test(records(logs, "main_system", yes),
                                      records(logs, "baseline_a", yes),
                                      records(logs, "main_system", no),
                                      records(logs, "baseline_a", no), fn))
        for m, fn in [("exact_match", exact_match), ("f1", f1_score)]}

    # Baseline A never sees D1's label: does it score differently on the two
    # subsets? Different questions, same system, so the test is unpaired.
    out["baseline_a_no_minus_yes"] = {
        "exact_match": summarise(unpaired_bootstrap([em(logs, "baseline_a", q) for q in no],
                                                    [em(logs, "baseline_a", q) for q in yes])),
        "f1": summarise(unpaired_bootstrap([f1(logs, "baseline_a", q) for q in no],
                                           [f1(logs, "baseline_a", q) for q in yes])),
        "supporting_facts_recall": summarise(unpaired_bootstrap(
            [supporting_facts_recall(logs["baseline_a"][q], gold[q]) for q in no],
            [supporting_facts_recall(logs["baseline_a"][q], gold[q]) for q in yes])),
    }

    # Counterfactual: Baseline B takes a second hop on every question and never
    # sees D1, so its gain over Baseline A shows what a second hop is worth on
    # each of D1's subsets.
    out["second_hop_gain_baseline_b_minus_a"] = {
        subset: {m: paired(logs, "baseline_b", "baseline_a", ids, m)
                 for m in ["exact_match", "f1"]}
        for subset, ids in [("d1_yes", yes), ("d1_no", no)]}
    out["correct_answers"] = {
        subset: {s: int(sum(em(logs, s, q) for q in ids))
                 for s in ["baseline_a", "baseline_b", "main_system"]}
        for subset, ids in [("d1_yes", yes), ("d1_no", no)]}
    out["interaction_baseline_b_vs_baseline_a"] = {
        m: summarise(interaction_test(records(logs, "baseline_b", yes),
                                      records(logs, "baseline_a", yes),
                                      records(logs, "baseline_b", no),
                                      records(logs, "baseline_a", no), fn))
        for m, fn in [("exact_match", exact_match), ("f1", f1_score)]}
    return out


def sufficiency_stops(logs, gold):
    ms = logs["main_system"]
    stops = sorted(q for q in ms if ms[q]["stop_condition_triggered"] == "sufficiency")
    sfr = {q: supporting_facts_recall(ms[q], gold[q]) for q in stops}
    complete = [q for q in stops if sfr[q] == 1.0]
    missing = [q for q in stops if sfr[q] < 1.0]

    def group(ids):
        return {"n": len(ids),
                "main_system_em": r4(np.mean([em(logs, "main_system", q) for q in ids])),
                "baseline_b_em_same_questions": r4(np.mean([em(logs, "baseline_b", q) for q in ids])),
                "baseline_b_minus_main_em": paired(logs, "baseline_b", "main_system", ids, "exact_match")}

    by_hop = {}
    for h in range(1, MAX_HOPS + 1):
        ids = [q for q in stops if ms[q]["num_hops"] == h]
        n_missing = sum(1 for q in ids if sfr[q] < 1.0)
        by_hop[str(h)] = {"n": len(ids), "n_missing": n_missing,
                          "share_missing": r4(n_missing / len(ids)) if ids else None}
    return {
        "definition": "A gold article is 'missing' when Supporting Facts Recall (article level) "
                      "is below 1 at the point D3 declared sufficiency.",
        "n_sufficiency_stops": len(stops),
        "n_missing": len(missing),
        "share_missing": r4(len(missing) / len(stops)),
        "sfr_distribution": {str(k): v for k, v in sorted(Counter(round(x, 2) for x in sfr.values()).items())},
        "both_articles_found": group(complete),
        "article_missing": group(missing),
        "by_stopping_hop": by_hop,
    }


def sfr_by_stop_group(logs, gold):
    ms = logs["main_system"]
    groups = {
        "single_hop_path_d1_no": lambda r: r["stop_condition_triggered"] == "single_hop",
        "ended_after_one_hop_any_reason": lambda r: r["num_hops"] == 1,
        "sufficiency_stop": lambda r: r["stop_condition_triggered"] == "sufficiency",
        "hop_limit": lambda r: r["stop_condition_triggered"] == "max_hops",
    }
    out = {}
    for name, keep in groups.items():
        ids = [q for q, r in ms.items() if keep(r)]
        out[name] = {"n": len(ids),
                     "supporting_facts_recall": r4(np.mean([supporting_facts_recall(ms[q], gold[q]) for q in ids]))}
    return out


def judge_baseline_b_vs_main():
    a = {r["question_id"]: r for r in load_judge_results("baseline_b", SPLIT)}
    b = {r["question_id"]: r for r in load_judge_results("main_system", SPLIT)}
    qids = sorted(set(a) & set(b))
    return {dim: summarise(paired_bootstrap([a[q][dim] for q in qids], [b[q][dim] for q in qids]))
            for dim in ["faithfulness", "relevance", "coherence"]}


def d1_agreement(logs):
    ms, ab = logs["main_system"], logs["ablation_1"]
    pairs = Counter(f"{ms[q]['hop_necessity_classification']}/{ab[q]['hop_necessity_classification']}"
                    for q in ms)
    disagree = sum(v for k, v in pairs.items() if k.split("/")[0] != k.split("/")[1])
    return {"main_system/ablation_1": dict(sorted(pairs.items())),
            "n_disagree": disagree, "n_questions": len(ms)}


def plan_length(logs):
    ms = logs["main_system"]
    yes = [q for q in ms if ms[q]["hop_necessity_classification"] == "YES"]
    plen = {q: len(ms[q]["sub_queries_generated"]) for q in yes}
    over = sorted(q for q in yes if plen[q] > MAX_HOPS)
    at = [q for q in yes if plen[q] == MAX_HOPS]
    within = sorted(q for q in yes if plen[q] <= MAX_HOPS)

    unexecuted_budget = sum(plen[q] - MAX_HOPS for q in over
                            if ms[q]["stop_condition_triggered"] == "max_hops")
    unexecuted_total = sum(max(0, plen[q] - ms[q]["num_hops"]) for q in yes)

    def em_mean(system, ids):
        return r4(np.mean([em(logs, system, q) for q in ids]))

    return {
        "distribution": {str(k): v for k, v in sorted(Counter(plen.values()).items())},
        "n_multi_hop": len(yes),
        "n_over_budget": len(over),
        "n_exactly_budget": len(at),
        "share_at_or_over_budget": r4((len(over) + len(at)) / len(yes)),
        "unexecuted_sub_queries": {
            "total": unexecuted_total,
            "because_plan_exceeded_budget": unexecuted_budget,
            "because_d3_stopped_early": unexecuted_total - unexecuted_budget,
        },
        "exact_match": {g: {s: em_mean(s, ids) for s in ["main_system", "ablation_1", "baseline_a"]}
                        for g, ids in [("over_budget", over), ("within_budget", within)]},
        "main_minus_ablation_em": {"over_budget": paired(logs, "main_system", "ablation_1", over, "exact_match"),
                                   "within_budget": paired(logs, "main_system", "ablation_1", within, "exact_match")},
        "difference_of_gaps_em": summarise(interaction_test(
            records(logs, "main_system", over), records(logs, "ablation_1", over),
            records(logs, "main_system", within), records(logs, "ablation_1", within), exact_match)),
    }


def decomposition_fallback(logs):
    ms = logs["main_system"]
    ids = sorted(q for q, r in ms.items() if r["hop_necessity_classification"] == "YES"
                 and r["sub_queries_generated"] == [r["question"]])
    return {"n": len(ids), "n_correct": int(sum(em(logs, "main_system", q) for q in ids)),
            "questions": [ms[q]["question"] for q in ids]}


def decomposition_by_type(logs, gold):
    out = {}
    types = {t: sorted(q for q in gold if gold[q]["type"] == t) for t in ["bridge", "comparison"]}
    for t, ids in types.items():
        out[t] = {"n": len(ids), **{m: paired(logs, "main_system", "ablation_1", ids, m)
                                    for m in ["exact_match", "f1"]}}
    out["difference_bridge_minus_comparison"] = {
        m: summarise(interaction_test(records(logs, "main_system", types["bridge"]),
                                      records(logs, "ablation_1", types["bridge"]),
                                      records(logs, "main_system", types["comparison"]),
                                      records(logs, "ablation_1", types["comparison"]), fn))
        for m, fn in [("exact_match", exact_match), ("f1", f1_score)]}
    return out


def retrieval_volume(logs):
    out = {}
    for s in SYSTEMS:
        counts = [len({c for hop in r["docs_retrieved_per_hop"] for c in hop}) for r in logs[s].values()]
        out[s] = r4(np.mean(counts))
    return {"mean_distinct_chunks_per_question": out}


def code_defects(logs):
    ms, ab = logs["main_system"], logs["ablation_1"]

    def multi(system):
        return [q for q, r in logs[system].items() if r["hop_necessity_classification"] == "YES"]

    fallback = {q for q in multi("main_system")
                if ms[q]["sub_queries_generated"] == [ms[q]["question"]]}
    extraction = {s: {q for q in multi(s)
                      if any(UNEXTRACTED_QUERY.search(x) for x in reactive_queries(logs[s][q]))}
                  for s in ["main_system", "ablation_1"]}
    n_reactive = {s: sum(len(reactive_queries(logs[s][q])) for q in multi(s))
                  for s in ["main_system", "ablation_1"]}
    n_unextracted = {s: sum(1 for q in multi(s) for x in reactive_queries(logs[s][q])
                            if UNEXTRACTED_QUERY.search(x))
                     for s in ["main_system", "ablation_1"]}
    self_ref_subqueries = [sq for q in multi("main_system")
                           for sq in ms[q]["sub_queries_generated"] if SELF_REFERENCE.search(sq)]
    self_ref = {q for q in multi("main_system")
                if any(SELF_REFERENCE.search(sq) for sq in ms[q]["sub_queries_generated"])}

    others = [q for q in multi("main_system") if q not in self_ref]
    affected = fallback | extraction["main_system"] | extraction["ablation_1"] | self_ref
    kept = sorted(q for q in ms if q not in affected)

    sensitivity = {}
    for name, a, b in [("RQ1", "main_system", "baseline_a"), ("RQ3a", "ablation_1", "baseline_b"),
                       ("RQ3b", "main_system", "ablation_1"), ("CTX", "baseline_b", "baseline_a")]:
        sensitivity[name] = {m: paired(logs, a, b, kept, m) for m in ["exact_match", "f1"]}
    yes_k = [q for q in kept if ms[q]["hop_necessity_classification"] == "YES"]
    no_k = [q for q in kept if ms[q]["hop_necessity_classification"] == "NO"]
    sensitivity["RQ2_interaction_em"] = summarise(interaction_test(
        records(logs, "main_system", yes_k), records(logs, "baseline_a", yes_k),
        records(logs, "main_system", no_k), records(logs, "baseline_a", no_k), exact_match))

    return {
        "decomposer_fallback": {"n_questions": len(fallback)},
        "reactive_query_extraction": {
            s: {"n_reactive_queries": n_reactive[s], "n_unextracted": n_unextracted[s],
                "n_questions": len(extraction[s])} for s in ["main_system", "ablation_1"]},
        "unvalidated_self_reference": {
            "n_sub_queries": len(self_ref_subqueries), "n_questions": len(self_ref),
            "main_system_em_on_these": r4(np.mean([em(logs, "main_system", q) for q in self_ref])),
            "main_system_em_other_multi_hop": r4(np.mean([em(logs, "main_system", q) for q in others])),
            "correct_answers_on_these": {s: int(sum(em(logs, s, q) for q in self_ref)) for s in SYSTEMS},
        },
        "n_affected_questions_any_defect": len(affected),
        "sensitivity_excluding_affected": {"n_questions": len(kept), **sensitivity},
    }


def phase7_cross_system(logs):
    with open(FAILURE_CASES_PATH) as f:
        cases = json.load(f)["cases"]
    comparators = ["baseline_a", "baseline_b", "ablation_1"]
    solved = {s: sorted(c["n"] for c in cases if em(logs, s, c["question_id"])) for s in comparators}
    escaped = {s: sorted(c["n"] for c in cases
                         if em(logs, s, c["question_id"]) or f1(logs, s, c["question_id"]) >= 0.3)
               for s in comparators}
    union_em = sorted(set().union(*solved.values()))
    union_escaped = sorted(set().union(*escaped.values()))
    identical = sorted(c["n"] for c in cases
                       if len({normalise(logs[s][c["question_id"]]["final_answer"])
                               for s in SYSTEMS}) == 1)
    return {
        "n_cases": len(cases),
        "solved_exact_match": solved,
        "solved_by_any_comparator": union_em,
        "all_four_failed_exact_match": len(cases) - len(union_em),
        "escaped_failure_criterion": escaped,
        "escaped_by_any_comparator": union_escaped,
        "all_four_failed_failure_criterion": len(cases) - len(union_escaped),
        "identical_answer_all_four": {"n": len(identical), "cases": identical},
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    logs, gold = load_all()
    out = {
        "generated_at": datetime.now().isoformat(),
        "split": SPLIT,
        "seed": RANDOM_SEED,
        "n_resamples": N_RESAMPLES,
        "note": "Supplementary analyses. None belongs to the Holm-corrected families of "
                "run_significance.py; every p-value is uncorrected.",
        "rq2": rq2(logs, gold),
        "sufficiency_stops": sufficiency_stops(logs, gold),
        "sfr_by_stop_group": sfr_by_stop_group(logs, gold),
        "judge_baseline_b_vs_main": judge_baseline_b_vs_main(),
        "d1_agreement": d1_agreement(logs),
        "plan_length": plan_length(logs),
        "decomposition_fallback": decomposition_fallback(logs),
        "decomposition_by_type": decomposition_by_type(logs, gold),
        "retrieval_volume": retrieval_volume(logs),
        "code_defects": code_defects(logs),
        "phase7_cross_system": phase7_cross_system(logs),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Supplementary analyses written to {OUTPUT_PATH}")
    return out


if __name__ == "__main__":
    main()