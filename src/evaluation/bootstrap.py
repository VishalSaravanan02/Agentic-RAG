# =============================================================================
# bootstrap.py — Paired bootstrap significance testing
#
# Implements the significance testing specified in the proposal (Section 7.4):
# paired bootstrap resampling, 10,000 iterations, p < 0.05, across answer
# quality, retrieval quality, and efficiency metrics.
#
# PAIRED, not independent: every system is evaluated on the same questions, so
# resampling draws a set of QUESTIONS and scores both systems on that same set.
# This preserves the pairing and is what makes the test sensitive to per-question
# differences rather than just the gap between two means.
#
# Reads saved .jsonl result files only. No LLM calls, no cost, deterministic
# given a fixed seed.
# =============================================================================

import numpy as np
from scipy import stats

from src.core.config import RANDOM_SEED
from src.core.logger import load_results
from src.evaluation.metrics import (
    exact_match,
    f1_score,
    recall_at_k,
    supporting_facts_recall,
    retrieval_precision,
)

N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

# Metrics requiring HotpotQA gold supporting facts
NEEDS_GOLD = {"supporting_facts_recall", "retrieval_precision"}

METRICS = [
    "exact_match",
    "f1",
    "recall_at_k",
    "supporting_facts_recall",
    "retrieval_precision",
    "latency_ms",
    "input_tokens",
    "output_tokens",
]


def per_question_values(system_name: str, split: str, metric: str,
                        gold_lookup: dict | None = None) -> dict:
    """
    Return {question_id: metric_value} for one system on one split.

    gold_lookup maps question_id -> HotpotQA item, and is required for
    supporting_facts_recall and retrieval_precision. Passing it for other
    metrics is harmless.
    """
    if metric in NEEDS_GOLD and gold_lookup is None:
        raise ValueError(
            f"metric '{metric}' requires gold data; pass gold_lookup "
            f"built from the split's question file"
        )

    values = {}
    for r in load_results(system_name, split):
        qid = r["question_id"]

        if metric == "exact_match":
            v = exact_match(r["final_answer"], r["gold_answer"])
        elif metric == "f1":
            v = f1_score(r["final_answer"], r["gold_answer"])
        elif metric == "recall_at_k":
            v = recall_at_k(r)
        elif metric == "latency_ms":
            v = r["total_latency_ms"]
        elif metric == "input_tokens":
            v = r["input_tokens"]
        elif metric == "output_tokens":
            v = r["output_tokens"]
        elif metric in NEEDS_GOLD:
            if qid not in gold_lookup:
                continue          # question absent from gold file; skip
            item = gold_lookup[qid]
            v = (supporting_facts_recall(r, item) if metric == "supporting_facts_recall"
                 else retrieval_precision(r, item))
        else:
            raise ValueError(f"unknown metric: {metric}")

        values[qid] = float(v)

    return values


def paired_bootstrap(values_a, values_b,
                     n_resamples: int = N_RESAMPLES,
                     confidence_level: float = CONFIDENCE_LEVEL,
                     seed: int = RANDOM_SEED) -> dict:
    """
    Paired bootstrap on the per-question differences (a - b).

    Resampling the difference array is equivalent to resampling questions and
    scoring both systems on the same draw, which is what makes the test paired.

    Returns observed difference, confidence interval, and a two-sided p-value.

    The p-value is twice the smaller tail of the bootstrap distribution about
    zero. Replicates falling exactly on zero are split evenly between the two
    tails rather than counted in both, which would inflate the smaller tail and
    bias p upward — a real effect for discrete metrics such as exact_match and
    recall_at_k, where exact ties are common.

    An add-one correction is applied so that p is strictly positive and bounded
    below by exactly 2 / (n_resamples + 1) — approximately 0.0002 at 10,000
    resamples. A comparison at that floor should be reported as "p < 0.0002",
    never "p = 0": zero replicates beyond the observed effect means p lies below
    the resolution of the resampling, not that it is zero. Use format_result(),
    which applies this convention.

    SINGLE SOURCE OF TRUTH: `significant` is derived from p < 0.05, matching the
    criterion pre-specified in the proposal (Section 7.4). The percentile CI is
    reported alongside as an effect-size estimate. The two are near-equivalent
    procedures but can differ marginally at the boundary (a CI that appears to
    exclude zero alongside p = 0.051); where they differ, p governs.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires equal-length inputs")
    if a.size == 0:
        raise ValueError("no questions to compare")

    d = a - b
    observed = float(d.mean())

    res = stats.bootstrap(
        (d,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    dist = res.bootstrap_distribution
    ci_low = float(res.confidence_interval.low)
    ci_high = float(res.confidence_interval.high)

    # Two-sided p from the bootstrap distribution about zero.
    #
    # Exact-zero replicates are split evenly between the tails rather than
    # counted in both, which would inflate the smaller tail and bias p upward.
    # This matters for discrete metrics (exact_match, recall_at_k) where exact
    # ties are common; continuous metrics almost never land exactly on zero.
    #
    # The +1 / (B+1) add-one correction (Davison & Hinkley 1997) keeps p
    # strictly positive. Without it, a comparison with no replicates on one
    # side returns p = 0.0 exactly, which is not a meaningful value: zero
    # replicates beyond the observed effect means p is below the resolution of
    # B resamples, not that it is zero. The correction makes the reported floor
    # 2 / (n_resamples + 1) exact rather than aspirational.
    n = dist.size
    n_below = int((dist < 0).sum())
    n_above = int((dist > 0).sum())
    n_ties = int((dist == 0).sum())

    tail_low = (n_below + n_ties / 2.0 + 1.0) / (n + 1.0)
    tail_high = (n_above + n_ties / 2.0 + 1.0) / (n + 1.0)
    # The two corrected tails sum to (B+2)/(B+1) > 1, so min() can marginally
    # exceed 0.5 when the distribution is entirely ties; clamp is required.
    p = min(2.0 * min(tail_low, tail_high), 1.0)
    p_floor = 2.0 / (n + 1.0)

    return {
        "observed_difference": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "p_floor": p_floor,
        "significant": p < 0.05,
        "n_questions": int(d.size),
        # Reported from the returned distribution, not the requested parameter,
        # so this and p_floor are derived from the same quantity and cannot
        # disagree if scipy ever returns fewer replicates than requested.
        "n_resamples": int(n),
    }


def compare(system_a: str, system_b: str, split: str, metric: str,
            gold_lookup: dict | None = None,
            question_ids: set | None = None,
            seed: int = RANDOM_SEED) -> dict:
    """
    Compare two systems on one metric, aligning strictly by question_id.

    question_ids optionally restricts the comparison to a subset (used for the
    RQ2 split by D1 classification). Both systems must cover every question in
    the comparison, otherwise the pairing is broken and this raises.
    """
    va = per_question_values(system_a, split, metric, gold_lookup)
    vb = per_question_values(system_b, split, metric, gold_lookup)

    shared = set(va) & set(vb)

    if question_ids is None:
        # Full-split comparison: both systems must cover exactly the same
        # questions. Silently intersecting would compare a partial run (e.g.
        # one interrupted and resumed under checkpointing) against a complete
        # one and report a clean-looking but under-powered result.
        mismatched = set(va) ^ set(vb)
        if mismatched:
            raise ValueError(
                f"{system_a} and {system_b} cover different question sets for "
                f"metric '{metric}' ({len(mismatched)} mismatched; "
                f"{len(va)} vs {len(vb)} questions). Pass question_ids "
                f"explicitly if the restriction is intentional."
            )
    else:
        missing = question_ids - shared
        if missing:
            raise ValueError(
                f"{len(missing)} requested question(s) missing from "
                f"{system_a} or {system_b} for metric '{metric}'"
            )
        shared = shared & question_ids

    if not shared:
        raise ValueError(f"no shared questions between {system_a} and {system_b}")

    qids = sorted(shared)                      # sorted => reproducible ordering
    result = paired_bootstrap([va[q] for q in qids],
                              [vb[q] for q in qids],
                              seed=seed)
    result.update({
        "system_a": system_a,
        "system_b": system_b,
        "metric": metric,
        "split": split,
        "mean_a": float(np.mean([va[q] for q in qids])),
        "mean_b": float(np.mean([vb[q] for q in qids])),
    })
    return result


def format_result(r: dict) -> str:
    """One-line human-readable summary of a comparison."""
    p = r["p_value"]
    p_str = f"< {r['p_floor']:.4f}" if p <= r["p_floor"] else f"= {p:.4f}"
    star = "*" if r["significant"] else " "
    return (
        f"{star} {r['metric']:<24} "
        f"{r['system_a']} {r['mean_a']:.4f} vs {r['system_b']} {r['mean_b']:.4f}   "
        f"diff {r['observed_difference']:+.4f}  "
        f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
        f"p {p_str}  (n={r['n_questions']})"
    )