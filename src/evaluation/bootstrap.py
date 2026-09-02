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

import json
import os

import numpy as np
from scipy import stats

from src.core.config import RANDOM_SEED, RESULTS_DIR
from src.core.logger import load_results
from src.evaluation.metrics import (
    exact_match,
    f1_score,
    recall_at_k,
    supporting_facts_recall,
    retrieval_precision,
    duplicate_retrieval_rate,
)

N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

# Metrics requiring HotpotQA gold supporting facts.
# recall_at_k belongs here since it was rescored against gold supporting facts
# (see metrics.recall_at_k). It must ALSO have no earlier branch of its own in
# per_question_values(), or that branch wins and this membership does nothing.
NEEDS_GOLD = {"recall_at_k", "supporting_facts_recall", "retrieval_precision"}

# Judge dimensions live in results/judge_{system}_{split}.jsonl rather than in
# the system's own log, because scoring was a separate pass over a 200-question
# subsample. They need no gold data: the judge scores the answer against the
# retrieved context, not against the gold answer.
JUDGE_METRICS = {"faithfulness", "relevance", "coherence"}

METRICS = [
    "exact_match",
    "f1",
    "recall_at_k",
    "supporting_facts_recall",
    "retrieval_precision",
    "duplicate_retrieval_rate",
    "latency_ms",
    "input_tokens",
    "output_tokens",
]


def load_judge_results(system_name: str, split: str) -> list[dict]:
    """
    Read results/judge_{system}_{split}.jsonl.

    Kept separate from logger.load_results() because the judge files follow a
    different naming convention and a different schema: one record per scored
    question, carrying the three 1-5 dimensions rather than the 14-field run
    log. Malformed lines are skipped rather than raising, matching the
    behaviour of load_results().
    """
    path = os.path.join(RESULTS_DIR, f"judge_{system_name}_{split}.jsonl")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Judge results not found at {path}. Run the judge scoring pass "
            f"before requesting judge metrics."
        )

    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def per_question_values(system_name: str, split: str, metric: str,
                        gold_lookup: dict | None = None) -> dict:
    """
    Return {question_id: metric_value} for one system on one split.

    gold_lookup maps question_id -> HotpotQA item, and is required for
    supporting_facts_recall and retrieval_precision. Passing it for other
    metrics is harmless.

    Judge metrics are read from the separate judge results file and cover only
    the 200-question subsample, so a judge comparison is paired over 200
    questions rather than 1,000.
    """
    if metric in NEEDS_GOLD and gold_lookup is None:
        raise ValueError(
            f"metric '{metric}' requires gold data; pass gold_lookup "
            f"built from the split's question file"
        )

    if metric in JUDGE_METRICS:
        values = {}
        for r in load_judge_results(system_name, split):
            v = r.get(metric)
            # A question whose scoring failed or was skipped has no score for
            # this dimension; omit it rather than substituting a value. The
            # coverage check in compare() will then catch any mismatch between
            # the two systems being compared.
            if v is not None:
                values[r["question_id"]] = float(v)
        return values

    values = {}
    for r in load_results(system_name, split):
        qid = r["question_id"]

        if metric == "exact_match":
            v = exact_match(r["final_answer"], r["gold_answer"])
        elif metric == "f1":
            v = f1_score(r["final_answer"], r["gold_answer"])
        elif metric == "duplicate_retrieval_rate":
            # Needs no gold data — computed from the logged retrievals alone.
            v = duplicate_retrieval_rate(r)
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
            if metric == "recall_at_k":
                v = recall_at_k(r, item)
            elif metric == "supporting_facts_recall":
                v = supporting_facts_recall(r, item)
            else:
                v = retrieval_precision(r, item)
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


def holm_correction(results: list[dict]) -> list[dict]:
    """
    Holm-Bonferroni step-down correction over a family of tests.

    Returns a new list, ordered by raw p-value ascending, with two fields added
    to each entry: 'p_holm' and 'significant_holm'.

    The procedure sorts the m raw p-values ascending and multiplies the i-th
    (zero-indexed) by (m - i), then enforces monotonicity by carrying forward
    the running maximum, so a corrected value can never fall below one ranked
    before it. Values are capped at 1.0.

    Holm is uniformly more powerful than Bonferroni -- only the smallest
    p-value is multiplied by the full family size m, where Bonferroni
    multiplies every one by m -- while controlling the same family-wise error
    rate. Note that the multiplier for a given test therefore depends on its
    RANK within the family, so adding or removing a test can change the
    corrected value of an unrelated one.

    The choice of family is a judgement, not a computation. Passing a
    different subset of comparisons here yields a different answer, which is
    why run_significance.py reports more than one family rather than
    presenting a single corrected figure as definitive.
    """
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r["p_value"])
    m = len(ordered)

    out = []
    running_max = 0.0
    for i, r in enumerate(ordered):
        adjusted = min(1.0, max(running_max, (m - i) * r["p_value"]))
        running_max = adjusted
        entry = dict(r)
        entry["p_holm"] = adjusted
        entry["significant_holm"] = adjusted < (1 - CONFIDENCE_LEVEL)
        out.append(entry)

    return out


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