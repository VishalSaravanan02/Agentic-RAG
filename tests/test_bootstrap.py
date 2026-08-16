# =============================================================================
# test_bootstrap.py — Paired bootstrap significance testing
#
# Mostly synthetic: the point of testing a statistical routine is to feed it
# inputs whose correct answer is known in advance, which real result files
# cannot provide. Real files are used only for structural checks (loading,
# question-ID alignment), never for asserting particular metric values, since
# those shift whenever a system is re-run.
#
# Run with: python -m pytest tests/ -v
# =============================================================================

import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import RESULTS_DIR
from src.core.logger import load_results
from src.evaluation.bootstrap import (
    paired_bootstrap,
    compare,
    per_question_values,
    format_result,
    NEEDS_GOLD,
)
from src.evaluation.metrics import (
    recall_at_k,
    retrieval_precision,
    duplicate_retrieval_rate,
)

FAST = 2000  # fewer resamples than production; keeps the suite quick


# --- Core statistical behaviour ---------------------------------------------

def test_identical_systems_show_no_difference():
    """Two identical systems: zero difference, never significant."""
    x = [0, 1, 1, 0, 1] * 40
    r = paired_bootstrap(x, x, n_resamples=FAST)
    assert r["observed_difference"] == 0.0
    assert not r["significant"]


def test_maximally_separated_systems_are_significant():
    """Perfect vs useless: difference of exactly 1.0, clearly significant."""
    r = paired_bootstrap([1] * 200, [0] * 200, n_resamples=FAST)
    assert r["observed_difference"] == 1.0
    assert r["significant"]


def test_confidence_interval_brackets_known_effect():
    """With a true effect of +0.10 built in, the CI should contain it."""
    rng = np.random.default_rng(7)
    a = rng.binomial(1, 0.55, 2000)
    b = rng.binomial(1, 0.45, 2000)
    r = paired_bootstrap(a, b, n_resamples=FAST)
    assert r["ci_low"] <= 0.10 <= r["ci_high"]


def test_small_difference_is_not_significant():
    """
    72 vs 70 correct on n=159 — the Main System vs Ablation 1 gap on dev.
    Constructed with maximal overlap, so this is the minimum-variance case:
    if it were ever going to reach significance, it would here.
    """
    a, b = np.zeros(159), np.zeros(159)
    a[:72] = 1
    b[:70] = 1
    r = paired_bootstrap(a, b, n_resamples=FAST)
    assert not r["significant"]


def test_seed_controls_reproducibility():
    """Same seed must reproduce exactly; a different seed must not."""
    rng = np.random.default_rng(7)
    a, b = rng.binomial(1, 0.55, 500), rng.binomial(1, 0.45, 500)
    assert paired_bootstrap(a, b, n_resamples=FAST, seed=42) == \
           paired_bootstrap(a, b, n_resamples=FAST, seed=42)
    assert paired_bootstrap(a, b, n_resamples=FAST, seed=42)["ci_low"] != \
           paired_bootstrap(a, b, n_resamples=FAST, seed=99)["ci_low"]


def test_pairing_detects_what_unpaired_would_miss():
    """
    THE test that distinguishes a paired bootstrap from an independent one.

    A small, perfectly consistent per-question difference (+100ms) buried in
    large between-question variance is detectable only if resampling preserves
    the pairing. An implementation that resampled each system independently
    would return a confidence interval spanning zero here.
    """
    base = np.random.default_rng(3).uniform(1000, 10000, 200)
    r = paired_bootstrap(base + 100, base, n_resamples=FAST)
    assert r["significant"]
    assert abs(r["observed_difference"] - 100.0) < 1e-9


# --- Regression tests for issues found in review ----------------------------

def test_p_value_never_exceeds_one():
    """
    Regression: with tails counted as (<=0, >=0), exact-zero replicates fall in
    both, and an all-ties distribution produced p = 2.0. Ties are now split and
    a clamp retained, because the add-one correction pushes the corrected tails
    to sum slightly above 1 again.
    """
    r = paired_bootstrap(np.zeros(200), np.zeros(200), n_resamples=FAST)
    assert 0.0 < r["p_value"] <= 1.0


def test_p_value_is_never_zero_and_respects_its_floor():
    """
    Regression: with no replicates on one side, p came out as exactly 0.0.
    Zero replicates beyond the observed effect means p is below the resolution
    of the resampling, not that it is zero. The add-one correction keeps p
    strictly positive and at or above 2 / (n_resamples + 1).
    """
    r = paired_bootstrap([1] * 200, [0] * 200, n_resamples=FAST)
    assert r["p_value"] > 0.0
    assert r["p_value"] >= r["p_floor"]


def test_significance_flag_always_matches_p_threshold():
    """
    Regression: `significant` was derived from the CI while `p_value` came from
    the tail proportions — near-equivalent procedures that could disagree at the
    boundary. `significant` is now defined as p < 0.05, one source of truth.
    """
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(50, 400))
        a = rng.binomial(1, 0.5, n)
        b = rng.binomial(1, min(0.5 + rng.uniform(0, 0.09), 1.0), n)
        r = paired_bootstrap(a, b, n_resamples=500)
        assert r["significant"] == (r["p_value"] < 0.05)


def test_p_floor_consistent_with_reported_resample_count():
    """
    Regression: p_floor was derived from the returned distribution while
    n_resamples echoed the requested parameter, so the two could in principle
    describe different quantities. Both now come from the returned distribution.
    """
    for nr in (500, 1000, 2000):
        r = paired_bootstrap(np.random.default_rng(1).normal(0, 1, 100),
                             np.zeros(100), n_resamples=nr)
        assert r["p_floor"] == pytest.approx(2.0 / (r["n_resamples"] + 1))


def test_p_value_bounds_hold_across_many_random_comparisons():
    """Sweep: p must stay within (0, 1] and never fall below its floor."""
    rng = np.random.default_rng(5)
    for _ in range(150):
        n = int(rng.integers(30, 400))
        a = rng.binomial(1, 0.5, n)
        b = rng.binomial(1, min(0.5 + rng.uniform(0, 0.6), 1.0), n)
        r = paired_bootstrap(a, b, n_resamples=500)
        assert 0.0 < r["p_value"] <= 1.0
        assert r["p_value"] >= r["p_floor"] - 1e-12


# --- Input guardrails -------------------------------------------------------

def test_unequal_length_inputs_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0, 1], [1, 0], n_resamples=FAST)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([], [], n_resamples=FAST)


@pytest.mark.parametrize("metric", sorted(NEEDS_GOLD))
def test_gold_dependent_metric_rejected_without_gold_data(metric):
    """
    All three retrieval metrics score against HotpotQA's gold supporting facts.
    Without them the metric would silently be unavailable rather than wrong, so
    this fails loudly instead.

    REGRESSION GUARD for recall_at_k specifically. recall_at_k was originally
    scored against the gold ANSWER string matched as a substring of the chunk
    ID, which needed no gold data and produced false positives (gold "no"
    matching "Christopher Nolan"). Driving this test off NEEDS_GOLD means that
    if recall_at_k is ever removed from that set, this test disappears silently
    -- so test_recall_at_k_is_gold_dependent below pins the membership itself.
    """
    with pytest.raises(ValueError):
        per_question_values("main_system", "dev", metric)


def test_recall_at_k_is_gold_dependent():
    """
    Pins recall_at_k's membership of NEEDS_GOLD, and that no earlier branch in
    per_question_values() shadows it. Both are required: adding the metric to
    NEEDS_GOLD while leaving a standalone `elif metric == "recall_at_k"` branch
    above the gold branch would leave the old, invalid scoring in place.
    """
    assert "recall_at_k" in NEEDS_GOLD

    gold = {"q1": {"supporting_facts": {"title": ["Gold Article"]}}}
    result = {"question_id": "q1",
              "docs_retrieved_per_hop": [["Gold Article_0", "Other_3"]]}
    assert recall_at_k(result, gold["q1"]) == 1

    miss = {"question_id": "q1",
            "docs_retrieved_per_hop": [["Other_3", "Another_1"]]}
    assert recall_at_k(miss, gold["q1"]) == 0


def test_recall_at_k_no_longer_matches_answer_substrings():
    """
    The exact defect that motivated the rescoring: gold answer "no" scored 1
    against article titles merely CONTAINING the letters n-o. Under gold-based
    scoring these must all be 0, since none of the titles is a gold article.
    """
    item = {"supporting_facts": {"title": ["Some Gold Article"]}}
    for title in ("Taxonomy of the Cactaceae_1",
                  "Christopher Nolan (author)_1",
                  "Harley Knoles_0"):
        result = {"gold_answer": "no", "docs_retrieved_per_hop": [[title]]}
        assert recall_at_k(result, item) == 0, f"false positive on {title!r}"


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        per_question_values("main_system", "dev", "not_a_real_metric")


# --- Integration with real result files -------------------------------------
# Structural only: no assertions on particular metric values, since those move
# whenever a system is re-run.

def test_compare_pairs_all_questions_across_two_systems():
    r = compare("main_system", "baseline_a", "dev", "exact_match")
    assert r["n_questions"] == len(load_results("baseline_a", "dev"))
    assert r["system_a"] == "main_system" and r["system_b"] == "baseline_a"
    assert r["ci_low"] <= r["observed_difference"] <= r["ci_high"]


def test_compare_restricted_to_a_subset():
    """The RQ2 path: restricting to questions D1 classified as multi-hop."""
    yes = {x["question_id"] for x in load_results("main_system", "dev")
           if x["hop_necessity_classification"] == "YES"}
    r = compare("main_system", "baseline_a", "dev", "exact_match",
                question_ids=yes)
    assert r["n_questions"] == len(yes)
    assert 0 < len(yes) < 200


def test_compare_rejects_unknown_question_ids():
    with pytest.raises(ValueError):
        compare("main_system", "baseline_a", "dev", "exact_match",
                question_ids={"not-a-real-question-id"})


@pytest.fixture
def partial_results_file():
    """
    Writes a deliberately incomplete results file, mimicking a run interrupted
    and resumed under checkpointing, then removes it.
    """
    name = "_tmp_partial_system"
    path = os.path.join(RESULTS_DIR, f"{name}_dev.jsonl")
    rows = load_results("main_system", "dev")[:192]
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    yield name
    os.remove(path)


def test_uneven_coverage_raises_rather_than_silently_intersecting(partial_results_file):
    """
    Regression, and the most consequential of the issues found in review.

    Comparing a 192-question run against a 200-question one previously took the
    intersection without complaint and reported a clean-looking result computed
    over fewer questions than either system held. Given that run_evaluation.py
    checkpoints and resumes, a partial results file is a realistic occurrence.
    """
    with pytest.raises(ValueError, match="different question sets"):
        compare(partial_results_file, "baseline_a", "dev", "exact_match")


def test_uneven_coverage_permitted_when_subset_is_explicit(partial_results_file):
    """The same mismatch is fine when the restriction is stated deliberately."""
    ids = {x["question_id"] for x in load_results(partial_results_file, "dev")}
    r = compare(partial_results_file, "baseline_a", "dev", "exact_match",
                question_ids=ids)
    assert r["n_questions"] == 192


# --- Output formatting ------------------------------------------------------

def test_format_result_reports_floor_rather_than_a_zero_p_value():
    """A p-value at the floor must display as '< floor', never as '= 0'."""
    r = paired_bootstrap([1] * 200, [0] * 200, n_resamples=FAST)
    line = format_result({**r, "system_a": "a", "system_b": "b",
                          "metric": "exact_match", "split": "dev",
                          "mean_a": 1.0, "mean_b": 0.0})
    assert "p <" in line
    assert "p = 0.0000" not in line

# --- Duplicate retrieval rate ------------------------------------------------

def test_duplicate_rate_needs_no_gold_data():
    """
    Computed from logged retrievals alone, so unlike the other retrieval
    metrics it must NOT be in NEEDS_GOLD and must run without gold_lookup.
    """
    assert "duplicate_retrieval_rate" not in NEEDS_GOLD
    v = per_question_values("main_system", "dev", "duplicate_retrieval_rate")
    assert len(v) > 0
    assert all(0.0 <= x <= 1.0 for x in v.values())


def test_duplicate_rate_counts_repeated_slots_not_repeated_docs():
    """
    The denominator is retrieval slots used, not distinct documents. A chunk
    fetched on three hops wastes two slots, not one.
    """
    # 6 slots, 2 distinct chunks -> 4 wasted
    r = {"docs_retrieved_per_hop": [["A_0", "B_0"], ["A_0", "B_0"], ["A_0", "B_0"]]}
    assert duplicate_retrieval_rate(r) == pytest.approx(4 / 6)

    # No repeats at all
    clean = {"docs_retrieved_per_hop": [["A_0", "B_0"], ["C_0", "D_0"]]}
    assert duplicate_retrieval_rate(clean) == 0.0

    # Single hop cannot repeat
    single = {"docs_retrieved_per_hop": [["A_0", "B_0", "C_0"]]}
    assert duplicate_retrieval_rate(single) == 0.0

    # No retrievals at all
    assert duplicate_retrieval_rate({"docs_retrieved_per_hop": []}) == 0.0


def test_retrieval_precision_counts_only_what_the_model_read():
    """
    Precision is scored over deduplicated chunks, matching the context the
    retrieval loop actually assembles. The worked case: 10 retrieved, 8
    distinct, 2 of them gold -> 0.25, not 3/10 = 0.30.
    """
    result = {"docs_retrieved_per_hop": [
        ["George Eliot_0", "Middlemarch_0", "Victorian novels_0",
         "Warwickshire_0", "Mary Ann Evans_0"],
        ["George Eliot_0", "Bedford College_0", "London_0",
         "Middlemarch_0", "Womens education_0"],
    ]}
    item = {"supporting_facts": {"title": ["George Eliot", "Bedford College"]}}
    assert retrieval_precision(result, item) == pytest.approx(0.25)