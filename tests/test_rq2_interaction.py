# =============================================================================
# test_rq2_interaction.py — The RQ2 interaction test
#
# These tests exist because the thing being replaced was not merely imprecise,
# it was uninformative: the previous verdict was `if yes_gap > no_gap`, which
# fires on roughly half of all comparisons between systems that differ in no
# way at all. The decisive test here is the false-positive-rate one, which
# would fail loudly if that rule ever returned.
#
# No API calls, no result files, no keys. Synthetic data throughout, so the
# correct answer is known in advance rather than assumed.
#
# Run with: python -m pytest tests/test_rq2_interaction.py -v
# =============================================================================

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

# run_rq2_analysis lives in scripts/ and is not an importable package, so load
# it by path rather than restructuring the repository for the tests' benefit.
_spec = importlib.util.spec_from_file_location(
    "run_rq2_analysis", os.path.join(REPO_ROOT, "scripts", "run_rq2_analysis.py"))
rq2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq2)

FAST = 2000  # fewer resamples than production; keeps the suite quick


def _records(scores):
    """
    Build minimal result records carrying a given binary score.

    interaction_test only reads final_answer and gold_answer through the metric
    function, so an exact-match metric can be driven by making the two fields
    equal (score 1) or different (score 0).
    """
    return [{"final_answer": "x", "gold_answer": "x" if s else "y"} for s in scores]


def _run(yes_main, yes_base, no_main, no_base, seed=0, n=FAST):
    return rq2.interaction_test(
        _records(yes_main), _records(yes_base),
        _records(no_main), _records(no_base),
        rq2.exact_match, n_resamples=n, seed=seed,
    )


# --- THE TEST THAT MATTERS ---------------------------------------------------

def test_false_positive_rate_is_near_alpha_not_near_half():
    """
    On systems with NO real difference and NO real interaction, the test must
    declare significance at roughly alpha, not at roughly 50%.

    The replaced `yes_gap > no_gap` rule scores about 48% here — it is a coin
    flip, not a test. This is the regression guard for that entire class of
    mistake.
    """
    rng = np.random.default_rng(12345)
    trials = 200
    fired = 0
    bare_rule_fired = 0

    for _ in range(trials):
        # Same underlying accuracy in both subsets, for both systems.
        yes_main = rng.binomial(1, 0.45, 400)
        yes_base = rng.binomial(1, 0.45, 400)
        no_main = rng.binomial(1, 0.45, 150)
        no_base = rng.binomial(1, 0.45, 150)

        r = _run(yes_main, yes_base, no_main, no_base, seed=int(rng.integers(1e6)), n=500)
        fired += r["significant"]
        bare_rule_fired += (r["yes_gap"] > r["no_gap"])

    rate = fired / trials
    bare_rate = bare_rule_fired / trials

    assert rate < 0.12, (
        f"False positive rate {rate:.1%} is far above alpha=5%. The interaction "
        f"test is not controlling error."
    )
    # Not an assertion about our code, but a documented contrast: the rule this
    # replaced fires about half the time on exactly this null data.
    assert 0.3 < bare_rate < 0.7, (
        f"Sanity check on the null data itself: the bare `yes_gap > no_gap` "
        f"rule fired {bare_rate:.1%} of the time, expected ~50%."
    )


# --- Basic correctness -------------------------------------------------------

def test_strong_interaction_is_detected():
    """A large, real concentration of advantage on the YES subset must be found."""
    r = _run(yes_main=[1] * 300, yes_base=[0] * 300,   # +1.00 gap on YES
             no_main=[1] * 150, no_base=[1] * 150)      #  0.00 gap on NO
    assert r["significant"]
    assert r["difference"] == pytest.approx(1.0)
    assert r["ci_low"] > 0


def test_reversed_interaction_is_reported_as_contrary_not_merely_null():
    """
    An advantage concentrated on the NO subset is a real finding contradicting
    RQ2, and must be distinguishable from an unresolved result.
    """
    r = _run(yes_main=[1] * 300, yes_base=[1] * 300,    # 0.00 gap on YES
             no_main=[1] * 150, no_base=[0] * 150)       # +1.00 gap on NO
    assert r["significant"]
    assert r["difference"] < 0


def test_identical_subsets_are_not_significant():
    """Identical behaviour in both subsets must not be flagged."""
    scores = [1, 0] * 100
    r = _run(yes_main=scores, yes_base=scores, no_main=scores, no_base=scores)
    assert not r["significant"]
    assert r["difference"] == pytest.approx(0.0)


# --- Reproducibility and reporting ------------------------------------------

def test_seed_controls_reproducibility():
    args = dict(yes_main=[1, 0] * 100, yes_base=[0, 0] * 100,
                no_main=[1, 0] * 50, no_base=[1, 0] * 50)
    assert _run(**args, seed=42) == _run(**args, seed=42)
    assert _run(**args, seed=42)["ci_low"] != _run(**args, seed=99)["ci_low"]


def test_p_value_is_never_zero_and_respects_its_floor():
    """
    Matches the convention in src/evaluation/bootstrap.py: the add-one
    correction keeps p strictly positive, so an extreme result is reported as
    below the resolution of the resampling rather than as exactly zero.
    """
    r = _run(yes_main=[1] * 300, yes_base=[0] * 300,
             no_main=[1] * 150, no_base=[1] * 150)
    assert r["p_value"] > 0
    assert r["p_value"] >= r["p_floor"]
    assert r["p_floor"] == pytest.approx(2.0 / (FAST + 1))


def test_subset_sizes_are_reported():
    r = _run(yes_main=[1] * 30, yes_base=[0] * 30,
             no_main=[1] * 17, no_base=[1] * 17)
    assert r["n_yes"] == 30
    assert r["n_no"] == 17