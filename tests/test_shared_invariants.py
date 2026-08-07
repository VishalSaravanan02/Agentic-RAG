# =============================================================================
# test_shared_invariants.py — Cross-system architectural invariants
# These tests belong to no single system: they assert properties that must
# hold ACROSS systems for the experimental comparisons to remain valid.
# Run with: python -m pytest tests/ -v
# =============================================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import answer_synthesizer
from src.systems import baseline_a, baseline_b, main_system, ablation_1


# --- Test 1: THE D5 INVARIANT — synthesis prompt identical everywhere --------
# RQ1 compares Baseline A against the Main System and attributes any measured
# difference to retrieval strategy alone. That attribution holds ONLY if the
# generation step is held constant, which requires the D5 synthesis prompt to
# be byte-for-byte identical in every system.
#
# Baseline A is the only system holding its own copy of the prompt string;
# every other system imports synthesize() from answer_synthesizer.py and so
# cannot drift. This test guards the single pair that CAN drift.
def test_d5_prompt_identical_in_baseline_a_and_synthesizer():
    assert (
        baseline_a.SYNTHESIS_PROMPT_TEMPLATE
        == answer_synthesizer.SYNTHESIS_PROMPT_TEMPLATE
    ), (
        "D5 synthesis prompt has drifted between baseline_a.py and "
        "answer_synthesizer.py. The generation step is no longer held constant "
        "across systems, so RQ1's comparison is invalid until they match again."
    )


# --- Test 2: every other system uses the SHARED synthesise function ----------
# Test 1 protects only the one known duplicate. If a future edit gave some
# other system its own copy of the prompt, Test 1 would still pass while the
# invariant silently broke. This asserts the three importing systems really do
# call the same function object, so no third copy of D5 can exist unnoticed.
def test_other_systems_use_shared_synthesize():
    for system in (baseline_b, main_system, ablation_1):
        assert system.synthesize is answer_synthesizer.synthesize, (
            f"{system.__name__} no longer uses the shared synthesize() from "
            "answer_synthesizer.py — the D5 generation step may have been "
            "duplicated or overridden."
        )