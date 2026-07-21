# =============================================================================
# test_baseline_b.py — Integration tests for Fixed 2-Hop RAG
# Run with: python -m pytest tests/ -v
# =============================================================================

import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.systems.baseline_b import run_baseline_b
from src.core.config import DEV_MODEL


# --- Test 1: baseline_b.py produces complete output --------------------------
def test_baseline_b_produces_required_fields():
    result = run_baseline_b(
        question="What is the capital of France?",
        question_id="pytest_baseline_b_test",
        gold_answer="Paris",
        model=DEV_MODEL
    )

    required_fields = [
        "question_id", "question", "system_name",
        "hop_necessity_classification", "num_hops",
        "sub_queries_generated", "docs_retrieved_per_hop",
        "stop_condition_triggered", "input_tokens", "output_tokens",
        "latency_per_hop_ms", "total_latency_ms",
        "final_answer", "gold_answer"
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    # Baseline-B-only diagnostic field
    assert "hop2_query" in result, "Missing field: hop2_query"


# --- Test 2: schema values are correct ---------------------------------------
def test_baseline_b_schema_values():
    result = run_baseline_b(
        question="What is the capital of France?",
        question_id="pytest_baseline_b_test2",
        gold_answer="Paris",
        model=DEV_MODEL
    )
    assert result["system_name"] == "baseline_b"

    # No D1, no D2, no D4 — Baseline B never classifies, decomposes, or decides
    # to stop. These placeholders must match Baseline A's so that the
    # evaluation pipeline reads every system's logs without special-casing.
    assert result["hop_necessity_classification"] == "N/A"
    assert result["sub_queries_generated"] == []
    assert result["stop_condition_triggered"] == "N/A"

    # The defining property of Baseline B: exactly two retrievals, always.
    assert result["num_hops"] == 2
    assert len(result["docs_retrieved_per_hop"]) == 2

    assert len(result["final_answer"]) > 0

    # hop2_query must be a real query, not an empty string — the fallback
    # returns the original question, so this holds on every code path.
    assert isinstance(result["hop2_query"], str)
    assert len(result["hop2_query"]) > 0