# =============================================================================
# test_main_system.py — Integration tests for the full agentic pipeline
# Run with: python -m pytest tests/ -v
# =============================================================================

import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.systems.main_system import run_main_system
from src.core.config import DEV_MODEL, MAX_HOPS


# --- Test 1: Full pipeline runs end-to-end without crashing ------------------
def test_main_system_runs_end_to_end():
    result = run_main_system(
        question="What is the capital of France?",
        question_id="pytest_main_001",
        gold_answer="Paris",
        model=DEV_MODEL
    )
    assert result is not None
    assert isinstance(result, dict)


# --- Test 2: All required log fields are present -----------------------------
def test_main_system_produces_required_fields():
    result = run_main_system(
        question="What is the capital of France?",
        question_id="pytest_main_002",
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


# --- Test 3: Schema values are correct types ---------------------------------
def test_main_system_schema_types():
    result = run_main_system(
        question="What is the capital of France?",
        question_id="pytest_main_003",
        gold_answer="Paris",
        model=DEV_MODEL
    )

    assert result["system_name"] == "main_system"
    assert result["hop_necessity_classification"] in ["YES", "NO"]
    assert isinstance(result["num_hops"], int)
    assert result["num_hops"] >= 1
    assert isinstance(result["sub_queries_generated"], list)
    assert isinstance(result["docs_retrieved_per_hop"], list)
    assert result["stop_condition_triggered"] in [
        "sufficiency", "max_hops", "single_hop"
    ]
    assert isinstance(result["input_tokens"], int)
    assert isinstance(result["output_tokens"], int)
    assert isinstance(result["total_latency_ms"], float)
    assert len(result["final_answer"]) > 0


# --- Test 4: Hop count never exceeds MAX_HOPS --------------------------------
def test_main_system_respects_max_hops():
    # Use a deliberately hard question to push the hop limit
    result = run_main_system(
        question="Which athlete won the most medals at the 1996 Atlanta Olympics and what country were they from?",
        question_id="pytest_main_004",
        gold_answer="test",
        model=DEV_MODEL
    )
    assert result["num_hops"] <= MAX_HOPS, (
        f"Hop count {result['num_hops']} exceeded MAX_HOPS={MAX_HOPS}"
    )


# --- Test 5: Single-hop path works when classifier says NO -------------------
def test_main_system_single_hop_path():
    # Simple question likely classified as NO
    result = run_main_system(
        question="What is the capital of France?",
        question_id="pytest_main_005",
        gold_answer="Paris",
        model=DEV_MODEL
    )
    # If classified NO, sub_queries should be empty and stop = single_hop
    if result["hop_necessity_classification"] == "NO":
        assert result["sub_queries_generated"] == []
        assert result["stop_condition_triggered"] == "single_hop"
        assert result["num_hops"] == 1


# --- Test 6: Multi-hop path produces sub-queries when classifier says YES ----
def test_main_system_multi_hop_path():
    result = run_main_system(
        question="Who was the director of the film that won the Academy Award for Best Picture in 2020?",
        question_id="pytest_main_006",
        gold_answer="Bong Joon-ho",
        model=DEV_MODEL
    )
    # If classified YES, sub_queries should be non-empty
    if result["hop_necessity_classification"] == "YES":
        assert len(result["sub_queries_generated"]) >= 1
        assert result["stop_condition_triggered"] in ["sufficiency", "max_hops"]