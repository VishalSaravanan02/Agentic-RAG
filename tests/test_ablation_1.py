# =============================================================================
# test_ablation_1.py — Integration tests for Agentic RAG without decomposition
# Run with: python -m pytest tests/ -v
# =============================================================================

import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.systems.ablation_1 import run_ablation_1
from src.core.config import DEV_MODEL


# --- Test 1: ablation_1 produces complete output -----------------------------
def test_ablation_1_produces_required_fields():
    result = run_ablation_1(
        question="What is the capital of France?",
        question_id="pytest_ablation_001",
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

    # Inherited from the shared pipeline
    assert "queries_per_hop" in result, "Missing field: queries_per_hop"


# --- Test 2: schema values are correct ---------------------------------------
def test_ablation_1_schema_values():
    result = run_ablation_1(
        question="What is the capital of France?",
        question_id="pytest_ablation_002",
        gold_answer="Paris",
        model=DEV_MODEL
    )
    assert result["system_name"] == "ablation_1"
    assert result["hop_necessity_classification"] in ["YES", "NO"]
    assert isinstance(result["num_hops"], int)
    assert len(result["final_answer"]) > 0


# --- Test 3: THE ablation invariant — never decomposes -----------------------
# This is what makes ablation_1 an ablation: D2 is disabled, so it must NEVER
# produce sub-queries, on any question, however it is classified. If this ever
# fails, decomposition has leaked back in and the Main-vs-Ablation1 comparison
# (RQ3) no longer isolates decomposition.
def test_ablation_1_never_decomposes():
    # A question the classifier will very likely route as multi-hop (YES),
    # which in the Main System WOULD trigger decomposition.
    result = run_ablation_1(
        question="Who was the director of the film that won the Academy Award for Best Picture in 2020?",
        question_id="pytest_ablation_003",
        gold_answer="Bong Joon-ho",
        model=DEV_MODEL
    )
    assert result["sub_queries_generated"] == [], \
        "ablation_1 must never decompose — sub_queries_generated must always be []"


# --- Test 4: reactive-from-hop-1 signature + queries aligned -----------------
# With no decomposition plan, hop 1 must retrieve on the ORIGINAL question,
# and queries_per_hop must stay aligned 1:1 with docs_retrieved_per_hop.
def test_ablation_1_hop1_is_original_question_and_aligned():
    question = "Who was the director of the film that won the Academy Award for Best Picture in 2020?"
    result = run_ablation_1(
        question=question,
        question_id="pytest_ablation_004",
        gold_answer="Bong Joon-ho",
        model=DEV_MODEL
    )
    assert isinstance(result["queries_per_hop"], list)
    assert len(result["queries_per_hop"]) == len(result["docs_retrieved_per_hop"]), \
        "queries_per_hop and docs_retrieved_per_hop must have one entry per hop"

    # Hop 1's query is always the original question (no plan to draw from)
    assert result["queries_per_hop"][0] == question, \
        "ablation_1 hop 1 must retrieve on the original question"