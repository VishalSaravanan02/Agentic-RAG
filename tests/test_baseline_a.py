# =============================================================================
# test_baseline_a.py — Unit tests for core infrastructure and Baseline A
# Run with: pytest tests/
# =============================================================================

import pytest
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.retriever import retrieve
from src.core.llm_client import call_llm
from src.core.logger import log_result, get_completed_ids, load_results
from src.systems.baseline_a import run_baseline_a
from src.core.config import TOP_K, DEV_MODEL


# --- Test 1: retriever.py returns exactly k documents -----------------------
def test_retriever_returns_correct_k():
    results = retrieve("Olympic games history", k=5)
    assert len(results) == 5, f"Expected 5 documents, got {len(results)}"

def test_retriever_returns_required_fields():
    results = retrieve("Olympic games history", k=3)
    for doc in results:
        assert "text" in doc
        assert "metadata" in doc
        assert "cosine_similarity" in doc
        assert "article_title" in doc["metadata"]
        assert "chunk_index" in doc["metadata"]


# --- Test 2: llm_client.py returns a non-empty string ------------------------
def test_llm_client_returns_text():
    result = call_llm("Say 'test successful' and nothing else.", model=DEV_MODEL, max_tokens=100)
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0

def test_llm_client_returns_token_counts():
    result = call_llm("Say hello.", model=DEV_MODEL, max_tokens=100)
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert len(result["text"]) > 0


# --- Test 3: logger.py writes valid JSON Lines -------------------------------
def test_logger_writes_and_reads_correctly():
    test_entry = {
        "question_id": "pytest_test_001",
        "question": "Test question?",
        "system_name": "baseline_a",
        "hop_necessity_classification": "N/A",
        "num_hops": 1,
        "sub_queries_generated": [],
        "docs_retrieved_per_hop": [["doc1"]],
        "stop_condition_triggered": "N/A",
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_per_hop_ms": [100.0],
        "total_latency_ms": 100.0,
        "final_answer": "Test answer",
        "gold_answer": "Test answer"
    }
    log_result(test_entry, system_name="baseline_a", split="pytest")

    completed = get_completed_ids("baseline_a", "pytest")
    assert "pytest_test_001" in completed

    results = load_results("baseline_a", "pytest")
    assert len(results) >= 1
    assert any(r["question_id"] == "pytest_test_001" for r in results)

    # Cleanup
    filepath = "results/baseline_a_pytest.jsonl"
    if os.path.exists(filepath):
        os.remove(filepath)

def test_logger_raises_on_missing_fields():
    incomplete_entry = {"question_id": "incomplete"}
    with pytest.raises(ValueError):
        log_result(incomplete_entry, system_name="baseline_a", split="pytest")


# --- Test 4: baseline_a.py produces complete output --------------------------
def test_baseline_a_produces_required_fields():
    result = run_baseline_a(
        question="What is the capital of France?",
        question_id="pytest_baseline_test",
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

def test_baseline_a_schema_values():
    result = run_baseline_a(
        question="What is the capital of France?",
        question_id="pytest_baseline_test2",
        gold_answer="Paris",
        model=DEV_MODEL
    )
    assert result["system_name"] == "baseline_a"
    assert result["hop_necessity_classification"] == "N/A"
    assert result["num_hops"] == 1
    assert result["sub_queries_generated"] == []
    assert len(result["final_answer"]) > 0