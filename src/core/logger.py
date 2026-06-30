# =============================================================================
# logger.py — Structured per-question logging
# Writes results as JSON Lines (.jsonl) — one JSON object per line.
# Every system uses this. Schema is identical across all systems.
# =============================================================================

import json
import os
from datetime import datetime
from src.core.config import RESULTS_DIR

def _get_filepath(system_name: str, split: str) -> str:
    """
    Returns the log file path for a given system and split.
    e.g. results/baseline_a_dev.jsonl
         results/main_system_eval.jsonl
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{system_name}_{split}.jsonl")

def log_result(data: dict, system_name: str, split: str) -> None:
    """
    Append one question result to the appropriate .jsonl log file.
    Creates the file if it doesn't exist.

    Args:
        data:        Dict containing all required log fields (see below)
        system_name: e.g. 'baseline_a' or 'main_system'
        split:       e.g. 'dev' or 'eval'

    Required fields in data:
        question_id, question, system_name,
        hop_necessity_classification, num_hops,
        sub_queries_generated, docs_retrieved_per_hop,
        stop_condition_triggered, input_tokens, output_tokens,
        latency_per_hop_ms, total_latency_ms,
        final_answer, gold_answer
    """
    # Add timestamp
    data["logged_at"] = datetime.utcnow().isoformat()

    # Validate required fields
    required_fields = [
        "question_id", "question", "system_name",
        "hop_necessity_classification", "num_hops",
        "sub_queries_generated", "docs_retrieved_per_hop",
        "stop_condition_triggered", "input_tokens", "output_tokens",
        "latency_per_hop_ms", "total_latency_ms",
        "final_answer", "gold_answer"
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Missing required log fields: {missing}")

    filepath = _get_filepath(system_name, split)
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")

def get_completed_ids(system_name: str, split: str) -> set:
    """
    Returns a set of question_ids already logged.
    Used by run_evaluation.py for checkpointing —
    skip questions that are already done.
    """
    filepath = _get_filepath(system_name, split)
    if not os.path.exists(filepath):
        return set()

    completed = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    completed.add(entry["question_id"])
                except json.JSONDecodeError:
                    continue
    return completed

def load_results(system_name: str, split: str) -> list[dict]:
    """
    Load all logged results for a system/split as a list of dicts.
    Used by metrics.py and bootstrap.py.
    """
    filepath = _get_filepath(system_name, split)
    if not os.path.exists(filepath):
        return []

    results = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results