# =============================================================================
# download_data.py — Downloads HotpotQA and creates evaluation samples
# Run once only. Never re-run after samples are created.
# =============================================================================

import json
import os
import random
from collections import defaultdict
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import (
    RAW_DIR, PROCESSED_DIR, DEV_DATA_PATH,
    EVAL_DATA_PATH, RANDOM_SEED, DEV_SAMPLE_SIZE, EVAL_SAMPLE_SIZE
)

def create_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_hotpotqa():
    print("Downloading HotpotQA...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
    validation = list(dataset["validation"])
    print(f"Total validation questions: {len(validation)}")

    # Show question type breakdown
    types = defaultdict(int)
    for q in validation:
        types[q["type"]] += 1
    print(f"Question types: {dict(types)}")

    # Save full validation set
    raw_path = os.path.join(RAW_DIR, "hotpotqa_validation.json")
    with open(raw_path, "w") as f:
        json.dump(validation, f)
    print(f"Saved full validation set to {raw_path}")

    return validation

def stratified_sample(questions, n, seed, exclude_ids=None):
    """Draw a stratified random sample by question type."""
    random.seed(seed)
    exclude_ids = exclude_ids or set()

    # Group by type
    by_type = defaultdict(list)
    for q in questions:
        if q["id"] not in exclude_ids:
            by_type[q["type"]].append(q)

    # Calculate proportions
    total_available = sum(len(v) for v in by_type.values())
    sampled = []
    for qtype, qs in by_type.items():
        proportion = len(qs) / total_available
        n_sample = round(n * proportion)
        sampled.extend(random.sample(qs, min(n_sample, len(qs))))

    # Trim or top up to exactly n
    random.shuffle(sampled)
    return sampled[:n]

def create_samples(validation):
    # Check if samples already exist
    if os.path.exists(DEV_DATA_PATH) and os.path.exists(EVAL_DATA_PATH):
        print("WARNING: Sample files already exist! Not regenerating.")
        print("Delete them manually if you really want to resample.")
        return

    print(f"\nCreating dev sample ({DEV_SAMPLE_SIZE} questions)...")
    dev_sample = stratified_sample(validation, DEV_SAMPLE_SIZE, RANDOM_SEED)
    dev_ids = {q["id"] for q in dev_sample}

    print(f"Creating eval sample ({EVAL_SAMPLE_SIZE} questions)...")
    eval_sample = stratified_sample(
        validation, EVAL_SAMPLE_SIZE, RANDOM_SEED + 1, exclude_ids=dev_ids
    )
    eval_ids = {q["id"] for q in eval_sample}

    # Verify no overlap
    overlap = dev_ids & eval_ids
    assert len(overlap) == 0, f"CRITICAL: {len(overlap)} questions appear in both samples!"
    print(f"Non-overlap verified: 0 questions shared between dev and eval sets")

    # Show type breakdowns
    dev_types = defaultdict(int)
    for q in dev_sample:
        dev_types[q["type"]] += 1
    eval_types = defaultdict(int)
    for q in eval_sample:
        eval_types[q["type"]] += 1

    print(f"Dev sample types:  {dict(dev_types)}")
    print(f"Eval sample types: {dict(eval_types)}")

    # Save
    with open(DEV_DATA_PATH, "w") as f:
        json.dump(dev_sample, f, indent=2)
    with open(EVAL_DATA_PATH, "w") as f:
        json.dump(eval_sample, f, indent=2)

    print(f"\nSaved dev sample  -> {DEV_DATA_PATH}")
    print(f"Saved eval sample -> {EVAL_DATA_PATH}")

if __name__ == "__main__":
    create_dirs()
    validation = download_hotpotqa()
    create_samples(validation)
    print("\nDone! Data download complete.")