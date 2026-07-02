# =============================================================================
# config.py — Single source of truth for ALL project settings
# Every other file imports from here. Never hardcode values elsewhere.
# =============================================================================

import os

# --- Retrieval settings ------------------------------------------------------
TOP_K           = 5      # documents retrieved per hop
CHUNK_SIZE      = 100    # words per chunk
CHUNK_OVERLAP   = 20     # word overlap between chunks
MAX_HOPS        = 4      # hard stop on retrieval loop
MAX_CONTEXT_TOKENS = 3000  # context window budget

# --- Sampling ----------------------------------------------------------------
RANDOM_SEED      = 42    # fixed forever — never change after sampling
DEV_SAMPLE_SIZE  = 200
EVAL_SAMPLE_SIZE = 1000

# --- Models ------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEV_MODEL       = "qwen/qwen3.6-27b"   # Groq — development only (free)
EVAL_MODEL      = "gpt-4o-mini"      # OpenAI — final evaluation
JUDGE_MODEL     = "gpt-4o"           # OpenAI — LLM-as-judge only

# --- Paths -------------------------------------------------------------------
DATA_DIR      = "data/"
RAW_DIR       = "data/raw/"
PROCESSED_DIR = "data/processed/"
CHROMA_DIR    = "data/chroma_db/"
RESULTS_DIR   = "results/"

# --- Derived paths -----------------------------------------------------------
DEV_DATA_PATH  = os.path.join(PROCESSED_DIR, "dev_200.json")
EVAL_DATA_PATH = os.path.join(PROCESSED_DIR, "eval_1000.json")
CHUNKS_PATH    = os.path.join(PROCESSED_DIR, "chunks.json")