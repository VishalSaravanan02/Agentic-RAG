# =============================================================================
# embedder.py — Wraps sentence-transformers embedding model
# Loads the model ONCE at module level. Never load inside a loop.
# =============================================================================

from sentence_transformers import SentenceTransformer
from src.core.config import EMBEDDING_MODEL

print(f"Loading embedding model: {EMBEDDING_MODEL}...")
_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded successfully!")

def embed(text: str) -> list[float]:
    """
    Embed a single text string into a vector.
    Returns a list of floats (the embedding vector).
    """
    return _model.encode(text, convert_to_tensor=False).tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in one batch (more efficient than calling embed() repeatedly).
    Used by build_index.py when embedding all chunks.
    """
    return _model.encode(texts, convert_to_tensor=False).tolist()