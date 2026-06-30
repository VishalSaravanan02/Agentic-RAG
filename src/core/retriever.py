# =============================================================================
# retriever.py — Wraps ChromaDB for document retrieval
# Every system calls retrieve() — never implement retrieval elsewhere.
# =============================================================================

import chromadb
from src.core.config import CHROMA_DIR, TOP_K

# Initialise ChromaDB client once at module level
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = None

def get_collection():
    """Get or create the ChromaDB collection."""
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name="hotpotqa_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        query: The search query string
        k: Number of documents to retrieve (defaults to TOP_K from config)

    Returns:
        List of dicts, each containing:
            - text: the chunk text
            - metadata: dict with article_title, article_id, chunk_index
            - cosine_similarity: float similarity score (higher = more relevant)
    """
    from src.core.embedder import embed

    # Embed the query
    query_embedding = embed(query)

    # Query ChromaDB
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    documents = []
    for i in range(len(results["documents"][0])):
        # ChromaDB returns cosine distance (0=identical, 2=opposite)
        # Convert to similarity (1=identical, -1=opposite)
        distance = results["distances"][0][i]
        similarity = 1 - distance

        documents.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "cosine_similarity": round(similarity, 4)
        })

    return documents

def get_collection_size() -> int:
    """Returns total number of chunks in the collection."""
    return get_collection().count()