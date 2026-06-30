# =============================================================================
# build_index.py — Chunks the corpus and builds the ChromaDB index
# Run once only. The index is shared by ALL systems.
# =============================================================================

import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import RAW_DIR, PROCESSED_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.core.embedder import embed_batch
from src.core.retriever import get_collection

def load_raw_questions():
    """Load the full HotpotQA validation set (contains all context paragraphs)."""
    raw_path = os.path.join(RAW_DIR, "hotpotqa_validation.json")
    print(f"Loading raw questions from {raw_path}...")
    with open(raw_path, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")
    return questions

def extract_unique_articles(questions):
    """
    Extract all unique Wikipedia articles (title + sentences) from the
    'context' field across all questions. Deduplicates by title.
    """
    articles = {}  # title -> list of sentences
    for q in questions:
        titles = q["context"]["title"]
        sentences_list = q["context"]["sentences"]
        for title, sentences in zip(titles, sentences_list):
            if title not in articles:
                articles[title] = sentences
    print(f"Extracted {len(articles)} unique articles")
    return articles

def chunk_article(title: str, sentences: list[str], article_id: int) -> list[dict]:
    """
    Chunk a single article's sentences into ~CHUNK_SIZE-word pieces
    with CHUNK_OVERLAP-word overlap. Returns list of chunk dicts.
    """
    # Join all sentences into one text, tracking word positions
    full_text = " ".join(s.strip() for s in sentences)
    words = full_text.split()

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "article_title": title,
            "article_id": article_id,
            "chunk_index": chunk_index
        })

        chunk_index += 1
        if end == len(words):
            break
        start += (CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks

def build_all_chunks(articles: dict) -> list[dict]:
    """Chunk every article and return a flat list of all chunks."""
    all_chunks = []
    for article_id, (title, sentences) in enumerate(tqdm(articles.items(), desc="Chunking articles")):
        chunks = chunk_article(title, sentences, article_id)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} total chunks")
    return all_chunks

def save_chunks(chunks: list[dict]):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    print(f"Saved chunks to {CHUNKS_PATH}")

def build_chroma_index(chunks: list[dict]):
    """Embed all chunks in batches and insert into ChromaDB."""
    collection = get_collection()

    existing_count = collection.count()
    if existing_count > 0:
        print(f"WARNING: Index already contains {existing_count} chunks. Skipping build.")
        print("Delete data/chroma_db/ manually if you really want to rebuild.")
        return

    batch_size = 256
    print(f"Embedding and inserting {len(chunks)} chunks in batches of {batch_size}...")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "article_title": c["article_title"],
                "article_id": c["article_id"],
                "chunk_index": c["chunk_index"]
            }
            for c in batch
        ]

        embeddings = embed_batch(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"Indexing complete! Total chunks in collection: {collection.count()}")

def sanity_check():
    """Run a few test queries and print top results for manual inspection."""
    from src.core.retriever import retrieve

    test_queries = [
        "What year was a famous bridge built?",
        "Who directed this award winning film?",
        "Which country is this historical figure from?",
    ]

    print("\n--- SANITY CHECK: Sample retrievals ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, k=3)
        for r in results:
            print(f"  [{r['cosine_similarity']:.3f}] {r['metadata']['article_title']}: {r['text'][:100]}...")

if __name__ == "__main__":
    questions = load_raw_questions()
    articles = extract_unique_articles(questions)
    chunks = build_all_chunks(articles)
    save_chunks(chunks)
    build_chroma_index(chunks)
    sanity_check()
    print("\nDone! Index build complete.")