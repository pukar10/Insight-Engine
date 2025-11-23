"""
search.py

Provides a small helper function to search the Chroma index.

Usage example (from a Python shell):
    from backend.search import search
    results = search("What did I write about project X?", n_results=3)
"""

from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

# Must match ingest.py
DB_DIR = "db"
COLLECTION_NAME = "notes"


def get_collection():
    """
    Connect to the existing Chroma collection.

    This assumes you've already run ingest.py at least once so that
    'db/' and the 'notes' collection exist.
    """
    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def search(query: str, n_results: int = 5) -> List[Dict]:
    """
    Given a text query, return up to n_results most similar chunks.

    Each result is a dict with:
      - text:         the chunk text
      - source:       which file it came from
      - chunk_index:  which number chunk in that file
      - distance:     similarity distance (smaller = more similar)
    """
    collection = get_collection()

    # We pass a list of queries; here it's just a single query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    documents = results["documents"][0]   # list of chunk texts
    metadatas = results["metadatas"][0]   # list of metadata dicts
    distances = results.get("distances", [[None] * len(documents)])[0]

    output = []
    for text, meta, dist in zip(documents, metadatas, distances):
        output.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            }
        )

    return output


if __name__ == "__main__":
    # Quick manual test when running: python backend/search.py
    example_query = "example question"
    hits = search(example_query, n_results=3)

    for hit in hits:
        print("-" * 40)
        print("Source:", hit["source"])
        print("Chunk index:", hit["chunk_index"])
        print("Distance:", hit["distance"])
        print("Text snippet:", hit["text"][:200], "...")
