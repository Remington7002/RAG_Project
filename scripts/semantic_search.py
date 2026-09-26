import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "crypto_news"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


def main():
    print("=" * 50)
    print("SEMANTIC SEARCH")
    print("=" * 50)

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Model loaded.")

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents: {collection.count():,}")

    query = input("\nEnter your search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    print("\nGenerating query embedding...")

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
    )

    print("Searching ChromaDB...")

    start_time = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    search_time = time.perf_counter() - start_time

    print("\n" + "=" * 50)
    print("SEARCH RESULTS")
    print("=" * 50)

    print(f"Query: {query}")
    print(f"Search latency: {search_time * 1000:.2f} ms")
    print(f"Results: {len(results['documents'][0])}")

    for index, document in enumerate(
        results["documents"][0],
        start=1,
    ):
        metadata = results["metadatas"][0][index - 1]
        distance = results["distances"][0][index - 1]

        print("\n" + "-" * 50)
        print(f"Result #{index}")
        print("-" * 50)

        print(f"Distance: {distance:.4f}")
        print(f"Source: {metadata.get('source')}")
        print(f"Published: {metadata.get('published_on')}")
        print(f"URL: {metadata.get('url')}")

        print("\nText:")
        print(document)


if __name__ == "__main__":
    main()