import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "crypto_news"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

QUERIES = [
    "How does Ethereum staking work?",
    "What caused Bitcoin price to increase?",
    "How does blockchain technology work?",
    "What are cryptocurrency regulations?",
    "What is DeFi?",
]


def main():
    print("=" * 50)
    print("RETRIEVAL PERFORMANCE BENCHMARK")
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

    results_log = []

    print("\nRunning queries...")

    for query in QUERIES:

        # Measure query embedding time
        embedding_start = time.perf_counter()

        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
        )

        embedding_time = (
            time.perf_counter() - embedding_start
        )

        # Measure ChromaDB search time
        search_start = time.perf_counter()

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        search_time = (
            time.perf_counter() - search_start
        )

        total_time = embedding_time + search_time

        result_count = len(
            results["documents"][0]
        )

        results_log.append({
            "query": query,
            "embedding_time_ms": round(
                embedding_time * 1000, 2
            ),
            "search_time_ms": round(
                search_time * 1000, 2
            ),
            "total_time_ms": round(
                total_time * 1000, 2
            ),
            "results": result_count,
        })

        print("\n" + "-" * 50)
        print(f"Query: {query}")
        print(
            f"Embedding time: {embedding_time * 1000:.2f} ms"
        )
        print(
            f"Search time:    {search_time * 1000:.2f} ms"
        )
        print(
            f"Total time:     {total_time * 1000:.2f} ms"
        )
        print(f"Results:        {result_count}")

    output_file = Path(
        "data/processed/retrieval_benchmark.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results_log,
            file,
            indent=4,
        )

    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Queries tested: {len(QUERIES)}")
    print(f"Results saved:  {output_file}")


if __name__ == "__main__":
    main()