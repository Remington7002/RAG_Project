import json
from pathlib import Path

import chromadb
import numpy as np


CHROMA_DIR = Path("data/chroma")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/embeddings.npz")

COLLECTION_NAME = "crypto_news"
CHROMA_BATCH_SIZE = 5000


def load_chunks(limit: int):
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunks.append(json.loads(line))

            if len(chunks) >= limit:
                break

    return chunks


def main():
    print("=" * 50)
    print("CHROMADB INGESTION")
    print("=" * 50)

    print("\nLoading embeddings...")

    data = np.load(EMBEDDINGS_FILE)
    embeddings = data["embeddings"]

    print(f"Embeddings shape: {embeddings.shape}")

    print("\nLoading chunk metadata...")

    chunks = load_chunks(len(embeddings))

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match "
            f"embedding count ({len(embeddings)})"
        )

    print(f"Chunks loaded: {len(chunks):,}")

    print("\nInitializing ChromaDB...")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Crypto news semantic search collection"
        },
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Existing documents: {collection.count():,}")

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))

        documents.append(chunk["text"])

        metadatas.append({
            "document_id": str(chunk["document_id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "published_on": str(chunk.get("published_on") or ""),
            "url": str(chunk.get("url") or ""),
            "source": str(chunk.get("source") or ""),
        })

    print("\nAdding documents to ChromaDB...")

    # ChromaDB has a maximum batch size.
    # Insert the 20,000 documents in smaller batches.
    for start in range(0, len(ids), CHROMA_BATCH_SIZE):

        end = min(
            start + CHROMA_BATCH_SIZE,
            len(ids)
        )

        print(
            f"Inserting documents "
            f"{start + 1:,} - {end:,}..."
        )

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    print("\n" + "=" * 50)
    print("CHROMADB INGESTION COMPLETE")
    print("=" * 50)

    print(f"Collection:          {COLLECTION_NAME}")
    print(f"Documents stored:    {collection.count():,}")
    print(f"Database location:   {CHROMA_DIR}")


if __name__ == "__main__":
    main()