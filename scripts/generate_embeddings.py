import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("data/processed/embeddings.npz")
METADATA_FILE = Path("data/processed/embedding_metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHUNKS = 20_000
BATCH_SIZE = 32


def load_chunks():
    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)

            chunks.append(document)

            if len(chunks) >= MAX_CHUNKS:
                break

    return chunks


def main():
    print("=" * 50)
    print("EMBEDDING GENERATION")
    print("=" * 50)

    print(f"Model:       {MODEL_NAME}")
    print(f"Max chunks:  {MAX_CHUNKS:,}")
    print(f"Batch size:  {BATCH_SIZE}")
    print("Device:      CPU")

    print("\nLoading chunks...")

    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks):,}")

    texts = [chunk["text"] for chunk in chunks]

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Model loaded.")

    print("\nGenerating embeddings...")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    elapsed_time = time.perf_counter() - start_time

    print("\nSaving embeddings...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
    )

    metadata = {
        "model": MODEL_NAME,
        "device": "cpu",
        "chunk_count": len(chunks),
        "embedding_dimensions": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE,
        "total_time_seconds": round(elapsed_time, 2),
        "chunks_per_second": round(
            len(chunks) / elapsed_time,
            2,
        ),
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 50)

    print(f"Chunks embedded:      {len(chunks):,}")
    print(f"Dimensions:           {embeddings.shape[1]}")
    print(f"Total time:           {elapsed_time:.2f} seconds")
    print(
        f"Chunks per second:    "
        f"{len(chunks) / elapsed_time:.2f}"
    )
    print(f"Embeddings:           {OUTPUT_FILE}")
    print(f"Performance metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()