import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/chunks.jsonl")

MODEL_NAME = "all-MiniLM-L6-v2"
SAMPLE_SIZE = 1_000
BATCH_SIZE = 32


def load_sample():
    texts = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document.get("text", "").strip()

            if text:
                texts.append(text)

            if len(texts) >= SAMPLE_SIZE:
                break

    return texts


def main():
    print("=" * 50)
    print("EMBEDDING BENCHMARK")
    print("=" * 50)

    print(f"Model: {MODEL_NAME}")
    print(f"Sample size: {SAMPLE_SIZE:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Device: CPU")

    print("\nLoading model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Model loaded.")

    texts = load_sample()

    print(f"Texts loaded: {len(texts):,}")

    print("\nGenerating embeddings...")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    elapsed_time = time.perf_counter() - start_time

    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)

    print(f"Texts embedded:       {len(texts):,}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Total time:           {elapsed_time:.2f} seconds")
    print(f"Time per document:    {elapsed_time / len(texts):.4f} seconds")
    print(
        f"Documents per second: "
        f"{len(texts) / elapsed_time:.2f}"
    )


if __name__ == "__main__":
    main()