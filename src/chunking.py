from pathlib import Path
import json

INPUT_FILE = Path("data/processed/clean_corpus.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def recursive_chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_dataset():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_documents = 0
    total_chunks = 0

    with (
        INPUT_FILE.open("r", encoding="utf-8") as input_file,
        OUTPUT_FILE.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document.get("text", "")

            chunks = recursive_chunk_text(text)

            for index, chunk in enumerate(chunks):
                chunk_document = {
                    "chunk_id": f"{document['id']}_{index}",
                    "document_id": document["id"],
                    "chunk_index": index,
                    "text": chunk,
                    "published_on": document.get("published_on"),
                    "url": document.get("url"),
                    "source": document.get("source"),
                }

                output_file.write(
                    json.dumps(chunk_document, ensure_ascii=False) + "\n"
                )

                total_chunks += 1

            total_documents += 1

            if total_documents % 10_000 == 0:
                print(
                    f"Documents processed: {total_documents:,} | "
                    f"Chunks generated: {total_chunks:,}"
                )

    print()
    print("=" * 50)
    print("CHUNKING COMPLETE")
    print("=" * 50)
    print(f"Documents processed: {total_documents:,}")
    print(f"Chunks generated:    {total_chunks:,}")
    print(f"Output:              {OUTPUT_FILE}")


if __name__ == "__main__":
    chunk_dataset()