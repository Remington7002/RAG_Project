import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect, LangDetectException


# Makes language detection deterministic
DetectorFactory.seed = 0


INPUT_FILE = Path("data/raw/crypto_news.jsonl")
OUTPUT_FILE = Path("data/processed/clean_corpus.jsonl")


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    """Normalize spaces, tabs, and line breaks."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Apply all text-cleaning operations."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)

    return text


def is_english(text: str) -> bool:
    """Return True if the text is detected as English."""
    if not text or len(text.strip()) < 20:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def process_document(document: dict) -> dict | None:
    """Clean and validate one document."""

    title = str(document.get("title") or "")
    body = str(document.get("body") or "")

    combined_text = f"{title}\n\n{body}"

    cleaned_text = clean_text(combined_text)

    if not cleaned_text:
        return None

    if not is_english(cleaned_text):
        return None

    return {
        "id": document.get("id"),
        "text": cleaned_text,
        "published_on": document.get("published_on"),
        "url": document.get("url"),
        "tags": document.get("tags"),
        "categories": document.get("categories"),
        "source": document.get("source"),
    }


def process_dataset():
    """Process the complete JSONL dataset."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_documents = 0
    processed_documents = 0
    filtered_documents = 0

    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    with (
        INPUT_FILE.open("r", encoding="utf-8") as input_file,
        OUTPUT_FILE.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:

            if not line.strip():
                continue

            total_documents += 1

            document = json.loads(line)

            processed = process_document(document)

            if processed is None:
                filtered_documents += 1
                continue

            output_file.write(
                json.dumps(
                    processed,
                    ensure_ascii=False,
                )
                + "\n"
            )

            processed_documents += 1

            if processed_documents % 10_000 == 0:
                print(
                    f"Processed: {processed_documents:,} "
                    f"| Read: {total_documents:,}"
                )

    print()
    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Documents read:      {total_documents:,}")
    print(f"Documents kept:      {processed_documents:,}")
    print(f"Documents filtered:  {filtered_documents:,}")


if __name__ == "__main__":
    process_dataset()