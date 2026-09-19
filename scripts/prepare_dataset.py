import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/Document_5000.csv")
OUTPUT_FILE = Path("data/raw/crypto_news.jsonl")


COLUMNS = [
    "id",
    "published_on",
    "title",
    "body",
    "url",
    "tags",
    "categories",
    "source",
]


def main():
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows: {len(df):,}")

    missing_columns = [column for column in COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    documents_written = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for _, row in df[COLUMNS].iterrows():

            document = {
                "id": row["id"],
                "published_on": row["published_on"],
                "title": row["title"],
                "body": row["body"],
                "url": row["url"],
                "tags": row["tags"],
                "categories": row["categories"],
                "source": row["source"],
            }

            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

            documents_written += 1

    print(f"Documents written: {documents_written:,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()