import pytest

from src.chunking import recursive_chunk_text


def test_empty_text():
    assert recursive_chunk_text("") == []


def test_short_text():
    text = "Hello world"
    chunks = recursive_chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_is_split():
    text = "A" * 250

    chunks = recursive_chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1


def test_chunk_size():
    text = "A" * 250

    chunks = recursive_chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    for chunk in chunks:
        assert len(chunk) <= 100


def test_overlap():
    text = "A" * 250

    chunks = recursive_chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks[0][-20:] == chunks[1][:20]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        recursive_chunk_text(
            "Hello world",
            chunk_size=100,
            chunk_overlap=100,
        )