from src.preprocessing import (
    clean_text,
    clean_whitespace,
    is_english,
    normalize_unicode,
    strip_html,
)


def test_strip_html():
    text = "<p>Hello <b>world</b></p>"
    result = strip_html(text)

    assert result == "Hello world"


def test_normalize_unicode():
    text = "Café"
    result = normalize_unicode(text)

    assert result == "Café"


def test_clean_whitespace():
    text = "Hello    world\n\nThis\tis a test."
    result = clean_whitespace(text)

    assert result == "Hello world This is a test."


def test_clean_text():
    text = "<p>Hello</p>    world\n"
    result = clean_text(text)

    assert result == "Hello world"


def test_english_language_detection():
    text = (
        "Bitcoin is a digital asset that uses blockchain "
        "technology to record transactions."
    )

    assert is_english(text) is True


def test_non_english_language_detection():
    text = (
        "Este es un texto escrito en español "
        "sobre criptomonedas y tecnología."
    )

    assert is_english(text) is False


def test_short_text_is_rejected():
    text = "Hello"

    assert is_english(text) is False