"""Span hashing preserves warm and uncached tokenization."""
import pytest

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer


@pytest.mark.parametrize("cache_size", [0, 1, 10000])
@pytest.mark.parametrize(
    "text",
    [
        "",
        "alpha beta alpha",
        "alpha\x00beta alpha beta alpha\x00beta",
        "NOUN VERB ORTH ROOT alpha alpha",
        " leading\twords\nwith\r\nspaces \v\f trailing ",
        "ASCII naïve 漢字 🙂 ASCII café",
        "a" * 4096 + " " + "b" * 4096,
        "\x01\x1f\x7f a \x00b \x00c",
        "word's word-thing 123.456 http://example.com",
    ],
)
def test_ascii_span_hash_matches_uncached(text, cache_size):
    cached = English().tokenizer
    cached.max_cache_size = cache_size
    reference = Tokenizer(
        cached.vocab,
        rules=cached.rules,
        prefix_search=cached.prefix_search,
        suffix_search=cached.suffix_search,
        infix_finditer=cached.infix_finditer,
        token_match=cached.token_match,
        url_match=cached.url_match,
        faster_heuristics=cached.faster_heuristics,
        max_cache_size=0,
    )
    for value in [text, "intervening new words", text, text]:
        assert cached(value).to_bytes() == reference(value).to_bytes()


def test_ascii_span_hash_keeps_string_subclass_rejection():
    class Text(str):
        pass

    tokenizer = English().tokenizer
    text = "ASCII words with\x00control and naïve"
    with pytest.raises(TypeError, match="incorrect type"):
        tokenizer(Text(text))


def test_ascii_span_hash_keeps_surrogate_failure():
    tokenizer = English().tokenizer
    with pytest.raises(UnicodeEncodeError):
        tokenizer("bad\ud800text")
