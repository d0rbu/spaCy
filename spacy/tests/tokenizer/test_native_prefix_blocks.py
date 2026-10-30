import gc
import re
import weakref

import numpy
import pytest
from hypothesis import given, settings, strategies as st

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer


ATTRS = ["ORTH", "IS_PUNCT", "IS_SPACE", "SPACY", "IDX", "SENT_START", "NORM", "LANG"]


def prefix_length(search, text):
    match = search(text)
    return match.end() - match.start() if match is not None else 0


def test_english_prefix_all_unicode_scalars():
    tokenizer = English().tokenizer
    search = tokenizer.prefix_search
    for value in range(0x110000):
        text = chr(value)
        assert tokenizer.find_prefix(text) == prefix_length(search, text), value


@settings(max_examples=400)
@given(st.text(max_size=128))
def test_english_prefix_generated_strings(text):
    tokenizer = English().tokenizer
    assert tokenizer.find_prefix(text) == prefix_length(tokenizer.prefix_search, text)


@pytest.mark.parametrize("head", [".", "..", "...", "....", "+", "A$", "C$", "US$", "…", "……"])
@pytest.mark.parametrize("tail", ["", "0", "9", "a", "９", "\u0301", "\U0001f600"])
def test_english_prefix_order_and_lookahead(head, tail):
    tokenizer = English().tokenizer
    text = head + tail
    assert tokenizer.find_prefix(text) == prefix_length(tokenizer.prefix_search, text)


def test_prefix_replacement_and_callback_fallback():
    tokenizer = English().tokenizer
    original = tokenizer.prefix_search
    changed = re.compile("^[a-z]+", re.IGNORECASE).search
    tokenizer.prefix_search = changed
    assert tokenizer.find_prefix("ABC..") == 3
    assert tokenizer.find_prefix("..ABC") == 0
    calls = []

    def wrapped(text):
        calls.append(text)
        return original(text)

    tokenizer.prefix_search = wrapped
    # The setter rebuilds special cases through the supplied callback.
    calls.clear()
    assert tokenizer.find_prefix("US$10") == 3
    assert calls == ["US$10"]
    tokenizer.prefix_search = original
    assert tokenizer.find_prefix("US$10") == 3


def test_prefix_cache_does_not_retain_pattern():
    tokenizer = English().tokenizer
    pattern_text = tokenizer.prefix_search.__self__.pattern
    re.purge()
    pattern = re.compile(pattern_text)
    reference = weakref.ref(pattern)
    tokenizer.prefix_search = pattern.search
    tokenizer.prefix_search = None
    del pattern
    re.purge()
    gc.collect()
    assert reference() is None


class ReferenceTokenizer(Tokenizer):
    """A subclass keeps the callback-based tokenizer path."""


@pytest.mark.parametrize("width", [1, 2, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize(
    "space",
    [
        " ",
        "\t",
        "\n",
        "\x1c",
        "\x85",
        "\xa0",
        "\u1680",
        "\u2000",
        "\u2028",
        "\u202f",
        "\u205f",
        "\u3000",
    ],
)
def test_block_whitespace_and_special_rules_match_subclass(width, space):
    tokenizer = English().tokenizer
    reference = ReferenceTokenizer(
        tokenizer.vocab,
        rules=dict(tokenizer.rules),
        prefix_search=tokenizer.prefix_search,
        suffix_search=tokenizer.suffix_search,
        infix_finditer=tokenizer.infix_finditer,
        token_match=tokenizer.token_match,
        url_match=tokenizer.url_match,
        faster_heuristics=tokenizer.faster_heuristics,
    )
    text = "x" * width + space + "can't US$10 ...\u0301" + space * 2 + "\U0001f600"
    for _ in range(2):
        actual, expected = tokenizer(text), reference(text)
        assert actual.text == expected.text == text
        assert numpy.array_equal(actual.to_array(ATTRS), expected.to_array(ATTRS))
