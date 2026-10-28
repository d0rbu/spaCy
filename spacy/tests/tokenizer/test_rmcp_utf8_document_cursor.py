"""Document byte cursors preserve the original Unicode tokenizer contract."""
import re

import pytest
from hypothesis import given, settings, strategies as st
from spacy.attrs import IDX, IS_PUNCT, IS_SPACE, LANG, NORM, ORTH, SENT_START, SPACY
from spacy.symbols import ORTH as SPECIAL_ORTH
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab


class ReferenceTokenizer(Tokenizer):
    """Subclasses retain the original codepoint/slice implementation."""


class StringSubclass(str):
    pass


WHITESPACE = tuple(chr(i) for i in (
    9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760,
    8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201,
    8202, 8232, 8233, 8239, 8287, 12288,
))
WIDTH_BOUNDARIES = tuple(chr(i) for i in (
    0, 1, 127, 128, 255, 256, 2047, 2048, 55295, 57344,
    65535, 65536, 1114111,
))
ATTRS = [ORTH, IS_PUNCT, IS_SPACE, SPACY, IDX, SENT_START, NORM, LANG]


def outcome(tokenizer, text):
    try:
        doc = tokenizer(text)
    except UnicodeEncodeError as error:
        result = (type(error), error.encoding, error.object, error.start,
                  error.end, error.reason)
    except TypeError as error:
        result = (type(error), str(error))
    else:
        result = (doc.text, [(t.text, t.idx, t.whitespace_) for t in doc],
                  doc.to_array(ATTRS).tolist())
    return result, len(tokenizer.vocab), len(tokenizer.vocab.strings)


def compare_sequence(texts, cache_size=10000):
    actual = Tokenizer(Vocab(), max_cache_size=cache_size)
    expected = ReferenceTokenizer(Vocab(), max_cache_size=cache_size)
    for text in texts:
        assert outcome(actual, text) == outcome(expected, text)


@settings(max_examples=300, deadline=None, database=None, derandomize=True)
@given(st.text(alphabet=st.characters(blacklist_categories=()), max_size=250))
def test_unicode_documents_and_warm_cache(text):
    compare_sequence([text, text, "prefix " + text, text + " suffix"])


@pytest.mark.parametrize("cache_size", [0, 1, 3, 10000])
def test_utf8_widths_whitespace_nul_and_cache_eviction(cache_size):
    compare_sequence(
        [left + space + "a " + left + space + right
         for left, right in zip(WIDTH_BOUNDARIES, reversed(WIDTH_BOUNDARIES))
         for space in WHITESPACE] * 2,
        cache_size,
    )


@pytest.mark.parametrize("text", [
    "\ud800", "\udfff", "alpha \ud800 beta", "é alpha \udfff",
    "\ud800\udfff", "é\ud800\tnewword", "newword \ud800",
])
def test_surrogate_error_and_partial_vocabulary_side_effects(text):
    compare_sequence(["existing", text, "newword existing", text])


def test_string_subclass_fallback():
    compare_sequence([StringSubclass("é \x00 中\u2003🙂 "),
                      StringSubclass("newword \ud800"), "after"])


def make_callbacks(cls):
    events = []

    def observed(name, callback):
        def call(text):
            events.append((name, text))
            return callback(text)
        return call

    token = cls(
        Vocab(),
        prefix_search=observed("prefix", re.compile(r"^[([]").search),
        suffix_search=observed("suffix", re.compile(r"[).!?]$").search),
        infix_finditer=observed("infix", re.compile(r"[-/]").finditer),
        token_match=observed("token", re.compile(r"^a-b$").match),
        max_cache_size=3,
    )
    return token, events


def test_callbacks_special_cases_rule_mutation_and_serialization():
    actual, actual_events = make_callbacks(Tokenizer)
    expected, expected_events = make_callbacks(ReferenceTokenizer)
    for turn in range(3):
        for text in ("(é/a-b) 中-🙂!", "a-b a-b", "alpha\talpha", "é a/b "):
            assert outcome(actual, text) == outcome(expected, text)
            assert actual_events == expected_events
        for token in (actual, expected):
            token.add_special_case("é-中", [{SPECIAL_ORTH: "é"},
                                             {SPECIAL_ORTH: "-中"}])
        assert outcome(actual, "é-中 é-中") == outcome(expected, "é-中 é-中")
        assert actual_events == expected_events
        for token in (actual, expected):
            token.rules = {}
    # Arbitrary Python callbacks are not serializable in the original API.
    for token in (actual, expected):
        with pytest.raises(AttributeError, match="__self__"):
            token.to_bytes()
        token.prefix_search = None
        token.suffix_search = None
        token.infix_finditer = None
        token.token_match = None
    assert actual.to_bytes() == expected.to_bytes()

