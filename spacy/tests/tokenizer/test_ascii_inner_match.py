"""Exact ASCII token-match specialization and setter publication regressions."""
import re

import numpy
import pytest

from spacy.lang.en import English


ATTRS = ["ORTH", "IS_PUNCT", "IS_SPACE", "SPACY", "IDX", "SENT_START", "NORM", "LANG"]
TEXTS = [
    "", "alpha", "UPPER lower", "A1", "123", "hello!", "((letters)).",
    "can't won't U.S. example.com", "abc-def", "alpha—beta", "é K İ ı ſ",
    "a\x00b", "a\tb\nc\rd", "  words  ", "Hello... World?!", "。漢字",
    "😀word😀", "A" * 1000, "abc123def", "...alpha...", "a\u2003b",
]


@pytest.fixture(scope="module", params=[
    (r"[A-Za-z]+\Z", 0),
    (r"[A-Za-z]+\Z", re.ASCII),
    (r"[A-Za-z]+\Z", re.IGNORECASE),
    (r"[A-Za-z]+$", 0),
    (r"\d+\Z", 0),
])
def tokenizers(request):
    pattern, flags = request.param
    compiled = re.compile(pattern, flags)
    reference, candidate = English().tokenizer, English().tokenizer
    # The wrapper deliberately takes the unchanged generic callback path.
    reference.token_match = lambda text: compiled.match(text)
    candidate.token_match = compiled.match
    return reference, candidate


@pytest.mark.parametrize("text", TEXTS)
def test_exact_pattern_and_generic_callback_have_equal_documents(tokenizers, text):
    reference, candidate = tokenizers
    for _ in range(2):
        left, right = reference(text), candidate(text)
        assert [(t.text, t.whitespace_) for t in left] == [
            (t.text, t.whitespace_) for t in right
        ]
        numpy.testing.assert_array_equal(left.to_array(ATTRS), right.to_array(ATTRS))


def test_exact_pattern_roundtrip_and_setter_reclassification():
    candidate = English().tokenizer
    candidate.token_match = re.compile(r"[A-Za-z]+\Z").match
    restored = English().tokenizer.from_bytes(candidate.to_bytes())
    for text in TEXTS:
        numpy.testing.assert_array_equal(
            candidate(text).to_array(ATTRS), restored(text).to_array(ATTRS)
        )
    for match in (None, re.compile(r"\d+\Z").match,
                  re.compile(r"[A-Za-z]+\Z").match):
        candidate.token_match = match
        reference = English().tokenizer
        reference.token_match = None if match is None else lambda text: match(text)
        for text in TEXTS:
            numpy.testing.assert_array_equal(
                reference(text).to_array(ATTRS), candidate(text).to_array(ATTRS)
            )


def test_reentrant_old_callback_destruction_keeps_the_actual_current_matcher():
    candidate = English().tokenizer
    digits = re.compile(r"\d+\Z")

    class Reenter:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, text):
            return None

        def __del__(self):
            self.tokenizer.token_match = digits.match

    candidate.token_match = Reenter(candidate)
    candidate.token_match = re.compile(r"[A-Za-z]+\Z").match
    assert candidate.token_match.__self__ is digits
    reference = English().tokenizer
    reference.token_match = lambda text: digits.match(text)
    for text in TEXTS:
        numpy.testing.assert_array_equal(
            reference(text).to_array(ATTRS), candidate(text).to_array(ATTRS)
        )
