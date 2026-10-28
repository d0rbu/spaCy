"""Exact default English suffix lengths and mutable callback contracts."""
import gc
import re
import weakref

import pytest
from hypothesis import given, settings, strategies as st
from spacy.lang.en import English
from spacy.tokenizer import RMCPSuffixSearch, Tokenizer
from spacy.util import compile_suffix_regex
from spacy.vocab import Vocab

SINGLES = ",:;!?()[]{}<>_#*&"
SINGLE_MATCH = re.compile("[" + re.escape(SINGLES) + "]$").match
PATTERN = compile_suffix_regex(English.Defaults.suffixes).pattern


def reference_search(text, callback):
    if text and text[-1] in SINGLES:
        return SINGLE_MATCH(text, len(text) - 1)
    return callback(text)


def signature(match):
    if match is None:
        return None
    return (match.span(), match.group(), match.groups(), match.groupdict(),
            match.pos, match.endpos, match.string, match.re.pattern, match.re.flags)


@settings(max_examples=250, deadline=None, database=None, derandomize=True)
@given(st.text(alphabet=st.characters(max_codepoint=127), max_size=300))
def test_ascii_default_lengths_and_public_matches(text):
    owner = RMCPSuffixSearch(PATTERN)
    token = Tokenizer(Vocab(), suffix_search=owner.search, max_cache_size=0)
    expected = reference_search(text, re.compile(PATTERN).search)
    assert token.find_suffix(text) == (expected.end() - expected.start() if expected else 0)
    assert signature(owner.search(text)) == signature(expected)


@settings(max_examples=150, deadline=None, database=None, derandomize=True)
@given(st.text(max_size=100))
def test_unicode_default_fallback(text):
    owner = RMCPSuffixSearch(PATTERN)
    token = Tokenizer(Vocab(), suffix_search=owner.search, max_cache_size=0)
    expected = reference_search(text, re.compile(PATTERN).search)
    assert token.find_suffix(text) == (expected.end() - expected.start() if expected else 0)
    assert signature(owner.search(text)) == signature(expected)


@pytest.mark.parametrize("text", ["a.\n", "AA.\n", "a..\n", "0km/h\n", "1US$\n", "x!\n", "\n", "\n\n", "a.\n\n", "." * 10000 + "\n"])
def test_consumed_match_width_excludes_final_newline(text):
    owner = RMCPSuffixSearch(PATTERN)
    token = Tokenizer(Vocab(), suffix_search=owner.search)
    match = re.compile(PATTERN).search(text)
    assert token.find_suffix(text) == (match.end() - match.start() if match else 0)


@pytest.mark.parametrize("flags", [re.ASCII, re.IGNORECASE, re.MULTILINE, re.DOTALL])
@pytest.mark.parametrize("text", ["a.\n", "a.\nx", "AA.", "0KM", "0km/h", "中.", "x!", ""])
def test_mutated_search_flags_use_original_callback(flags, text):
    owner = RMCPSuffixSearch(PATTERN)
    callback = re.compile(PATTERN, flags).search
    owner._search = callback
    token = Tokenizer(Vocab(), suffix_search=owner.search)
    match = reference_search(text, callback)
    assert token.find_suffix(text) == (match.end() - match.start() if match else 0)
    assert signature(owner.search(text)) == signature(match)


def test_deleted_search_error_and_unconditional_single_branch():
    owner = RMCPSuffixSearch(PATTERN)
    token = Tokenizer(Vocab(), suffix_search=owner.search)
    del owner._search
    assert token.find_suffix("x!") == 1
    for callback in (lambda: owner.search("x"), lambda: token.find_suffix("x")):
        with pytest.raises(AttributeError) as caught:
            callback()
        assert str(caught.value) == "'_SuffixSearch' object has no attribute '_search'"
        assert caught.value.name == "_search"
        assert caught.value.obj is owner


def test_mutable_callback_releases_owner_before_match_methods():
    events = []
    holder = {}

    class Match:
        def end(self):
            events.append("end")
            return 2

        def start(self):
            events.append("start")
            return 1

    def callback(text):
        holder["token"].suffix_search = None
        events.append("cleared")
        return Match()

    owner = RMCPSuffixSearch(PATTERN)
    owner._search = callback
    ref = weakref.ref(owner, lambda unused: events.append("finalized"))
    token = Tokenizer(Vocab(), suffix_search=owner.search)
    holder["token"] = token
    del owner
    assert token.find_suffix("negative") == 1
    assert events == ["cleared", "finalized", "end", "start"]
    assert ref() is None


def test_replaced_search_does_not_keep_old_pattern_alive():
    re.purge()
    owner = RMCPSuffixSearch(PATTERN)
    pattern = owner._search.__self__
    ref = weakref.ref(pattern)
    del pattern
    re.purge()
    owner._search = lambda text: None
    gc.collect()
    assert ref() is None
