"""Exact English prefix shortcut and mutable callback publication regressions."""
import pickle
import re
import weakref

import pytest
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex
from spacy.vocab import Vocab


def prefix_regex():
    return compile_prefix_regex(English.Defaults.prefixes)


@pytest.mark.parametrize("initial", [chr(i) for i in range(128)] + ["é", "中", "🙂", "𐐀", "\ud800"])
def test_prefix_first_character(initial):
    pattern = prefix_regex()
    token = Tokenizer(Vocab(), prefix_search=pattern.search)
    for tail in ("", "word", "12", "$", "USD", "+", "\n", "中🙂", "\x00", "x" * 10000):
        text = initial + tail
        match = pattern.search(text)
        assert token.find_prefix(text) == (0 if match is None else match.end() - match.start())


@pytest.mark.parametrize("limit", [0, 1, 10000])
def test_prefix_callback_replacement(limit):
    regex = prefix_regex()
    token = Tokenizer(Vocab(), prefix_search=regex.search, max_cache_size=limit)
    calls = []
    def custom(text):
        calls.append(text)
        return re.match(".", text)
    for callback in (None, custom, re.compile("a").search, regex.match, re.compile(regex.pattern, re.I).search, regex.search):
        token.prefix_search = callback
        for text in ("abc", "0test", "A$12", "C$12", "US$12", "$12", "🙂hi", "x\n$12", ""):
            match = None if callback is None else callback(text)
            expected = 0 if match is None else match.end() - match.start()
            assert token.find_prefix(text) == expected
    assert calls


@pytest.mark.parametrize("outer", [None, "eligible", "generic"])
@pytest.mark.parametrize("inner", [None, "eligible", "generic"])
def test_prefix_destructive_setter_reentry(outer, inner):
    regex = prefix_regex()
    token = Tokenizer(Vocab(), max_cache_size=0)
    old = regex.search
    events = []
    def generic(text):
        return re.match(".", text)
    def value(kind):
        return regex.search if kind == "eligible" else generic if kind == "generic" else None
    replacement = value(outer)
    def released(unused):
        events.append("released")
        assert token.prefix_search is replacement
        token.prefix_search = value(inner)
        assert token.find_prefix("abc") == (1 if inner == "generic" else 0)
    reference = weakref.ref(old, released)
    token.prefix_search = old
    del old
    token.prefix_search = replacement
    assert reference() is None
    assert events == ["released"]
    assert token.find_prefix("abc") == (1 if inner == "generic" else 0)


def test_prefix_serialization_and_subclasses():
    regex = prefix_regex()
    token = Tokenizer(Vocab(), prefix_search=regex.search)
    clones = [token, Tokenizer(token.vocab).from_bytes(token.to_bytes()), pickle.loads(pickle.dumps(token))]
    class CustomIndex(str):
        def __getitem__(self, key):
            raise AssertionError("text subclass must use original regex semantics")
    for text in ("abc", "0word", "A$12", "$12", "éclair"):
        match = regex.search(text)
        expected = 0 if match is None else match.end() - match.start()
        assert all(t.find_prefix(text) == expected for t in clones)
        for t in clones:
            with pytest.raises(TypeError, match="Argument 'string' has incorrect type"):
                t.find_prefix(CustomIndex(text))
    class CustomTokenizer(Tokenizer):
        @property
        def prefix_search(self):
            return re.compile(".").search
        @prefix_search.setter
        def prefix_search(self, callback):
            Tokenizer.prefix_search.__set__(self, callback)
    assert CustomTokenizer(Vocab(), prefix_search=regex.search).find_prefix("abc") == 1
