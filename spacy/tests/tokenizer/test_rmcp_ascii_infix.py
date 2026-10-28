"""Private default-English ASCII infix emission and public callback contracts."""
import gc
import pickle
import re
import weakref

import pytest
from hypothesis import given, settings, strategies as st
from spacy.attrs import NORM
from spacy.lang.en import English
from spacy.tokenizer import RMCPInfixFinditer, Tokenizer
from spacy.util import compile_infix_regex
from spacy.vocab import Vocab

PATTERN = compile_infix_regex(English.Defaults.infixes).pattern


def signature(doc):
    return [(t.text, t.idx, t.whitespace_, t.norm_, t.is_space, t.is_punct) for t in doc]


def match_signature(match):
    return (match.span(), match.group(), match.groups(), match.groupdict(),
            match.pos, match.endpos, match.string, match.re.pattern, match.re.flags)


def compare_text(text, pattern=PATTERN, flags=0):
    owner = RMCPInfixFinditer(pattern, flags)
    regex = re.compile(pattern, flags)
    candidate = Tokenizer(Vocab(), infix_finditer=owner.finditer, max_cache_size=0)
    reference = Tokenizer(Vocab(), infix_finditer=regex.finditer, max_cache_size=0)
    assert [match_signature(m) for m in candidate.find_infix(text)] == [
        match_signature(m) for m in reference.find_infix(text)
    ]
    for _ in range(2):
        assert signature(candidate(text)) == signature(reference(text))


@settings(max_examples=400, deadline=None, database=None, derandomize=True)
@given(st.text(alphabet=st.characters(max_codepoint=127), max_size=250))
def test_complete_ascii_infix_emission_and_public_match_objects(text):
    compare_text(text)


@settings(max_examples=200, deadline=None, database=None, derandomize=True)
@given(st.text(max_size=100))
def test_unicode_uses_generic_infix_behavior(text):
    compare_text(text)


@pytest.mark.parametrize("text", [
    "", "abc123", "a-b-c", "1--2", "1---a", "a---b", "a----b", "a~b",
    "a.B", "A.b", "a,b", "1/a", "1/2", "..abc", "...abc", "a...Z",
    "a\x00-b", "a'.,Z", "." * 10000, "a" + "." * 10000 + "Z",
])
def test_ordered_contexts_dot_runs_and_initial_match_skip(text):
    compare_text(text)


@pytest.mark.parametrize("flags", [re.ASCII, re.IGNORECASE, re.MULTILINE, re.DOTALL])
@pytest.mark.parametrize("text", ["a.b", "a.B", "é-a", "中/a", "a\n.B"])
def test_nondefault_flags_preserve_original_regex(flags, text):
    compare_text(text, flags=flags)


@pytest.mark.parametrize("pattern", ["", "(?=a)", "-", "[a-z]+", "(?<=a).*?(?=b)"])
def test_custom_patterns_and_zero_width_matches_remain_generic(pattern):
    for text in ["", "abc", "a-b", "aa-bb", "é-a"]:
        compare_text(text, pattern)


def test_subclass_find_infix_override_is_called():
    calls = []
    class Derived(Tokenizer):
        def find_infix(self, text):
            calls.append(text)
            return super().find_infix(text)
    candidate = Derived(Vocab(), infix_finditer=RMCPInfixFinditer(PATTERN).finditer,
                        max_cache_size=0)
    assert [t.text for t in candidate("a-b")] == ["a", "-", "b"]
    assert calls == ["a-b"]


@pytest.mark.parametrize("text,pattern_after_callback", [
    ("a-b-c", True), ("...abc", True), ("abc123", False),
])
def test_pattern_lifetime_matches_nonempty_list_but_owner_releases_in_callback(
    text, pattern_after_callback
):
    re.purge()
    events = []
    holder = {}
    owner = RMCPInfixFinditer(PATTERN)
    regex = next(owner.finditer("a-b")).re
    regex_ref = weakref.ref(regex, lambda unused: events.append("pattern-finalized"))
    owner_ref = weakref.ref(owner, lambda unused: events.append("owner-finalized"))
    def norm(value):
        events.append(("norm", value))
        if holder["token"].infix_finditer is not None:
            holder["token"].infix_finditer = None
            events.append("callback-cleared")
        return value
    token = Tokenizer(Vocab(lex_attr_getters={NORM: norm}),
                      infix_finditer=owner.finditer, max_cache_size=0)
    holder["token"] = token
    del owner, regex
    re.purge()
    gc.collect()
    events.clear()
    token(text)
    events.append("returned")
    assert owner_ref() is None and regex_ref() is None
    assert events.index("owner-finalized") < events.index("callback-cleared")
    if pattern_after_callback:
        last_norm = max(i for i, event in enumerate(events) if isinstance(event, tuple))
        assert last_norm < events.index("pattern-finalized") < events.index("returned")
    else:
        assert events.index("pattern-finalized") < events.index("callback-cleared")


def test_public_finditer_bounds_and_pickle_keep_original_matches():
    owner = RMCPInfixFinditer(PATTERN)
    restored = pickle.loads(pickle.dumps(owner))
    text = "a.B--Z"
    expected = [match_signature(m) for m in re.compile(PATTERN).finditer(text, 1, 5)]
    assert [match_signature(m) for m in restored.finditer(text, 1, 5)] == expected
    token = Tokenizer(Vocab(), infix_finditer=owner.finditer)
    copied = Tokenizer(token.vocab).from_bytes(token.to_bytes())
    pickled = pickle.loads(pickle.dumps(token))
    assert signature(token(text)) == signature(copied(text)) == signature(pickled(text))


@pytest.mark.parametrize("field", ["pattern", "_regex", "_english"])
def test_marker_state_cannot_be_arbitrarily_published(field):
    owner = RMCPInfixFinditer(PATTERN)
    with pytest.raises(AttributeError):
        setattr(owner, field, None)
