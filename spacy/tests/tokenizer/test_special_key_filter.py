import pickle

import pytest

from spacy.strings import hash_string
from spacy.symbols import NORM, ORTH
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab


def observed(tokenizer, text):
    return [(token.text, token.norm_, token.idx) for token in tokenizer(text)]


def test_new_special_replaces_warm_regular_cache():
    tokenizer = Tokenizer(Vocab())
    assert observed(tokenizer, "abcd") == [("abcd", "", 0)]
    tokenizer.add_special_case("abcd", [{ORTH: "ab", NORM: "left"}, {ORTH: "cd"}])
    assert observed(tokenizer, "abcd") == [("ab", "left", 0), ("cd", "", 2)]
    tokenizer.add_special_case("abcd", [{ORTH: "abcd", NORM: "replacement"}])
    assert observed(tokenizer, "abcd") == [("abcd", "replacement", 0)]


def test_filter_collision_preserves_exact_lookup():
    by_bits = {}
    pair = None
    for i in range(2000):
        text = f"collision{i}"
        bits = hash_string(text) & 65535
        if bits in by_bits:
            pair = (by_bits[bits], text)
            break
        by_bits[bits] = text
    assert pair is not None
    first, second = pair
    tokenizer = Tokenizer(Vocab(), rules={first: [{ORTH: first, NORM: "special"}]})
    assert observed(tokenizer, first) == [(first, "special", 0)]
    assert observed(tokenizer, second) == [(second, "", 0)]
    tokenizer.add_special_case(second, [{ORTH: second, NORM: "second"}])
    assert observed(tokenizer, second) == [(second, "second", 0)]


def test_rules_replacement_and_reinitialization():
    tokenizer = Tokenizer(Vocab(), rules={"old": [{ORTH: "old", NORM: "special"}]})
    assert observed(tokenizer, "old")[0][1] == "special"
    tokenizer.rules = {"new": [{ORTH: "new", NORM: "next"}]}
    assert observed(tokenizer, "new")[0][1] == "next"
    assert observed(tokenizer, "old")[0][1] != "special"
    tokenizer.__init__(Vocab(), rules={"fresh": [{ORTH: "fresh", NORM: "reset"}]})
    assert observed(tokenizer, "fresh")[0][1] == "reset"
    # Existing reinitialization reloads the previous rules while publishing
    # callbacks. Preserve that behavior, including its surviving special.
    assert observed(tokenizer, "new")[0][1] == "next"


@pytest.mark.parametrize("setter", ["prefix_search", "suffix_search", "infix_finditer", "token_match", "url_match", "faster_heuristics"])
def test_callback_reload_keeps_specials(setter):
    tokenizer = Tokenizer(Vocab(), rules={"keep": [{ORTH: "keep", NORM: "special"}]})
    setattr(tokenizer, setter, getattr(tokenizer, setter))
    assert observed(tokenizer, "keep") == [("keep", "special", 0)]


def test_serialization_and_pickle_reconstruct_filter():
    tokenizer = Tokenizer(Vocab(), rules={"keep": [{ORTH: "keep", NORM: "special"}]})
    restored = Tokenizer(Vocab()).from_bytes(tokenizer.to_bytes())
    pickled = pickle.loads(pickle.dumps(tokenizer))
    for value in [restored, pickled]:
        assert observed(value, "keep") == [("keep", "special", 0)]
        assert observed(value, "ordinary") == [("ordinary", "", 0)]


def test_subclass_special_behavior():
    class Subclass(Tokenizer):
        pass

    tokenizer = Subclass(Vocab(), rules={"keep": [{ORTH: "keep", NORM: "special"}]})
    assert observed(tokenizer, "keep") == [("keep", "special", 0)]


def test_reentrant_token_match_can_add_special():
    tokenizer = Tokenizer(Vocab())

    def match(text):
        if text == "trigger":
            tokenizer.add_special_case("added", [{ORTH: "added", NORM: "nested"}])
        return None

    tokenizer.token_match = match
    tokenizer("trigger")
    assert observed(tokenizer, "added") == [("added", "nested", 0)]
