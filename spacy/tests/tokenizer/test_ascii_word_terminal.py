"""Warm word/terminal emission retains cache, rule and callback behavior."""

import re

import pytest

from spacy.attrs import NORM
from spacy.lang.en import English
from spacy.tokenizer import RMCPSuffixSearch, Tokenizer
from spacy.vocab import Vocab


class GenericTokenizer(Tokenizer):
    pass


def make_pair(cache_size=0):
    result = []
    for cls in (Tokenizer, GenericTokenizer):
        original = English().tokenizer
        tok = cls(
            original.vocab,
            rules=original.rules,
            prefix_search=original.prefix_search,
            suffix_search=RMCPSuffixSearch(original.suffix_search.__self__.pattern).search,
            infix_finditer=original.infix_finditer,
            token_match=re.compile(r"[A-Za-z]+\Z").match,
            url_match=original.url_match,
            max_cache_size=cache_size,
        )
        result.append(tok)
    # Both constructors may publish the module-level weak prefix witness.
    # Share this immutable builtin callback so the optimized arm is eligible.
    result[1].prefix_search = result[0].prefix_search
    return result


@pytest.mark.parametrize("cache_size", [0, 1, 10000])
@pytest.mark.parametrize("terminal", list(",:;!?()[]{}<>_#*&"))
def test_terminal_word_matches_generic_before_and_after_warming(cache_size, terminal):
    fast, generic = make_pair(cache_size)
    texts = [
        "alpha" + terminal,
        "UPPER" + terminal,
        "cannot" + terminal,
        "NOUN" + terminal,
        "ORTH" + terminal,
        "I'm alpha" + terminal + " beta" + terminal,
        "é alpha" + terminal + "\n漢字",
    ]
    for text in [*texts, *texts]:
        assert fast(text).to_bytes() == generic(text).to_bytes()


@pytest.mark.parametrize("cache_size", [0, 1, 10000])
def test_warm_word_rules_and_reinitialization(cache_size):
    fast, generic = make_pair(cache_size)
    for tok in (fast, generic):
        for value in ["alpha", "cannot", "special", ",", "!", ";"]:
            tok.vocab[value]
        tok.add_special_case("alpha", [{"ORTH": "alpha", "NORM": "changed"}])
        tok.add_special_case("special,", [{"ORTH": "special,", "NORM": "whole"}])
    for text in ["alpha, cannot! special,", "alpha; other, alpha,", "cannot! cannot!"] * 2:
        assert fast(text).to_bytes() == generic(text).to_bytes()
    for tok in (fast, generic):
        tok.token_match = None
    assert fast("alpha, cannot!").to_bytes() == generic("alpha, cannot!").to_bytes()


def test_custom_prefix_and_suffix_callbacks_keep_their_order():
    fast, generic = make_pair()
    logs = [[], []]
    for tok, log in zip((fast, generic), logs):
        for text in ["alpha", ","]:
            tok.vocab[text]

        def prefix(text, log=log):
            log.append(("prefix", text))
            return re.match("^x", text)

        def suffix(text, log=log):
            log.append(("suffix", text))
            return re.search(",$", text)

        tok.prefix_search = prefix
        tok.suffix_search = suffix
    logs[0].clear()
    logs[1].clear()
    assert fast("alpha, xalpha,").to_bytes() == generic("alpha, xalpha,").to_bytes()
    assert logs[0] == logs[1]


def test_cold_suffix_lexical_reentry_uses_original_path():
    prefix = English().tokenizer.prefix_search
    suffix_pattern = English().tokenizer.suffix_search.__self__.pattern
    outputs = []
    logs = []
    for cls in (Tokenizer, GenericTokenizer):
        events = []
        holder = []

        def norm(text):
            events.append(text)
            if text == ",":
                holder[0].token_match = None
            return text.lower()

        vocab = Vocab(lex_attr_getters={NORM: norm})
        tok = cls(
            vocab,
            rules={},
            prefix_search=prefix,
            suffix_search=RMCPSuffixSearch(suffix_pattern).search,
            token_match=re.compile(r"[A-Za-z]+\Z").match,
            max_cache_size=0,
        )
        holder.append(tok)
        vocab["alpha"]
        events.clear()
        outputs.append(tok("alpha,").to_bytes())
        logs.append(events)
    assert outputs[0] == outputs[1]
    assert logs[0] == logs[1]


@pytest.mark.parametrize("method", ["bytes", "pickle"])
def test_word_terminal_roundtrip(method):
    import pickle

    tok, generic = make_pair()
    expected = generic("I'm alpha, beta!").to_bytes()
    assert tok("I'm alpha, beta!").to_bytes() == expected
    if method == "bytes":
        tok.from_bytes(tok.to_bytes())
    else:
        tok = pickle.loads(pickle.dumps(tok))
    assert tok("I'm alpha, beta!").to_bytes() == expected
