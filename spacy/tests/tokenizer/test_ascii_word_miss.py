"""Plain ASCII misses preserve the generic tokenizer contract."""
import re

import pytest

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer


class GenericTokenizer(Tokenizer):
    pass


def generic_copy(tokenizer):
    return GenericTokenizer(
        tokenizer.vocab,
        rules=tokenizer.rules,
        prefix_search=tokenizer.prefix_search,
        suffix_search=tokenizer.suffix_search,
        infix_finditer=tokenizer.infix_finditer,
        token_match=tokenizer.token_match,
        url_match=tokenizer.url_match,
        faster_heuristics=tokenizer.faster_heuristics,
        max_cache_size=tokenizer.max_cache_size,
    )


@pytest.mark.parametrize("cache_size", [0, 1, 10000])
@pytest.mark.parametrize("text", [
    "", "lower UPPER Mixed", "I'm repeated repeated another repeated",
    " plain  words\tand\nmore ", "word-thing (word), word123",
    "naïve 漢字 🙂 word café", "a\x00b word", "x" * 10000,
    "special ordinary special ordinary", "http://example.com user@example.org",
])
def test_ascii_word_misses_match_generic(text, cache_size):
    fast = English().tokenizer
    fast.token_match = re.compile(r"[A-Za-z]+\Z").match
    fast.max_cache_size = cache_size
    fast.add_special_case("special", [{"ORTH": "special", "NORM": "changed"}])
    reference = generic_copy(fast)
    for value in [text, text, "other new words", text]:
        assert fast(value).to_bytes() == reference(value).to_bytes()


def test_custom_token_match_keeps_callback_sequence():
    fast = English().tokenizer
    reference = generic_copy(fast)
    logs = [[], []]
    for tokenizer, log in zip((fast, reference), logs):
        def match(text, log=log):
            log.append(text)
            return re.match(r"[A-Za-z]+\Z", text)
        tokenizer.token_match = match
        tokenizer.max_cache_size = 0
    for text in ["alpha beta alpha", "I'm plain after special", "ab-cd"]:
        for log in logs:
            log.clear()
        assert fast(text).to_bytes() == reference(text).to_bytes()
        assert logs[0] == logs[1]


def test_word_match_setter_and_rule_mutations():
    fast = English().tokenizer
    reference = generic_copy(fast)
    for match in [re.compile(r"[A-Za-z]+\Z").match, None,
                  re.compile(r"[A-Za-z]+$").match,
                  re.compile(r"[A-Za-z]+\Z").match]:
        for tokenizer in (fast, reference):
            tokenizer.token_match = match
            tokenizer.add_special_case("alpha", [{"ORTH": "alpha", "NORM": "new"}])
        for text in ["alpha beta", "alpha\n beta", "abc-def"]:
            assert fast(text).to_bytes() == reference(text).to_bytes()


@pytest.mark.parametrize("cache_size", [0, 1, 10000])
def test_existing_word_lexemes_reserved_symbols_and_memory_zones(cache_size):
    fast = English().tokenizer
    fast.token_match = re.compile(r"[A-Za-z]+\Z").match
    fast.max_cache_size = cache_size
    reference = generic_copy(fast)
    words = ["freshword", "NOUN", "VERB", "ADJ", "ROOT", "ORTH", "alpha", "beta"]
    for word in words:
        fast.vocab[word].is_stop = True
    text = "I'm " + " ".join(words + words)
    assert fast(text).to_bytes() == reference(text).to_bytes()
    with fast.vocab.memory_zone():
        value = text + " zoneonlyword zoneonlyword"
        assert fast(value).to_bytes() == reference(value).to_bytes()
    for word in words:
        fast.vocab[word].is_stop = False
    assert fast(text).to_bytes() == reference(text).to_bytes()
