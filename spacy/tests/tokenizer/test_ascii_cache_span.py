"""Cache hits preserve token state and custom tokenizer behavior."""
import re

import pytest

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer


class GenericTokenizer(Tokenizer):
    pass


def generic_copy(tokenizer, cls=GenericTokenizer):
    return cls(
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
    "", "hello", " hello  hello ", "can't U.S. can't U.S.",
    "a\x00b a\x00b", "a\tb\nc\rd\ve\ff\x1cg\x1dh\x1ei\x1fj",
    "http://example.com?q=one user@example.org", "(hello), (world)!",
    "ASCII words with café and \u2003 space", "漢字 🙂 \U00010000",
    "x" * 65 + " " + "x" * 65,
    "café repeat repeat", "漢字 repeat repeat", "🙂 repeat repeat",
    "café " + "x" * 127 + " " + "x" * 127,
    "漢字 " + "x" * 128 + " " + "x" * 128,
    "🙂 " + "x" * 129 + " " + "x" * 129,
    "word\u0301 word\u0301 repeat repeat", "é \x00word \x00word",
    "漢字\u2003ASCII\u2003ASCII", "🙂\u00a0ASCII\u00a0ASCII",
])
def test_cache_hits_match_custom_subclass(text, cache_size):
    fast = English().tokenizer
    fast.max_cache_size = cache_size
    reference = generic_copy(fast)
    for value in [text, text, "different words", text]:
        assert fast(value).to_bytes() == reference(value).to_bytes()


def test_cache_miss_callbacks_and_special_case_changes():
    fast = English().tokenizer
    reference = generic_copy(fast)
    events = [[], []]
    for tokenizer, log in zip((fast, reference), events):
        def prefix(text, log=log):
            log.append(("prefix", text))
            return re.search(r"^\(", text)

        def suffix(text, log=log):
            log.append(("suffix", text))
            return re.search(r"[).]$", text)

        tokenizer.prefix_search = prefix
        tokenizer.suffix_search = suffix
    for text in ["(newword). newword", "(newword). newword"]:
        for log in events:
            log.clear()
        assert fast(text).to_bytes() == reference(text).to_bytes()
        assert events[0] == events[1]
    for tokenizer in (fast, reference):
        tokenizer.add_special_case("newword", [{"ORTH": "newword", "NORM": "new"}])
    assert fast("newword (newword).").to_bytes() == reference("newword (newword).").to_bytes()


def test_string_subclass_rejection_matches_generic_tokenizer():
    calls = []

    class Text(str):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def __getitem__(self, key):
            calls.append(("getitem", repr(key)))
            return super().__getitem__(key)

    fast = English().tokenizer
    reference = generic_copy(fast)
    text = Text("first second first")
    messages = []
    for tokenizer in (reference, fast):
        with pytest.raises(TypeError) as error:
            tokenizer(text)
        messages.append(str(error.value))
    assert messages[0] == messages[1]
    assert calls == []
