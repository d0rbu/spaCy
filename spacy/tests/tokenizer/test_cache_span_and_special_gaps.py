"""Public tokenizer contracts for cache-hit slicing and ordered retokenization."""
import pytest
from spacy.symbols import NORM, ORTH
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab

@pytest.mark.parametrize("prefix", ["prefix", "éprefix", "中prefix", "🙂prefix"])
@pytest.mark.parametrize("size", [1, 255, 256, 257])
@pytest.mark.parametrize("cache_size", [0, 10000])
def test_ascii_span_cache_preserves_wide_text_offsets(prefix, size, cache_size):
    tokenizer = Tokenizer(Vocab(), max_cache_size=cache_size)
    word = "x" * size
    text = prefix + " " + word + " tail"
    expected = [(prefix, 0, " "), (word, len(prefix) + 1, " "),
                ("tail", len(prefix) + size + 2, "")]
    for _ in range(3):
        doc = tokenizer(text)
        assert doc.text == text
        assert [(t.text, t.idx, t.whitespace_) for t in doc] == expected

class DerivedTokenizer(Tokenizer):
    pass

@pytest.mark.parametrize("kind", [Tokenizer, DerivedTokenizer])
@pytest.mark.parametrize("mode", ["merge", "expand", "mixed"])
@pytest.mark.parametrize("gap", [0, 1, 1000])
def test_special_case_gap_copy_preserves_expansion_and_reduction(kind, mode, gap):
    left = ["a b"] if mode == "merge" else ["a", " ", "b"]
    right = ["c", " ", "d"] if mode == "expand" else ["c d"]
    rules = {"a b": [{ORTH: word, NORM: word.upper()} for word in left],
             "c d": [{ORTH: word, NORM: word.upper()} for word in right]}
    tokenizer = kind(Vocab(), rules=rules, faster_heuristics=False, max_cache_size=0)
    text = "start a b " + "gap " * gap + "c d end"
    words = ["start", *left, *["gap"] * gap, *right, "end"]
    expected_offsets = []
    end = 0
    for word in words:
        start = text.index(word, end)
        expected_offsets.append(start)
        end = start + len(word)
    restored = Tokenizer(tokenizer.vocab).from_bytes(tokenizer.to_bytes())
    for implementation in [tokenizer, restored]:
        for _ in range(2):
            doc = implementation(text)
            assert doc.text == text
            assert [t.text for t in doc] == words
            assert [t.idx for t in doc] == expected_offsets
            assert [t.norm_ for t in doc if t.text in left + right] == [w.upper() for w in left + right]
            assert [t.whitespace_ for t in doc] == [
                text[start + len(word):next_start]
                for start, word, next_start in zip(expected_offsets, words, expected_offsets[1:] + [len(text)])]
