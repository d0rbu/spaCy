"""The native state loop consumes the existing ORTH/punctuation snapshot exactly."""
import numpy
from hypothesis import example, given, settings, strategies as st

from spacy.pipeline.sentencizer import _rmcp_sentence_starts_from_arrays


def _scalar(data, endings):
    result = [False] * len(data)
    if result:
        result[0] = True
    seen_period = False
    for i, (orth, punctuation) in enumerate(data):
        is_ending = orth in endings
        if seen_period and not punctuation and not is_ending:
            result[i] = True
            seen_period = False
        elif is_ending:
            seen_period = True
    return result


@settings(max_examples=1000, deadline=None, derandomize=True, database=None)
@given(
    rows=st.lists(st.tuples(st.integers(0, 2**64 - 1), st.booleans()), max_size=512),
    endings=st.lists(st.integers(0, 2**64 - 1), max_size=128),
)
@example(rows=[], endings=[])
@example(rows=[(1, False), (2, True), (3, False)], endings=[2])
def test_state_matches_scalar_without_mutating_arrays(rows, endings):
    data = numpy.asarray(rows, dtype=numpy.uint64).reshape((-1, 2))
    marks = numpy.asarray(endings + [r[0] for r in rows[::3]], dtype=numpy.uint64)
    old_data, old_marks = data.copy(), marks.copy()
    data.flags.writeable = False
    marks.flags.writeable = False
    assert _rmcp_sentence_starts_from_arrays(data, marks) == _scalar(rows, set(marks))
    numpy.testing.assert_array_equal(data, old_data)
    numpy.testing.assert_array_equal(marks, old_marks)
