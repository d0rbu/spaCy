# Exact ASCII token-match specialization

This fork starts at spaCy 3.8.11 release commit
`e7a662acf8eda206a091aa3e7316700e0e6c699d`.

Only exact native Tokenizer instances with the exact immutable compiled
`[A-Za-z]+\Z` match method and default Unicode flags use the native letter
predicate at the original two inner truth tests. All other callbacks, patterns,
flags and subclasses keep the original expressions and call order. Setter
publication retains the old callback until its replacement is visible, then
classifies the actual current callback after possible destructor reentry.

The tokenizer has one additional private Boolean field; every spaCy extension
must be built together. Linux uses generic `-O3` without fast-math or native-CPU
targeting. Build dependencies are pinned to the versions used for qualification.
This is a combined source/build change, not a compiler-independent timing claim.
