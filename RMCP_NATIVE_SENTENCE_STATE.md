# Native sentence-state construction

This change builds on the immutable inner-ASCII-match/O3 fork at
`b852c3ffc9ecd516b81f30ad701153c006621300`.

A private function in the existing sentencizer extension consumes the caller's
ordinary uint64 ORTH/punctuation array and current uint64 ending hashes. An
uncached per-call native set and one state loop produce the existing Boolean
sentence-start list. No tokenizer/class layout, annotations, vocabulary, GC,
worker count, source validation or model behavior changes. The caller retains
its original attribute snapshot before iterating punctuation, and applies
annotations using the original Sentencizer method. No GIL release is added.

The fixed real-text component passed; full ordinary source installation,
64-job/eight-worker timing/parity and project gates are still required.
