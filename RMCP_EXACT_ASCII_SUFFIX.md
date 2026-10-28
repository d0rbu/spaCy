# Exact ASCII English suffix lengths

This fork extends the accepted O3/inner-ASCII-token matcher baseline
`b852c3ffc9ecd516b81f30ad701153c006621300`.

The private `RMCPSuffixSearch` producer preserves its original direct search
Match semantics and mutable fields. Exact Tokenizer/default-English/ASCII
calls can compute `Match.end() - Match.start()` without creating a regex Match.
A final newline changes the dollar-anchor position, not the consumed width.
Non-ASCII strings, subclasses, custom searches and non-default flags use the
original callback path. Classification confines its temporary owner reference
so a custom callback can release the wrapper before arbitrary Match methods.
No Tokenizer field/layout or general regex engine changes are introduced.

The finite helper is generated from the ASCII projection of all default suffix
alternatives: fixed-width lookbehinds, explicit units/currencies, punctuation,
and the original unbounded final dot run. The runtime performs no parser,
source fingerprint, Unicode scan or input checksum. The standalone helper
passed 3,570,552 literal comparisons against Python's actual matched width.
`spacy/tests/tokenizer/test_rmcp_ascii_suffix.py` tests property-based ASCII and
Unicode behavior, terminal newlines, flags, deleted fields and callback lifetime.

A fixed real component screen used 1,944 documents/28,192,346 characters in eight
fresh processes, order A1 B1 B2 A2 A3 B3 B4 A4, default GC and cache size 10,000.
The corrected source reduced median cold wall time by 20.23% and warm wall time
by 30.61%; cold CPU by 24.45% and warm CPU by 26.19%. Both halves improved on
all four metrics and peak RSS ratio was 0.999605. All exact Text/native outputs
matched. These are component results on a shared host, not an end-to-end claim;
full installed-package and complete-workload qualification remain separate.
