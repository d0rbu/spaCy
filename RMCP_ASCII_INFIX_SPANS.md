# Exact ASCII infix spans

This CPU fork adds an optional `RMCPInfixFinditer` owner for the exact default
English infix expression. The private RMCP boundary pipeline can bind its
`finditer` method. Public `find_infix` and `finditer` still return ordinary regex
Match objects with their original bounds, groups, input and compiled Pattern.

For the exact Tokenizer class, exact ASCII strings and this exact immutable
default-pattern owner, `_attach_tokens` computes nonoverlapping ranges directly.
It avoids the regex scanner, intermediate Python list, Match objects and temporary
native span vector. Other callback owners, Tokenizer subclasses, nondefault
patterns/flags and non-ASCII input keep the existing implementation. The
Tokenizer layout and `.pxd` remain unchanged.

The ASCII grammar preserves original alternative order: greedy dot runs;
digit/operator/digit-or-hyphen; lower-or-quote/dot/upper-or-quote; letter/comma/letter;
letter-or-digit/hyphen-or-tilde/letter; and letter-or-digit/colon-angle-equals-slash/letter.
The ordered hyphen alternatives include one, two and three hyphens. Matching at
position zero retains the existing attachment skip. The default ASCII grammar
has no zero-width alternatives; custom zero-width expressions remain generic.

The owner stores a compiled immutable regex and read-only pattern. A nonempty
original Match list retains its Pattern throughout attachment. The native path
holds that Pattern for the same scope, including when the only match is skipped
at position zero. It releases the callback owner at the original point when a
vocabulary callback clears it. With no matches, it adds no Pattern keeper.
Reinitialization or callback replacement does not change the already-selected
immutable grammar for the current attachment. Public serialization retains the
original pattern and callback behavior.

A standalone build of the identical C helper passed 3,460,827 comparisons,
including every ASCII string through length three, all ASCII contexts for 13
multi-character operator strings, every Unicode codepoint for fallback, random
strings and long runs. The functional comparison retained 2,571 document cases,
the earlier dispatch/cache/serialization controls and 30,180 new infix cases.
All 55 new native regressions passed against the component binary, including
nonvacuous Pattern and owner finalization checks. Complete installed-package
qualification is a separate required step.

The fixed eight-call component comparison used 1,944 real documents and
28,192,346 characters. Median candidate/baseline cold-inclusive wall/CPU ratios
were 0.920629/0.918791; warm ratios were 0.879216/0.878952. Both halves improved
all four metrics; the peak individual RSS ratio was 0.998921. These component
results do not qualify the full corpus workflow, production use or GPU work.
The shared host may have concurrent cleanup I/O.

An earlier vector prototype passed the speed screen but finalized its Pattern
inside the first vocabulary callback. That failed lifetime witness and the
original timings remain preserved. An initial checker also stopped in the
baseline on a lone-surrogate exception; its corrected fresh namespace compares
that exception. No earlier timings are pooled with this implementation.

Detailed local evidence is under
`corpus-native-lm-fast-20261023-v1/spacy-ascii-infix-spans-component-20261028-v4`.
