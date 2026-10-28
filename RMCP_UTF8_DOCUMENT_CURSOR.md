# Exact Unicode document span cursor

For exact built-in strings and the native `Tokenizer`, scan one borrowed cached
UTF-8 document view alongside the original codepoint cursor. At each unchanged
whitespace boundary, hash the byte span and check the existing special/cache maps
before allocating a substring. Misses follow the original affix and vocabulary
logic. The Unicode input owns the view for the entire call; no pointer escapes.

Native whitespace and branchless UTF-8 widths cover all Unicode codepoints.
An initial width/surrogate prepass prevents eager UTF-8 conversion from changing
the original partial vocabulary/cache side effects on invalid surrogate input.
Surrogates, oversized hash inputs, tokenizer subclasses, and string subclasses
use the original implementation. A zero hash retains the original slice/hash
fallback. The existing document length limit remains in force.

This changes no regex callback order, hash algorithm, public layout, cache
capacity, worker scheduling, or garbage-collection policy. Whole-document UTF-8
storage remains attached to its immutable input and is included in RSS evidence.

The source regression compares the native path with the unchanged subclass
fallback, including UTF-8 width boundaries, Unicode whitespace, NUL, errors and
partial vocabulary state, cache eviction, callbacks, rules, and serialization.
Fixed real-corpus component measurements passed; installed full-workload
qualification is required independently before application adoption.
