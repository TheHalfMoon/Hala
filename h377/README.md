# H377 — Close remaining invisible format controls in model-visible tasks

**Pure native-action source contract fix; NO TRAIN or novel blind OOD.**
Stacked on H376 Draft PR #406 after exact-head 56/56 GitHub CI.

## Reproduced overlooked input defect

The new H376 Unicode guard correctly rejected bidi overrides and surrogates
but STILL accepted several invisible bidirectional/formatting marks:
U+061C Arabic Letter Mark, U+200E LRM, U+200F RLM, U+2060 Word Joiner
and U+00AD Soft Hyphen. It also accepted line/paragraph separator U+2028
and U+2029 that bypass conventional newline rules. This was reproduced on
the owner Windows Python3.11 H199 source_task_prompt BEFORE this change,
with no source media, labels, model weights or inference.

## Tight fix, preserving Arabic and Persian

Reject all Unicode Cf format-control characters **except U+200C Zero Width
Non-Joiner and U+200D Zero Width Joiner** required for legitimate shaping.
Reject Zl/Zp Unicode line and paragraph separators. Keep ordinary Arabic
letters and vowel marks, Persian ZWNJ usage, emoji, English, real newline/tab
instructions and the original task bytes unchanged. This does not silently
strip or normalize text, so signed or audited source identities cannot drift.

New invented-only tests include exact controls and an exhaustive scan over
Unicode Cf codepoints, paired with positive bilingual sentence tests and
the unchanged H199 action/message schema tests. Dedicated 4-way GitHub
Windows/Ubuntu Python3.11/3.12 source-only workflow, no model weights/GPU,
private media, screenshots or gold.

## Still blocked from actual model training

This fixes only untrusted task Unicode input, not independent grants for
screenshot ML or derivative weights. H199 official legally admitted TRAIN
digest allowlist and H355/H356 independent signer registries remain empty;
rights GitHub Issue #384 and independent never-exposed family-disjoint
bilingual EN/AR blind OOD Issue #385 remain open. Do not merge main or
claim actual Qwen model improvement/release.
