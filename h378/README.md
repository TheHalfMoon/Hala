# H378 — Native task private-use and unassigned Unicode (NO TRAIN)

## Reproduced defect
At H377 exact head d10898a00b82ed8927a9fce9a3ebe6fdf13b75ff,
H199 validate_model_visible_task still accepted Unicode Co private-use
(U+E000) and Cn unassigned/noncharacters (U+0378, U+FDD0).
The same H199 native type_text validator already rejects both categories.
Their glyph identities are not portable model-visible task instructions.

## Narrow repair
Reject Co and Cn in H199 model-visible task text alongside existing
Cs, Zl, Zp and unsafe controls. Preserve accepted task bytes, Arabic
harakat, Persian ZWNJ, Arabic ZWJ, emoji, newline/tab and native actions.
Invented-only negative and positive tests; no screenshots, private gold,
weights, GPU, training or blind OOD. This is NOT a legal rights grant,
model learning result or release qualification. H199 official TRAIN
allowlist and independent H355/H356 trust registries remain EMPTY;
rights Issue #384 and sealed bilingual OOD Issue #385 remain unresolved.
