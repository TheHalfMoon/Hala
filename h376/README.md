# H376 — Bilingual native task Unicode safety before Qwen processor input

**Source-only H199 input correctness, not legal TRAIN admission, model training,
inference or learned gain.** Stacked on fully CI-qualified H375 Draft #405.

## Real reproduced input bug

H199 `source_task_prompt` rejected NUL and CR but accepted a lone UTF-16
surrogate (not UTF-8 encodable) and Unicode directional override/isolate
control characters. Verified on owner Windows before this change. Later
`processor.apply_chat_template`, JSON encoding, or tokenizer handling could
fail or show a different visual ordering of instructions. Malformed input
cannot silently become an assistant-supervised training example.

## Minimal bounded fix

`validate_model_visible_task` now enforces the existing 1..500 Unicode
codepoint limit and strict UTF-8 encodability (up to 2,000 bytes; all scalar
sequences within 500 characters already meet this bound). It rejects
unpaired Unicode surrogates, dangerous C0/C1 terminal controls, bidi
embedding/override/isolation controls (U+202A..U+202E and U+2066..U+2069),
BOM (U+FEFF) and zero-width space (U+200B), while preserving normal Arabic
right-to-left letters, diacritics, legitimate Persian ZWNJ, English,
emoji, newlines and tabs. It **does not normalize, sanitize or rewrite**
source tasks or re-interpret raw image/action metadata.

All existing H199 action grammar, assistant-only masking, source-file SHA,
processor pinning, official admission checks and LO RA/weights gates remain
unchanged. The same validation automatically applies to
`training_messages_for_synthetic_contract` via `source_task_prompt`;
legacy logical-action records are never silently converted.

New synthetic/adversarial tests cover all 9 bidi controls, four lone
surrogates, terminal escape/C1, invisible formatting, valid Arabic and
English including line breaks and tab labels, and fail-closed H199 SFT
message construction. A dedicated four-way Windows/Ubuntu Python3.11/3.12
stdlib-only GitHub CI also runs the original H199 action tests. No images,
model weights, token generation, gold, private data or optimizer.

## Project release still NO-GO

H199 production approved TRAIN SHA registry EMPTY, H355 independently
trusted rights reviewer keys EMPTY and rights Issue #384 OPEN, H356 independent
never-exposed source-disjoint bilingual blind OOD custodian keys EMPTY and
Issue #385 OPEN. This input validation does not provide either permission,
nor demonstrate a trained Qwen gain or new OOD. Keep DRAFT and no merge
to main pending the separate release process.
