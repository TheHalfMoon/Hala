# H368 — independent right-padding input integrity (research only)

**Stacked on H367; no training, weights, model forward, private data, or release.**

## Observed defect

The H367 independent four-row Qwen3VLProcessor batch-order verifier checked
that right-padded positions had `attention_mask=0` and `labels=-100`,
but did **not** check the actual `input_ids` in those positions.
An adversarial CPU-only mutation replacing a PAD input ID with 999 was
accepted by `verify_pack_order` while its labels and attention stayed masked.
The H364 batch *builder* itself constructed correct padding; this was a
**verification blind spot**, not evidence of corrupt source data or Qwen learning.

## Repair and bounded evidence

H368 passes the pinned tokenizer's `pad_token_id` into the H367 verifier,
checks full row lengths and exact `full_ids + [pad_id] * right_pad`,
and keeps the existing source-image, visual-grid, assistant-label, and
attention-mask association checks. Eight invented-token stdlib regressions
cover valid padding, no padding, masked non-PAD input, wrong pad ID,
unmasked/supervised PAD, truncated attention, and bool-as-int pad ID.
The inherited Windows/Ubuntu Python 3.11/3.12 source-only workflow now
includes these tests. The original source images/gold are never committed.

This protects a CPU **processor-input research proof**. It is not legal
TRAIN admission, Qwen model execution, optimizer training, new blind OOD,
or a learned improvement. H199 TRAIN allowlist, H355 reviewer registry,
and H356 custodian registry remain empty; Issues #384 and #385 require
independent qualified evidence.
