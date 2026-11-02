# H364 — H199-native two-image batch and assistant-only label preflight

**This PR is an engineering exercise on already-exposed, self-authored
review candidates. It is not an authorized TRAIN release, optimizer,
model forward pass, adapter, blind DEV/TEST or trained Hala.**

## Why H364 is needed

H361 proved that 12 individual H357/H358 source-only H199 native
actions tokenized correctly under the **actual pinned Qwen3VLProcessor**
with unchanged vision inputs and assistant-only completion mask.
Individual rows are not a supervised multimodal batch. A real Qwen
LoRA path also needs right-padded image+text batches whose image
patch features and image grid order remain aligned with the
question/answer tokens, while **neither user prompt nor padding**
ever contributes to the model's supervised labels.

## Scope and guarantees

`h364/native_batch.py` implements a bounded **two-to-four row,
zero-gradient, CPU-only, research preflight**:

- Requires exact H199 prefix of the full assistant completion.
- Applies H199's `assistant_completion_mask`, then right-pads
  both token IDs and attention mask with a pinned tokenizer PAD ID.
- Labels are `-100` for the complete image/user prefix and for
  **every padding token**; only the native assistant action JSON
  continuation receives supervision.
- Checks all lengths and exact prefix preservation, refuses
  malformed/non-integer tokens, missing assistant continuation,
  extra multimodal fields, CUDA tensors, NaN or Inf pixel features,
  missing/invalid `image_grid_thw`, mixed patch feature width,
  missing images and gradient-carrying visual features.
- Concatenates `pixel_values` over the *Qwen image-patch axis* and
  `image_grid_thw` in the **same source row order** as padded
  `input_ids`, `labels` and `attention_mask`.
- Validates that each original visual grid remains a separate
  image and every source row retains its original token lengths.

`h364/run_real_cpu_batch.py` is an **opt-in owner-private CPU
processor-only QA**, not a batch intended for training. It verifies
all 192 public source-candidate rows and their PNG/source identities
via H359 (then discards the source bundle), selects one original
ControlGarden English native `click` and one original NoteRelay
Arabic native `type_text`, builds their distinct real Qwen chat
templates, verifies image tensor/prompt prefix identity and strict
assistant JSON roundtrip, then performs the two-row CPU tensor batch
preflight. Uses the already-verified original 8 public Qwen3VL
processor assets pinned at revision
`89644892e4d85e24eaac8bacfd4f463576704203`.
Temporary screenshots/labels are destroyed *before* a
bounded aggregate audit receipt is printed. Never prints tasks,
click coordinates, model-visible prompts, source screenshots,
raw tokens, model logs or gold labels.

## Evidence boundaries

The default GitHub CI matrix uses stdlib-only pure token padding
and fail-closed tests on Windows/Ubuntu Python3.11/3.12, and **does
not load torch, transformers, Qwen model weights or images**.
It cannot be cited as proof of actual real Qwen tensor batch parity.
The **actual owner Windows CPU run completed successfully** on
2026-11-02 with pinned real `Qwen3VLProcessor` (not a fake),
strictly offline H218-verified public assets, and two updated H362
original source candidates, EN click + AR type_text. It re-audited
all 192 source-only candidate rows and original PNG identities
before two real image/text/answer tokenizations. Aggregate,
NON-MODEL result: `image_grids=2`, `visual_patches=5280`,
`assistant_supervised_tokens=39`, `padded_tokens=12`,
`batch_tensor_dimensions_valid=true`, and all source images/gold
deleted at completion. `model_weights_loaded=false`,
`model_forward_calls=0`, `optimizer_steps=0`,
`official_train_admitted=false`, `can_train=false`,
`can_release=false`. This is **genuine processor/tensor
batch-parity engineering**, not actual supervised model learning
or new held-out evaluation.

No optimizer/loss/LoRA/model.load/model.generate is called, no
model weights downloaded, no GPU, no paid service and no new
independent blind OOD. H199 official TRAIN digest allowlist is
EMPTY; H355 rights reviewers and H356 OOD custodian trust lists
are EMPTY. Independent legal rights/title for ML screenshot
training/derivative weights, and truly independent sealed EN/AR
new source-disjoint blind OOD remain unresolved in #384 and #385.
Passing this primitive alone **cannot** grant training authorization
or justify Hala release/merge.
