# H301 — terminal latch for unexpected model/runtime exceptions

This source-only change hardens the **existing** H298 preloaded two-stage
screenshot policy. Previously its outer terminal-failure latch caught only
`ValueError`, `TypeError`, `RuntimeError`, and `OSError`. An unexpected
`KeyError`, `AttributeError`, or `LookupError` from a tokenizer, multimodal
processor, or model bridge could escape **without latching** the policy,
leaving the same object eligible for another action attempt.

The outer boundary now catches **ordinary `Exception`** (not
`BaseException`), sets the terminal latch, and returns only the existing
categorical `ModelContractRefused` error. No exception message, private
tokens, model output, screenshot, coordinates, or task labels are emitted.
A subsequent callback cannot invoke the model again. The inner model
wrapper remains unchanged; existing successful two-action execution and
abstention semantics are preserved.

Three invented-only CPU regression cases inject unexpected tokenizer,
processor, and model-bridge errors. This is **not** a new Qwen run, new
optimizer step, or demonstration of GUI accuracy. The previous H299
pretrained first-party FieldPort 2/2 is unchanged, as is H292's zero
paired LoRA gain. Public original research scenes are not blind OOD;
H199 official admitted TRAIN remains empty. All work remains on a
stacked **DRAFT** branch, with no main merge, model release, paid GPU,
or private artifacts.
