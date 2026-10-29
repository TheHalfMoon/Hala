# H209 — fail-closed preloaded multimodal prompt identity across device transfer

**Scope:** source-only hardening of H208's preloaded model protocol. Parent is
H208 Draft #229 at `ba8adfbe0fac4ce95ff52d3799076681d815cce0`.
Keep every PR Draft and never merge, publish, or deploy.

## Why this matters

H208 extracted the original prompt token prefix before
`BatchEncoding.to(model.device)` and verified the generated continuation
against that saved prefix. It did **not** verify that the actual token batch
passed to `generate()` still represented the same screenshot-only task.
A faulty or adversarial processor/batch wrapper could replace or remove
`input_ids` during transfer, or introduce an extra task row. A generated
action could then be accepted against the original prefix despite the model
having received a different task.

## Change

- Reject any processor batch with more or fewer than exactly one input row.
- After device transfer, require exactly one input row and the **same exact
  integer token sequence** as before transfer.
- Preserve H208's existing requirement that the transferred batch still
  contains nonempty image values.
- Reject all malformed batches **before** calling `model.eval()` or
  `model.generate()`; redact details from public error messages.
- Add synthetic negative tests for extra input rows and three distinct
  transfer corruptions: replacement, removal, and duplication.
- Existing H208 hosted CPU Python 3.11/3.12 workflow is triggered by
  changes to `h208/**` and `tests/test_h208_preloaded_vlm_policy.py`.
  **CI success must be verified at the exact new PR head.**

## Evidence and limits

All tests use synthetic PNGs and a fake processor/model. No checkpoint
is loaded, no actual Qwen inference, no optimizer, no GPU, no training,
no new OOD, and no model improvement is claimed. This does not prove
that image tensors contain the right pixels *after* transfer or that a
model uses the visual tokens; it proves only prompt-token identity and
batch shape at the adapter boundary.

H173 original OOD is permanently consumed; do not open or reuse it.
The H199 official TRAIN SHA allowlist remains empty, and historical
source/fork and decoded-RGBA provenance remain incomplete. The 20
original GUI research candidates remain unadmitted. No owner media,
gold, predictions, or private weights are added to GitHub.
