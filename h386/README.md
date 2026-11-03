# H386 — reject nonfinite visual tensors and invalid one-image Qwen grids

**CPU-only processor interface safety, NOT model training, NOT learned
model quality, NOT legal TRAIN/DEV/OOD admission.**

## Real source-backed contract deficiency

`h216.real_qwen_processor_alignment.visual_tensor_receipt`, shared
by H216 real pinned Qwen3-VL processor alignment and H361 real original
H357/H358 Qwen screenshot+native-action processor parity, previously
checked only nonzero pixel tensor element counts, `torch.equal` between
prompt/answer tensors, and identical `image_grid_thw` tensors.

`torch.equal` permits a tensor containing **identical positive or
negative infinity** on both sides. It also permits identical malformed
one-image geometry metadata such as `[0,2,3]`,
`[1,-2,3]` and non-integral, multi-frame or malformed grid arrays.
Such values are not safe Qwen input even when unchanged between prompt
and supervised answer. This is a *processor-output validation
deficiency*, **not** evidence the official pinned Qwen processor
produces corrupt tensors on genuine source data.

On owner Windows, five newly authored pure fake-tensor regression tests
were executed against the **unchanged** original H216 validator at
commit `4e2ae4a`: **11 expected-refusal failures**, including equality
of nonfinite pixel data and malformed vision grids. No model, weights,
GPU, private screenshots or third-party data were involved.

## Fix

The visual receipt still demands an exact nonzero image tensor
element count and `torch.equal` for prompt and full tokens. It now
also requires that `torch.isfinite(prompt["pixel_values"]).all().item()`
return literal True. This checks real torch tensors in place without
materializing a huge Python list or mutating the input.

For the **one original screenshot** source contract, it validates
`image_grid_thw` as one positive integer triplet `[T,H,W]` (unit
fakes) or `[[T,H,W]]` (actual Qwen3-VL) with each component bounded
to 1..4096. Malformed, zero, negative, non-integral or multi-image
grid vectors fail closed. The original equality of prompt/answer
grid tensors remains in place. No normalization, grid inference,
screenshot substitution, tokenizer/prompt change, loss masking,
model image tensor mutation or extra model calls.

Existing simulated H216/H361 torch modules were updated only to
provide a source-only `isfinite` stub; a passing fake cannot
substitute for actual pinned CPU Qwen3VLProcessor qualification.

## Tests and qualification

- New 5 tests: identical finite pixels and positive fake/real-shaped
  grids PASS; +/-Infinity and NaN FAIL; zero/negative/noninteger,
  multi-image and malformed grid FAIL; changed grids and pixels FAIL.
- Owner Windows fixed-source Python3.11 **5/5 PASS** and Python3.12
  **5/5 PASS** with no package dependency.
- Owner Windows pinned Pillow CPU environment combined H386 + H216
  + H361 simulated processor source tests **20 total, 19 PASS and
  one intentional original-browser opt-in SKIP**, zero failures.
- New four-way hosted Windows/Ubuntu × Python3.11/3.12
  source-only tests require pinned CPU Pillow; no model packages
  are installed in hosted CI. An opt-in real offline pinned
  Qwen3VLProcessor CPU source run is required to check actual
  image_grid_thw layout and finiteness with the real processor.
  Until it and all exact-head GitHub runs pass, do not call H386
  fully qualified.
- All images and tasks used in optional real source validation are
  openly exposed self-authored synthetic candidates and may only
  live in an ephemeral temporary directory.

The legal H199 officially admitted TRAIN manifest digest allowlist
remains EMPTY. H355 independent external source screenshot/title
and derivative weights reviewer key registry EMPTY, rights Issue #384
OPEN. H356 genuinely new, never-exposed, family-disjoint,
independently sealed bilingual EN/AR OOD custodian keys EMPTY,
issue #385 OPEN. No lawful LoRA adaptation, learned model improvement,
blind OOD audit, trained weights or release. Keep PR DRAFT; do not
merge the protected main branch.
