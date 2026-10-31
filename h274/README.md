# H274 — Private Qwen relative-output format diagnostics

**Purpose:** Diagnose why the preregistered H273 normalized 0..999
image-coordinate arm did not reach native action validation, without
disclosing a single model prediction, target point or screenshot.

## Verified motivation

H273 owner-private FREE Kaggle T4x2 v1 ran **four genuine pretrained
Qwen3-VL generations** and used strict model-time token grammar.
Pixel arm produced valid clicks in both original first-party Arabic
Stockroom scenes but missed the first `+1` target (0/2).
Qwen-relative arm produced two genuine generations, but neither
passed its exact normalized schema (0/2 valid, 0/2 dispatched).
This **does not** establish whether the model can ground better with
relative coordinates; the relative predictions were not scored as
clicks. The exact 2,655-byte H273 aggregate is documented in
[h273/README.md](../h273/README.md), commit
`53d7fb25e113f8f2cd337bb4aebeafac026d1a22`.
H272 earlier genuine 8-step joint visual-language rank4 LoRA also
failed actual task-state (0/2 both before and after).

## Exact prospective private diagnostic

Run the **same pinned pretrained Qwen3-VL-2B-Instruct revision**
`89644892e4d85e24eaac8bacfd4f463576704203`, same
two first-party author-owned RTL screenshots, and SAME H273
model-time constrained normalized token grammar with **no changes
to the model, task prompt, decoder settings or grammar**.
One genuine `model.generate` on each scene, without issuing any
browser action. Read the generated text only in ephemeral private
Kaggle memory and output only one of the closed category labels:

- `STRICT_EXACT_VALID`
- `NON_STRING`, `EMPTY`, `OVER_128_CHARS`
- `MARKDOWN_FENCE`
- `INCOMPLETE_OR_UNCLOSED_JSON`
- `NOT_JSON_OR_NONCANONICAL_PREFIX`
- `JSON_DIFFERENT_TOP_LEVEL_KEYS`
- `JSON_DIFFERENT_ACTION_COUNT`
- `JSON_CLICK_INVALID_RELATIVE_POINT`
- `JSON_VALID_CLICK_NONCANONICAL_SYNTAX`
- `JSON_VALID_ABSTAIN_NONCANONICAL_SYNTAX`
- `JSON_OTHER_ACTION`

These labels are narrowly defined and contain **no model text,
token stream, coordinates or images**. The only report permitted is
one owner-private aggregate JSON
`hala-h274-private-relative-format-reason-aggregate.json`
under Kaggle's output root. No screenshots, weights, checkpoints,
private app data, gold labels, or raw model generations are saved.
The full pinned Qwen model cache and all original source media remain
in an ephemeral private `/tmp` directory and are deleted at exit.
Original screenshot hash must remain unchanged during inference.
If any real generation, tokenizer grammar or source pin fails,
the report is refused instead of fabricated.

Source is frozen to H273 head
`53d7fb25e113f8f2cd337bb4aebeafac026d1a22` on
`ps/h273-qwen-native-normalized-point-guard`. The original
H262 HTML SHA256 must still equal
`3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.
Transformers==5.16.1 and the existing pinned MIT
lm-format-enforcer source commit
`817f944fcc8851917c40300a47f62d1defc3ffc3`
are required. Existing free Kaggle CUDA PyTorch must stay unchanged.
No local Windows/Mac pretrained model weights.

## Owner-private free T4

Public code-only files are this README,
`h274/private_relative_format_diagnostic.py`,
`h274/build_private_payload.py` and synthetic-only
`tests/test_h274_private_relative_format_diagnostic.py`.
Create exactly one Python script + private metadata with no
data/model/kernel attachments, then submit using the owner's
authenticated Kaggle CLI:

    python -m h274.build_private_payload --source h274/private_relative_format_diagnostic.py --destination D:\Hala-H274-Private-Upload
    kaggle kernels push -p D:\Hala-H274-Private-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h274-private-relative-format-reasons-t4

Never run unfiltered `kaggle kernels output`: use ONLY
`--file-pattern hala-h274-private-relative-format-reason-aggregate.json`
into a fresh private scratch folder, then delete that exact small
local file after documenting sanitized evidence.

**Scientific stop gates remain mandatory:** H199
`ADMITTED_TRAIN_MANIFEST_SHAS` is EMPTY; the H173 scored 32-case
OOD is consumed and permanently SEALED; H212 procedural scored
rows barred; original H250/H251 third-party visual media not
admitted until source/media lineage and transitive licensing are
qualified. H262 is a single previously exposed author-owned
source, NOT fresh blind OOD. No proof of generalized Hala skill or
model release follows from this diagnostic.
