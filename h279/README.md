# H279 — owner-private pretrained Qwen relative-to-native action probe

**Pre-registered BEFORE running any H279 GPU experiment.**
This is a bounded original-source diagnostic, **NOT independently blind
OOD, model improvement, admitted training, adapted-model comparison, or
model release**. All H279 files in the public repository are code and
synthetic-only tests; the genuine model and first-party rendered media
remain inside owner-private free Kaggle T4 volatile memory.

## Motivation and frozen protocol

H273's previously exposed H262 Stockroom RTL scenes produced four
real pretrained Qwen3-VL image-conditioned generations: two native-pixel
predictions yielded valid clicks but both missed the first `+1` button;
the two prospective relative-frame predictions did not pass strict JSON.
H274 v2 later produced `STRICT_EXACT_VALID` grammar-constrained relative
JSON **2/2** on these same known scenes; there were **zero native clicks**
in that run, so it could not establish localization or action success.

H279 freezes the **same** H274 pretrained model and Qwen tokenizer
revision `89644892e4d85e24eaac8bacfd4f463576704203`,
`Qwen/Qwen3-VL-2B-Instruct`, Transformers 5.16.1, model-time
lm-format-enforcer grammar from pinned H273, max output 160 tokens,
fixed relative prompt, torch manual seed 274, first-party screenshot
960x720, H271 original two scenes in EXACT order:

1. `ar_sorted_tape`: Arabic RTL, sorted, filtered Paper tape.
2. `ar_sorted_gloves`: Arabic RTL, sorted, filtered Exam gloves.

Source checkout must be H273 exact
`53d7fb25e113f8f2cd337bb4aebeafac026d1a22` and the H262 original
HTML SHA256 `3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.
No new test scene, adaptation, source or prompt change after observing
results. These examples are PUBLICLY EXPOSED and NOT a new independent OOD.

## Real action protocol (precommitted)

For each scene, fresh isolated offline Chromium context, source-owned
sort/language/filter, blur the known search input without reading task
targets, take the fully decoded actual PNG, then one real pretrained
vision-conditioned model generation with exactly the H274 RELATIVE_PROMPT
and pinned H273 normalized token grammar. No DOM, target box, gold,
selector or known coordinates enter the model policy callback.

Fail closed on source/model/hardware/viewport/PNG drift and unchanged
screen guard. Require **strict exact normalized JSON** (0..999)
and use the ALREADY preregistered H273 relative-to-pixel converter.
Malformed JSON triggers **zero** actions; do not repair it or use a
fallback target. On valid action, dispatch at most one genuinely native
H263 browser action with screenshot locked; `abstain` is non-action.

Only **after model inference and action** a separate original-source
oracle checks: first visible stock item count increases by exactly
one, other visible source items are unchanged, and the *model-chosen*
pixel lies in the first `+1` button bounding box. Never alter the
model's click to the target box; a wrong click stays wrong. If no
action, score task failure, not an inferred hit. Report exact aggregate
booleans by the two preregistered scene *labels only*, no coordinates.

Private output:
`hala-h279-private-relative-native-aggregate.json`, only after
two real GPU model generations, two grammar callbacks and a fully
verified schema. Public GitHub may receive ONLY independently
validated sanitized booleans/counts; never the raw private JSON,
model response, coordinate, screenshot/pixels, or model weights.

## Budget / privacy requirements

- Owner-private Kaggle `is_private=true`, **free NVIDIA T4**,
  non-commercial only. New dedicated code-only notebook
  `abdulazizshehri/hala-h279-private-relative-native-actions-t4`.
- Exactly one Python script and one private Kaggle metadata file,
  **zero** attached datasets, model, kernel sources or external media.
- Model HF weights and source scenes load only into ephemeral
  remote `/tmp`, deleted at exit. Never download Qwen weights to
  owner's Windows (C:/D:) or Mac.
- On owner Windows, use short-lived dedicated D: checkout+packaging
  folders; delete only those owned folders after submission.
- Retrieve Kaggle output **only** through
  `--file-pattern '^hala-h279-private-relative-native-aggregate[.]json$'`
  and delete the tiny downloaded JSON after validating safe categories.
  Do not fetch unfiltered Kaggle output/logs or persist media.
- Test locally with Python stdlib synthetic guards before any private
  upload; verify real GPU results separately. Hosted CPU CI
  success alone **cannot** certify actual model inference.
- Remain DRAFT/unmerged. H199 training manifest stays EMPTY; H173
  original 32 scored OOD permanently SEALED; H212 procedural scored
  cases BARRED; H250/H251 third-party media QUARANTINED until complete
  provenance/licensing and PNG/decoded-RGBA exclusion.

## Results

At plan freeze, **no H279 GPU result yet**. All future claims must
cite exact Kaggle completion and strictly validated aggregate.
