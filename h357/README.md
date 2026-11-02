# H357 — from-scratch original ControlGarden TRAIN/DEV research candidate

**NO TRAIN ADMISSION. NO MODEL TRAINING. NO BLIND OOD. NO RELEASE.**

This PR moves Hala toward eventual lawful bilingual model adaptation without
reusing any H173 consumed evaluation, H319 publicly exposed comparison
scenes, third-party HTML/JS, logos or model outputs. The small
`controlgarden.html` application is authored here entirely from scratch:
four stateful native buttons, each showing one CSS-drawn shape and color,
no external image/font download, tracking, network dependency or third-party
copied source. Its four actual native browser buttons update the selected
state when clicked. The screenshot does not show the gold answer, native
button IDs, selectors or other oracle data. Only original pixel raster and
the EN/AR user instruction are intended for any eventual model input.

`source_candidate.py` enumerates **64 unique visual scenes** in one
original application family. It chooses shape and color permutations
without random collisions, half English and half Arabic, and 64 actual
real-browser reference clicks. It checks the original native state
after each click and requires two identical screenshots of the same
pre-action state. It produces an ephemeral, private/caller-owned
`candidate-manifest.json`, `candidate-rows.jsonl` and PNG files
with SHA256 content identities. All these generated files live only
inside the test's temporary working folder and are deleted after
qualification. **Only source and tests are committed to Git**.
Generated labels are TRAIN/DEV *candidates*, never held-out tests.

Arabic is supplied as UTF-8 prompt text, not as unqualified Arabic
glyph rendering in the GUI screenshot; CSS shapes need no Arabic
font shaping. The original app may still need independent human
legal review of screenshot image rights, platform system fonts,
and full ML/model weight distribution permission. It is *one*
single family and provides neither source-family-diverse training
nor independently source-disjoint DEV or test.

## Original browser qualification

```sh
python -m unittest tests.test_h357_original_controlgarden_candidate -v
```

On Windows the test uses the locally installed Edge (headless).
On Ubuntu the dedicated CI matrix uses pinned Playwright Chromium.
The public original source is in Git; no private benchmark content
is consulted or uploaded. A test repeats all 64 pre-action screenshots
byte-for-byte and drives the reference click through the browser:
64 native state transitions and 64 verified original-screen captures.
Rows are strict H199 one-click action JSON in screenshot coordinates.

## Safety and research distinction

`candidate-manifest.json` uses explicit schema
`hala.h357.controlgarden.original-candidate.v1`, role
`TRAIN_DEV_REVIEW_CANDIDATE_ONLY_NOT_ADMITTED`, and flag
`official_train_admitted:false`, `independent_rights_reviewed:false`,
`fresh_blind_ood:false`, `can_train:false`, `can_release:false`.
It is **not** a `hala.training.manifest.v1` official TRAIN manifest
and cannot reach any official H199 Qwen GUI LoRA training entrypoint.
The production `ADMITTED_TRAIN_MANIFEST_SHAS` registry is EMPTY;
even `rights_cleared:true` in a self-declared JSON is refused.

No training step, free or paid GPU, inference, model weights, OOD
benchmark, external source license grant, real patient account or
patient data are used. No claim that 64 examples would generalize or
yield learned improvement. Further human work: independent legal
rights/title signoff (Issue #384), independent sealed unseen
bilingual OOD custodian (Issue #385), multiple genuinely disjoint
source-family TRAIN/DEV candidates and separate approved official
H199 manifest admission before any actual training/evaluation.
