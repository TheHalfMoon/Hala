# H359 — Native H199 message parity across two real original candidate families

**STATUS: SOURCE-ONLY QUALIFICATION — NOT A TRAINER OR RELEASE.**

H357 ControlGarden and H358 NoteRelay are two authored original visual
TRAIN/DEV *candidates* with 64 and 128 screenshot/action rows,
respectively. They are distinct original app families but neither
has been independently approved for rendered screenshot ML training,
and both are openly exposed, not blind OOD.

## Confirmed incompatible legacy training format

The old `training.artifacts.validate_action_completion` expects a
logical action such as `{"type":"click",...}`, with DOM element
reference hints. H199 requires screenshot-native strict one-action
JSON such as `{"actions":[{"kind":"click","point":[x,y]}]}` and
separately focused `type_text`. Converting by guessing element
references would violate the screenshot-only/native action contract.
Neither old `training/scripts/train_adapter.py` nor its legacy
manifest pathway should be fed these candidate rows without a
separate, qualified H199-native trainer.

## Implemented bounded, read-only bridge

`h359/native_candidate_bridge.py` uses H199's
`training_messages_for_synthetic_contract` to validate exactly
the visible screenshot placeholder, user-only task, one native
assistant completion and action grammar. It validates original
source SHA256 of BOTH frozen authored Git HTML assets, manifest's
**unadmitted** candidate role/split, 64/128 unique screenshot
content digests and IDs, actual PNG dimensions and file SHA256,
real 32/32 and 64/64 EN/AR row totals, and expected action kinds.
H357 deliberately uses a different not-admitted `split` than
its candidate-manifest `role`; both must match their precise
original contracts. H358 expects 96 `click` plus 32
`type_text` actions. No secret DOM, gold or model output enters
the model-visible message. It explicitly tests that H199 native
completions are rejected by the legacy `type` logical action
validator, never implicitly mapped or promoted.

Each test creates H357/H358 real browser PNGs and labels only
inside temporary folders and deletes them. The bridge builds only
in-memory prompt/action message pairs, discards them and returns a
small counts-only audit. It does **not** save a model-training
manifest, training dataset, JSONL export, Qwen processor input,
tokenized records, checkpoints or weights. It never invokes GPU,
model, private blind OOD or training.

### Qualification

```sh
HALA_H359_BROWSER_QA=1 python -m unittest tests.test_h359_native_candidate_bridge -v
```

Dedicated CI requires both original real browser sources and Pillow
on Windows and Ubuntu with Python3.11/3.12. It rechecks 192 actual
original native reference rows (96 EN/96 AR), 192 PNG byte identities,
mask-free H199 messages and same-task semantics, verifies refusal
of forged TRAIN admission, changed media, and legacy completion
conversion. Tests without installed browser/Pillow may skip only the
actual-browser case in normal repository-wide unittest discovery;
the dedicated CI installs required dependencies and opts in.

## NO-GO gates

An original source plus successfully validated H199 messages is
still **NOT independent legal permission**, official TRAIN
admission or a prepared Hugging Face Qwen processor training batch.
The production H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY,
H355 reviewer trust registry EMPTY, H356 blind OOD custodian
registry EMPTY. Issues #384 (independent screenshot/weight rights)
and #385 (genuinely unseen source-disjoint bilingual benchmark)
remain unresolved. No Qwen weights improved, no new model
training, no observed learned adapted-vs-base gains. Prior H351–H354
exposed-source pixel postprocessing gains are unrelated to learned
weights. Hala remains NO-GO for release.
