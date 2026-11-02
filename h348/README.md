# H348 — decode exactly the hashed training image bytes

**Status: engineering-only Draft. Hala model and release remain NO-GO.**

## Reproduced source-code issue

The training entrypoint previously verified each image's path by calling
`sha256_file(path)`, and then independently called `PIL.Image.open(path)`.
A mutable path could change between those operations: the hash could
validate an admitted image while the processor receives different pixels.
This affected both the preprocessor audit (`check_media`) and the
training-time processor (`prepare_inputs`). It was separate from H347's
fixed manifest/checkpoint identity issue.

## Repair

`verified_media_bytes` reads the candidate image **once**, capped to the
existing `MAX_IMAGE_BYTES` budget (32 MiB), validates SHA-256 against the
row's reviewed media digest, and returns that immutable byte snapshot.
Both `check_media` and `prepare_inputs` decode from
`PIL.Image.open(BytesIO(verified_bytes))`, never from a later path read.
The existing PNG/JPEG and pixel budget checks remain. Other consumers
of `sha256_file` are unchanged.

## Evidence and limits

`tests/test_h348_verified_training_media_snapshot.py` includes four
entirely invented CPU tests. Two deterministically replace the file
on disk at the exact moment Pillow opens it; they prove both the
preflight image decoder and the actual processor input use the verified
original pixels, not the replacement. Negative tests reject changed
digests and out-of-budget bytes. No torch GPU, model weights, real
screenshots, private labels or external data enter these tests.

Owner-local Windows tests at this initial checkpoint:
- H348 tests: **4/4 PASS**
- Related H346/H347/integrity/H54/H61/H199/H82 tests: **78 tests;
  75 PASS, 3 SKIPPED**
- `git diff --check`: clean

The already existing pinned, read-only H346 CI matrix runs the new tests
on Ubuntu and Windows with Python 3.11 and 3.12, in addition to the
inherited PR checks. Exact-head hosted CI on H348 is required before
calling H348 qualified. No shortcut or fake source/browser pass.

## Unresolved external model blockers

Production H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains empty: no
independently authenticated legal rendered-image ML rights, lawful
separate source-family TRAIN/DEV, new custodied never-public blind OOD
or bilingual BASE-vs-ADAPTED native-state Qwen improvement exists.
No free GPU dispatched. Do not merge, publish, release or deploy.
