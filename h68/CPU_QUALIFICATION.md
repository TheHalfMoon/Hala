# H68 — CPU first-party bilingual qualification and strict full-TRAIN gate

Status: **2,000 ORIGINAL BILINGUAL CPU SCENES COMPLETE; PRIVATE KAGGLE BYTE-READBACK AND RESTORE VERIFIED; NO TRAINED H61 CHECKPOINT**.
Parent: H67 Draft PR #51, head `37635b76a4a7125d89359563d4c1b1a92a565e79`.

## Why H68 exists

H67 proved the CPU stage transport but used synthetic data. H68 adds an
operator-configurable, strictly validated number of real first-party browser
scenes, and a full-TRAIN readiness check that never treats a smoke-test subset
as an approved 10k-row diagnostic curriculum.

The genuine `h61.firstparty_drag` generator authors real local HTML pages,
renders them in Microsoft Edge, audits colored drag handles and targets
against exact screenshot pixels, and creates a bilingual original TRAIN
family. None of these assets are official held-out evaluation answers.

## Real Windows CPU smoke test on October 23, 2026

```powershell
python -m h61.colab_run --repo <reviewed-H68-checkout> \
  --work <isolated-temporary-work> \
  --steps firstparty,baseline,curriculum \
  --firstparty-scenes 8 --target-rows 8 --max-rows-per-image 1 \
  --family-targets-file <JSON mapping firstparty family to 8>
python -m h67.stage build --work <isolated-temporary-work> \
  --bundle <separate-output.zip> --git-sha <exact-reviewed-H68-SHA>
python -m h67.stage restore --bundle <separate-output.zip> \
  --expected-sha256 <full-64-hex-SHA256> --dest <new-empty-directory> \
  --git-sha <same-exact-reviewed-H68-SHA>
```

Observed in the first completed run:
- 8 of 8 first-party scenes accepted, 4 Arabic and 4 English.
- 8 H61 curriculum rows created from actual Edge screenshots.
- H67 exact transport: `CPU_STAGE_CREATED` -> `RESTORED_TRAIN_STAGE`.
- Archive: 152,971 bytes, 22 SHA-verified files.
- ZIP SHA256 `3af891fd46511e346990d471a7cfc64f9e2e7e1eaeb3b41edf29d58555f99d5b`.
- H61 derived dataset SHA256 `075529de3f809495459be3eeb9e5df8b5f7d82a9b7a09c4c63ba6be6b443255c`.
- Underlying source Git commit `f92d42de8d820f7997953399e9812a75381a53b1`.
- **This is a TRANSPORT PILOT and not a model training run.**

## Complete 2,000-scene run — real images and independent private readback

On the Windows host using Microsoft Edge batch browser capture, the H61
first-party generator **completed all 2,000 scenes** (1,000 Arabic and
1,000 English), with **zero rejected scenes**. The full CPU session
completed in **772.04 seconds**, with no Colab/GPU allocation.

- Original first-party labels SHA-256:
  `94c9b943f79264ceccf5d04a2daf8278e7cb02589ce5edbddee8a3e557daa4e6`.
- H61 CPU curriculum: exactly 2,000 rows and 2,000 unique screenshot media,
  SHA-256 `935a5b90be42701b572f6d2e5dea94237f18e1883121c5c9dd045a15eebf464e`.
- H67 immutable CPU bundle: 4,006 files, 36,601,923 bytes, ZIP-wrapper
  SHA-256 `e396ae08e7061411580801577a763f5dc2f15e872e2c2a508a256d368e982a0a`.
- Source pinned inside the archive to `bcb26b589f79797c00a73d43378c29f299a4c25f`
  (the immutable verified commit used to build it, not a future branch tip).
- Kaggle dataset private owner resource:
  `abdulazizshehri/hala-h68-bilingual-drag-2000-private`.
- Real Kaggle CLI `datasets create`: exit 0, private dataset successfully
  added. Authenticated `datasets files` lists the full binary (36,601,923
  bytes). Anonymous `GET /api/v1/datasets/view/...`: **HTTP 403**.
- Independently downloaded the actual provider binary using authenticated
  `kaggle datasets download -f` into a new directory and compared its full
  SHA-256: **byte-identical** to the original checksum.
- `python -m h67.stage restore` on the **provider-returned** archive passed
  the original H61 curriculum re-derivation and member integrity checks,
  restoring 2,000 TRAIN rows and 2,000 screenshot media.

**Claim boundary:** this is a separately qualified *first-party bilingual
family* and private recoverable dataset, **not** the full H61 TRAIN
dataset. It contains no completed original DragOn four-domain 11,000-image
CPU export, no finished 10,000-row mixed curriculum, no trained H61
checkpoint, and no improvement in GUI accuracy. A `PRIVATE` word in a
Kaggle dataset name is not itself a privacy proof: the anonymous 403 and
owner-authenticated byte return are the evidence.

## New full-readiness gate

```bash
python -m h68.readiness --work <cpu-stage-work>
```

`READY_FOR_PRIVATE_STAGE_BUILD` requires all the following:
1. Exactly 3,500 slider, 2,500 slide_resize, 2,500 sheet,
   2,500 text_highlight original screenshots; each counted on disk and
   bound to the pinned upstream original TRAIN revision and per-domain
   source receipt and labels hash.
2. Exactly 2,000 audited first-party images: 1,000 Arabic/1,000 English,
   with complete first-party labels and no rejected scene.
3. A source-rederived H61 TRAIN curriculum with exactly 10,000 rows
   and its original per-media SHA integrity proof; the approved 2,000
   first-party family target and English/Arabic breakdown are checked.
4. Original source TRAIN-only receipts and explicit absence of held-out
   evaluation/test sources. It must not run weights or allocate compute.

Anything missing returns **NOT_READY** and a nonzero exit. Do not overwrite
existing data or combine different source SHA versions silently.

## Operating boundaries

- The stage ZIP is private training material. Never commit screenshots,
  generated HTML, credentials, checkpoint bytes, or held-out answers.
- Kaggle's old H50/H57/H60 private datasets were enumerated read-only and
  may contain reusable original source, but **do not assume** they contain
  this exact H61 curriculum. Their byte-level contents need qualification.
- Private publication of the eventual full H67 ZIP requires real authenticated
  privacy AND full byte-for-byte provider readback before any host deletion.
- `--firstparty-scenes` supports exact multiples of 8 between 8 and 2000.
  On already-populated output it refuses a mismatched scene-count receipt.
  Incomplete accepted images are not reported as a completed CPU build.
- The 2,000-scene Edge CPU qualification must be given a real completion
  receipt before being called qualified; partial PNG counts are progress,
  not model data maturity.
- Colab status at previous final check: 0 active sessions, 0.00 units/hour.
  No GPU training is launched by H68.
- Full original 11k images, 10k-row training set and live H61 model
  convergence/evaluation remain separate uncompleted steps.

The CPU smoke test and H68 readiness tests are reproducible with no GPU.
Keep this change as Draft pending final hosted CI and review.
