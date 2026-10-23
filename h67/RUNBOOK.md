# H67 — CPU-first TRAIN staging, private storage, pinned T4 handoff

**Status:** implemented and mock/real-curriculum CPU tested; **real 10k-image stage:** NOT YET PRODUCED.  
**Parent:** H66 Draft PR #50, commit `1860e516509cfb5bcc0a17aa19cace4a9a6072b5`.  
**No public release or model improvement claim.**

## Why this exists

Actual T4 `hala-h64` was allocated on 2026-10-23 but the provider
terminated it during the dependency setup phase. No completed optimizer
step or newly trained checkpoint was measured. H65 reduced uncontrolled
dependency resolution; H66 fixed the real source-export bug where requesting
5 images produced 68. But the old GPU path still downloaded original TRAIN
ranges and rendered 2,000 first-party scenes, consuming short-lived GPU time
for CPU-only work.

H67 makes training data preparation an explicit, verifiable CPU artifact.
It does not allocate GPU, rerun H61 training, or publish data automatically.

## Operating sequence

### A. Prepare on an existing authorized CPU-only machine

Use a clean **pinned H67 source commit SHA** for the entire workflow,
not a moving branch. Prepare genuine first-party Edge screenshots and
original source TRAIN ranges using H61's existing CPU steps:

```bash
python -m h61.colab_run --repo <pinned-checkout> --work <cpu-stage-work> \
  --steps export,firstparty,curriculum,baseline \
  --images slider=3500,slide_resize=2500,sheet=2500,text_highlight=2500 \
  --target-rows 10000 --max-rows-per-image 1 \
  --family-targets '{"h61_firstparty_bilingual_drag_train":2000}'
```

**Environment qualification:** `firstparty_drag` currently has a Windows
Edge executable constant; an arbitrary WSL CPU container is NOT qualified
to render the complete first-party corpus. Exporting the 4 pinned original
domains on WSL CPU was demonstrated in H66 (a 5-image pilot), but a full
10k-row CPU stage has not yet been created or verified in this turn.
Before running the full sequence, confirm Microsoft Edge and Pillow,
available storage, and permitted temporary retention. The operation can
download many GB; do not start blindly.

H61's baseline, source manifest, media hashes and first-party receipt must
all remain in the same `<cpu-stage-work>` directory.

### B. Build and verify the transport artifact on CPU

```bash
python -m h67.stage build --work <cpu-stage-work> \
  --bundle <work>/hala-h67-train-stage.zip --git-sha <PINNED_H67_COMMIT>
```

The command re-derives the H61 curriculum with its own source verifier,
reads every media byte, and creates a deterministic ZIP and SHA256 receipt.
It refuses symlinks, missing original media, invalid TRAIN metadata, duplicate
paths, out-of-tree paths, excessive bytes, mismatched JPEG/PNG content hashes
and unverified source-derived rows. This ZIP is intended to be **private**;
do not commit it to public GitHub. A matching archive SHA is required on import.

The `h67.stage restore` command is for controlled tests and integration,
not an instruction to overwrite an existing experiment:

```bash
python -m h67.stage restore --bundle <ZIP> --expected-sha256 <ZIP_SHA256> \
  --dest <new-empty-run-directory> --git-sha <PINNED_H67_COMMIT>
```

### C. Authorize and verify private storage before removing staging

If the owner has authorized Kaggle **private dataset storage**, not Kaggle
GPU training, use the already configured owner API token from the environment:

```bash
python -m h67.private_stage \
  --bundle <ZIP> --sha256 <ZIP_SHA256> --git-sha <PINNED_H67_COMMIT> \
  --receipt-dir <durable-receipts>
```

No provider upload was executed for H67 during development. The publisher
requires the dataset's private flag, anonymous HTTP 403, and authenticated
byte-level readback of **every** archive and metadata member. Index-only or
partial byte checks are **NOT** sufficient. Without both checks, retain local
staging and report blocked. The optional
`--delete-archive-after-readback` is only for the specific verified ZIP;
it does not remove source directories and is never implicit.

The current transport is host-upload-based, not direct provider-to-GPU:
an authorized operator must securely retrieve the private bundle before
uploading it to the selected free GPU session. The exact, largest-bundle
`colab upload` size limit has **not** been qualified.

### D. Use exactly one existing, authorized free Colab T4

First independently confirm free-tier conditions, balance, ownership, current
provider sessions and that no other Hala watcher is running. Do not create
duplicate allocations. The following is valid only when a T4 **already exists**.

```bash
python -m h67.upload_stage --bundle <ZIP> --session <EXISTING_T4> \
  --sha256 <ZIP_SHA256> --pinned-sha <PINNED_H67_COMMIT>

python -m h65.safe_diagnostic --session <EXISTING_T4> \
  --pinned-sha <PINNED_H67_COMMIT> --destination <host-evidence-dir> \
  --steps train,evaluate --stage-sha256 <ZIP_SHA256>
```

Host upload success is **only** a transport signal. Inside Colab, the
pinned `h62.remote_h62_launch` rechecks the ZIP SHA, pinned Git SHA,
every archived member's SHA, training-only labels, original source
provenance and full H61 curriculum derivation before touching GPU weights.
It atomically creates the new `run` directory. `export`, `firstparty`,
`curriculum` and `baseline` are prohibited in this staged T4 run.

H65 supervises the existing GPU run and periodically egresses completed
adapter, optimizer and RNG checkpoints. If a T4 disappears, or the
checkpoint is not recoverable, it refuses to report success. Publish the
checkpoint privately and verify actual readback *before* deleting host
recovery files. Qualified evaluation, H63 separate image intervention and
real GUI task success remain required to claim learning.

## Measured evidence (not model progress)

- H66 real source CPU control: 5 requested images -> 5 accepted, 47 original
  actions, 905,969,664 task bytes streamed, 77.300 seconds; no GPU.
- H67 uses deterministic mock byte fixtures to reject changed ZIP media,
  partial private readback, wrong SHA, wrong source commit, unsafe paths
  and unwanted CPU steps on GPU.
- H67 also successfully builds and imports a tiny real H61 curriculum export
  via the actual H61 `load_export` verifier.
- Only tests and source are committed. No real H61 adapter checkpoint,
  full CPU dataset ZIP, Kaggle private storage upload or T4 model result
  exists because of H67 implementation alone.

## Known limits

- A 10k-image CPU-stage build, end-to-end provider upload/download, maximum
  upload sizes and real T4 imports are not yet qualified.
- Host CPU preparation may take substantial time and local staging space;
  only delete after verified durable storage.
- H61 remains a drag-grounding diagnostic. Full click, type, scroll, recovery,
  abstention and multistep computer use are not validated by this stage.
- Keep H67 as a stacked Draft PR until hosted CI passes and independent review
  checks the provider/ZIP data path. Do not merge to main or publish a model.
