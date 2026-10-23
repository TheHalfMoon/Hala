# H72 — Resumable and byte-verified Colab source-stage transfer

**Parent:** H71 PR #55. **No merge, no model release.** Source-bound final
H61 TRAIN ZIP is already owner-private in Kaggle and independently recovered.

## Measured reason for H72

The official Colab CLI's single-file `colab upload` rejected the
1,759,484,545-byte H71 archive with **HTTP 400**. No original full ZIP
appeared on the T4. A free T4 allocation was stopped after that refusal:
no optimizer update, model checkpoint or evaluation was performed.

An independent transfer test on a **Colab CPU session** then proved that
the same first-party CLI can upload one **16,777,216-byte** source segment.
Source/remote SHA256 of that segment were identical:
`75aa756e6a4d398181f99b9b26dbfc92f906769810972991f7915e20273bb2b9`.

## Code and qualification

`h72/chunk_transfer.py` reads a source archive without creating a second
full local copy, checks its **complete** pinned H71 SHA256 before using any
provider, enumerates consecutive bounded 16MiB pieces with SHA256 digests,
creates an exact manifest bound to the **original** H71 source Git commit
and ZIP byte size, and uses official `colab exec`/`colab upload`.
Any existing remote chunk is *skipped only after Colab computes its matching
SHA256*, so interruption can be resumed without trusting filenames.

`h72/remote_stage.py` runs on the existing authorized Colab session,
validates exact paths/counts/contiguous size and per-chunk digest, and
atomically reassembles the ZIP only after the complete final SHA256 matches.
The final source archive path is the unchanged H67 contract
`/content/hala-h67-stage.zip`. It refuses invalid or mismatching
existing ZIPs, missing/corrupt chunks, unbounded manifests and insufficient
reassembly space. H67 must **still independently rederive H61 TRAIN media
before any weight update**.

### Actual real Colab CPU integration

Using 16 MiB plus 101 extra source bytes from the immutable H71 ZIP as
a controlled small transport sample, the end-to-end host CLI produced:

- 2 contiguous parts, 16,777,317 total bytes;
- `REMOTE_ZIP_SHA_VERIFIED`, exact sample SHA256
  `fae3857e0f34a48b0be4bebf7092b2e749f3d929fd13eedad69a32ebaab8046f`;
- an independent repeat reported **2/2 already verified** chunks and did
  not upload them again.
- New deterministic adversarial local tests: **7/7 PASS** including
  corrupted-chunk rejection, manifest path traversal, atomic reassembly,
  wrong pinned original SHA and exact byte-offset continuation.
- No GPU model training, official benchmark or inference was run on these
  sample bytes. Do not confuse this sample with an entire ZIP.

### Full H71 source-stage transfer commands

For a previously allocated, owner-authorized and genuinely free **T4**,
run on the trusted host where this repo and the approved private ZIP exist:

```bash
python -m h72.chunk_transfer \
  --source /path/to/hala-h71-train-10000.hala.bin \
  --sha256 4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a \
  --git-sha acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe \
  --session <EXISTING_AUTHORIZED_FREE_T4>
```

This will require **105** files of at most 16MiB and can consume meaningful
runtime on an ephemeral free VM. No availability or running time is
guaranteed; do not allocate a paid tier or auto-loop new sessions.

Once the remote ZIP reports `H72_STAGE_READY` and matching final SHA,
the H65 supervised diagnostic's existing staged-data path may run
`--steps train,evaluate --stage-sha256 <ZIP_SHA256>`.
H67 on the remote VM then verifies all ~15k stored members and
re-derives the original 10k H61 training rows before any optimizer
step. H65 host snapshot must verify actual checkpoint readback.

**Current status at H72 publication:** full 1.76 GB GPU transfer not yet
confirmed and no new H61 adapter checkpoint exists. The Kaggle original
remains the source of truth, not the ephemeral Colab VM. Free managed
Colab runtimes may be terminated and have usage restrictions. Avoid
unbounded automation or unsupervised long-lived sessions.

The host uploader does not send Kaggle credentials or copy the owner's
API tokens into the Colab instance; all bytes originate from the already
qualified local source ZIP.
