# H73 — Kaggle private CPU dataset admission (research, not training)

**Stacked on H72 PR #56.** This is a no-GPU, no-model-weights,
no-commercial-training verification flow to eliminate repeated transfers of
the separately stored and independently verified H71 private TRAIN archive.

## Measured predecessor

H72 successfully transported the complete **1,759,484,545-byte** H71
archive into a real free Colab T4 in 105 bounded 16 MiB pieces (3 concurrent
CLI operations), with exact remote SHA256
`4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a`.
The subsequent bounded 3-step GPU attempt, despite a successful source
transfer and package bootstrap repair, lasted **1,227.06 seconds** and
returned **FAILED / NO_PUBLISHED_CHECKPOINT**, because the remote session
became unavailable before any host-verifiable optimizer step.

No new H61 adapter, final GUI metric, or release artifact exists.
The H71 Kaggle dataset is still independently recoverable, so no source
reconstruction is needed.

## Architecture

Kaggle Notebooks can attach owner-private datasets by source ID and
persistence is separate from the lifetime of Colab. An attached
`abdulazizshehri/hala-h71-drag-train-10000-private` means the CPU
notebook need not upload the 1.76GB ZIP over Colab CLI.

`h73/kaggle_h73_cpu.py` is a self-contained notebook SCRIPT with:
- owner-private metadata (`is_private=true`), **GPU disabled**, pinned
  private H71 dataset source, research-only note
- a strict **full ZIP byte-length / SHA256** check before source inspection
- pinned exact H71 Git commit `acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe`
  fetched detached with no bearer token
- `h67.stage.restore_stage` in a temporary Kaggle scratch directory,
  including validation of every member and H61 TRAIN rederivation
- immutable 10,000-row curriculum SHA
  `9888ac9d3b5ec7b8efe677287d54dc350c222a1804619e6217a84ec8f95eddcd`
- a small deterministic `/kaggle/working/hala-h73-cpu-admission.json`
  with provenance and pass/fail; **never publish screenshots or weights**.

Any unavailable mounted input, GPU, wrong source digest, wrong pinned Git
SHA, inadequate disk or invalid rederivation fails closed. H73 does not
change the intentionally strict `reject_kaggle_runtime` guard in H61's
actual model trainer. No Kaggle GPU training authorization is implied.

## Execution frontier

Actual host unit tests on Ubuntu CPU: **7/7 PASS**. Owner Kaggle private
kernel: `abdulazizshehri/hala-h73-cpu-train-source-admission`, version 1
created CPU-only. Version 1 failed the attached source path lookup and
created no training output; version 2 adds bounded attachment name diagnostics
to resolve the provider's actual input layout without reading private
content into logs.

**Acceptance criterion:** successful private kernel result and an H67
`RESTORED_TRAIN_STAGE` from the *attached* Kaggle input bytes. Until then
this is a prepared CPU admission candidate, not validated portable storage.

## Compliance gate

Kaggle's current general Terms of Use include a personal/non-commercial
services restriction. H73 is a non-training source-integrity research
experiment only. Commercial Hala or Aladdin model development requires a
separately eligible training provider or a specific authorized arrangement.
Do not silently remove H61's Kaggle training rejection or mischaracterize
free Kaggle GPU eligibility as a general commercial compute grant.

Sources:
- https://www.kaggle.com/terms
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- https://www.kaggle.com/docs/notebooks

Keep this PR **Draft**, no model release and no protected-branch merge.
