# H185 — owner-private old-manifest SHA-field coverage (no old media)

**Status: partial historical metadata-only census, not dataset admission.**

## Purpose

H182 predeclared Pickr prospective TRAIN and interact.js prospective DEV,
with Gantt reserved as a *tentative, never-rendered* OOD source. H184
blocked nine additional prior-public original GUI repository families.
However, an old source-repository census alone cannot catch reusing the
same exact old screenshot in a new purported TRAIN/DEV case.

H185 conservatively extracts only SHA-256 values with exact JSON field
names `image_sha256`, `png_sha256`, `visual_rgba_sha256`, and
`rgba_sha256` from owner-local historical files **named exactly
`manifest.json`**. It does not decode the surrounding JSON into case
records, inspect reference gold/coordinates, open original PNG/JPG/WebP,
read old model predictions/weights, extract labels, or transfer bytes to
GitHub, Kaggle or any other service.

It scans only bounded `private-hNN` phase directories, shallow
depth <= 4, while pruning Git/vendor/source/model/media/gold/prediction
directories. Junctions, symlinks, unexpected root layouts, oversized
manifests and ambiguous file paths fail closed. No files are modified.

Returned raw digest sets are **owner process-memory only**; the CLI
prints aggregate counts and a SHA-only fingerprint, not the actual
private hashes, historical source names or file paths.

## Exact owner Windows source-only finding (2026-10-29)

Ran the scan with CPython 3.11.15 against the authorized existing local
historical Hala workspace:

- Historical phase folders enumerated: **37**
- Shallow allowlisted historical `manifest.json` files read: **64**
- Manifests with explicitly recognized PNG SHA fields: **63**
- Distinct old PNG SHA candidates found: **2,661**
- Manifests with explicitly recognized decoded-RGBA SHA fields: **0**
- Distinct old decoded-RGBA SHA candidates found: **0**
- Metadata-only hash-set fingerprint:
  `35ab1b0757e964cdabcee5f55c91ff6418231609c79325cbd1be03f14c33d278`

The earlier filename-only directory inventory counted 65 manifests;
the narrowed H185 allowlisted reader inspected 64. The difference
reinforces that **this is an incomplete subset**, not a full historical
source or media registry. An old manifest may use different SHA field
names; a digest may be fabricated or stale. No comparison against old
image bytes is performed or authorized.

## New conservative collision fence

`h185.manifest_sha_only.screen_unadmitted_new_case_hashes` accepts only
fresh `image_sha256` and `visual_rgba_sha256` digest pairs, with
**no** prompt, gold or reference action payload. It can reject a future
candidate that matches any owner-local known old PNG or RGBA SHA field
or duplicates another proposed case. It **cannot** prove safety against
unobserved historical images; it never promotes real TRAIN/DEV data.

This is useful immediately as an extra known-reuse exclusion, but
`legacy_rgba_sha_coverage_missing=True` against the current owner
metadata is an explicit unresolved blocker.

## Running only on authorized owner Windows

From the exact branch's checkout:

    python -m h185.audit_manifest_hash_coverage --historical-root C:/Users/Shehr/work/Hala

Expected high-level status:

    H185_OWNER_MANIFEST_SHA_FIELD_COVERAGE_ONLY_NOT_HISTORY_COMPLETE
    H185_HISTORY_COMPLETENESS_UNPROVEN_NO_IMAGE_MEDIA_OPENED_NO_TRAINING

The owner remains responsible for keeping future candidate image bytes
private. Do not check old manifest contents, SHA arrays, images, model
responses or gold/labels into public GitHub. The only committed files in
this PR are source code, synthetic tests and this aggregate evidence.

## Absolute remaining blockers

1. Independently prove complete historical original source/fork/npm/archived
   family coverage and recover **trusted old PNG plus decoded-RGBA SHA-only
   registries** from existing authorized SHA-only attestations (without
   reopening consumed benchmark image/media or evaluation gold).
2. Admit only genuine new native TRAIN/DEV captures from independent
   pinned sources, with bilingual tasks, verified original action state,
   unique original image/RGBA hashes, and strict no-gold model inputs.
3. Only after all data and runtime gates pass: free-only GPU training,
   freeze checkpoint, compare frozen DEV, then ONE independent truly
   untouched Gantt OOD evaluation if its source is still qualified.
4. H173 scored one-shot v3 (2/32 vs 3/32, exact McNemar p=1.0)
   stays permanently consumed. Never reuse it as model selection/training.

**No main merge, release, public model/data, paid inference, deployment or
human-review requirement. Keep Draft.**
