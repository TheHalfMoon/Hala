# H179 — owner Windows original-source novelty preflight

**Source-only candidate screening. Zero new TRAIN, DEV, OOD images, native
rewards, model inferences or model training. No release authorization.**

## Why this phase is necessary

H174 pinned three immutable MIT original-source candidates (Pickr,
interact.js dropzones, medium-zoom vanilla); H175 independently verified
real Windows Edge behavior for interact.js. H176–H178 built progressively
stronger source-history barriers and identified 22 historical original
Git checkouts on owner Windows. Before collecting any new dataset, screen
the actual three donors against those historical source Git identities
and against the publicly documented consumed original-app families.

H179 **reuses**, rather than rewrites:

- `h174/source_candidates.py` for the three frozen prospective Git, HTML,
  LICENSE bytes and original checkout validation.
- `h178/owner_git_inventory.py` for owner-local, read-only history Git
  repository identity and SHA-40 census; no old media access.
- `h176/source_family_attribution.py` and
  `h177/public_origin_pins.py` for known previous original repositories
  including tingle, tiny-slider, Split.js npm, nine publicly frozen past
  original apps, and old H173 Git/HTML hashes.

The screen rejects exact repo reuse, case variants, reused Git SHA-40,
reused known H173 HTML SHA, duplicated prospective repo/source IDs and
malformed immutable source pins. An optional independent byte check requires
**all three** original pristine donor checkouts simultaneously. Only
tracked original HTML/LICENSE files with immutable committed bytes qualify.
Nothing is copied into Hala's Git repo.

## Reproduce locally on authorized Windows owner machine

Run the CLI from the Hala checkout (adapt the four paths to the machine):

    python -m h179.audit_pinned_candidates \
      --historical-root C:/Users/Shehr/work/Hala \
      --pickr-donor D:/Hala-H174-Donor-Pickr-SourceOnly \
      --interact-donor D:/Hala-H174-Donor-Interact-SourceOnly \
      --mediumzoom-donor D:/Hala-H174-Donor-MediumZoom-SourceOnly

The donor flags are optional as a **group**; if any donor flag is present,
all three must be specified. History-only source identity screening without
the three flags reports `donor_file_pins=0` and never claims original
byte qualification. Missing historical roots or any mismatch fail closed.

## Exact owner result (2026-10-29)

Source identity preflight on original donor trees returned:

- Three prospective immutable original-source Git/LICENSE/HTML byte
  attestations passed. Every tree reported a pristine checkout at its
  frozen revision and canonical original Git remote.
- 22 locally retained earlier historical original Git checkouts belonging
  to 22 distinct upstream repositories were compared, without opening
  old screenshots, gold, predictions, trace files or datasets.
- Candidate source SHA-256 identity fingerprint:
  `3b11c53357fac7ed39b02fea0697f96f18f92eebe7c0bbe4dce001e520fd55ab`
- Owner original history SHA-256 identity fingerprint:
  `bac6e942f199453e2f254af049ddbc68c61c755a31622a703f321bcdc35eb67a`
- The command explicitly returned
  `HISTORICAL_COMPLETENESS_UNPROVEN_NO_TRAIN_DEV_OOD_ADMISSION`.

These fingerprints are identities of selected *metadata*, not proof of
whole-history completeness, historical original license validity, app
novelty under forks, independence at the image level or a model score.

## Remaining gates

1. Independently establish all historical original repo/fork, npm, Git,
   original HTML and owner-only previous image/RGBA hash identities.
   Present local checkouts can miss deleted/moved or archived source
   families. Never reopen or recreate consumed H173/OOD media to infer them.
2. Preregister **source-disjoint** TRAIN, DEV and OOD_RESERVED source
   families; original OOD is *source reservation only*, no captures before
   training freeze. Current H174 three pins are candidates, not assigned.
3. New genuine owner-native reference actions and distinct visual state
   captures for only TRAIN and DEV. Enforce bilingual instructions, source
   identity manifests, unique screenshot/decoded-RGBA fingerprints and
   independent exact replay under the same source and rendering setup.
   No owner-only reference coordinates or labels in model inputs.
4. Only after all gates pass: adapt on allowed zero-cost free GPU,
   select only from DEV, freeze candidate/model/eval protocol, and assess
   genuinely fresh OOD once. Do not reuse any consumed H173 v3 / earlier
   OOD or claim significant generalization based on 2/32 versus 3/32
   (exact McNemar p=1.0).

No paid compute, publication, protected-main merge, dataset admission,
model weights, release, or human review gate. Keep this PR Draft.
