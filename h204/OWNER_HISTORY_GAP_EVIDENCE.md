# H204 — actual training-readiness blocker: legacy PNG-only provenance

**Status: DATA AND TRAINING ARE STILL NOT ADMITTED.** H204 is an
owner-only metadata audit, stacked exactly on H203 Draft #224 head
`a02ad99c065313f61b76e3ae72ba86eb8a9e3b46`. It never uses
a GPU/model, old source media, saved owner images, sealed gold or OOD.

## Concrete new control

H198 previously counted the 20 original GUI source prospect manifests
and verified new case PNG/RGBA metadata. H185 separately counted old
metadata-only PNG/RGBA fields, but H198's optional history argument
admitted just one historical root, and the observed field counts did
not enforce a new-case collision screen for the subset of explicitly
selected historical phases.

H204 composes **both existing source-only validators**, and runs the
H185 *actual known-old SHA exclusion* against all **20** new quarantined
candidate hashes. It prints only bounded aggregate counts. It accepts
only explicitly supplied safe **pre-H173 historical phase directories**,
not a broad parent drive; specifically rejects `private-h173` and
`Hala-H173-Private` so the one-shot consumed H173 OOD is never scanned.
Source and reference action sidecars remain unopened; test fixtures
intentionally put invalid content in reference files.

This gate explicitly **NEVER** sets `complete_source_history_proven`,
`complete_old_rgba_proven`, or an official TRAIN release SHA. Even if
all selected old manifests happen to have RGBA fields, that does not
prove complete historical source/media coverage.

## Actual owner Windows source-only evidence — October 29, 2026

With the H204 owner-only CLI on Windows CPython 3.11.15, and ONLY the six
explicit pre-H173 metadata phases H167–H172:

    H204_TRAINING_RELEASE_BLOCKED_PARTIAL_SHA_ONLY_HISTORY
    new=20 legacy_phases=6 manifests=7
    legacy_png=130 legacy_rgba=0 manifests_missing_rgba=7
    NO_ADMISSION_NO_GOLD_NO_MEDIA_NO_OLD_OOD_NO_MODEL

This is a **partial historical subset**: 6 older phases, seven bounded
manifest files, 130 distinct historical PNG SHA-256 candidates, and zero
recorded old decoded-RGBA SHA-256 candidates. **All seven manifests**
in this subset lack the expected old decoded-RGBA fields. H204 checked
that the 20 new prospect pairs do not collide with these 130 known old
PNG hashes. This DOES NOT establish that they are novel relative to
all prior media or source forks; it must not promote any image to TRAIN.

Remaining independently necessary source/data qualification:

1. Prove a complete historical source-family/fork/npm/Git inventory.
2. Establish complete historical decoded-RGBA exclusion by a separately
   approved, privacy-preserving provenance process; do not replay old
   consumed H173 OOD or open old gold to manufacture that proof.
3. Freeze a newly verified and source-disjoint TRAIN/DEV role manifest
   with explicit identity, SHA evidence and independently sealed DEV gold.
4. Only after those checks, review a nonempty immutable manifest SHA
   allowlist in H199 on a future Draft PR, and then perform true
   multimodal model fine-tuning and fresh independent evaluation.

H204's initial Windows tests: 10/10 synthetic mutation/privacy cases
on CPython 3.11.15, normal and `python -O`. Hosted CI is a separate
gate and must not be counted as successful before an actual completion.

No raw SHA, owner paths, original instruction/gold or image bytes were
included in GitHub. No owner media uploaded, no new checkpoint, no new
training steps, no old H173 OOD, no paid inference, no merge/publish.
