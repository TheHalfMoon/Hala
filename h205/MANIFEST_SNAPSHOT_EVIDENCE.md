# H206 — strict original-source JSON plus immutable audit snapshot

**Draft-only, source metadata only. Not a dataset, new checkpoint, model
training, accuracy evaluation, release or publication.**

H206 integrates the parallel H205 strict manifest parser Draft #226
(commit `05b5959d50163ba4df20dacd6eab00b4d2b443cc`) with the
snapshot-first H205 work on the historical
`ps/h205-fail-closed-manifest-snapshot` branch. The two changes are
both preserved in the resulting *two-parent merge commit*. Crucially,
the SHA pairs are now extracted from **the same snapshotted bytes**
using `parse_source_manifest`, so the H205 duplicate-key/NaN
firewall cannot be bypassed in the independent H204 collision check.
This is stacked on #226, not a competing sibling to H204.

## Real bug repaired

H204 called the independent H198 `census(owner_root)`, then opened
the same five `source-prospects-not-admitted.json` manifest files again
to assemble the SHA pairs compared against H185 known historical hashes.
No check tied the first validated manifest content to the second pass.
A concurrent edit of a source task, provenance flag, or encoded PNG/
decoded-RGBA hash could make a single H204 response describe
**different candidate data from different audit stages** even though
the number of candidates remained 20.

H205 adds a **bounded in-memory exact-bytes snapshot** of only those
five strictly named source-metadata JSON files. Each source file is
checked with the inherited safe-path/reparse/size guards. Its bytes
must match after the H198 census and again after the H185 historical
SHA comparison. H185 now receives SHA pairs parsed from the same
original validated manifest snapshot, not a separate unverified read.
Any differences fail closed with a redacted error; private metadata
bytes, identities, filenames beyond public fixture names, and SHA
values are never printed or uploaded by H205.

This detects mutation across these audit stages, **not** an atomic
filesystem transaction or malicious byte replacement after final
verification. Do not claim completeness or true source/label
novelty. The 20 existing owner images and sealed action-reference
sidecars remain UNOPENED and NOT admitted to TRAIN/DEV/OOD.

## Qualification

H205 synthetic negative controls include two equally counted but
changed valid original-source manifests (task content or PNG hash),
an edit occurring while the historical SHA scan is running, a
redaction canary, and an attempted forged TRAIN release; these
must fail closed without opening any owner media. The inherited
H204 no-media test was updated to admit reads of **only**
`source-prospects-not-admitted.json` and legacy `manifest.json`;
it explicitly forbids reading image files or sealed owner gold.

**Actually observed on owner Windows Python 3.11.15 AFTER
integrating #226 and the snapshot fix:** H185 14 discovered
(2 symlink-privilege skips), H198 14/14, H199 16/16,
H204 13/13 (including 3 strict duplicate-key/NaN mutation tests)
and the snapshot tests 6/6, all in normal and `python -O` modes;
**61 executed passes and 2 skips per mode**, 122 executed passes
plus 4 skips total. `compileall` passed.
The owner-only original-source and historical SHA audit also passed
with the exact current metadata snapshot: **20** unadmitted case
metadata, **6** explicitly selected pre-H173 historical phases,
**7** old manifests, **130** known PNG SHA candidates and
**0** historical decoded-RGBA SHA candidates; all seven selected
legacy manifests lack decoded-RGBA SHA fields.
A separate offline H178 original Git metadata census on a selected
pre-H173 phase observed **3** surviving independent Git checkouts
(3 repositories/revision pairs). This cannot certify the history of
deleted/forked source applications.

These tests never read or upload old media, private reference goals
or old OOD. The free hosted Linux CPU Python 3.11/3.12 workflow is a
**separate pending gate**; no hosted pass is claimed until observed.

## Model readiness continues to be blocked

H204's owner-only selected historical H167–H172 sample found
seven manifests, 130 distinct known PNG SHA values and **zero**
old RGBA SHA fields. This is not the entire past source history.
No prior consumed H173 OOD or sealed future Gantt tasks are read.
H199's immutable admitted TRAIN manifest SHA allowlist is empty.
The next substantive model-training work must independently establish
source-family/old decoded-RGBA provenance, eligible TRAIN/DEV release,
and a fresh genuinely independent OOD before a trained model can
claim improvements. No paid inference or GPU is used in H205.
