# H186 — H141 historical drag-phase PNG exclusion aliases

**Source-identity and SHA-metadata hardening only. No dataset admission,
training, model calls, benchmark OOD capture or publishing.**

## Why this was necessary

H185 inspected only PNG hash fields `image_sha256` and `png_sha256`;
the owner-local H185 manifest scan returned 2,661 distinct old PNG SHA
candidates from 64 allowlisted manifests in 37 historical phases.

Earlier public original-source H141 documents reveal a different source
capture schema. `h141/interactive_source.py` stored two genuine original
.PNG frame names and their sha256 per case:

- `at_rest_image` + `at_rest_sha256`
- `during_drag_image` + `during_drag_sha256`

`h141/audit.py` tied both SHA values to the original PNG paths, and
`h141/audit_visible.py` explicitly excluded **16 previously seen
original frames**, since the H141 oracle-guided diagnostics were neither
TRAIN nor sealed BLIND eligible. These are old *PNG* byte hashes, not
historical decoded RGBA SHA hashes.

H186 copy-first expands only the allowlisted H185 PNG hash-key regex to
include these two precise aliases. It does **not** accept generic
`sha256`, `model_sha256`, `license_sha256`, artifact SHA,
source Git SHA or unrelated cryptographic digests as image fingerprints.

## Reproducible source-only owner results — 2026-10-29

On the authorized Windows machine, re-running the existing H185
read-only manifest SHA census on H186's code found:

- Historical phase folders enumerated: **37**
- Allowlisted `manifest.json` files: **64**
- Distinct old PNG hash candidates: **2,677** (previous 2,661 + **16**)
- H141-style `at_rest_sha256` field occurrences: **8**
- H141-style `during_drag_sha256` field occurrences: **8**
- Recognized old decoded-RGBA SHA hashes: **0** (unchanged)
- Revised SHA-only metadata-set fingerprint:
  `d007f09b919c285acae936737cb3c3294cc48ede863bbf5e67e2bfc60c4ff030`

The 16 additional old SHA values stay only in private process memory;
neither their bytes nor any old screenshots/gold/reference paths or
predictions are disclosed to GitHub.

The H185 conservative `screen_unadmitted_new_case_hashes` now rejects a
new proposed TRAIN/DEV case if it matches either old H141 alias, even
if the proposed case uses the ordinary `image_sha256` field name.
The result remains **no data admission**, regardless of unique hashes.

## Owner-only command

    python -m h186.audit_old_drag_png_aliases --historical-root C:/Users/Shehr/work/Hala

This displays only aggregate old SHA-field counts and the revised
SHA-set fingerprint. The companion H185 command continues to work
with the expanded exact SHA-field allowlist. Both return no model/data
approval.

## Remaining absolute blockers

Observed source families and stored legacy manifest SHA fields are only
partial, neither a trustworthy complete old PNG provenance index nor a
complete old decoded RGBA SHA index. The unchanged zero-RGBA count
**does not mean historical PNG images were unique**: different encoded
PNG bytes can share identical decoded pixels.

Do not re-open consumed historical evaluation screenshots, read gold,
retrain against old DEV/OOD, claim new TRAIN/DEV data admission, render
the tentative Frappe Gantt OOD source, or consume a paid GPU. The
previous H173 v3 one-shot test remains permanently consumed. Only
a complete independently authenticated old-source and old-visual
registry plus genuinely new original native TRAIN/DEV scenes may
authorize an actual new training experiment. Keep this Draft, no
publication, protected-main merge or release.
