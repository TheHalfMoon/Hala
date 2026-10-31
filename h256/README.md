# H256 — surviving PRE-H173 PNG and decoded-RGBA history, owner-only

**Research quarantine / DENIAL only. No TRAIN, DEV, OOD, model, or release.**

H241 established 130 distinct historical PNG SHA values across seven metadata
manifests in six recognized pre-H173 private phase roots, but **zero** historical
decoded-RGBA digests. H256 now performs a *separate, opt-in owner-only* bounded
scan of **surviving** pre-H173 PNG files to recover H231-compatible visual
hash exclusions. It never traverses H173; it excludes gold, labels, scored,
OOD, model, predictions, Git and other sensitive subdirectories.

The scanner verifies safe directories and regular files, rejects reparse
points, limits traversal depth, file count, PNG byte sizes and decoded pixel
dimensions, and uses the same `SHA256(width_be32 + height_be32 + RGBA bytes)`
as H231. A `.png` filename does not establish decodability: invalid PNGs
retain raw-byte SHA exclusions but receive **no fabricated RGBA digest**.

The private output contains file names and raw PNG/RGBA SHA values. It MUST
stay outside Git, never be uploaded, pasted, committed, or included in PRs.
The CLI emits **aggregate counts only** and refuses overwrite.

## Owner Windows result, 2026-10-31

On the authorized owner Windows machine, under six recognized strictly
pre-H173 private phase roots:

| Audit | Observed |
|---|---:|
| Surviving PNG-named files | 256 |
| Decodable PNG files | 231 |
| Undecodable PNG-named files | 25 |
| Distinct surviving PNG byte digests | 144 |
| Distinct decoded RGBA digests | **137** |
| Previously known manifest PNG SHA found on disk | **130 / 130** |
| Additional surviving distinct PNG SHA not in manifests | **14** |
| Historical phase roots | 6 |
| H173 scored OOD accessed | No |
| Gold files opened | No |
| Known PNG/RGBA collisions with 20 H198 quarantined candidates | **0 / 0** |

The 137 visual digests are **historical exclusions**, not new training images.
Some different PNG bytes decode to identical RGBA pixels. The 25 undecodable
files cannot be given visual digests without additional verified evidence.

This is a material improvement in *surviving historical visual exclusion
coverage*, not proof of complete historical history. Deleted images, old
unrecognized phase names, source forks/npm/commit identities, and any missing
external images remain unresolved. No new data is admitted, H199 TRAIN
allowlist stays empty, and H173 scored OOD remains permanently sealed.

## Reproduction (authorized owner Windows only)

Use a fresh owner-private directory outside all Git repositories. Never run
against H173 or an OOD/gold path. This example is the already verified owner
private location; do **not** overwrite the existing receipt:

```powershell
python -m h256.pre173_visual_history --historical-parent D:\ --historical-parent C:\ --private-output D:\Hala-H256-Private\H256_OWNER_PRIVATE_PRE173_VISUAL_HASHES_NOT_ADMITTED.json
```

For a future independent rerun, use a new private directory and the same
required receipt basename. The code refuses existing output and writes no
media files. Synthetic regression tests:

```powershell
python -m pytest -q tests/test_h256_pre173_visual_history.py --basetemp D:\Hala-H256-Private\pytest-h256-new
python -O -m pytest -q tests/test_h256_pre173_visual_history.py --basetemp D:\Hala-H256-Private\pytest-h256-optimized-new
```

**Never** treat a successful scan or hosted synthetic CI as license clearance,
source-history completeness, training approval, model inference, or accuracy.
H15's earlier free Colab two-step LoRA optimizer engineering is separate.
Next: privacy-safe integration of the owner-only expanded visual exclusion set
into new-source collision rejection, complete source/fork/npm history and
independent rights-cleared TRAIN/DEV with fresh sealed OOD before model work.
