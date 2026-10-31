# H258 — recover tall pre-H173 decoded-RGBA exclusions (owner-only)

**Research quarantine; DENIAL ONLY. No TRAIN, DEV, OOD admission or model improvement.**

H256's original 256-file inventory remains immutable. A read-only owner Windows
triage found that its 25 PNG-named files without RGBA hashes comprise:

- **19 files with no PNG signature**, not decodable PNG media (one historical
  name/extension is not evidence of valid pixels).
- **6 structurally valid, CRC-verified PNGs**, whose height exceeds H256's
  conservative 4096-pixel side bound. These are not corrupted images. Each
  fits H258's 8192-pixel side and 16,777,216-pixel total bounds.

H258 validates the original private H256 receipt, rehashes only missing-RGBA
files from recognized strictly pre-H173 private phase roots, validates PNG
chunk CRCs and complete IEND, then verifies and decodes only bounded tall
PNG media. It writes a **new** owner-private receipt with raw hashes and
relative paths, never modifies the H256 receipt, and prints only aggregates.

## Verified owner Windows result — 2026-10-31

| Finding | Count |
|---|---:|
| H256 surviving PNG-named files | 256 |
| H256 missing decoded-RGBA entries | 25 |
| Not actually PNG signature | 19 |
| Newly safely decoded tall PNG files | **6** |
| New distinct decoded-RGBA exclusions | **6** |
| Combined known surviving decoded-RGBA exclusions | **143** |
| Still undecoded PNG-signature files within audited subset | 0 |

The 143 digests are **exclusions**, not eligible data. They are limited to
surviving known pre-H173 files. Deleted images, unrecognized roots,
old forks/npm/Git source families, historical visual completeness and
transitive asset rights remain unproven. H199 admitted TRAIN allowlist is
EMPTY; H173 scored OOD is irrevocably sealed; H212 procedural research
scenes are not optimizer data. No inference, optimizer steps, weights or
accuracy improvement in H258.

## Owner-only reproduction

Run from the Hala checkout on authorized Windows, using a **new private
output directory** outside every Git checkout. Never overwrite or upload
the existing H256 receipt or any H258 receipt.

```powershell
python -m h258.recover_tall_rgba `
  --h256-receipt $env:HALA_H256_PRIVATE_RECEIPT `
  --historical-parent D:\ `
  --historical-parent C:\ `
  --private-output $env:HALA_H258_NEW_PRIVATE_OUTPUT
```

The output basename must be
`H258_OWNER_PRIVATE_TALL_RGBA_EXCLUSIONS_NOT_ADMITTED.json`.
The CLI rejects overwrite, unsafe paths, tampered source hashes, scored
H173 references, Git destinations, and unbounded pixels. Public tests use
only generated synthetic PNG bytes and fabricated source receipts:

```powershell
python -m pytest -q tests/test_h258_tall_historical_rgba.py --basetemp D:\Hala-H258-Private\pytest-new
```

Hosted CI runs synthetic tests only. Neither CI nor private historical
hash recovery constitutes source-license clearance, a valid independent
TRAIN/DEV/OOD split, or real model quality. Next substantive gate:
rights-clear independent original-app data and sealed fresh OOD, then
measure unadapted Qwen3-VL before any eligible free LoRA training.
