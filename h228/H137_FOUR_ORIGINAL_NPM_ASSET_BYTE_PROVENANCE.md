# H228 — Verify original H137 four MIT npm vendor bytes, including CSS

**Private, DRAFT only**, stacked on H227 Draft PR #248.
No merge/deploy/release/publication, Mac, private old OOD/gold, model
weights, free/paid GPUs or optimizer steps. Never upload the owner
`provenance.json`, actual binary files, or historical owner source metadata.

## Identified gap in H137/H176

The original `h137/source.py::VENDOR_SOURCES` specified **four**
owner-private copies of published MIT npm vendor assets:

| Original H137 runtime vendor filename | Original npm name | Fixed version | H176 prior row |
| --- | --- | --- | --- |
| `nouislider.min.js` | `nouislider` | `15.8.1` | `nouislider` |
| `nouislider.min.css` | `nouislider` | `15.8.1` | none (separate CSS) |
| `sortable.min.js` | `sortablejs` | `1.15.6` | `sortablejs` |
| `split.min.js` | `split.js` | `1.6.5` | `splitjs` |

The H176 source-family attribution checks that the **three** JS vendor
sha256 values are formally structured, but it cannot tell if the
claimed hash actually belongs to the byte stream used by the
original owner browser runtime, and never separately checks
`nouislider.min.css`. A source SHA metadata format check is
**not** a verified vendor file-byte provenance receipt.

## Implementation

`h228/original_npm_asset_integrity.py` can take an explicitly provided
read-only owner-local H137 `_runtime` path plus its private owner H176
paired attribution manifest, and does the following without touching
historical images, gold, predictions or old OOD:

1. Verifies the checked-in `h137/source.py` original
   `VENDOR_SOURCES` static AST declaration exactly matches all four
   pinned original filenames, package names, versions and relative
   published paths; does **not import or execute** H137.
2. Demands an original H137 runtime with a regular `vendor`
   directory containing **exactly** four recognized file names.
   Extra/absent files and symlinked files or directory are rejected.
3. Reads only original H137 `provenance.json` (bounded 4MiB,
   UTF-8 strict duplicate-key and nonfinite-constant validation).
   Requires all four asset names and exact original
   `package/version/license=MIT/sha256` fields.
4. Hashes actual four vendor JS/CSS bytes using bounded 1MiB reads
   (max 8MiB each), compares with each original H137 provenance
   `sha256`. This separately checks noUiSlider's CSS. Rejects even
   a same-length one-byte change.
5. Requires the original three H176 prior npm rows with exact
   `app/package/version/artifact_sha256` fields, and matches
   their three original JS SHA256s to the actual H137 bytes.
   An npm row cannot masquerade as Git/HTML provenance.
6. Prints **only source-byte counts and a research-only/no-admission
   status**, never original paths, images, metadata, raw artifact
   contents or owner SHA values.

## Running against a real already-created private owner runtime

Only from an authorized owner Windows repository checkout:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m h228.original_npm_asset_integrity `
  --original-h137-runtime 'D:\path\to\your-existing-H137\_runtime' `
  --owner-h176-attribution 'D:\path\to\your-private\paired-source-history.json'
```

These are **examples of arguments, NOT claims that such real paths
exist on the owner device**. The command checks only public copied
JS/CSS files and the two metadata records; it does not read H137
case screenshots or gold and does not create a dataset or checkpoint.
If actual old H137 runtime or owner H176 paired history is missing,
report **BLOCKED**, never synthesize provenance or substitute fake
data. The owner Windows environment has not yet been proven to
contain these two specific original artifacts, so **no real H137
owner byte-check success is claimed** for H228.

## Synthetic CI and scientific limits

`tests/test_h228_h137_four_original_assets.py` creates only small
fabricated fixture bytes. It tests all 4 assets, missing/modified
CSS, same-length mutated JS, missing/extra vendor and manifest files,
duplicate JSON keys, fake npm Git fields, changed historical JS
SHA, malformed license/version/package, symlinked inputs,
oversized/zero-length bytes, false historical source completeness
and explicit no-training status.

The CI workflow covers Windows/Linux Python 3.11/3.12, normal and
optimized Python. **Passing CI is only evidence the verifier rejects
fabricated bad inputs**, not that the owner's original H137 browser
runtime still exists or that the four original assets have been
authenticated against the npm registry.

Even a byte-perfect match between owner-supplied vendor bytes and
owner-supplied `provenance.json` is not independently trusted
publisher provenance: a coordinated rewrite could change both.
This check is a new necessary byte-integrity condition, not
historical source completeness proof or official dataset admission.

Original H137 browser demonstrations are evaluation-only wrappers
around three genuine MIT interaction libraries, **not independent
original production applications**. H173 consumed original owner OOD
remains permanently sealed (2/32 to 3/32 paired, exact McNemar p=1,
not significant). H199's official TRAIN manifest allowlist remains
empty. H212 96 synthetic research GUI candidates are permanently
barred from gradients. Never claim actual Hala VLM fine-tuning,
fresh sealed OOD, improved model performance or a release from
this receipt.
