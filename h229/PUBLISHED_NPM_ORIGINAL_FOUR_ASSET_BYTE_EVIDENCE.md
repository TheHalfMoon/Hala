# H229 — Independent published npm registry bytes for four original H137 assets

Private Draft stacked on H228 PR #249, with no merge, publishing, weight
download, historical OOD use, original owner media or model training.

## Why H228 was not sufficient

H228's owner-local verifier compares four H137 vendor JS/CSS files to an
owner-supplied `provenance.json`, and matches three JS digest values to
owner H176 historical npm rows. An adversary able to rewrite both files
and records could make those checks pass. The owner H137 original
runtime/vendor and paired H176 registry have not yet been found or
verified: **real original owner asset byte integrity remains blocked.**

H229 instead fetches three **public, previously published** exact-version
npm registry tarballs on a disposable GitHub-hosted Linux runner, verifies
tarball SHA-1 against previously independently recorded npm published
registry shasum, fetches exact registry version metadata and verifies
tarball SHA-512 against its npm dist.integrity SRI, and reads **only**
four exact member files plus the bundled public package.json.
No npm install, package scripts, compiler, runtime execution,
arbitrary path extraction or local owner access occurs.

| Fixed npm public package | Previously recorded published registry SHA-1 | Original published assets checked |
| --- | --- | --- |
| `nouislider@15.8.1` | `18729ed29738d2d328a1b5e0429d38eb8be8a696` | `dist/nouislider.min.js`, `dist/nouislider.min.css` |
| `sortablejs@1.15.6` | `ff93699493f5b8ab8d828f933227b4988df1d393` | `Sortable.min.js` |
| `split.js@1.6.5` | `f7f61da1044c9984cb42947df4de4fadb5a3f300` | `dist/split.min.js` |

Source of those three version/shasum/license entries: public npm exact-version
registry metadata, crosschecked via a read-only npm registry metadata lookup
on 2026-10-30 (all versions exist, all declare MIT). These are the exact
versions and relative vendor member paths previously pinned in H137.

## Implemented evidence

- `h229/verified_published_npm_assets.py` provides a hardcoded
  three-package, four-asset published registry auditor and a CLI
  explicitly restricted to a GitHub-hosted job, with HTTPS fixed npm
  registry host, exact URL matching and no redirects.
- The original published tarballs are downloaded in memory with
  bounded reads, SHA-1 **and** exact registry SHA-512 SRI checks,
  then inspected as a `tar.gz` in memory without calling
  `extractall`, `npm install`, `pip install`, `require` or scripts.
- It reads only the requested original JS/CSS member streams and
  bundled `package/package.json`. Rejects member path traversal,
  links, missing/duplicate assets, malformed package identity/license,
  bad registry names, versions, URLs, SHA, JSON and SRI.
- Emits a JSON source-only receipt with all four **original published
  asset SHA-256 values**, three archive SHA-256 values and three
  archive SHA-512 SRI values. The receipt contains **only PUBLIC
  npm artifact identities**, never owner paths/data, old screenshots,
  gold, model outputs or secret tokens.
- `tests/test_h229_published_npm_tarball_integrity.py` uses only
  fabricated tiny tarballs with synthetic data. Windows/Linux
  Python 3.11/3.12 CI runs those offline in normal and optimized
  Python modes. A separate GitHub Actions job executes the
  live *public* npm registry audit; it never executes a package.
  No artifacts need to be uploaded.

## What real npm archive verification DOES and DOES NOT establish

A successful hosted published npm artifact audit means the four
named bytes genuinely occurred inside registry tarballs matching
the fixed registry shasums/SRI as of this inspection. This is a
stronger independent benchmark than owner-modifiable H137 metadata.

This does **not** establish Sigstore-backed publisher provenance,
that the owner H137 copied those actual npm bytes, that the entire
historical source family/old decoded RGBA history is complete, or
that TRAIN/DEV/OOD are admitted. When actual private original
H137 runtime and prior attribution are independently available, a
subsequent owner-only check must compare H228's actual measured
four vendor SHA-256s to H229's *published* SHA-256s. **No historical
owner files are to be fabricated, regenerated or uploaded to GitHub.**

H199's official TRAIN allowlist stays EMPTY. H212 96 synthetic research
screenshots are permanently barred from gradient training by H222.
H173 original OOD was permanently consumed and must never be read
or reused for training or model selection. No new Hala optimizer
step, pretrained weights, qualified DEV/OOD, or model accuracy
improvement is claimed here.

If the live registry job is offline, fails an original member file
lookup or differs from the pinned public digest, H229 MUST be
reported blocked and its job must FAIL, not changed to a demo
success. Initial authoring is NOT proof that the hosted job ran.
