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

## First **real** independent published npm run — SUCCESS

[GitHub Actions run 41039507011, original published-registry job](https://github.com/TheHalfMoon/Hala/actions/runs/41039507011)
independently fetched all three **actual** npm registry tarballs (not
GitHub source and not owner files), validated their previously recorded
official registry SHA-1, observed exact `dist.integrity` SHA-512 SRI,
and extracted the following original **published JS/CSS bytes**.
The runner printed only public digests; no original owner data was
read or uploaded.

| npm package@version | Exact original H137 vendor filename | Published file SHA-256 |
| --- | --- | --- |
| nouislider@15.8.1 | `nouislider.min.js` | `79222927cb7695f403144b58e31da6a7082d82e08d310b0cd776264d311abe7d` |
| nouislider@15.8.1 | `nouislider.min.css` | `9dc9155cfd6331d1f5629881ac0bc63928243fa4c1d83b17ce4921b5200be0ae` |
| sortablejs@1.15.6 | `sortable.min.js` | `6d0a831fc19b4bae851797ad3393157e861afb7862459c11226359b27e2c4337` |
| split.js@1.6.5 | `split.min.js` | `c32bd277fed98cd8a04033d4eef37201d9b50886189b4bb4d099d133feaf16dc` |

Pinned entire published tarball SHA-256 receipts from that run:

- `nouislider@15.8.1` `1d51198034f4df9f9496cd5a0a04c74efba281f6b365b105a1a5f6c2922913b6`
- `sortablejs@1.15.6` `7a5de41e4a184f43178f3b7cfd4e22f9ac2945a4b7c715ddc3e1c752b5a631dd`
- `split.js@1.6.5` `6c193ff0c1670b95c91c39b9ea18f7ec175b009e52b581a17e4d196e65469cfd`

The publication's exact SHA-512 SRI strings were also pinned
in `h229/verified_published_npm_assets.py`. A subsequent publisher
audit rejects a registry tarball or individual original JS/CSS asset
that drifts even if registry metadata changed consistently.

`h229/verify_original_h137_against_published.py` now integrates
H228's strict owner original four-asset/provenance/three-H176-app
checks and compares EACH of those four owner-local public binary
SHA-256s to these independently verified, hardcoded **published**
SHA-256s. There are **no CLI runtime flags** to override them.
Its unit tests use synthetic fabricated bytes and patch their mock
SHA constants only inside the disposable test fixture.
**Do not claim the owner original H137 runtime passed this
final comparison**; the real H137 runtime and private H176
paired attribution have not been independently located.

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
