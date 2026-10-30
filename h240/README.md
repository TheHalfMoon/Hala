# H240 — close owner-history GitHub clone-alias identity gap

**Private-history identity firewall only. No new admitted dataset, neural model
call, optimizer update, historical OOD access, deployment or model release.**

## Actual failure mode

H238 normalized repeated GitHub clone `.git` suffixes when checking PUBLIC
historical original sources. H178's owner-only Git checkout inventory,
however, removed just ONE trailing `.git` from a historical
`remote.origin.url`. Thus `owner/example.git.git` and
`owner/example` could be treated as different families inside the
historical owner Git exclusion. H178 also casefolded candidate and prior
registry names without canonicalizing their clone suffixes.

This can bypass the intended **early rejection** of a reused historical
private source at the owner history intersection. An eventual failure
elsewhere is not equivalent: all source admission paths must deny the same
source at the historical exclusion boundary.

## Repair

`h178/owner_git_inventory.py` now reuses the exact H174 canonical
repository-family function for owner Git remotes, candidate source
metadata, and the owner-supplied prior source registry. This accepts only
the already-allowlisted HTTPS and GitHub SSH forms, never embeds
credentials, and normalizes case plus repeated `.git` suffixes.

The owner-private in-memory `_owner_only_repository_shas` set stays
private. Only aggregate counts appear in a CLI/status result. This
does not add private repo identities or hashes to the public Hala
source floor, and it does not create a completeness attestation.

The synthetic H240 tests use disposable local Git repositories with
fabricated origin strings. They require historical repository reuse
to be refused for both HTTPS and GitHub SSH origins, all 3 candidate
aliases, and positive owner-registry matches to remain explicitly
**NOT admitted**.

## Reproduction

```powershell
python -m pytest -q tests/test_h240_owner_clone_alias_firewall.py tests/test_h178_owner_git_inventory.py tests/test_h179_source_novelty.py
python -O -m pytest -q tests/test_h240_owner_clone_alias_firewall.py
```

No network, real source donor download, original screenshot, old gold,
checkpoint, sensitive source name or owner media is needed to run these
tests.

## Scientific dependencies still unresolved

- Public H238 historical family floor: 27 known families, not complete.
- Owner H178 census: seven historical Git checkouts, including two
  private families missing from public records. Their private identities
  are neither disclosed nor assumed to cover deleted checkouts.
- H168/H171 old TRAIN metadata: old PNG SHA fields exist, but the scoped
  legacy manifests lack the corresponding decoded RGBA SHA fields.
- H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains **EMPTY**.
- H212 research procedural scenes remain optimizer-blocked; H173 scored
  32-case OOD is permanently consumed and sealed.

Passing H240 means **identity consistency**, not proven source history,
licensing, media/visual independence, or Hala model improvement.
