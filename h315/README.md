# H315 — offline metadata-only asset custody audit

**Research-only. All assets remain quarantined.** This stacked Draft follows H314 and does not claim source-owner rights, model training permissions, blind OOD custody, model improvement or release readiness.

Run with invented or independently prepared **metadata-only** JSON:

```sh
python -m unittest tests.test_h315_asset_custody_audit -v
python -m h315.asset_custody_audit --manifest path/to/metadata-only.json
```

Manifest root must contain only `assets` (max 1000). Each asset requires a repository identifier, immutable 40-hex source commit, relative asset identifier, SHA256 content digest, SPDX license expression, immutable 40-hex license evidence commit, SHA256 rendered digest, and three boolean flags: author attestation, historical provenance review, and independent rights review. Unexpected keys, duplicate identities, missing fields, malformed values, or oversized manifests fail closed. The checker does not open asset paths, resolve links, query networks, ingest screenshots, or print identifying metadata. A fully attested input is **still NOT admitted**: independent review and existing H199 custody controls are separate mandatory gates.

No actual external assets or third-party media were ingested. H250/H251 remain quarantined; H173 scored OOD sealed, H212 barred; H199 admitted TRAIN manifest empty. H293/H294 safety restrictions must remain enforced. H300 exposed-public original-source 12/12 does not establish generalization or adapted gain. Main untouched; no paid GPU, weights, deployment, release, or merge.
