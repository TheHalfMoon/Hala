# H238 — unify all *publicly recorded* historical UI source identities

**Research-only source exclusion hardening. This stage does not qualify any
TRAIN/DEV/OOD dataset or any neural model.**

## Why this matters

H237 discovered that H171 TRAIN/H172 DEV already used Awesomplete, which H232
had incorrectly described as new. The follow-up review found another gap:
the H174 base preflight did not itself consult two old published source
registries, `h177/public_origin_pins.py` and
`h184/public_prior_considered_origins.py`; the latter covers historical
Kanban, MiniWoB and rejected/source-pilot UI candidates. A pathway using
the base H174 metadata preflight alone could treat one of those upstream
GitHub families as novel under a new app identifier and Git revision.

## What is now blocked

`h238/historical_family_floor.py` computes a **fail-closed union of 27
distinct canonical original GitHub source families** already published
across H173, H176, H177, H184 and H237. Those are *minimum known* historical
source families, not independently established global completeness.

The common floor is checked by four public code entrypoints:

- `h174.split_firewall.validate`: proposed TRAIN, DEV, and reserved OOD
  source metadata; denies old family even when app ID and Git SHA differ
- `h179.source_novelty.screen_pins_against_history`: proposed original
  source candidates; normalizes case and repeated `.git` clone suffixes
- `h231.native_replay.preflight`: source-native replay/capture manifest
- `h236.original_dragula_native.preflight`: old-family gate for the
  genuine original Dragula research-only QA

The H176, H177 and H184 original source catalogs stay unchanged and are
imported rather than manually replicated. H237's H168/H171 original
TRAIN/consumed DEV and both Awesomplete GitHub aliases remain frozen.
The canonical floor includes nine H184 prior/considered original source
repos that had been missing from H174.

## Verification

Run on Windows or hosted CPU, with only synthetic fixture tests:

```powershell
python -m pytest -q tests/test_h238_public_history_union.py tests/test_h179_source_novelty.py tests/test_h174_split_firewall.py tests/test_h176_source_family_attribution.py tests/test_h177_public_origin_pins.py tests/test_h184_public_prior_considered_origins.py
python -O -m pytest -q tests/test_h238_public_history_union.py
```

Test only source identity and fabricated metadata. No historical owner PNG
bytes, decoded RGBA files, gold, labels, predictions or consumed OOD are
opened. Hosted browser execution is optional for other modules.

## Mandatory unproven gates

The union does **NOT** prove coverage of all deleted source checkouts,
private prior donor histories, GitHub transfers/forks, npm bundles, old
decoded RGBA/PNG SHA metadata, transitive media licensing or clean new
TRAIN/DEV split provenance. No positive new-source result is a dataset
admission. H199's `ADMITTED_TRAIN_MANIFEST_SHAS` remains empty,
H212 procedural scenes remain optimizer-blocked, and H173 original OOD
remains permanently consumed and sealed.

**No paid compute, no model calls or training, no owner-private captures or
metadata uploads, no merge, main push or model release.**
