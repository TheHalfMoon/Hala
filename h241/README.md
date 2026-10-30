# H241 — expanded PRE-H173 owner history SHA-only exclusion

**This is a reproducible owner-only *DENIAL* gate, not TRAIN, DEV or OOD
admission and not Hala model training/evaluation.**

## Why H241 matters

Earlier owner metadata-only checks passed two explicitly supplied old
TRAIN phase roots (H168 and H171). On the current authorized Windows
host these yielded **4 manifests, 65 distinct known historical PNG SHA
values, and zero decoded-RGBA SHA values**. That excludes only the
known old pixels from those two phase directories.

An H241 source-only, read-only owner audit of all **recognized** safely
accessible PRE-H173 private phase roots found **6 valid historical phase
folders, 7 allowlisted `manifest.json` files, and 130 distinct prior
PNG SHA values, versus the earlier 65**. All 7 manifests lack decoded
RGBA SHA fields. This is **65 additional known PNG exclusions**, not
65 newly admitted TRAIN images. Historical completeness remains false.

H241 now makes the broader comparison repeatable: discover immediate
phase folders with exactly H178/H185's `PHASE_NAME` naming contract
(`private-hNN` or `Hala-HNN-Private`) and numeric phase strictly
less than H173. In particular, `Hala-H173-Private` is not traversed,
and neither are H174+ research/owner captures, random folder names,
source donor contents, images or outputs. Unrecognized folder naming
variants are not counted as verified historical phases and may represent
additional unexamined history; they do not establish completeness.
H241 reports their aggregate number as
`potential_old_phase_name_variants_unscanned` without enumerating or
reading their contents (two such naming variants were observed in the
owner-root discovery).

The module reuses H204's exact snapshot of the 20 owner-only
unadmitted H188/H190/H191/H194/H197 candidate source manifests and
H185's SHA-only field parser. It compares each candidate's known
PNG/decoded-RGBA digest against all available discovered old SHA
exclusions, detects any known collision and requires source manifests
to remain unchanged during the audit. No old media bytes or gold
are opened. Output contains **aggregate counts only**, no source
identity, historical private path, raw hash or patient data.

## Owner Windows execution (offline, source-only)

Use existing Python and installed ordinary testing deps:

```powershell
python -m h241.expanded_pre173_metadata --owner-root D:\ --historical-parent D:\ --historical-parent C:\
python -m pytest -q tests/test_h241_expanded_pre173_metadata.py tests/test_h204_training_readiness_gap.py tests/test_h185_manifest_sha_only.py
python -O -m pytest -q tests/test_h241_expanded_pre173_metadata.py
```

Every aggregate report MUST retain
`training_approved=false`, `dev_approved=false`,
`ood_approved=false`, `visual_history_complete=false`,
`historical_history_complete=false`, `h173_ood_accessed=false`,
and zero media/gold/model/optimizer activity. A missing parent root,
unsafe symlink/junction, modified candidate metadata, historical
collision, invalid phase identity or inconsistent scanner scope is
an explicit **FAIL CLOSED**, not a reason to infer qualification.

## Scientific blockers

A larger known exclusion list is important but does **not** prove the
full historical image population, all previously deleted checkouts,
all source/fork/npm history, license/media rights, or coverage of
decoded RGBA hashes missing in the original manifests. The H238 public
27-source exclusion is a minimum; H240 found 2 owner-private historical
Git families outside that minimum. H199 officially admitted TRAIN
manifest allowlist remains **EMPTY**. H212 procedural candidates
remain optimizer-blocked by H222. H173's original scored 32-case OOD
is permanently consumed and sealed. No actual Qwen3-VL LoRA
training or new independent blind evaluation was done.

The H241 owner-private aggregate report remains on the Windows owner
machine; only source code, docs, and synthetic tests belong to public
GitHub. Do not publish screenshots, source owner metadata, raw image
hashes, private source identities, gold, model weights or OOD records.
