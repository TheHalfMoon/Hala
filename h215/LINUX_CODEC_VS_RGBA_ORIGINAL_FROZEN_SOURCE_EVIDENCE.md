# H215 — fix false cross-OS PNG-byte reproducibility requirement without weakening pixel integrity

**Research source-only, Draft stack only.** Parent H214 Draft #235 at
`47116cb81d300566657d22d4e9a8b54511a6be8e`.
No main merge, model release, paid compute, owner screenshot access,
training run, DEV admission, sealed gold, or H173 consumed OOD.

## Empirically identified root cause (hosted CI)

The H213/H214 workflow did **NOT** pass on Linux; Windows-2025
Python 3.11/3.12 passed the same full 96-image source checks.
At exact earlier H214 source head `f33461facf01bc3f2176685a1ef45bbdac3fa80b`,
both Ubuntu jobs failed, and a dedicated fail-closed aggregate
diagnostic on `47116cb81d300566657d22d4e9a8b54511a6be8e`
reported on Ubuntu Python 3.12:

`png=96 rgba=0 generator_source_changed=False`

That is **96 different PNG byte streams but 96 identical decoded RGBA
pixel streams**, with identical original generator Python SHA.
Mac is NOT a permitted owner Hala workstation or future CI runner;
H214 explicitly removed Mac jobs. Do not re-enable Mac.

PNG encoding/DEFLATE/Zlib details are not source pixel identity.
The prior all-platform PNG-byte parity claim must be corrected.
Never replace the original 96 frozen Windows PNG SHA values by
Linux-specific values to make a test pass.

## New two-tier immutable provenance verification

The existing H213 helper `verify_exact_metadata_text` is
unchanged and remains **strict**. On the owner Windows platform,
`verify_checked_in_source` continues to require exact 96/96
PNG-byte SHA, 96/96 RGBA SHA, source SHA, roles/languages/flags.

On Linux, H215:
- Uses source-authored `h215/portable_png_rgba_gate.py`.
- Requires the **exact original H212 frozen JSON bytes**, independently
  hard-pinned to SHA-256
  `6a008992c51f6fd01d37fa321bf9f5fa1a346d8d8fb506cb52f3abf001111c8b`,
  confirmed in the committed Git blob and Windows clone of the original
  32,460-byte research metadata file (LF line endings).
- Recreates H212's 96 research screenshots in-memory from the
  unchanged source. Requires **all 96** newly decoded RGBA SHA-256
  values match the immutable frozen owner Windows metadata exactly.
- Enforces full equality of every other field: normalized generator
  source SHA, candidate indices/order, four source families,
  Arabic/English assignments, candidate-only roles,
  provenance flags and zero model/training activity. Both original and
  regenerated local PNG hashes must remain 96 distinct valid SHA256.
- **Only platform-local computed PNG-byte SHA** may differ from the
  original Windows values; the original JSON is byte-pinned and a
  modified/reordered/forged frozen PNG SHA cannot pass.
- Reports exact PNG-byte differences explicitly as an aggregate count,
  never silently calls them `png_sha_matches=96` on Linux.
- Rejects non-Windows/non-Linux platforms; no Mac CI is requested.

Thus the Linux finding is a **codec portability issue**, not a
pixel-content mismatch. The observed source pixel identity is the
RGBA fingerprint; original frozen Windows PNG provenance remains
intact with an independent file SHA seal.

## Tests and required completion evidence

- H215 synthetic adversarial tests mutate original frozen PNG/RGBA
  SHA, generated source, language, role, release flags, candidate order,
  generated pixel SHA and duplicate/missing fields; all must fail.
- Simulated different local codec SHA with unchanged RGBA can pass,
  *only* against the immutable original frozen manifest.
- H213 original strict Windows negative tests remain, and H214's
  Windows/Linux-only host restrictions remain.
- Free hosted CI runs Windows-2025 and ubuntu-24.04 on Python 3.11
  and 3.12, with pinned Pillow 12.3.0, no neural weights or GPU.
  **Do not claim hosted green before all four jobs pass** at
  the exact new H215 Draft HEAD.
- Owner Windows QA must pass both normal and `python -O` suites;
  any missing CI job remains a blocker.

## Scientific limits retained

H212's 96 generated cases remain RESEARCH-only, NOT officially
admitted TRAIN/DEV/OOD. It does not establish lineage disjointness from
all pre-H173 sources/forks and old decoded RGBA hashes, or independently
sealed DEV gold. H199 training SHA allowlist stays EMPTY; no
Qwen3-VL training, fine-tuning, evaluated GUI reward, browser action
or model release is claimed. H173 consumed OOD and reserved Gantt are
strictly excluded.
