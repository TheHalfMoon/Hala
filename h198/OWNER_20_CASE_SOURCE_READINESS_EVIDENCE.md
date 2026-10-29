# H198 — original GUI 20-prospect dataset readiness: independent source-only audit

**Zero TRAIN, DEV, or OOD admission; zero model training or scoring.**

## Why this gate was needed

The previous stages H188, H190, H191, H194 and H197
independently captured genuine original GUI preaction images privately
on the authorized owner Windows device, but no one single gate
checked all five cohorts' exact source identities, language balance,
known PNG or decoded RGBA collisions, and admission flags together.
H198 makes this cross-cohort check executable and fail-closed.

The audit reads only five small
`source-prospects-not-admitted.json` manifests and optionally the
H185/H186 **SHA fields only** in older allowlisted manifests.
Private image/gold sidecar files are checked for safe
regular-file presence but **never read or decoded**.
No old H173 scored OOD, Gantt tentative OOD, model, training output,
private path, image SHA, case ID or reference action is printed.

## Real owner result — 2026-10-29

On authorized Windows, CPython 3.11.15, exact original owner
data at the five already-quarantined paths:

    H198_20_PRIVATE_GUI_PROSPECTS_RESEARCH_ONLY
    total=20 families=3 ar=10 en=10
    distinct_png=20 distinct_rgba=20 known_old_png=2677
    NO_DATA_ADMISSION_NO_PRIVATE_MEDIA_NO_OLD_OOD_NO_MODEL

| Original source family | Research-only source role | Images | EN | AR |
| --- | --- | ---: | ---: | ---: |
| `simonwep/pickr` | Prospective TRAIN, **not admitted** | 14 | 7 | 7 |
| `taye/interact.js` | Prospective DEV, **not admitted** | 4 | 2 | 2 |
| `mymth/vanillajs-datepicker` | Unassigned, **not admitted** | 2 | 1 | 1 |
| **Total** | **Zero admitted** | **20** | **10** | **10** |

**Actual observed result:** all 20 new source SHA fields and all 20
decoded RGBA SHA fields are distinct across all five cohorts,
none match the *known* 2,677 older PNG SHA values, source repo/pins
and bilingual counts match, all eligibility flags remain false.
This **does not** prove that two different image bytes always
depict different semantic tasks, or that unknown historical data
cannot overlap.

Owner executable negative-control tests: **14/14** under
normal Python and **14/14** under `python -O` at H198
source head `14e3663e236ba0bc818b1b5175dec7b2a2438699`.
They reject a missing provenance manifest, missing
sealed private sidecar, repeated case identity, reused PNG hash,
reused decoded RGBA hash, changed upstream source/Git pin,
non-English/Arabic case, changed cohort size, changed instruction,
known historical screenshot reuse, unsafe missing history,
or promotion of *any* case to TRAIN/DEV/OOD.
The test verifies that an intentionally unparseable private
gold sidecar is **not** opened. An initial test found a raw
`FileNotFoundError` that exposed a synthetic owner path for a
missing private manifest; H198 now detects the missing leaf
before calling the inherited reparse guard, returning a
generic redacted failure.

## Why training is still explicitly blocked

- Historical source identity coverage is incomplete: some
  historical original Git checkouts/forks/npm families may be
  missing, even when the **22 surviving Git checkouts** were
  inspected in source-novelty preflight.
- The old decoded RGBA SHA register is incomplete; the 2,677
  old hashes only capture allowlisted encoded PNG metadata.
- Zero current original source screenshots have been
  independently admitted to TRAIN or DEV.
- Prospective TRAIN uses only **one original implementation**
  (Pickr), DEV another (interact.js), and the third original
  Datepicker source is still deliberately unassigned. Balanced
  bilingual labels alone do not establish task or app coverage.
- The already-scored H173 OOD benchmark is consumed and may
  never be reused for model selection. The tentative original
  Frappe Gantt OOD remains source-only and unrendered.

**Next qualified work:** enlarge *independent genuine GUI source
and action families* with explicit future source-role assignments;
close prior-image/old-source gaps or preregister a genuinely new
independent benchmark with sealed split controls; build a
strict TRAIN/DEV admission gate for *new* data; then train and
compare a new 2B policy on free Kaggle/Colab GPU under the
original-data permission rules and score exactly once on a
new untouched OOD source. No training result, accuracy figure,
or quality gain is claimed by H198.

Maintain Draft PR stack, protected main, zero paid compute,
no public owner media, and no publication.
