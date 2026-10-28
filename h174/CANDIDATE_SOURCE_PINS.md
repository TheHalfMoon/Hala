# H174 — three prospective original-app donor source pins

**Status: ORIGINAL SOURCE / LICENSE BYTE CANDIDATES ONLY.** No TRAIN, DEV, or OOD assignment, no owner-private images or gold, no browser-native action qualification, no training, and no model evaluation. This is not a qualified H174 dataset or a claim of improvement.

The following three distinct public repositories were independently inspected at immutable Git commits on 2026-10-29. GitHub reports MIT licensing and the pinned original LICENSE files contain the MIT grant. On owner Windows Abdulaziz, the original upstream HTML and LICENSE were downloaded **as bytes directly from the immutable raw GitHub URLs**, hashed using SHA-256 in memory, and not persisted. This did not touch any consumed H173/H172 or earlier benchmark material.

| Candidate (unassigned) | Immutable upstream Git SHA | Original HTML path | Original HTML SHA-256 | LICENSE SHA-256 |
| --- | --- | --- | --- | --- |
| [Pickr](https://github.com/simonwep/pickr) | `e8ff88d9161ffc6e3c46a4430be8c328c42d6415` | `index.html` | `4cc4f0dbcb26e79bbf09202ba3896d5d66953c6d40fb6347eac2c15522712d1e` | `7129bc74b781d3db2979d4617940b00e75885914b8d636a3155a64e57481e5e5` |
| [interact.js dropzones](https://github.com/taye/interact.js/tree/a993eb8cd5cbc65f3a19e45ee24b7fedc026174f/examples/dropzones) | `a993eb8cd5cbc65f3a19e45ee24b7fedc026174f` | `examples/dropzones/index.html` | `9bc4dbe2a8f4e63b2f78a589b54adcd822ba552ecf481528a3b1d65f79430049` | `e805e2367aed149858263a4ae0a62168146af837a9ba3095aa01d02c8559e7fe` |
| [medium-zoom vanilla](https://github.com/francoischalifour/medium-zoom/tree/dccda8ab5a1f1677cf1f141d1ead4f14780f0ca4/examples/vanilla) | `dccda8ab5a1f1677cf1f141d1ead4f14780f0ca4` | `examples/vanilla/index.html` | `0e85b970cc9a4f489945ff4f325c7d3380855aa7433223d24f2bd3336ab1a5ed` | `166ba718b70c6278cce9c9758396e25e35c7f1a940d39b2c8eeaceed18cc03fb` |

All three are **candidates**, not admitted source-native apps. The original demos may require their own bundled dependencies and exact local serving setup. Neither a GitHub MIT label nor a successful hash check proves browser operability, task novelty, source-disjoint historical coverage, screenshot reproducibility, or permission to distribute bundled third-party assets. Keep upstream licenses and notices intact for any permitted derivative; no donor content is copied into this branch.

`h174/source_candidates.py` verifies immutable Git HEAD, clean checkout, exact canonical origin, original HTML bytes, and original LICENSE bytes against these source pins. It returns only a **source-bytes-only** receipt. The negative tests use synthetic local fixtures, never historical media.

**Required next gates before any dataset:** independently audit the complete owner SHA-only historical identity registry from prior metadata (do not reopen old media); reproduce original donor demos with real native actions; verify separate TRAIN/DEV source families and exact independent decoded RGBA replays; reserve a completely different OOD source identity **before training** and never capture its images during TRAIN/DEV. Only then may fresh TRAIN-only learning and new DEV-only selection be considered. The frozen H164/H171 checkpoints and all permanently consumed H173 v3, H172 and earlier evaluations are untouched.
