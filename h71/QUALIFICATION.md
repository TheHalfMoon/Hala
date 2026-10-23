# H71 — Verified 10,000-row CPU H61 TRAIN curriculum

**Status:** local source/curriculum qualification PASS; private Kaggle publication
and independent full byte readback must be confirmed separately; real model
training and GUI evaluation NOT EXECUTED.

Stacked on H70 Draft PR #54. No branch merge and no public model release.

## Real CPU source qualification (2026-10-23)

The operator restored H69's original `slider` source (3,500 images, 30,840
original native actions) and H68's authentic Edge-rendered bilingual family
(2,000 audited scenes; Arabic 1,000 / English 1,000) from owner-private
Kaggle. Both provider-returned archives matched exact published SHA256 and
passed independent source-level H69/H67 revalidation. H70's three remaining
source domains (`sheet`, `slide_resize`, `text_highlight`, 2,500
images each) also passed independently authenticated private byte readback
and source validation. H71 assembled all **five** verified CPU-source
shards in one canonical TRAIN workspace.

The direct H61 `curriculum audit` and subsequent create-only export, with
`max_rows_per_image=1`, `target_rows=10000`, and first-party family cap
`h61_firstparty_bilingual_drag_train: 2000`, agree **byte-identically**:

| Metric | Actual CPU result |
| --- | ---: |
| Original pinned DragOn TRAIN images available | 11,000 |
| Audited browser-rendered first-party bilingual images available | 2,000 |
| Actual selected H61 TRAIN rows | **10,000** |
| Unique selected source screenshots | **10,000** |
| English | **9,000** |
| Arabic | **1,000** |
| Selected bilingual family rows | **2,000** |
| Selected source `sheet` rows | 2,408 |
| Selected source `slide_resize` rows | 1,864 |
| Selected source `slider` rows | 1,864 |
| Selected source `text_highlight` rows | 1,864 |
| Ordered drag annotations | 7,636 |
| Unordered drag annotations | 2,364 |
| Final dataset SHA256 | `9888ac9d3b5ec7b8efe677287d54dc350c222a1804619e6217a84ec8f95eddcd` |

Capability subcounts from the actual H61 curriculum: `sheet/ordered`
500, `sheet/ordered/column_resize` 2,365,
`sheet/ordered/row_resize` **43**, `slide_resize/ordered` 2,364,
`slider/ordered` 2,364, `text_highlight/unordered` 2,364.

**Distribution limitation:** only 43 row_resize rows. The validated source
is a drag-first diagnostic, not a sufficient benchmark for general-purpose
computer-use clicks, typing, scrolling, multi-step recovery, or abstention.
Training quality gains and SOTA status are not implied by these counts.

## Qualification defect fixed

A real H68 full-readiness run initially produced `NOT_READY` solely
because a 10k-image SHA media manifest is **2,306,124 bytes**, while
the earlier generic JSON reader limited every JSON receipt to 2 MiB.
H71 now permits up to **16 MiB only for the derived TRAIN manifest**,
using H67's existing manifest metadata bound; small source receipts
retain their 2 MiB bound.

After the fix, actual `python -m h68.readiness --work <cpu-work>` returned
`READY_FOR_PRIVATE_STAGE_BUILD`, all four 11k source image counts validated,
2k first-party bilingual images validated, **10,000** H61 selected rows
rederived, and **zero** failures. H71 + H68 + H67 focused regression suite
PASS **27/27** on real Ubuntu CPU.

## Final immutable CPU bundle

`python -m h67.stage build --work <cpu-work> --bundle <archive> --git-sha
<reviewed-commit>` passed the full source/media digest and H61 rederive
gate; it created a deterministic binary ZIP containing **15,224 files**,
**1,759,484,545 bytes**, and all original source provenance required to
re-derive the chosen curriculum. No source screenshots are committed here.

- Immutable archive SHA256:
  `4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a`.
- Frozen H71 source Git SHA inside archive:
  `acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe`.
- H61 dataset/content SHA:
  `9888ac9d3b5ec7b8efe677287d54dc350c222a1804619e6217a84ec8f95eddcd`.
- Bundle status after CPU build:
  `CPU_STAGE_CREATED`, **not** verified private publication.

Target owner-private Kaggle dataset:
`abdulazizshehri/hala-h71-drag-train-10000-private`.

### Owner-private provider proof required

A successful upload CLI exit is not sufficient. Require all of:
authenticated exact file byte size, anonymous Kaggle API HTTP **403**,
an independently authenticated **complete** binary download with archive
SHA256 equality, and an independent `h67.stage restore` on the provider
returned bytes that executes H61's actual source-bound curriculum rederive.
Record `RESTORED_TRAIN_STAGE` and exact 10k rows. Only then may disposable
CPU staging be removed. Do not claim these proofs until measured.

### GPU and release NO-GO

No free T4 session has been allocated for H71. Prior H64 T4 was removed by
Colab during dependency setup with zero optimizer success. H65 bounded
package bootstrap, host checkpoint egress and H67 verified source bundle
import are available, but do not justify a new GPU attempt without a
current valid free allocation and tested stage transfer. The first actual
optimizer checkpoint and matched H60/H61 held-out GUI comparison must
be recorded before any improvement claim or release. Keep H71 Draft.
