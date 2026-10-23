# H69 — Real pinned DragOn TRAIN slider domain, private reversible storage

**Status:** VERIFIED **3500/3500 original TRAIN images**, authenticated private
Kaggle byte-readback and source revalidation, **no model training or public release**.

Parent: H68 Draft PR #52 (head `5b41331799f6cb87e0ca3bc6a055eb5aed4936e8`).

## CPU-only source export (October 23, 2026)

Ran H61 original source-export on Windows WSL Ubuntu, Python 3.11.16
and Pillow 11.3.0, no GPU or Colab allocation.

```bash
python -m h61.colab_run \
  --repo <reviewed-H69-source> \
  --work <isolated-CPU-original-export> \
  --steps export --images slider=3500 --chunk-bytes 16777216
```

Real H61 `export/slider/receipt.json` records:

| Property | Verified value |
| --- | ---: |
| Upstream source pinned revision | `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5` |
| Original domain | `slider` / TRAIN only |
| Requested original screenshots | **3500** |
| Exported original screenshots | **3500** |
| Images with original native actions | **3500** |
| Native original drag actions | **30,840** |
| Exact source image byte range read | 234,881,024 |
| Exact source task byte range read | 915,404,288 |
| HTTP range SHA receipts | **42** |
| Elapsed CPU export seconds | 162.594 |
| Native labels SHA-256 | `833a72840c7cdddf021905c12aac7ea1102272af57d960d39429599690b86c7b` |
| `official_eval_or_test_used` | **false** |
| `benchmark_answer_leakage` | **false** |

No underlying complete 7.1GB archive full-SHA verification is implied; the
bounded pinned HTTP206 ranges carry individual exact byte-range receipts.
The stored original JPEGs and source task records are verifiable from those
receipts. This is the actual original TRAIN source, not synthetic examples.

## New bounded domain archive

`h69/domain_archive.py` independently verifies all original screenshot
bytes against each label, the pinned upstream source, the exact requested
domain count, action coverage and per-range proof count. It creates a
deterministic ZIP-as-binary without adding a model or official held-out
answers. Recovery requires the exact ZIP SHA256 and Git SHA; files are
checked individually, non-TRAIN data, symlinks, path traversal, duplicated
ZIP entries, corrupt media and unchecked overwrites are refused.

CPU pack result:
- **3,500 original images** and **30,840 native original drag actions**.
- Private transport binary `hala-h69-slider-3500.hala.bin`:
  **293,732,443 bytes**.
- ZIP SHA-256:
  `5283f6f936081d802e48e36ade921680d0d21e3ecab7fd2679ba3bd83e0a595e`.
- Source code SHA explicitly bound to archive:
  `218517cfc98816b98e8c9f6912945a4551c504c8`.
- `h69/domain_archive` dedicated regressions: **9/9 PASS** on real
  Ubuntu CPU with synthetic fault-injection fixtures.

## Kaggle verified private storage

Authenticated owner Kaggle dataset:
`abdulazizshehri/hala-h69-original-slider-3500-private`

1. Kaggle CLI private create succeeded. No `--public` flag; owner metadata
   explicitly sets `isPrivate=true`.
2. Authenticated `kaggle datasets files` reported the actual binary
   `hala-h69-slider-3500.hala.bin`, **293,732,443 bytes**.
3. Anonymous Kaggle API dataset view returned **HTTP 403**.
4. An authenticated `kaggle datasets download -f` pulled the binary into
   a brand-new local readback location, and its full SHA-256 was **identical**
   to `5283f6f936081d802e48e36ade921680d0d21e3ecab7fd2679ba3bd83e0a595e`.
5. `python -m h69.domain_archive unpack` on the provider-returned archive
   successfully revalidated **all 3500 original JPEGs**, labels and exact
   upstream source revision. It returned `VERIFIED_CPU_SOURCE_RESTORED`,
   images 3500 and actions 30,840.

Provider private readback and a full real source validator both passed.
No local copy is needed for long-term persistence after this step. This
commit stores **only public-safe evidence and code**; actual screenshots
are private with the owner, not GitHub.

## Remaining H61 frontiers

- Original three TRAIN domains `sheet`, `slide_resize` and
  `text_highlight`: **not yet H69-qualified or stored** at their required
  counts of 2500 each.
- H68's first-party bilingual family (2000 real scenes) is separately
  verified, stored privately, and must later join the complete source set.
- H67's final 10,000-row balanced mixed original TRAIN curriculum has not
  been created or authenticated on private storage.
- H65's actual real T4 checkpoint, optimizer step receipts, H63 visual
  intervention and held-out GUI success still do not exist. H60 remains
  0/48 true GUI task successes; no H61 model progress may be fabricated.
- Never allocate GPU for source TAR download; do all source-domain exports
  on CPU and use verified private storage before cleanup.
- Do not merge this or any stacked parent PR, or release a model, without
  final independent review and hosted CI.
