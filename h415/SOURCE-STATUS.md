# H415 — original DragOn source curation (private)

This is a source-ingestion checkpoint, **not** a trained model or permission to release.

Four existing, owner-controlled H60 **TRAIN-only** archives were selected from `D:\hala-h60\staging` to stage a new owner-private Kaggle dataset. The sources record upstream pinned revision `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`. The owner states reuse permission.

| Domain / shard | ZIP bytes | Original images recorded | Native drag training actions recorded |
|---|---:|---:|---:|
| sheet/0000 | 885,615,166 | 2,390 | 61,381 |
| slider/0005 | 964,435,690 | 14,430 | 126,671 |
| slide_resize/0000 | 901,959,176 | 7,806 | 82,713 |
| text_highlight/0000 | 874,370,114 | 2,969 | 29,665 |

The four source ZIP files were linked (hardlinks, no duplicate disk storage) into temporary owner Windows D: staging, named `*.hala.bin` so Kaggle does **not** automatically expand them. Dataset ID `abdulazizshehri/hala-h415-dragon-original-4domain-private` is declared `isPrivate:true`. The free Kaggle upload was started; **do not claim provider verification until actual upload completes and source/readback checks pass**.

Source caveats: full upstream TAR SHA verification is marked **false** by the H60 source manifests, even though source original TRAIN labels/images and pinned revision are recorded. Source-image privacy and independent rights signing are not a production release qualification. No EVAL/TEST rows are in these declared training shards.

Next: verify Kaggle privacy/readback, select per-source-image diverse native-drag rows (not 61k repetitive annotations from 2390 images), preserve `ordered` and endpoint geometry, train separately from H414 click/refusal LoRA, use coordinate-focused weighted loss, and evaluate on reserved sources. **Do not publish raw screenshots or model weights to GitHub.**

## Verified H415 original source intake — 2026-11-04

Kaggle private dataset creation succeeded. Authenticated provider file listing independently showed EXACTLY four binary file entries with matching source sizes: sheet 885615166, slider 964435690, slide_resize 901959176, text_highlight 874370114. These are size/index readbacks, NOT independently verified remote full-byte SHA readbacks. Source original full-TAR SHA flag remains false; owner states permission.

The existing h61.curriculum tool was run twice on the four original local H60 train ZIP hardlinks with target_rows=10000 and max_rows_per_image=1. First AUDIT and then create-only EXPORT both succeeded on owner Windows. The resulting frozen TRAIN dataset has 10000 rows and 10000 distinct original screenshot IDs; no benchmark eval/test inputs.

- Source-balanced domain rows: sheet=2361, slide_resize=2547, slider=2546, text_highlight=2546.
- Ordered native drag examples=5093; unordered=4907. Never convert these into clicks.
- Capability rows: sheet/cell_range=77; sheet/multi_row=2284; slide_resize=2547; slider=2546; text_highlight=2546.
- Language rows: EN=10000, AR=0. A separate original bilingual first-party corpus is necessary.
- Exact frozen selected training JSONL SHA256: 294cfb953457549a4315819557d9346e6bb81a243657bd940fd654cc29474f23.
- Private source material and exported labels are on owner D: drive, not committed to public GitHub.

This is verified PREPARATION, not 10000 gradient optimizer examples consumed. No H415 native-drag weights or external blind evaluator result yet. H414 existing click/refusal Kaggle training was RUNNING at last check. Privacy and release gates still apply.
