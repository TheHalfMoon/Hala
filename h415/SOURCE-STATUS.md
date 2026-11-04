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
