# H57 — Original H Company DragOn Three-Domain TRAIN Source and Native LoRA SFT

**Date 2026-10-21. Genuine original externally sourced TRAIN-only supervision. NO neural model training or independent model evaluation was run.**

## 1. Scientific source and immutable provenance

Official [Hcompany/DragOn](https://huggingface.co/datasets/Hcompany/DragOn), Apache-2.0, pinned original commit `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`, contains four distinct drag task domains with about 286,012 TRAIN images and 3,560,142 TRAIN instructions. The user confirms authorization to reuse H Company research and training source. H53 previously imported a small authentic `slider` TRAIN subset: **87 genuine images, 760 native ORDERED actions**, stored in existing **PRIVATE** [H50/H52/H53 library](https://www.kaggle.com/datasets/abdulazizshehri/hala-gui-training-library-private). These original H53 source files were **not changed or re-uploaded**.

H57 adds genuinely NEW original source domains: `sheet`, `slide_resize`, and `text_highlight`. Authentic original source uncompressed TRAIN TARs are **14,258,661,376**, **9,915,637,248**, and **29,874,024,960** bytes respectively. The importer fetches only 2×64 MiB precisely confirmed HTTP 206 byte windows of each, the first for original JPEG image headers, the second from the pinned original source TAR task section. Six actual bounded windows, total **384 MiB** streamed. Each has a distinct SHA256 plus original byte offset/length/content range and original source commit in `h57/source_evidence.json`. **No full upstream TAR checksum was independently verified**. This narrow range sample is intentionally non-representative and not the full 3.56 million examples.

The TAR original image and task filenames have different insertion order. Joining must use their exact original UUID filenames and each task's `image_id`; NEVER trust TAR order. Original TAR-header checksum, regular-file type, source domain identity, original JPEG format/completion, original image size, UTF-8 JSON shape, all `start_bbox`/`end_bbox` original integer image-pixel bounds and original prompt content were checked; credential-like instructions rejected. One text-highlighting original image/task pair failed image-geometry validation and was **excluded with explicit rejection evidence**, not silently repaired or relabeled.

**New real original evidence:**

| Official TRAIN domain | Original images | Original drag instructions | `ordered=true` | `ordered=false` |
|---|---:|---:|---:|---:|
| `sheet` (cell selection) | 23 | 598 | 224 | 374 |
| `slide_resize` (element resizing/rotation/crop) | 66 | 708 | 708 | 0 |
| `text_highlight` (text selection) | 28 | 280 | 0 | 280 |
| **H57 addition** | **117** | **1,586** | **932** | **654** |
| **H53 + H57 combined source subsets** | **204** | **2,346** | **1,692** | **654** |

The real H57 original `train_drag_multidomain.jsonl` byte SHA256 is `6f7e59cb86d1c7ac80fe70ade8447e11990b420586e505631fb7a4a6a69c38bf`.

**Critical scientific design:** H Company documents `ordered=false` as genuinely direction-insensitive for text highlighting and cell selection: start/end can be interchanged while achieving the same selection. `ordered=true` is directional for slider manipulation and resizing. The H57 importer **preserves exactly the source boolean**, associated `subtype`, original image width/height, boxes and image source SHA. It never trains a false ordered label for an unordered task. H57 model TRAIN prompt explicitly represents directional significance; future independent testing must allow reverse endpoints ONLY on source tasks marked `ordered=false`, while evaluating all ordered actions direction-sensitively. Do **not** use original DragOn public `eval/` or private TEST labels as TRAIN.

## 2. Private Kaggle storage, actually independently checked

**New, separate, verified PRIVATE Kaggle dataset only:** https://www.kaggle.com/datasets/abdulazizshehri/hala-dragon-multidomain-train-private

It is separate from the previous H50/H52/H53 library and is **create-only**. The Kaggle API independently returned `is_private=True`; an unauthenticated Kaggle API request was rejected with **HTTP 403**. The original 117 JPEGs, 117 original JSON task files, 1,586 training rows and full per-original hash manifest are in the new private dataset, not TheHalfMoon/Hala's PUBLIC GitHub repository.

Original private ZIP (locally verified CRC and all members before upload) SHA256: `fccddbf9dc95489fc65c1b11691db28f497dc34c7535f08866dbb7257b5e29c7`; ZIP is 19,614,204 bytes. Kaggle expands uploaded ZIPs into virtual files: downloading the ZIP itself by original name returned provider **404**, NOT evidence of source corruption. Source validation instead independently downloaded virtual `train_drag_multidomain.jsonl`, the entire original manifest, one genuine original JPEG and its matching original task JSON directly from the new private provider. **All downloaded files' applicable SHA256 matched immutable local source receipts**. Do not claim the entire Kaggle reconstructed ZIP was externally rehashed. No privacy metadata update was performed; previous H50/H52/H53 private provider entries remain unchanged. The new public H57 source receipt explicitly records remote visibility, what was read back, and the unverified full-archive boundary.

## 3. Actual original-to-Qwen LoRA candidate source path

`h57/native_multidomain_sft.py` rereads and revalidates all **117 original private JPEG + task JSON source pairs**, per-original source SHA/offset receipts and all 1,586 original TRAIN labels. It creates a **separate** `h57_native_drag_sft.jsonl` and `h57_native_drag_training_manifest.json` in the private original data directory, without changing the Kaggle archive or any original source. Exact candidate SFT JSONL SHA256: `10cdcee91ae219e45f74850a165cb1ce1228d7fd45f8b1b2274b5e69f934f99a`, 1,586 native drag rows and 117 source-bound JPEGs.

The existing `training/scripts/train_adapter.py` Qwen3-VL LoRA source dispatcher now recognizes only this new pinned `hala.h57.native-multidomain-drag-sft.v1` schema, accepting genuine source-bound JPEG images for audit and eventual optimizer microbatches. The old H54 `slider` 760-annotation native path remains untouched; all older H48/H49/H50/H52 CLICK paths remain strict PNG. H57 requires a separately observed **real, unchanged-model, source-disjoint multi-domain** base inference with at least 24 native drag tasks across ordered/unordered domains, explicit source hash and measured deficit before running real gradient optimization. An **existing public H55 24-case slider DEV is not an independent multi-domain blind source**.

The original H53 and H57 model candidate loaders are **separate**. Combined 2,346 source labels have not been trained together and no multi-domain mixture sampling schedule has been qualified. Full H Company corpus has not been copied.

### Reproducible source-only audit on authorized private data

```powershell
# Authenticated access to the private Kaggle H57 dataset required; NEVER publicly host JPEG/JSON source.
$source = '<private-extracted-H57-original-117-image-task-train>'
python -m h57.import_train_multidomain audit --source $source
python -m h57.native_multidomain_sft --source-private-train-dir $source
python -m unittest tests.test_h57_original_multidomain -v
```

The raw original H57 data are **private** and neither imported into the public GitHub tree nor returned in PR comments. `h57/source_evidence.json` includes only IDs, hashes, numerical counts and original byte offsets, never source screenshot bytes or text task bodies.

## 4. Exact status (truthful model science)

- **DONE:** authentic H57 original external TRAIN data acquisition, strict JPEG/task UUID join and action semantics; source-hash readback; create-only private Kaggle with provider privacy verified; true 1,586-row source-verified Qwen LoRA candidate loader.
- **NOT DONE:** manual visual privacy review of ALL 117 external screenshots (automated source/media checks passed only); original full TAR checksum; source-disjoint multi-domain visual DEV/TEST; genuine Colab CUDA allocator; pinned-Qwen real source-target drag predictions; LoRA gradient steps, tuned weights, actual new model success, matched benchmark comparison or Holo4/UI-TARS superiority.
- **H47 original private Kaggle notebook** remains a separate queued experiment, and must not be duplicated or restarted. Earlier H46 Qwen click target 0/8 negative evidence remains unchanged.
- **No paid services, APIs or GPU resources created.** Original source TARS were read by original HF public TRAIN HTTP bounded byte ranges. Underlying third-party source attribution, provenance and immutable scientific evidence remain accurate while product branding stays Hala.

## 5. Test/review evidence

`tests/test_h57_original_multidomain.py` uses only **self-generated fictional JPEG and task JSON**. It tests ordered-false preservation, domain dispatch, original receipt integrity, fail-closed image and JSON mutation, duplicate/tampered supervised completion, create-only outputs and baseline-schema rejection. Original external media stay completely outside GitHub.

Original focused source tests **8/8 PASS**, and full source suite, Graft wiring-only review and exact signed GitHub Python 3.11/3.12 source CI will be recorded only after actual completion. Neither passing code nor Kaggle source storage alone demonstrates any trained neural model improvement.

## 6. Exact live engineering qualification (2026-10-21)

- Real Windows source-only **686 tests total, 684 PASS, 2 existing SKIP, ZERO failures/errors**, 238.198 seconds, exit 0. Original `reviews/h57-full-source-suite-20261021.log` SHA256 `8b0d7d1f2883c7d63c57103709406e81fd52ce29e895a5c419c1b1dfeee7a5a1`.
- Real focused `H57+H54+data-library` regression tests **19/19 PASS**, original `reviews/h57-focused-source-and-regression-20261021.log` SHA256 `2f82c05a12e49ecd37903ad536cb0311a40a805541399fae1e1982afbf519aad`. One initial focused run returned nonzero solely for README trailing blank whitespace detected by `git diff --check`; README was corrected and the entire 19-test gate rerun green.
- Genuine Graft **wiring-only** build/check PASS on **264 source files, 2,132 nodes and 5,667 edges**. Original build log `reviews/h57-graft-build-20261021.log` SHA256 `a7c61f6ada5d213d9f64906e9ad76f5c8a9a1096f79405c5d90b6257248dc017`; check log `reviews/h57-graft-check-20261021.log` SHA256 `d8bc5f17d86f6e17ed84f61808fb4f3ba7ecbd6cffe0b03d7cf5cda8c587efdb`. Graft meaning/deep semantic layer is **NOT** run (0% deep meaning); D1, PStack or Alibaba model-based semantic reviews **NOT** performed.
- Public hash-only `h57/source_evidence.json` SHA256 `9a2f15de4ca69c4b3fd8cecc26108dfd61340de112c78f8abd2ccaa22d848510`. Actual Kaggle authenticated `is_private=True`, anonymous HTTP 403, and 1,586 actual label JSONL plus one original JPEG and source task JSON remote SHA matches.
- **GitHub exact HEAD CI:** not yet qualified until actual signed commit is pushed and Python 3.11/3.12 hosted jobs have all finished. Keep PR DRAFT/unmerged and do not claim neural success or release-ready model.
