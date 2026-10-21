# H53 — Authentic H Company DragOn TRAIN-only Native Drag Examples

**2026-10-21 · Research-only source qualification. No model training, no weights, no neural success claimed.**

## Materially new learned skill supported

H53 is the first **external authentic GUI action-training data** actually imported for Hala. It has **87 distinct original Hcompany/DragOn TRAIN slider JPEG screenshots and 760 image-grounded drag action annotations**, not 760 unique screenshots. Unlike H49/H50/H52 click targets, each action keeps the original **start_bbox, end_bbox, ordered=true** and its distinct image-pixel start and end midpoints; NEVER relabel a drag as a click. Exact source original task JSON and JPEGs are stored only in a **PRIVATE Kaggle** dataset, because TheHalfMoon/Hala GitHub repository is PUBLIC.

Official source: https://huggingface.co/datasets/Hcompany/DragOn , declared Apache-2.0, source commit `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`, file `train/slider.tar`. The user asserts full H Company permission. DragOn's public `eval/` and private benchmark TEST were **never read**. The 7,146,830,848-byte original slider TAR was **NOT** fully downloaded or independently rehashed. Its provider-declared LFS SHA256 `2455d5b4a164d4ff3d6cf41d92d24ad2e27db29e28ea16f5585b55956356a33e` is recorded explicitly as a provider pointer, NOT independently checked full binary integrity.

## Authentic bounded source-first extraction

`h53/dragon_train_shard.py` requests two exactly bounded HTTP **206 Partial Content** windows from the pinned official original TRAIN TAR: one starting at byte 0, one at byte 6,231,427,584, each 64 MiB. The provider returned the exact `Content-Range` and `Content-Length` required; per-window SHA256 and original offsets are in `h53/source_evidence.json`. The source TAR is uncompressed, so valid 512-byte TAR headers can be located in the bounded windows and validated by original header checksum and file type. The code rejects `._` macOS artifacts, symlinks, PAX wrappers, path traversal, truncation, missing original JPEG/JSON source pair, wrong domain, invalid image_id and out-of-image bounding boxes.

**Critically, file ordering differs between `slider/images/*.jpg` and `slider/tasks/*.json`**. Pair only by the source UUID basename and each task's `image_id`, never by encounter order. The true raw original task JSON, its own SHA and original JPEG SHA are preserved in private storage. Generated `train_drag.jsonl` includes one original slider instruction per action, source image dimensions, ordered flag, start/end original boxes and action:
```json
{"type":"drag","start_point":[98,310],"end_point":[208,310],"coordinate_space":"source_image_pixels","ordered":true}
```
(These example numbers above are **unit-test synthetic**, not a quoted original source action.) The original file manifest SHA and all per-item hashes are in the signed source-evidence receipt, without publishing third-party media.

**Authentic results:** 87 matched and geometry-verified source screenshots; 760 individual ordered slider drag examples. The original shard `train_drag.jsonl` SHA256: `9abab62624b45ead5db1ed0c1fdb5d7123f7c761c0a32b8a35f9ecfdebc96c09`; privately stored source ZIP SHA256: `6176dfe2641f76b50759b38c7f8973625850fdc254e9111730ac50abf4cab959`. A sample visual inspection of one source screenshot showed a synthetic video-player slider. **Only one image was manually visually inspected; do not mark complete all-media privacy vetting or full upstream TAR verification.** PII-sensitive target prompt heuristics and valid complete JPEG checks were applied, but do not independently guarantee all source screenshot contents.

## Actual provider retention and required privacy correction

The original Kaggle library `abdulazizshehri/hala-gui-training-library` contained H50/H52 and the new H53 subset, but an API metadata/license update resulted in it **becoming PUBLIC**. Authenticated Kaggle `dataset_list` returned `is_private=False` and unauthenticated Kaggle API `datasets/view` returned HTTP 200. This was **not** an acceptable private-storage state and was not concealed.

**Fixed:** All three source groups were migrated to a separate newly created PRIVATE dataset:

- **ONLY CURRENT Kaggle source**: https://www.kaggle.com/datasets/abdulazizshehri/hala-gui-training-library-private
- New Kaggle provider dataset ID: **12752110**.
- Authenticated Kaggle listing returned `is_private=True`, license `Apache 2.0`.
- Unauthenticated independent API request to the new dataset returned **HTTP 403**.
- Remotely downloaded original **H50 384-row labels**, **H52 288-row labels**, **H53 760-row labels** from this new private dataset, verifying SHA256 `cb1a8529eaf0f82272d75a459c1634766ee5ec7694f6b21affa2045b11d02dd6`, `bca090f0a31aee88fba0cf47aaadaa895ca1fc7ede7c41fbf123d65d6ccb4c73` and `9abab62624b45ead5db1ed0c1fdb5d7123f7c761c0a32b8a35f9ecfdebc96c09`.
- Original historic H50 and H52 compressed source archive SHA256 stayed unchanged; the original H53 ZIP was retained.
- The **old public Kaggle dataset was deleted** successfully with provider response. Final authenticated listing showed ONLY the new private dataset. This action mitigates current exposure, but does not prove that no third party downloaded any prior public version. Original source privacy should be reviewed if sensitive content is discovered.
- Mixed-origin attribution is explicit: original H50/H52 synthetic data **CC0-1.0**; H53 H Company DragOn **Apache-2.0**. Kaggle dataset license field uses `apache-2.0`; this does not change each source's underlying licensing.

**Never reuse the deleted public Kaggle slug**, re-publicize the private dataset, or send source images to the public GitHub. `research/data_library/h53/build_private_kaggle_v3.py` prepares uploads only for the **new private** destination and explicitly sets Apache-2.0 attribution in the Kaggle metadata. Its create-only archive builder and rights claims are not a substitute for checking actual provider privacy after each update.

## Next independent model development gates

- **Stored genuine external TRAIN subset: DONE** (87 original screenshots, 760 original drag labels). Full DragOn 3.56-million-row training corpus is NOT in Hala, and H51 metadata pinning work remains separate.
- **Native drag-action schema in source shard: DONE**. The existing click-only Qwen LoRA trainer does not learn this drag sequence yet. Develop a new bounded drag-completion tokenizer/trainer and evaluate an actual learned action sequence on source-disjoint novel slider/resize tasks without converting drag to click.
- **Complete independent visual privacy review: NOT DONE** (one image preview only).
- **Training GPU / weights / gradient update: NOT DONE**, no free approved GPU allocated.
- **Independent model inference / drag task success / official held-out TEST: NOT DONE**, no win over Holo4 or any competitor claimed.
- H46 original Qwen DEV 0/8 remains negative; original H47 Kaggle private notebook remained queued when checked and was not resubmitted. Never train on Hcompany/trajectories evaluation rollouts, DragOn public `eval/`, OSWorld 2.1 or ScreenSpot test answers.

Model progress checklist and source evidence are in `research/data_library/catalog.json`, `h53/source_evidence.json` and this file. The raw user-requested dataset library is in authenticated **private Kaggle** only; the public GitHub branch contains code, synthetic-only unit tests, receipts and methodology.

## Exact H53 engineering evidence (2026-10-21)

All source tests were run from the isolated Windows `ps/h53-dragon-train-shard` worktree, without changing H51's concurrently staged source inventory in a separate worktree, the user's untracked `.ignore`, H14/H41/H46/H47 frozen experiments, or any unrelated repository.

- **9/9 Windows focused tests PASS**, original `reviews/h53-dragon-focused-20261021.log` SHA256 `a8b91218fe6e7b4956fcf3cfa7c9ed3961008e06d63784abcf8988334558edd7`.
- **644 Windows full-source tests total: 642 PASS, 2 existing SKIP, zero failures/errors**, original `reviews/h53-dragon-full-suite-20261021.log` SHA256 `3de562b8fef04a21deaa371a8b5edead7ff5174275f76cf6f4854c90adf2fb42`; 154.919 seconds, exit 0. The full-source test suite verifies prior H14/H49/H50/H52 and all other existing source gates on this branch.
- Genuine **Graft wiring-only** build and check both PASS on 246 files, 1,992 nodes, 5,351 edges. Build log SHA256 `44399846b5eef99e95059a5d7feef25e7350a644def6e7766998e4eb6e254e6a`, check log SHA256 `7d1c62003ef9b5956b71cfa1c7d7db1595e1522fe191899d7743f79472c472c3`. Deep semantic graph tier **NOT RUN**. D1/PStack/Alibaba LLM-backed semantic reviews **NOT RUN**.
- **Hosted GitHub exact signed-head Python 3.11/3.12 CI: PENDING** until its new commit is published and checks finish; local Windows results do not imply hosted success.
- Private provider evidence: exact original **H50 384**, **H52 288** and **H53 760** TRAIN labels re-downloaded and SHA-verified from migrated Kaggle ID `12752110`. Authenticated listing reported `is_private=True`, unauthenticated API GET reported `403`; prior public dataset deleted and absent from authenticated listing. No claims about earlier public-version downloads.
