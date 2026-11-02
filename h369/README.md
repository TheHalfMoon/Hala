# H369 — Require real Qwen image-placeholder / vision-patch cardinality

**Research-only, no approved ML TRAIN, source-disjoint blind OOD, weights, or model forward.** Stacked on H368 Draft PR #398 exact pinned source branch. Does not merge into main.

## Scientific / engineering defect

H368 protects full token-prefix equality, assistant-only labels, right-pad input IDs and attention masks, and H366 enforces exact image-grid volume against visual patch count. But a forged prompt+full token pair can silently replace one or more Qwen **image placeholder tokens** with ordinary token IDs while leaving the REAL CPU `pixel_values` tensor and `image_grid_thw` untouched. The previously approved token padding and vision-grid checks still accept this mutually consistent but **model-invalid multimodal encoding**.

The owner CPU-only **real pinned** `Qwen3VLProcessor` with H218 eight SHA-verified PUBLIC tokenizer/processor files at revision `89644892e4d85e24eaac8bacfd4f463576704203` was independently inspected using a fresh solid-color synthetic 960x720 image, never a private image or dataset. It returned `image_grid_thw=[1,44,60]` = 2,640 visual patch rows, `merge_size=2`, and exactly **660** image-placeholder token IDs of value `151655`, one per 2×2 merged visual patch. No model weights or inference used.

## H369 fail-closed fix

A pure-stdlib source-only helper rejects any one-image native source row unless the exact PINNED `<|image_pad|>` token ID appears exactly `pixel_values.shape[0] / (2*2)` times in BOTH prompt prefix and full completion, and never in the completion suffix. The four-row and two-row opt-in CPU batch builders both inherit this guard. It also enforces positive divisible patch counts, exact prefix match, nonboolean integer IDs and input bounds.

Tests use invented token IDs to exercise forged, missing, extra, wrongly shifted, malformed and suffix-added image placeholders. The opt-in real original browser 2-row CPU H364 runner injects a single forged placeholder mutation into BOTH original prefix/full while retaining the true screenshot tensor and grid, verifies H368's old pure token-pad helper would accept it, then requires the new multimodal batch to refuse. Original images and gold remain in TemporaryDirectory and are deleted before aggregate receipt. The H367 four-row real CPU proof remains a separately runnable integration check, not a new learned model evaluation.

## Owner Windows independent evidence (2026-11-02)

- **38/38 stdlib tests PASS on both Python 3.11 and Python 3.12**; no private data and no torch required in the tests.
- **Actual pinned offline Qwen3VLProcessor CPU-only H364 integrated Edge run PASS**: 192/192 original public synthetic candidate media/action SHAs audited; EN click + AR type_text, 2 image grids, 5,280 visual patches, 39 assistant-supervised tokens, 12 masked PAD; original H365/H366 negative guards true and NEW `missing_image_placeholder_refused=true` even though the old pure token-plan accepted the same prefix-consistent forged source row. No weights, no model forward, no optimizer, no GPU or paid service.
- **Actual pinned offline Qwen3VLProcessor CPU-only H367 four-row integrated Edge run PASS** after adding H369 guard: 4 genuine source images, 10,560 visual patches, 78 assistant tokens, 23 masked PAD; independent original+reversed image token/pixel-grid/source-row association verified. Same 192 original source candidate media/action SHAs audited.
- 960x720 solid-color synthetic pinned processor `image_grid=[1,44,60]`, 2,640 patches, 660 `<|image_pad|>` tokens was an **owner CPU-only public processor geometry observation**, not a new original GUI app, and not model quality evidence.

**GitHub CI runs stdlib tests only**, never actual Qwen processor or GPU. H369 exact-head GitHub CI must be qualified separately after opening a DRAFT PR. H368 parent #398 currently has one unrelated inherited **H319 `beacon_11` original-browser failure**, with exact-head 55/56 SUCCESS, and must not be mislabeled as green. No legal TRAIN admission: Issues #384 rights and #385 independent sealed EN/AR OOD still open. No model learning or release.
