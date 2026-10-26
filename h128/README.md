# H128 — fresh original TRAIN16 pretrained GUI zoom-refinement pilot

H127: 64 genuine new original TRAIN screenshots and private blind source labels
have been separated, but actual independent human identifiability review is
still pending. **H127 is NOT a measured human or model success.**

H125: two real pretrained 2B native GUI baselines each scored 0/64 exact
full drag. Their error often spans coarse regions, so any two-stage method
must measure both **wrong region selection** and **within-region refinement**.

This H128 exploratory *inference-only* trial follows visual test-time scaling
research (RegionFocus, ICCV 2025) and visual feedback refinement (Microsoft
2026). Neither paper proves H128 will work.

## Frozen experimental design

- Derive 16 entirely **NEW original H71 TRAIN** screenshots, four from each
  family, excluding ALL 2,688 original sources used or withheld by H105,
  H123, H124, H125, H126, by both source ID and image SHA. No public
  EVAL/TEST or any earlier withheld scenes are reused.
- Use one frozen pretrained GUI grounding model:
  `showlab/ShowUI-2B` at revision
  `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` (MIT license).
- For each START and END point, run its native single-shot prompt on the
  original image. From that **model-only** coordinate, crop a half-width,
  half-height window using the exact deterministic, border-safe transform
  in `zoom.py`. Request the *same* point from the *same* frozen model on
  the crop and map the result back into original screenshot pixels.
- Total **64 real pretrained neural calls** (16 scenes × 2 roles × 2
  passes), paired direct/zoom evaluation on exactly the same 16 screenshots.
- No ground-truth box, original correct coordinates, human mark or H127
  labels are allowed in a model prompt or crop selection. Original ground
  truth is used for scoring only after both arms finish.
- Preserve separate START/END strict native bbox hits, full drag successes,
  coarse quadrant accuracy and whether model-selected crop included target.
  Compare paired baseline versus crop-refine; do not compare 16-scene
  percentages directly to H125's different 64-scene set.
- A 16-scene pilot may expose a failure mode or a promising direction but
  **CANNOT establish population-level improvement or qualify release**.
  A credible follow-up requires heldout, adequately powered evaluation,
  task identifiability validation, and predeclared acceptance thresholds.

## Execution and resource controls

- Owner-private Kaggle free T4 only; exact model/revision/source/ML package
  pins. If unavailable, fail closed. No fallback paid compute, no
  publication/deployment, optimizer steps, training, or PR merge.
- Original image pixels, instructions, case IDs, model responses and weights
  remain in the **private owner Kaggle result directory** and are never
  committed or posted to public GitHub.
- Source and score summaries preserve provenance and explicit negative
  outcomes. Keep this PR Draft.

Owner-independent evidence: the separate private **CPU** kernel in
`kernel-source-metadata.json` recreates the exact same 16 original TRAIN
sources by frozen schedule and actual image SHA, exports source identities
and START/END boxes privately, and makes no model calls. Once the GPU
kernel has finished, run `h128.owner_replay.run_files(source, raw, report,
private_output)` on Windows to independently parse each native direct/zoom
model response, recompute every crop from the model's baseline alone, verify
SHA/source/model identity, and recompute paired score. No private sources,
predictions, labels or reviewer results are to be posted in the PR.

Tests:
`python -m unittest discover -s tests -p "test_h128*.py" -v`
