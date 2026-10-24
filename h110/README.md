# H110 — 16 unseen original DragOn scenes, model-guided crop refinement

**Pre-registered experiment, not proof of model gain.**

H107 genuine original strict success: 0/64 for base, H99, H106. H106
produced 63/64 native JSON, while predicted drag displacement median
20px versus target 117px. H108 simplistic transparent pixel grid on 32
fresh scenes gave 0/32 full drags on all four arms; owner independent
rescoring verified raw model outputs, source media and original strict
scorer. Only ruler image *cross-platform byte identity* remains
unverified; don't claim it passed.

H109 retrospective H108-source feasibility: start and end **320×320**
prediction-centered crops captured both true target centers on only
12/32 scenes, whereas **512×512** captured 22/32 centers (21/32 full
target bboxes). All target-box readings were after candidate crop
selection. This is **retrospective development diagnostic, not ability
and not a blinded future test**.

## Frozen H110 A/B

On genuinely fresh DragOn public **EVAL 44–47 in each of 4 domains**:
16 original scenes, never trained on or inferenced by H93–H109.

- **H106 original**: the exact owner SHA-qualified 256-step joint LoRA
  frozen checkpoint, one original H61 source image/prompt, no sampling.
- **H106 two 512 source-pixel crops**: same first-pass H106 drag
  prediction. If it strictly parses, crop one source 512×512 image
  around each **MODEL-PREDICTED** start/end point and enlarge each
  by 2× (source-to-crop mapping invertible); infer a start/end
  local crop position with the same frozen model. Map each local point
  back to original source pixel coordinates. Invalid crop output
  falls back to original model point, never uses a GT box or
  hidden answer. Invalid first-pass actions are unchanged.
  Additional model inference: <=2 per first-pass valid case, not
  claimed as equal compute budget to original baseline.

The crop may fail to include a target. Measure that separately by
examining original reference boxes **after** the model predictions.
Report exact strict original H61 action JSON, start/end hit, complete
drag success, degenerate drags, model valid JSON and ROI candidate
coverage. Do not claim that a parsed cropped coordinate necessarily
improves grounding.

Keep model training weights/checkpoint, original H71 data and original
strict H61 scorer unchanged; local Kaggle owner noncommercial free T4
only, no paid cloud, release, merged PR or model/public data
publication. The crop technique is inspired by iterative refinement,
but is **not the original GUI-Spotlight implementation**.

Explicit conservative acceptance: if both variants remain zero strict
task success, **NO-GO for this H110 crop refinement**. If pilot finds
nonzero gains, independent larger matched source-disjoint comparison
and visual debugging are still needed before a model-wide claim.
