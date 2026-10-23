# H63 — Image Intervention Validity Gate

**Status:** IMPLEMENTED (CPU-only); **model evidence:** NOT YET PRODUCED.
**Parent:** H62 Draft PR #46 at `f45eb66bff43f64c769893bf877a50c4445fc562`.

## Material H62 scientific defect

Before H63, `h62/collapse_probe.py:vision_pathway_probe` declared
`VISION_USED` when a fixed collection of predictions had greater accuracy
against their own target boxes than against a cyclic permutation of target
boxes. It did **not** rerun the model with alternative images. A text-only
position rule, or a prompt with position information, can yield the same
positive outcome without the model looking at pixels. The historical result
could therefore misrepresent a target-association statistic as causal evidence.

H63 retains the useful static matching/permuted-target descriptive figures,
but makes their verdict `VISION_INCONCLUSIVE` under all conditions.
The old `mean_shift_px` name was also misleading: it was a distance from
the static output to the permuted *target*, not a shift between predictions.
It is now `mean_distance_to_permuted_target_px`.

## A stronger separate-inference protocol

`h63/visual_intervention.py` reads newline-delimited paired receipts from
**separate model inference calls**, with **three arms per task**:

1. `reference`: original screenshot, fixed prompt, fixed checkpoint.
2. `repeat`: original screenshot again, same prompt/checkpoint; measures
   baseline stability without changing pixels.
3. `intervention`: truly different screenshot, same prompt/checkpoint and
   a different, non-overlapping task target. Measures changed prediction and
   whether the changed result is correct for the intervened image.

All three carry input image SHA-256, model SHA-256, prompt SHA-256, output
coordinates, target boxes and an explicit valid-output flag. The gate refuses
duplicate case IDs, invalid digests, mismatched prompt/model identities,
intervention images identical to the reference image, changed repeat images,
overlapping targets and malformed coordinates. Undecodable outputs count as
failures, **not** filtered successes.

The positive `PAIRED_VISUAL_SENSITIVITY_OBSERVED` verdict needs at least
eight valid pairs, 95% stability in same-image repeats and 75% paired success.
These are pragmatic initial gates, **not** calibrated p-values, SOTA evidence
or a general model-quality threshold. Smaller, unstable or inconclusive
runs are not promoted.

**Crucial provenance limitation:** self-reported SHA fields are not proof of
what image bytes were really sent to the model. An independent runner must
record the exact input pixel-byte digest, full prompt-template digest,
checkpoint digest, decoding settings, inference timestamps and raw outputs,
and inspect them independently before marking an H63 verdict evidence-backed.
Run the three arms as truly separate inference calls; never synthesize
intervention predictions by swapping target labels or boxes after the fact.
Changing prompts, cropping, resizing or switching model versions between arms
invalidates the visual-sensitivity claim. Repeat-run variability must remain
a first-class negative signal.

## CLI

```bash
python -m unittest tests.test_h62_collapse_probe tests.test_h63_visual_intervention -v
python -m h63.visual_intervention paired-model-inferences.jsonl --output paired-report.json
```

Each JSONL row should have this structure (hashes shown as symbolic
placeholders; real inputs require exactly 64 lowercase hex characters):

```json
{
  "case_id": "unique-id",
  "reference": {
    "image_sha256": "64 hex digits",
    "model_sha256": "64 hex digits",
    "prompt_sha256": "64 hex digits",
    "target_box": [10, 20, 30, 40],
    "point": [20, 30],
    "valid_output": true
  },
  "repeat": {
    "image_sha256": "same image hash as reference",
    "model_sha256": "same model hash",
    "prompt_sha256": "same prompt hash",
    "target_box": [10, 20, 30, 40],
    "point": [20, 30],
    "valid_output": true
  },
  "intervention": {
    "image_sha256": "different image hash",
    "model_sha256": "same model hash",
    "prompt_sha256": "same prompt hash",
    "target_box": [400, 20, 430, 40],
    "point": [420, 30],
    "valid_output": true
  }
}
```

## Experimental policy

- The original H60 0/48 live GUI result remains the baseline; H63 adds no
  measured model score.
- H61 training, Colab allocation watchers, Kaggle restrictions and external
  GPU sessions remain untouched. Nothing here runs or pays for a GPU.
- Actual counterfactual model inference is the NEXT blocked operation until
  compute is available. The implemented module and tests alone do not
  establish visual sensitivity.
- A valid paired-image result must be followed by external held-out and live
  task-success evaluations. Geometry is not full computer use.
- The H63 change is isolated on a stacked Draft PR, never merged automatically.
