# H102 — Original TRAIN geometry and EVAL frame audit

H100 and H101 both scored 0/16 strict GUI successes per arm. H101 improved
native JSON but generated even more degenerate drag outputs; **no evidence of
visual grounding improvement**. H102 is an independent **data/coordinate
pipeline** audit, not another model or training run.

Two strictly separate qualifications:

1. **Original public EVAL**: owner-only SHA-256 checks of source/framed images,
   independent recomputation of source→1024×640 frame dimensions, exact bbox
   scaling and H61 prompt. Ground-truth box-center actions must score 100%
   with the **unchanged H61 scorer**. This only qualifies evaluation geometry,
   never model performance or source-free test results.
2. **Original private TRAIN**: free CPU Kaggle only. SHA-admit the original
   1.76 GB H71 owner-private archive, independently rederive the immutable
   10,000-row H61 curriculum, inspect EVERY original label's bounds,
   coordinate-in-box, prompt/source/ordered identity and degeneracy.
   For the **actual H99 128 TRAIN rows**, SHA-check each original JPEG/PNG and
   actual decoded dimensions against the original label, strict action syntax,
   and original H61 geometry scorer. Emit aggregate counts only.
   **Do not rebalance, mutate, train or publish any TRAIN records.**

The source TRAIN is attached as a private Kaggle dataset, not downloaded
to Windows or committed to GitHub. No GPU required; original 10k/128 source
and selected audit quantities must be independently verifiable.

**H102 does not claim Qwen image processor tensors/tokenization are audited.**
The remaining hard gate is real processor image-token alignment and supervision
mask audit, then subsequent 2B supervised grounding experiments. Any H102
oracle success is *label/geometry correctness*, not model intelligence.
No private model publication, deployment or merges.

## Completed owner-private provider findings — 2026-10-24

- Owner Windows verified **80/80** official public EVAL source→frame media
  SHA-256, image dimensions, source boxes, original H61 prompt and unmodified
  strict H61 **oracle task success**. No model evaluation was run here.
- Owner Kaggle free CPU version 4 (private H71 1,759,484,545-byte source)
  rederived **10,000/10,000** original TRAIN curriculum rows, scanned all
  10,000 original-source label pairs, and SHA/decoded-media/oracle audited the
  actual **128/128 H99 optimizer samples** against the immutable source.
  Checkpoint/dataset plan SHA matched the owner-downloaded H99 training receipt.
- **One strict bounding-box boundary anomaly exists among the 10,000 TRAIN
  labels**: original sheet TRAIN's end_bbox has right edge **equal to image
  width** (a plausible source half-open x2 convention; NOT established as a
  corrupted label). It is **not** among the H99 selected 128. Nothing was
  changed, clipped, filtered or re-trained.
- Because `in_bounds` admits only genuine pixel indexes `x < width`,
  the full 10k strict source-label qualification deliberately returned
  `TRAIN_LABEL_GEOMETRY_GATE_FAIL`, and provider notebook status ERROR.
  This is **not** a failed model inference or a failed H99 checkpoint.
  The subset of actual H99 TRAIN examples separately passed 128/128.
- Owner Windows independent aggregate attestation compared Kaggle's receipt
  to H99's independently downloaded training receipt: **12/12 PASS**,
  explicitly **not full TRAIN qualification**.
- Source model/input/processor visual-token alignment remains **unqualified**.
  The next safe work item is a private CPU-only Qwen3-VL processor+masking audit
  with actual H71 TRAIN screenshots, then genuine fresh-holdout grounding
  experiments. Quarantine or specify handling for the single half-open boundary
  edge *before* any broader training and keep original source immutable.
- Kaggle kernel: `abdulazizshehri/hala-h102-private-original-train-geometry-cpu`.
  No GPU, optimizer, hidden TEST, model publication, paid provider, PR merge or
  private source artifacts were used/published during H102.
